"""
Visualisation interactive des predictions HWM.

Affiche pour chaque echantillon : l'image originale, le texte ground truth,
le texte predit par le modele, et le CER individuel. Navigation page par page
avec fleches ou boutons Prev/Next.

Usage:
    # Mode validation (split seedé, comme recognize.py)
    python visualize.py --model hwm_v17.pt --model-version v17
    python visualize.py --model hwm_v17.pt --model-version v17 --sort-by-cer --top-n 20

    # Mode fichier : fichier .xml unique ou répertoire de .xml
    python visualize.py --model hwm_v17.pt --model-version v17 --alto-file page.xml
    python visualize.py --model hwm_v17.pt --model-version v17 --alto-file /media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie/Alto/<dir>
"""

import sys
import os
import argparse
import glob
import random

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from functools import partial

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy
from matplotlib.widgets import Button

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from model_registry import get_spec, known_versions, default_train_args
from data_alto import (
    AltoLineDataset, _parse_page, build_alphabet,
    collate_alto_fn, collate_alto_v5_fn,
)
from recognize import ctc_greedy_decode_conf, align_pred_gt, levenshtein

# Colormap commune (rouge = faible confiance, vert = haute confiance)
_CONF_CMAP = plt.get_cmap("RdYlGn")


class _DirectDataset(Dataset):
    """Dataset léger construit à partir de samples déjà parsés (pas de cache)."""

    def __init__(self, samples, img_height):
        self.samples = samples
        self.img_height = img_height
        self.augment = False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, text = self.samples[idx]
        img = torch.from_numpy(arr.copy()) / 255.0
        img = 1.0 - img
        return img, text

    def filter_unlearnable(self, char_to_idx, width_stride=8, min_frames_per_char=1.0):
        """Identique à AltoLineDataset.filter_unlearnable."""
        import unicodedata as _ud
        kept = []
        for arr, text in self.samples:
            encoded_len = sum(
                1 for c in _ud.normalize("NFD", text) if c in char_to_idx
            )
            if encoded_len == 0:
                continue
            frames = arr.shape[1] // width_stride
            if frames >= min_frames_per_char * encoded_len:
                kept.append((arr, text))
        removed = len(self.samples) - len(kept)
        self.samples = kept
        return removed, len(kept)


def _load_model(args, device):
    """Charge le checkpoint, construit le modele, retourne (model, char_to_idx, idx_to_char, spec, saved_config)."""
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    saved_config = ckpt.get("config", {})

    ckpt_char_to_idx = ckpt.get("char_to_idx")
    if ckpt_char_to_idx:
        char_to_idx = ckpt_char_to_idx
        idx_to_char = {v: k for k, v in char_to_idx.items()}
        print(f"Alphabet from checkpoint: {len(char_to_idx)} characters")
    else:
        char_to_idx, idx_to_char = build_alphabet(args.alto_dirs)
        print(f"Alphabet from data: {len(char_to_idx)} characters")

    ckpt_num_classes = saved_config.get("num_classes")
    num_classes = ckpt_num_classes if ckpt_num_classes else len(char_to_idx) + 1

    ver = args.model_version
    spec = get_spec(ver)
    model = spec.builder(default_train_args(), num_classes).to(device)

    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if result.missing_keys:
        print(f"  Warning: missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Warning: unexpected keys: {result.unexpected_keys}")
    model.eval()
    print(f"Model {ver}: {model.count_parameters():,} params")

    return model, char_to_idx, idx_to_char, spec, saved_config


def _build_collate(spec, char_to_idx):
    return (
        partial(collate_alto_v5_fn, char_to_idx=char_to_idx)
        if spec.collate_style == "v5"
        else partial(
            collate_alto_fn,
            window_size=spec.window_size,
            stride=spec.stride,
            char_to_idx=char_to_idx,
        )
    )


def _load_val_split(args):
    """Mode validation : meme pipeline que recognize.py (split seedé)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, char_to_idx, idx_to_char, spec, saved_config = _load_model(args, device)

    img_h = saved_config.get("img_height", spec.img_height)
    no_gt = getattr(args, "no_gt", False)
    dataset = AltoLineDataset(
        args.alto_dirs, img_height=img_h, keep_empty=no_gt)

    if no_gt:
        # Séparer GT et no-GT, splitter les GT comme train.py, garder toutes les no-GT
        gt_samples = [(a, t) for a, t in dataset.samples if t and t.strip()]
        no_gt_samples = [(a, t) for a, t in dataset.samples
                         if not t or not t.strip()]

        # filter_unlearnable sur GT seulement
        if args.min_frames_per_char > 0.0 and gt_samples:
            gt_ds = _DirectDataset(gt_samples, img_h)
            removed, kept = gt_ds.filter_unlearnable(
                char_to_idx, width_stride=spec.cnn_width_stride,
                min_frames_per_char=args.min_frames_per_char)
            gt_samples = gt_ds.samples
            print(f"Filtered {removed} unlearnable GT lines; {kept} GT remain")

        n_gt = len(gt_samples)
        if n_gt > 0:
            train_size = int(0.8 * n_gt)
            val_size = n_gt - train_size
            _train_split, val_split = random_split(
                range(n_gt), [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            if args.split == "val":
                shown = [gt_samples[i] for i in val_split.indices] + no_gt_samples
            else:
                shown = gt_samples + no_gt_samples
        else:
            shown = no_gt_samples

        eval_ds = _DirectDataset(shown, img_h)
        n_val_gt = (len(val_split.indices) if (n_gt > 0 and args.split == "val")
                    else n_gt)
        print(f"  --no-gt: {len(no_gt_samples)} unannotated + "
              f"{n_val_gt} {args.split} GT = {len(shown)} shown")
        collate = _collate_no_gt_fn
        return model, eval_ds, collate, idx_to_char, device, spec

    # Chemin normal (avec GT)
    if args.min_frames_per_char > 0.0:
        removed, kept = dataset.filter_unlearnable(
            char_to_idx,
            width_stride=spec.cnn_width_stride,
            min_frames_per_char=args.min_frames_per_char,
        )
        print(f"Filtered {removed} unlearnable lines; {kept} remain")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    eval_ds = val_ds if args.split == "val" else train_ds
    print(f"Split: {args.split} ({len(eval_ds)} lines)")

    collate = _build_collate(spec, char_to_idx)
    return model, eval_ds, collate, idx_to_char, device, spec


def _collate_no_gt_fn(batch):
    """Collate pour mode sans GT : pad images, targets factices vides.
    Retourne le meme 5-tuple que collate_alto_v5_fn pour compatibilité."""
    imgs = []
    input_lengths = []
    for img, text in batch:
        imgs.append(img)
        input_lengths.append(img.shape[1] // 8)

    B = len(imgs)
    H = imgs[0].shape[0]
    W_max = max(img.shape[1] for img in imgs)

    padded = torch.zeros(B, H, W_max)
    for i, img in enumerate(imgs):
        padded[i, :, :img.shape[1]] = img

    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    raw_texts = [""] * B
    return padded, torch.tensor([], dtype=torch.long), input_lengths, \
        torch.zeros(B, dtype=torch.long), raw_texts


def _load_alto_file(args):
    """Mode fichier : parse un fichier .xml ou tous les .xml d'un répertoire."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, char_to_idx, idx_to_char, spec, saved_config = _load_model(args, device)

    img_h = saved_config.get("img_height", spec.img_height)

    path = args.alto_file
    if os.path.isfile(path):
        xml_files = [path]
    elif os.path.isdir(path):
        xml_files = sorted(
            f for f in glob.glob(os.path.join(path, "*.xml"))
            if os.path.basename(f) != "METS.xml"
        )
    else:
        print(f"Error: {path} is neither a file nor a directory")
        sys.exit(1)

    all_samples = []
    keep_empty = getattr(args, "no_gt", False)
    for xml_path in xml_files:
        page_samples, _chars = _parse_page((xml_path, img_h, 4000, keep_empty))
        all_samples.extend(page_samples)
        print(f"  {os.path.basename(xml_path)}: {len(page_samples)} lines")

    if not all_samples:
        print("No lines found")
        sys.exit(1)

    print(f"Total: {len(all_samples)} lines from {len(xml_files)} file(s)")

    split = getattr(args, "split", "val")
    mfc = getattr(args, "min_frames_per_char", 0.0)
    width_stride = spec.cnn_width_stride

    if keep_empty:
        # Mode --no-gt : exclure le split d'entraînement des lignes AVEC GT.
        # Les lignes sans GT n'ont jamais été vues à l'entraînement, donc
        # toutes sont affichées. On splitte les lignes GT avec le même seed 42
        # que _load_val_split / train.py pour cohérence.
        gt_samples = [(a, t) for a, t in all_samples if t and t.strip()]
        no_gt_samples = [(a, t) for a, t in all_samples if not t or not t.strip()]
        n_gt_total = len(gt_samples)

        # Appliquer filter_unlearnable sur les lignes GT AVANT le split
        # pour reproduire exactement la partition de train.py
        if mfc > 0.0 and gt_samples:
            gt_ds = _DirectDataset(gt_samples, img_h)
            removed, kept = gt_ds.filter_unlearnable(
                char_to_idx, width_stride=width_stride, min_frames_per_char=mfc)
            gt_samples = gt_ds.samples
            print(f"  Filtered {removed} unlearnable GT lines; {kept} GT remain")

        n_gt = len(gt_samples)
        if n_gt > 0:
            train_size = int(0.8 * n_gt)
            val_size = n_gt - train_size
            _train_split, val_split = random_split(
                range(n_gt), [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            if split == "val":
                shown = [gt_samples[i] for i in val_split.indices] + no_gt_samples
            else:
                shown = gt_samples + no_gt_samples
        else:
            shown = no_gt_samples

        eval_ds = _DirectDataset(shown, img_h)
        if split == "val":
            n_val_gt = len(val_split.indices) if n_gt > 0 else 0
        else:
            n_val_gt = n_gt
        print(f"  --no-gt: {len(no_gt_samples)} unannotated + "
              f"{n_val_gt} {split} GT "
              f"= {len(shown)} shown")
        collate = _collate_no_gt_fn
    else:
        # Même split seedé que _load_val_split (80/20, seed 42)
        dataset = _DirectDataset(all_samples, img_h)

        # Appliquer filter_unlearnable AVANT le split (comme train.py)
        if mfc > 0.0:
            removed, kept = dataset.filter_unlearnable(
                char_to_idx, width_stride=width_stride, min_frames_per_char=mfc)
            print(f"  Filtered {removed} unlearnable lines; {kept} remain")

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        eval_ds = val_ds if split == "val" else train_ds
        print(f"Split: {split} ({len(eval_ds)} lines of {len(dataset)})")
        collate = _build_collate(spec, char_to_idx)
    return model, eval_ds, collate, idx_to_char, device, spec


def _predict_all(model, eval_ds, collate, idx_to_char, device, spec,
                 use_beam=False, lm=None, beam_width=20, lm_weight=0.3,
                 workers=0):
    """Forward pass sur tout le split, retourne (images, gt, pred, cer) par echantillon.

    Les images sont extraites directement du tensor du batch (apres collate),
    ce qui garantit l'alignement 1:1 avec les predictions meme si le collate
    a filtre des echantillons (OOV, CTC impossible).

    Si use_beam=True, lance aussi un CTC prefix beam search (optionnellement
    biaise par un n-gram LM) et stocke le resultat dans chaque sample.
    Si workers>1, le beam search utilise un Pool multiprocessing (comme
    recognize.py) — utile sur un gros CPU pour accelerer le decode.
    """
    loader = DataLoader(
        eval_ds,
        batch_size=32,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    use_amp = device.type == "cuda"
    width_stride = spec.cnn_width_stride

    samples = []
    beam_tasks = []  # (seq_np, raw_text) collected for pool decode

    with torch.no_grad():
        for batch in loader:
            img_seqs, targets, input_lengths, target_lengths, raw_texts = batch
            img_seqs_dev = img_seqs.to(device, non_blocking=True)
            input_lengths_cpu = input_lengths.clone()

            with torch.amp.autocast("cuda", enabled=use_amp):
                _, _, ctc_logits = model(
                    img_seqs_dev, input_lengths=input_lengths.to(device)
                )

            decoded = ctc_greedy_decode_conf(
                ctc_logits.cpu().float(), input_lengths_cpu, idx_to_char
            )

            # Beam decode: inline (single or small batch) or deferred to pool
            beam_texts = None
            if use_beam:
                if workers > 1:
                    # Collect logits for the pool — decode after the loop
                    logits_np = ctc_logits.cpu().float().numpy()
                    for b in range(len(raw_texts)):
                        L = int(input_lengths_cpu[b])
                        beam_tasks.append((logits_np[b, :L, :], raw_texts[b]))
                else:
                    from beam_decode import ctc_beam_search_decode
                    beam_texts = ctc_beam_search_decode(
                        ctc_logits.cpu(), input_lengths_cpu, idx_to_char,
                        lm=lm, beam_width=beam_width, lm_weight=lm_weight,
                    )

            img_seqs_cpu = img_seqs.cpu()
            for i, (dec, gt) in enumerate(zip(decoded, raw_texts)):
                pred = dec["text"]
                has_gt = bool(gt)
                if has_gt:
                    gt_len = max(len(gt), 1)
                    cer = levenshtein(pred, gt) / gt_len
                    wrong = align_pred_gt(pred, gt)
                else:
                    gt_len = 1
                    cer = None
                    wrong = [False] * len(pred)

                real_w = input_lengths_cpu[i].item() * width_stride
                img_t = img_seqs_cpu[i, :, :real_w]
                img_arr = ((1.0 - img_t) * 255).numpy().astype("uint8")

                if use_beam and beam_texts is not None:
                    bt = beam_texts[i]
                    beam_cer = levenshtein(bt, gt) / gt_len if has_gt else None
                else:
                    bt = None
                    beam_cer = None

                samples.append({
                    "image": img_arr,
                    "gt": gt,
                    "pred": pred,
                    "cer": cer,
                    "char_confs": dec["char_confs"],
                    "frame_conf": dec["frame_conf"],
                    "line_conf": dec["line_conf"],
                    "wrong": wrong,
                    "beam": bt,
                    "beam_cer": beam_cer,
                })

    # Multiprocessing beam decode (all logits collected, decode in one pool)
    if use_beam and workers > 1 and beam_tasks:
        from beam_decode import _init_worker, _worker_decode
        import multiprocessing as mp

        n_chars = beam_tasks[0][0].shape[-1]
        chars = [idx_to_char.get(i, "") for i in range(n_chars)]
        n_procs = min(workers, len(beam_tasks))
        chunksize = max(1, len(beam_tasks) // (n_procs * 4))

        print(f"  Beam decoding {len(beam_tasks)} samples with {n_procs} workers...",
              flush=True)
        with mp.Pool(
            n_procs,
            initializer=_init_worker,
            initargs=(lm, chars, beam_width, lm_weight, 15, 0),
        ) as pool:
            beam_results = pool.map(_worker_decode, beam_tasks, chunksize=chunksize)

        # Assign beam results back to samples (order preserved)
        for s, bt in zip(samples, beam_results):
            s["beam"] = bt
            if s["gt"]:
                s["beam_cer"] = levenshtein(bt, s["gt"]) / max(len(s["gt"]), 1)
            else:
                s["beam_cer"] = None

    torch.cuda.empty_cache()
    return samples


def _select_samples(samples, args):
    """Sous-echantillonne et/ou trie les resultats selon les options CLI."""
    n = len(samples)

    if args.max_samples < n:
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(n), args.max_samples))
        samples = [samples[i] for i in indices]

    if args.sort_by_cer:
        samples.sort(key=lambda s: s["cer"] if s["cer"] is not None else 1.0,
                     reverse=True)

    if args.top_n is not None:
        samples = samples[: args.top_n]

    return samples


_PER_PAGE = 6
_FONTSIZE = 9


def _conf_color(conf):
    """Couleur RdYlGn assombrie pour rester lisible sur fond blanc."""
    r, g, b, _ = _CONF_CMAP(float(conf))
    f = 0.82
    return (r * f, g * f, b * f)


class _Viewer:
    """Navigateur matplotlib page par page."""

    def __init__(self, samples, per_page=_PER_PAGE, has_beam=False):
        self.samples = samples
        self.per_page = per_page
        self.has_beam = has_beam
        self.page = 0
        self.total_pages = max(1, -(-len(samples) // per_page))

        fig_h = per_page * 3.3 if has_beam else per_page * 2.9
        self.fig, self.axes = plt.subplots(
            per_page, 1, figsize=(16, fig_h)
        )
        if per_page == 1:
            self.axes = [self.axes]
        plt.subplots_adjust(bottom=0.10, top=0.93, hspace=1.0)

        ax_prev = self.fig.add_axes([0.15, 0.01, 0.15, 0.035])
        ax_next = self.fig.add_axes([0.70, 0.01, 0.15, 0.035])
        self.btn_prev = Button(ax_prev, "\u2190 Prev")
        self.btn_next = Button(ax_next, "Next \u2192")
        self.btn_prev.on_clicked(self._prev)
        self.btn_next.on_clicked(self._next)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        # Le texte PRED est positionne en pixels/points ; un redimensionnement de
        # la fenetre deplace les boites d'axes, donc on retrace pour rester aligne.
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)
        self._drawing = False

        self._draw()

    def _on_resize(self, _event=None):
        if self._drawing:
            return
        self._draw()

    def _draw(self):
        self._drawing = True
        try:
            self._draw_impl()
        finally:
            self._drawing = False

    def _draw_impl(self):
        start = self.page * self.per_page
        page_samples = self.samples[start : start + self.per_page]

        cer_vals = [s["cer"] for s in page_samples if s["cer"] is not None]
        mean_cer = sum(cer_vals) / max(len(cer_vals), 1) if cer_vals else None
        mean_conf = sum(s["line_conf"] for s in page_samples) / max(len(page_samples), 1)
        cer_txt = f"Page CER: {mean_cer:.1%}  |  " if mean_cer is not None else ""
        title = (
            f"Page {self.page + 1}/{self.total_pages}  |  "
            f"Samples {start + 1}\u2013{min(start + self.per_page, len(self.samples))}"
            f"/{len(self.samples)}  |  "
            f"{cer_txt}Page conf: {mean_conf:.1%}  |  "
            f"confiance: rouge=faible \u2192 vert=haute"
        )
        self.fig.suptitle(title, fontsize=12, fontweight="bold")

        # --- passe 1 : images + frises (sans texte) ---
        drawn = []
        for i, ax in enumerate(self.axes):
            ax.clear()
            if i >= len(page_samples):
                ax.axis("off")
                continue

            s = page_samples[i]
            img = s["image"]
            H, real_w = img.shape[0], img.shape[1]

            # image de la ligne (en haut)
            ax.imshow(img, cmap="gray", aspect="equal", vmin=0, vmax=255,
                      interpolation="nearest", extent=(0, real_w, H, 0))

            # frise de confiance par frame (juste en dessous)
            fc = np.asarray(s["frame_conf"], dtype=float)
            if fc.size > 0:
                fh = max(6.0, H * 0.18)
                gap = max(2.0, H * 0.05)
                ax.imshow(fc[None, :], cmap=_CONF_CMAP, vmin=0.0, vmax=1.0,
                          aspect="equal", interpolation="nearest",
                          extent=(0, real_w, H + gap + fh, H + gap))
                total_h = H + gap + fh
            else:
                total_h = H

            ax.set_xlim(0, max(real_w, 1))
            ax.set_ylim(total_h, 0)
            # ratio pixel preserve (pas d'etirement), boite ancree en haut a gauche
            ax.set_aspect("equal", adjustable="box", anchor="NW")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            drawn.append((ax, s))

        # Fige la mise en page : avec aspect='equal' la boite de chaque axe est
        # redimensionnee au trace. On force ce calcul avant de mesurer/placer le
        # texte, sinon les largeurs de caracteres seraient fausses (chevauchement).
        self.fig.canvas.draw()

        # --- passe 2 : texte (GT + PRED colore) sur des boites finalisees ---
        for ax, s in drawn:
            if s["gt"]:
                gt_color = "#2e7d32" if s["cer"] == 0.0 else "#c62828"
                ax.text(0.0, -0.07, f"GT:   {s['gt']}", transform=ax.transAxes,
                        fontsize=_FONTSIZE, color=gt_color, fontfamily="monospace",
                        va="top", ha="left", clip_on=False)
            else:
                ax.text(0.0, -0.07, "GT:   [pas de transcription]", transform=ax.transAxes,
                        fontsize=_FONTSIZE, color="#90a4ae", fontfamily="monospace",
                        va="top", ha="left", clip_on=False)
            self._draw_pred_line(ax, -0.27, s, _FONTSIZE,
                                 show_errors=not self.has_beam)

            if s.get("beam") is not None:
                if s["beam_cer"] is not None:
                    beam_color = "#2e7d32" if s["beam_cer"] == 0.0 else "#c62828"
                    suffix = f"   (CER {s['beam_cer']:.0%})"
                else:
                    beam_color = "#37474f"
                    suffix = ""
                ax.text(0.0, -0.47,
                        f"BEAM: {s['beam']}{suffix}",
                        transform=ax.transAxes, fontsize=_FONTSIZE,
                        color=beam_color, fontfamily="monospace",
                        va="top", ha="left", clip_on=False)

        self.fig.canvas.draw_idle()

    def _draw_pred_line(self, ax, y_ax, s, fontsize, show_errors=True):
        """Affiche la ligne PRED : chaque caractere colore selon sa confiance,
        les caracteres faux (vs GT) soulignes en rouge, et un score de ligne.
        Si show_errors=False, le soulignement est desactive (utile quand le
        beam search offre sa propre comparaison GT vs BEAM).

        Chaque lettre est avancee en POINTS depuis le bord gauche de l'axe via
        offset_copy(..., units='points', fig=...), qui recalcule l'offset au
        moment du trace. L'espacement reproduit donc exactement celui d'une
        chaine monospace normale (alignement avec la GT, pas de chevauchement)
        quelle que soit la taille de la fenetre ou le DPI."""
        renderer = self.fig.canvas.get_renderer()
        dpi = self.fig.dpi

        prefix = "PRED: "  # meme largeur que "GT:   " -> texte aligne
        t = ax.text(0.0, y_ax, prefix, transform=ax.transAxes, color="#37474f",
                    fontfamily="monospace", fontsize=fontsize, va="top", ha="left",
                    clip_on=False)
        t.draw(renderer)
        e0 = t.get_window_extent(renderer=renderer)
        ax_bb = ax.get_window_extent(renderer=renderer)

        adv_px = e0.width / max(len(prefix), 1)         # avance / caractere (px)
        adv_pts = adv_px * 72.0 / dpi                   # ... en points
        prefix_pts = e0.width * 72.0 / dpi
        lh = e0.height / ax_bb.height                   # hauteur de ligne (fraction)
        y_ul = y_ax - lh * 1.5                          # souligne juste sous le texte
        adv_frac = adv_px / ax_bb.width                 # longueur du souligne (fraction)

        def at(off_pts):
            return offset_copy(ax.transAxes, fig=self.fig, x=off_pts, y=0.0,
                               units="points")

        for j, (ch, conf, wr) in enumerate(zip(s["pred"], s["char_confs"], s["wrong"])):
            tr = at(prefix_pts + j * adv_pts)
            ax.text(0.0, y_ax, ch, transform=tr, color=_conf_color(conf),
                    fontfamily="monospace", fontsize=fontsize, va="top", ha="left",
                    clip_on=False)
            if show_errors and wr:
                ln = Line2D([0.0, adv_frac], [y_ul, y_ul], transform=tr,
                            color="#d50000", lw=1.6, clip_on=False,
                            solid_capstyle="butt")
                ax.add_line(ln)

        # score de confiance + CER en fin de ligne (ne perturbe pas l'alignement)
        conf_txt = f"{s['line_conf']:.0%}" if s["char_confs"] else "--"
        if s["cer"] is not None:
            suffix = f"   (conf {conf_txt} | CER {s['cer']:.0%})"
        else:
            suffix = f"   (conf {conf_txt})"
        ax.text(0.0, y_ax, suffix, transform=at(prefix_pts + len(s["pred"]) * adv_pts),
                color="#90a4ae", fontfamily="monospace", fontsize=fontsize,
                va="top", ha="left", clip_on=False)

    def _prev(self, _event=None):
        if self.page > 0:
            self.page -= 1
            self._draw()

    def _next(self, _event=None):
        if self.page < self.total_pages - 1:
            self.page += 1
            self._draw()

    def _on_key(self, event):
        if event.key in ("right", "n"):
            self._next()
        elif event.key in ("left", "p"):
            self._prev()

    def show(self):
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualisation interactive des predictions HWM"
    )
    parser.add_argument("--model", default="hwm_v17.pt", help="Model checkpoint")
    parser.add_argument("--model-version", choices=known_versions(), default="v17")
    parser.add_argument(
        "--alto-dirs", nargs="+", default=config.ALTO_DIRS,
        help="Repertoires ALTO (mode validation, avec split)",
    )
    parser.add_argument(
        "--alto-file", default=None,
        help="Fichier .xml ALTO unique ou répertoire de .xml "
             "(pas de split, toutes les lignes sont affichees)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--split", choices=["val", "train"], default="val",
        help="Split a evaluer en mode validation (default: val)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=50,
        help="Nombre d'echantillons a afficher (default: 50)",
    )
    parser.add_argument(
        "--sort-by-cer", action="store_true",
        help="Trier par CER descendant (pires cas d'abord)",
    )
    parser.add_argument(
        "--top-n", type=int, default=None,
        help="Avec --sort-by-cer, n'afficher que les N pires",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed pour le sous-echantillonnage (default: 42)",
    )
    parser.add_argument(
        "--min-frames-per-char", type=float, default=0.0,
        help="Doit correspondre a la valeur utilisee a l'entrainement (default: 0.0)",
    )
    parser.add_argument(
        "--beam-search", action="store_true",
        help="Activer le CTC beam search (en plus du greedy).",
    )
    parser.add_argument(
        "--lm-path", default=None,
        help="Chemin vers un n-gram LM (.pkl) pour biaiser le beam search.",
    )
    parser.add_argument(
        "--beam-width", type=int, default=20,
        help="Largeur du beam (default: 20).",
    )
    parser.add_argument(
        "--lm-weight", type=float, default=0.3,
        help="Poids du LM dans le score du beam (default: 0.3).",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Worker processes pour le beam search (0=sequentiel). "
             "Au-dela de 1, utilise un Pool multiprocessing comme recognize.py.",
    )
    parser.add_argument(
        "--no-gt", action="store_true",
        help="Mode sans ground truth : predit toutes les lignes meme sans "
             "transcription. Necessite --alto-file.",
    )
    args = parser.parse_args()

    lm = None
    if args.lm_path:
        from beam_decode import CharNgramLM
        lm = CharNgramLM(args.lm_path)
        print(f"Loaded n-gram LM (order {lm.order}) from {args.lm_path}")

    if args.alto_file:
        model, eval_ds, collate, idx_to_char, device, spec = _load_alto_file(args)
        args.max_samples = len(eval_ds)
    else:
        model, eval_ds, collate, idx_to_char, device, spec = _load_val_split(args)

    print("Running predictions...")
    samples = _predict_all(
        model, eval_ds, collate, idx_to_char, device, spec,
        use_beam=args.beam_search, lm=lm,
        beam_width=args.beam_width, lm_weight=args.lm_weight,
        workers=args.workers,
    )
    print(f"Collected {len(samples)} predictions")

    samples = _select_samples(samples, args)
    print(f"Displaying {len(samples)} samples ({'sorted by CER' if args.sort_by_cer else 'random order'})")

    cer_vals = [s["cer"] for s in samples if s["cer"] is not None]
    if cer_vals:
        overall_cer = sum(cer_vals) / len(cer_vals)
        print(f"Subset CER: {overall_cer:.1%} ({len(cer_vals)} with GT)")
    else:
        print(f"No GT available for {len(samples)} samples (no CER computed)")
    if args.beam_search:
        beam_vals = [s["beam_cer"] for s in samples if s["beam_cer"] is not None]
        if beam_vals and cer_vals:
            beam_cer = sum(beam_vals) / len(beam_vals)
            print(f"Subset beam CER: {beam_cer:.1%} (greedy: {sum(cer_vals)/len(cer_vals):.1%})")

    viewer = _Viewer(samples, has_beam=args.beam_search)
    viewer.show()
