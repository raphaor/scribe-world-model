#!/usr/bin/env python3
"""
Outil d'annotation interactive pour g\u00e9n\u00e9rer/corriger la ground truth ALTO.

Mode annotate (d\u00e9faut) : affiche les lignes SANS ground truth et propose
la pr\u00e9diction du mod\u00e8le comme texte \u00e0 valider/\u00e9diter.

Mode review : affiche les lignes AVEC ground truth existante pour correction.

Les *_gt.xml sont la source de v\u00e9rit\u00e9 : ils sont charg\u00e9s automatiquement
si pr\u00e9sents. Aucun flag n\'est n\u00e9cessaire pour reprendre une session.
\u00c0 la sauvegarde, un *_gt.xml est cr\u00e9\u00e9 pour chaque fichier source,
contenant la GT initiale + les ajustements de la session.

Usage:
    python annotate.py --model hwm_v17.pt --model-version v17 \\
        --alto "D:/OCR/alto_dir" --beam-search

    python annotate.py --model hwm_v17.pt --model-version v17 \\
        --alto page.xml --mode review
"""

import sys
import os
import argparse
import glob
import json

import torch

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_alto import _parse_page
from model_registry import known_versions
from visualize import _load_model, _predict_all, _DirectDataset

try:
    from lxml import etree
except ImportError:
    print("Error: lxml is required. Install with: pip install lxml")
    sys.exit(1)


PER_PAGE = 6
_FONTSIZE = 8


# ─── Collate ────────────────────────────────────────────────────────────

def _annotate_collate(batch):
    """Collate pour annotation : pad images, pr\u00e9serve raw_texts, pas de filtrage.

    Contrairement \u00e0 ``collate_alto_v5_fn``, cette fonction ne filtre aucun
    \u00e9chantillon (pas de contr\u00f4le OOV ni CTC T>=L), garantissant un alignement
    1:1 entre les pr\u00e9dictions et les m\u00e9tadonn\u00e9es de ligne.
    """
    imgs = []
    input_lengths = []
    raw_texts = []
    for img, text in batch:
        imgs.append(img)
        input_lengths.append(img.shape[1] // 8)
        raw_texts.append(text)

    B = len(imgs)
    H = imgs[0].shape[0]
    W_max = max(img.shape[1] for img in imgs)

    padded = torch.zeros(B, H, W_max)
    for i, img in enumerate(imgs):
        padded[i, :, :img.shape[1]] = img

    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    targets = torch.tensor([], dtype=torch.long)
    target_lengths = torch.zeros(B, dtype=torch.long)

    return padded, targets, input_lengths, target_lengths, raw_texts


# ─── Chemins de sortie ──────────────────────────────────────────────────

def _derive_output_path(xml_path):
    """page.xml -> page_gt.xml | page_gt.xml -> page_gt.xml (deja GT)"""
    base, ext = os.path.splitext(xml_path)
    if base.endswith("_gt"):
        return xml_path
    return base + "_gt" + ext


def _derive_progress_path(xml_path):
    """Base sur l'output, pas l'input -> meme fichier dans les deux modes."""
    return _derive_output_path(xml_path).replace(".xml", ".progress.json")


def _derive_original_path(xml_path):
    """page_gt.xml -> page.xml | page.xml -> page.xml"""
    base, ext = os.path.splitext(xml_path)
    if base.endswith("_gt"):
        return base[:-3] + ext
    return xml_path


# ─── Progress tracking ──────────────────────────────────────────────────

def load_progress(progress_path):
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"accepted": {}, "last_page": 0}


def save_progress(progress_path, accepted, last_page):
    data = {"accepted": accepted, "last_page": last_page}
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── \u00c9criture XML ───────────────────────────────────────────────────────

def _read_gt_texts(gt_path):
    """Lit {line_id: text} depuis un fichier *_gt.xml.

    Retourne un dict vide si le fichier n'existe pas.
    """
    if not os.path.exists(gt_path):
        return {}
    tree = etree.parse(gt_path)
    texts = {}
    for tl in tree.iter():
        if not isinstance(tl.tag, str):
            continue
        if tl.tag.split('}')[-1] != 'TextLine':
            continue
        line_id = tl.get('ID')
        for child in tl:
            if not isinstance(child.tag, str):
                continue
            if child.tag.split('}')[-1] == 'String':
                texts[line_id] = child.get('CONTENT', '')
                break
    return texts


def write_gt_xml(samples, accepted, xml_files):
    """\u00c9crit un *_gt.xml complet pour chaque fichier source.

    Chaque *_gt.xml est construit depuis le *_gt.xml existant s'il est
    pr\u00e9sent (pr\u00e9serve le travail des sessions pr\u00e9c\u00e9dentes), sinon depuis
    l'original. Seules les lignes accept\u00e9es (modifi\u00e9es) sont mises \u00e0 jour ;
    les autres conservent leur texte source (GT pr\u00e9c\u00e9dent ou original).

    Tous les fichiers de xml_files re\u00e7oivent un *_gt.xml, m\u00eame ceux
    sans aucune ligne accept\u00e9e, pour garantir un jeu coh\u00e9rent.
    """
    by_file = {}
    for idx, text in accepted.items():
        s = samples[idx]
        xml_path = s["xml_path"]
        line_id = s["line_id"]
        by_file.setdefault(xml_path, {})[line_id] = text

    for xml_path in xml_files:
        output_path = _derive_output_path(xml_path)
        original_path = _derive_original_path(xml_path)
        line_texts = by_file.get(xml_path, {})

        # Source : _gt.xml existant (préserve les sessions précédentes),
        # sinon l'original.
        source_path = output_path if os.path.exists(output_path) else original_path

        tree = etree.parse(source_path)
        root = tree.getroot()

        modified = 0
        for tl in root.iter():
            if not isinstance(tl.tag, str):
                continue
            if tl.tag.split('}')[-1] != 'TextLine':
                continue
            line_id = tl.get('ID')
            if line_id not in line_texts:
                continue

            text = line_texts[line_id]
            tag = tl.tag
            ns = tag.split('}')[0] + '}' if '}' in tag else ''

            for child in list(tl):
                if not isinstance(child.tag, str):
                    continue
                child_tag = child.tag.split('}')[-1]
                if child_tag in ('String', 'SP'):
                    tl.remove(child)

            hpos = tl.get('HPOS', '0')
            vpos = tl.get('VPOS', '0')
            width = tl.get('WIDTH', '0')
            height = tl.get('HEIGHT', '0')

            string_el = etree.SubElement(tl, f'{ns}String')
            string_el.set('CONTENT', text)
            string_el.set('ID', f'{line_id}_gt')
            string_el.set('HPOS', hpos)
            string_el.set('VPOS', vpos)
            string_el.set('WIDTH', width)
            string_el.set('HEIGHT', height)
            modified += 1

        tree.write(output_path, xml_declaration=True, encoding='utf-8')
        print(f"  {os.path.basename(output_path)}: {modified} ligne(s) mise(s) \u00e0 jour")


# ─── Viewer matplotlib ──────────────────────────────────────────────────

class AnnotateViewer:
    """Navigateur matplotlib page par page pour annotation de ground truth.

    Chaque page affiche PER_PAGE lignes avec :
    - l'image de la ligne (grayscale)
    - les pr\u00e9dictions greedy/beam (read-only, sous l'image)
    - un TextBox \u00e9ditable pr\u00e9-rempli
    - un bouton de validation (toggle \u2713/\u25cb)

    Boutons globaux : Accept All, Clear All, Prev, Next, Save & Quit.
    """

    def __init__(self, samples, xml_files, mode, has_beam,
                 initial_accepted=None, initial_page=0,
                 on_save=None):
        self.samples = samples
        self.xml_files = xml_files
        self.mode = mode
        self.has_beam = has_beam
        self.per_page = PER_PAGE
        self.total_pages = max(1, -(-len(samples) // PER_PAGE))
        self.page = min(initial_page, self.total_pages - 1)

        # State: {global_line_index: accepted_text}
        self.accepted = dict(initial_accepted) if initial_accepted else {}
        self.on_save = on_save
        self._drawing = False
        self._dirty = False

        self._build_ui()
        self._draw()

    def _build_ui(self):
        n = self.per_page
        fig_h = n * 4.0 + 1.5
        self.fig = plt.figure(figsize=(16, fig_h))

        self.img_axes = []
        self.textboxes = []
        self.accept_btns = []

        row_h = 0.88 / n

        for i in range(n):
            y_top = 0.93 - i * row_h
            img_h = row_h * 0.52
            img_bottom = y_top - img_h

            img_ax = self.fig.add_axes(
                [0.02, img_bottom, 0.80, img_h])
            self.img_axes.append(img_ax)

            tb_bottom = y_top - row_h * 0.92
            tb_h = max(row_h * 0.11, 0.025)
            tb_ax = self.fig.add_axes([0.04, tb_bottom, 0.58, tb_h])
            tb = TextBox(tb_ax, "", initial="")
            self.textboxes.append(tb)

            btn_ax = self.fig.add_axes([0.64, tb_bottom, 0.08, tb_h])
            btn = Button(btn_ax, "\u25cb")
            self.accept_btns.append(btn)

            btn.on_clicked(self._make_accept_handler(i))

        nav_y = 0.015
        nav_h = 0.035
        self.btn_accept_all = Button(
            self.fig.add_axes([0.04, nav_y, 0.09, nav_h]), "Accept All")
        self.btn_clear_all = Button(
            self.fig.add_axes([0.15, nav_y, 0.09, nav_h]), "Clear All")
        self.btn_prev = Button(
            self.fig.add_axes([0.30, nav_y, 0.09, nav_h]), "< Prev")
        self.btn_next = Button(
            self.fig.add_axes([0.41, nav_y, 0.09, nav_h]), "Next >")
        self.btn_save = Button(
            self.fig.add_axes([0.72, nav_y, 0.13, nav_h]), "Save & Quit")

        self.btn_accept_all.on_clicked(self._accept_all)
        self.btn_clear_all.on_clicked(self._clear_all)
        self.btn_prev.on_clicked(self._prev)
        self.btn_next.on_clicked(self._next)
        self.btn_save.on_clicked(self._save_quit)

        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _make_accept_handler(self, row_idx):
        def handler(_event=None):
            if self._drawing:
                return
            self._toggle_accept(row_idx)
        return handler

    def _toggle_accept(self, row_idx):
        start = self.page * self.per_page
        global_idx = start + row_idx
        if global_idx >= len(self.samples):
            return

        if global_idx in self.accepted:
            del self.accepted[global_idx]
            self.accept_btns[row_idx].ax.set_facecolor("#e0e0e0")
            self.accept_btns[row_idx].label.set_text("\u25cb")
        else:
            self.accepted[global_idx] = self.textboxes[row_idx].text
            self.accept_btns[row_idx].ax.set_facecolor("#66bb6a")
            self.accept_btns[row_idx].label.set_text("\u2713")
        self._dirty = True
        self._update_title()
        self.fig.canvas.draw_idle()

    def _accept_all(self, _event=None):
        if self._drawing:
            return
        start = self.page * self.per_page
        for i in range(min(self.per_page, len(self.samples) - start)):
            global_idx = start + i
            self.accepted[global_idx] = self.textboxes[i].text
            self.accept_btns[i].ax.set_facecolor("#66bb6a")
            self.accept_btns[i].label.set_text("\u2713")
        self._dirty = True
        self._update_title()
        self.fig.canvas.draw_idle()

    def _clear_all(self, _event=None):
        if self._drawing:
            return
        start = self.page * self.per_page
        for i in range(min(self.per_page, len(self.samples) - start)):
            global_idx = start + i
            self.accepted.pop(global_idx, None)
            self.accept_btns[i].ax.set_facecolor("#e0e0e0")
            self.accept_btns[i].label.set_text("\u25cb")
        self._dirty = True
        self._update_title()
        self.fig.canvas.draw_idle()

    def _sync_page_texts(self):
        """Met \u00e0 jour les textes accept\u00e9s depuis les TextBox avant navigation."""
        start = self.page * self.per_page
        for i in range(min(self.per_page, len(self.samples) - start)):
            global_idx = start + i
            if global_idx in self.accepted:
                self.accepted[global_idx] = self.textboxes[i].text

    def _prev(self, _event=None):
        if self._drawing or self.page <= 0:
            return
        self._sync_page_texts()
        if self._dirty:
            self._do_save()
            self._dirty = False
        self.page -= 1
        self._draw()

    def _next(self, _event=None):
        if self._drawing:
            return
        self._sync_page_texts()
        if self._dirty:
            self._do_save()
            self._dirty = False
        if self.page < self.total_pages - 1:
            self.page += 1
            self._draw()

    def _save_quit(self, _event=None):
        self._do_save()
        plt.close(self.fig)

    def _on_close(self, _event=None):
        self._do_save()

    def _do_save(self):
        self._sync_page_texts()
        if self.on_save:
            self.on_save(self.accepted, self.page)

    def _update_title(self):
        n_accepted = len(self.accepted)
        self.fig.suptitle(
            f"Page {self.page + 1}/{self.total_pages}  |  "
            f"Mode: {self.mode}  |  "
            f"{n_accepted} ligne(s) accept\u00e9e(s)",
            fontsize=10, fontweight="bold")

    def _draw(self):
        self._drawing = True
        try:
            self._draw_impl()
        finally:
            self._drawing = False

    def _draw_impl(self):
        start = self.page * self.per_page
        page_samples = self.samples[start : start + self.per_page]

        self._update_title()

        for i in range(self.per_page):
            ax = self.img_axes[i]
            ax.clear()

            if i >= len(page_samples):
                ax.axis("off")
                self.textboxes[i].ax.set_visible(False)
                self.accept_btns[i].ax.set_visible(False)
                continue

            self.textboxes[i].ax.set_visible(True)
            self.accept_btns[i].ax.set_visible(True)

            s = page_samples[i]
            global_idx = start + i
            img = s["image"]
            H, real_w = img.shape

            ax.imshow(img, cmap="gray", aspect="equal", vmin=0, vmax=255,
                      extent=(0, real_w, H, 0))
            ax.set_xlim(0, max(real_w, 1))
            ax.set_ylim(H, 0)
            ax.set_aspect("equal", adjustable="box", anchor="NW")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            gt = s.get("gt", "")
            pred = s["pred"]
            beam = s.get("beam")

            y_off = -0.08
            if self.mode == "review" and gt:
                cer = s.get("cer")
                gt_color = "#2e7d32" if cer == 0.0 else "#c62828"
                cer_txt = f"  (CER {cer:.0%})" if cer is not None else ""
                ax.text(0.0, y_off, f"GT:   {gt}{cer_txt}",
                        transform=ax.transAxes,
                        fontsize=_FONTSIZE, color=gt_color,
                        fontfamily="monospace",
                        va="top", ha="left", clip_on=False)
                y_off -= 0.12

            ax.text(0.0, y_off, f"PRED: {pred}",
                    transform=ax.transAxes,
                    fontsize=_FONTSIZE, color="#37474f",
                    fontfamily="monospace",
                    va="top", ha="left", clip_on=False)
            y_off -= 0.12

            if beam:
                ax.text(0.0, y_off, f"BEAM: {beam}",
                        transform=ax.transAxes,
                        fontsize=_FONTSIZE, color="#1565c0",
                        fontfamily="monospace",
                        va="top", ha="left", clip_on=False)

            if global_idx in self.accepted:
                self.textboxes[i].set_val(self.accepted[global_idx])
            elif self.mode == "review" and gt:
                self.textboxes[i].set_val(gt)
            else:
                best = beam if beam else pred
                self.textboxes[i].set_val(best)

            if global_idx in self.accepted:
                self.accept_btns[i].ax.set_facecolor("#66bb6a")
                self.accept_btns[i].label.set_text("\u2713")
            else:
                self.accept_btns[i].ax.set_facecolor("#e0e0e0")
                self.accept_btns[i].label.set_text("\u25cb")

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Annotation interactive de ground truth ALTO"
    )
    parser.add_argument("--model", default="hwm_v17.pt",
                        help="Checkpoint du mod\u00e8le")
    parser.add_argument("--model-version", choices=known_versions(),
                        default="v17")
    parser.add_argument("--alto", required=True,
                        help="Fichier .xml ALTO ou r\u00e9pertoire de .xml")
    parser.add_argument("--mode", choices=["annotate", "review"],
                        default="annotate",
                        help="annotate: lignes sans GT | review: lignes avec GT")
    parser.add_argument("--beam-search", action="store_true",
                        help="Activer le CTC beam search")
    parser.add_argument("--lm-path", default=None,
                        help="Chemin vers un n-gram LM (.pkl)")
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--lm-weight", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=0,
                        help="Worker processes pour le beam search (0=séquentiel)")
    args = parser.parse_args()

    args.alto_dirs = config.ALTO_DIRS

    if args.lm_path and not args.beam_search:
        args.beam_search = True
        print("--lm-path fourni : beam search active automatiquement")
    if args.workers > 0 and not args.beam_search:
        args.beam_search = True
        print("--workers > 0 : beam search active automatiquement")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, char_to_idx, idx_to_char, spec, saved_config = _load_model(args, device)
    img_h = saved_config.get("img_height", spec.img_height)

    path = args.alto
    if os.path.isfile(path):
        xml_files = [_derive_original_path(path)]
    elif os.path.isdir(path):
        all_xmls = sorted(
            f for f in glob.glob(os.path.join(path, "*.xml"))
            if os.path.basename(f) != "METS.xml"
        )
        xml_files = [f for f in all_xmls
                     if not os.path.splitext(f)[0].endswith("_gt")]
    else:
        print(f"Error: {path} is neither a file nor a directory")
        sys.exit(1)

    all_samples = []
    all_metas = []
    for xml_path in xml_files:
        samples, _chars, metas = _parse_page(
            (xml_path, img_h, 4000, True), return_meta=True)

        gt_path = _derive_output_path(xml_path)
        if os.path.exists(gt_path):
            gt_texts = _read_gt_texts(gt_path)
            n_gt = 0
            for i, meta in enumerate(metas):
                lid = meta["line_id"]
                if lid in gt_texts:
                    arr, _ = samples[i]
                    samples[i] = (arr, gt_texts[lid])
                    n_gt += 1
            print(f"  {os.path.basename(xml_path)}: {len(samples)} lignes "
                  f"({n_gt} depuis GT existant)")
        else:
            print(f"  {os.path.basename(xml_path)}: {len(samples)} lignes")

        all_samples.extend(samples)
        all_metas.extend(metas)

    if not all_samples:
        print("Aucune ligne trouv\u00e9e")
        sys.exit(1)

    if args.mode == "annotate":
        indices = [i for i, (a, t) in enumerate(all_samples)
                   if not t or not t.strip()]
        print(f"Mode annotate: {len(indices)} ligne(s) sans GT")
    else:
        indices = [i for i, (a, t) in enumerate(all_samples)
                   if t and t.strip()]
        print(f"Mode review: {len(indices)} ligne(s) avec GT")

    if not indices:
        action = "annoter" if args.mode == "annotate" else "r\u00e9viser"
        print(f"Aucune ligne \u00e0 {action}")
        sys.exit(0)

    shown_samples = [all_samples[i] for i in indices]
    shown_metas = [all_metas[i] for i in indices]

    eval_ds = _DirectDataset(shown_samples, img_h)

    print(f"Total: {len(shown_samples)} ligne(s) depuis {len(xml_files)} fichier(s)")

    lm = None
    if args.lm_path:
        from beam_decode import CharNgramLM
        lm = CharNgramLM(args.lm_path)
        print(f"Loaded n-gram LM (order {lm.order}) from {args.lm_path}")

    print("Running predictions...")
    samples = _predict_all(
        model, eval_ds, _annotate_collate, idx_to_char, device, spec,
        use_beam=args.beam_search, lm=lm,
        beam_width=args.beam_width, lm_weight=args.lm_weight,
        workers=args.workers,
    )

    for i, s in enumerate(samples):
        s["line_id"] = shown_metas[i]["line_id"]
        s["xml_path"] = shown_metas[i]["xml_path"]

    print(f"Collected {len(samples)} predictions")

    first_xml = xml_files[0]
    progress_path = _derive_progress_path(first_xml)
    progress = load_progress(progress_path)

    initial_accepted = {}
    for idx, s in enumerate(samples):
        key = f"{_derive_output_path(s['xml_path'])}::{s['line_id']}"
        if key in progress.get("accepted", {}):
            initial_accepted[idx] = progress["accepted"][key]["text"]

    if initial_accepted:
        print(f"Reprise: {len(initial_accepted)} ligne(s) d\u00e9j\u00e0 accept\u00e9e(s)")

    def on_save(accepted, last_page):
        prog_accepted = {}
        for idx, text in accepted.items():
            s = samples[idx]
            key = f"{_derive_output_path(s['xml_path'])}::{s['line_id']}"
            prog_accepted[key] = {"text": text, "xml_path": s["xml_path"]}

        save_progress(progress_path, prog_accepted, last_page)
        write_gt_xml(samples, accepted, xml_files)
        print(f"Sauvegard\u00e9: {len(accepted)} ligne(s) accept\u00e9e(s)")

    viewer = AnnotateViewer(
        samples, xml_files, args.mode, args.beam_search,
        initial_accepted=initial_accepted,
        initial_page=progress.get("last_page", 0),
        on_save=on_save,
    )
    viewer.show()


if __name__ == "__main__":
    main()
