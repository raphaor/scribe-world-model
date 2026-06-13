"""
CTC Recognition Evaluation for HWM-v2
Greedy CTC decoding and Character Error Rate (CER) evaluation.
"""

import sys
import random

import torch


def _safe_print(s):
    enc = sys.stdout.encoding or "utf-8"
    return s.encode(enc, errors="replace").decode(enc)


def ctc_greedy_decode(log_probs, lengths, idx_to_char):
    """
    Greedy CTC decoding.

    Args:
        log_probs: (B, T, C) log-probabilities
        lengths: (B,) actual sequence lengths
        idx_to_char: dict mapping indices to characters
    Returns:
        list of decoded strings
    """
    results = []
    preds = log_probs.argmax(dim=-1)

    for i in range(preds.size(0)):
        seq = preds[i, : lengths[i]].tolist()
        decoded = []
        prev = None
        for idx in seq:
            if idx != 0 and idx != prev:
                decoded.append(idx_to_char.get(idx, "?"))
            prev = idx
        results.append("".join(decoded))

    return results


def ctc_greedy_decode_conf(log_probs, lengths, idx_to_char):
    """Greedy CTC decoding qui renvoie aussi les confiances.

    Args:
        log_probs: (B, T, C) log-probabilites (sortie du CTC head, deja log_softmax)
        lengths: (B,) longueurs reelles (nombre de frames valides)
        idx_to_char: dict index -> caractere
    Returns:
        list de dicts, un par echantillon :
          - "text":       str decode
          - "char_confs": list[float] proba de chaque caractere emis (la frame "pic")
          - "frame_conf": list[float] proba max a chaque frame (len = lengths[i])
          - "line_conf":  float moyenne geometrique des char_confs (0.0 si vide)
    """
    probs = log_probs.exp()                     # (B, T, C)
    preds = log_probs.argmax(dim=-1)            # (B, T)
    max_p = probs.max(dim=-1).values            # (B, T) confiance par frame

    results = []
    for i in range(preds.size(0)):
        L = int(lengths[i])
        seq = preds[i, :L].tolist()
        frame_conf = max_p[i, :L].tolist()

        chars, confs = [], []
        prev = None
        for t, idx in enumerate(seq):
            if idx != 0 and idx != prev:
                chars.append(idx_to_char.get(idx, "?"))
                confs.append(float(probs[i, t, idx]))
            prev = idx

        if confs:
            log_mean = sum(torch.log(torch.tensor(confs))).item() / len(confs)
            line_conf = float(torch.exp(torch.tensor(log_mean)))
        else:
            line_conf = 0.0

        results.append({
            "text": "".join(chars),
            "char_confs": confs,
            "frame_conf": frame_conf,
            "line_conf": line_conf,
        })

    return results


def align_pred_gt(pred, gt):
    """Aligne pred sur gt (edit distance) et marque les caracteres errones de pred.

    Renvoie une liste de bool de longueur len(pred) : True si le caractere predit
    est une substitution ou une insertion (donc faux). Les suppressions (caracteres
    manquants) ne correspondent a aucune position de pred et ne sont pas marquees.
    """
    n, m = len(pred), len(gt)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    wrong = [False] * n
    i, j = n, m
    while i > 0 and j > 0:
        cost = 0 if pred[i - 1] == gt[j - 1] else 1
        if dp[i][j] == dp[i - 1][j - 1] + cost:
            if cost == 1:
                wrong[i - 1] = True          # substitution
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            wrong[i - 1] = True              # insertion (caractere predit en trop)
            i -= 1
        else:
            j -= 1                           # suppression (pas de slot dans pred)
    while i > 0:
        wrong[i - 1] = True
        i -= 1
    return wrong


def levenshtein(s1, s2):
    """Edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def compute_cer(predictions, ground_truths):
    """Character Error Rate = sum(edit_distances) / sum(gt_lengths)."""
    total_dist = 0
    total_len = 0
    for pred, gt in zip(predictions, ground_truths):
        total_dist += levenshtein(pred, gt)
        total_len += len(gt)
    return total_dist / max(total_len, 1)


# Fixed seed: the displayed sample is the SAME 10 lines every epoch and
# across runs. The dataset order is deterministic and the train/val
# split is seeded, so fixed indices map to fixed lines.
_SAMPLE_SEED = 20240517
_NUM_SAMPLES = 10
# Fixed seed for the truncated-CER subset (see evaluate_cer).
_CER_SEED = 770414


def _show_fixed_samples(model, loader, device, idx_to_char, use_amp):
    """
    Print predictions for a fixed random subset of the loader's dataset.

    Replaces the old "first 10 in loader order": with a width-bucketed
    loader the first batch holds only the narrowest (trivial) lines.
    Indices are drawn once from a fixed seed so the sample is stable
    epoch-to-epoch and run-to-run; augmentation is disabled for the
    fetch so the pixels — and predictions — don't jitter.
    """
    ds = loader.dataset
    n = len(ds)
    if n == 0:
        return
    k = min(_NUM_SAMPLES, n)
    pick = sorted(random.Random(_SAMPLE_SEED).sample(range(n), k))

    base = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    old_aug = getattr(base, "augment", False)
    if hasattr(base, "augment"):
        base.augment = False
    try:
        items = [ds[i] for i in pick]
    finally:
        if hasattr(base, "augment"):
            base.augment = old_aug

    img_seqs, _t, input_lengths, _tl, raw_texts = loader.collate_fn(items)
    img_seqs = img_seqs.to(device, non_blocking=True)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        _, _, ctc_logits = model(img_seqs, input_lengths=input_lengths.to(device))
    decoded = ctc_greedy_decode(ctc_logits.cpu(), input_lengths.clone(), idx_to_char)

    print(f"\nExamples ({k} fixed random lines, idx {pick}):")
    for pred, gt in zip(decoded, raw_texts):
        mark = "OK" if pred == gt else "ERR"
        print(f"  [{mark}] GT:   {_safe_print(gt)}")
        print(f"        PRED: {_safe_print(pred)}")


def evaluate_cer(
    model, loader, device, idx_to_char,
    max_samples=None, verbose=True,
    use_beam=False, lm=None, beam_width=20, lm_weight=0.3,
):
    """
    Run CTC evaluation on a DataLoader.

    With max_samples=None the whole dataset is evaluated. When
    max_samples truncates, a FIXED seeded random subset is evaluated —
    NOT the first max_samples in loader order: the width-bucket sampler
    iterates shortest-first, so loader order would make the CER a sample
    of the easiest (shortest) lines — optimistically biased and not
    representative. The seeded subset is also stable epoch-to-epoch.

    If use_beam=True, CTC prefix beam search is used instead of greedy.
    A loaded CharNgramLM can be passed as *lm* to bias the beam search.
    """
    model.eval()
    use_amp = device.type == "cuda"

    ds = loader.dataset
    n = len(ds)
    if max_samples is not None and max_samples < n:
        idx = sorted(random.Random(_CER_SEED).sample(range(n), max_samples))
        eval_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx),
            batch_size=64,
            collate_fn=loader.collate_fn,
        )
    else:
        eval_loader = loader  # full pass — loader order is irrelevant

    if use_beam:
        from beam_decode import ctc_beam_search_decode

    all_preds = []
    all_gts = []
    num_batches = len(eval_loader)
    with torch.no_grad():
        for bi, batch in enumerate(eval_loader):
            img_seqs, targets, input_lengths, target_lengths, raw_texts = batch
            img_seqs = img_seqs.to(device, non_blocking=True)
            input_lengths_cpu = input_lengths.clone()

            with torch.amp.autocast("cuda", enabled=use_amp):
                _, z_seq, ctc_logits = model(
                    img_seqs, input_lengths=input_lengths.to(device)
                )

            if use_beam:
                decoded = ctc_beam_search_decode(
                    ctc_logits.cpu(),
                    input_lengths_cpu,
                    idx_to_char,
                    lm=lm,
                    beam_width=beam_width,
                    lm_weight=lm_weight,
                )
                if num_batches > 5 and (bi % 5 == 0 or bi == num_batches - 1):
                    print(f"  beam: batch {bi+1}/{num_batches}", flush=True)
            else:
                decoded = ctc_greedy_decode(
                    ctc_logits.cpu(), input_lengths_cpu, idx_to_char
                )
            all_preds.extend(decoded)
            all_gts.extend(raw_texts)

    cer = compute_cer(all_preds, all_gts)

    if verbose:
        _show_fixed_samples(model, loader, device, idx_to_char, use_amp)

    torch.cuda.empty_cache()
    return cer


if __name__ == "__main__":
    import argparse
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import config
    from model_registry import get_spec, known_versions, default_train_args
    from data_alto import AltoLineDataset, build_alphabet, collate_alto_fn, collate_alto_v5_fn
    from functools import partial
    from torch.utils.data import DataLoader, random_split

    parser = argparse.ArgumentParser(description="Evaluate HWM CER")
    parser.add_argument("--model", default="hwm_v4.pt", help="Model checkpoint")
    parser.add_argument("--model-version", choices=known_versions(), default="v5")
    parser.add_argument("--alto-dirs", nargs="+", default=config.ALTO_DIRS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Evaluate on a fixed seeded random subset (faster for beam search).",
    )
    parser.add_argument("--split", choices=["all", "val", "train"], default="val",
                        help="Which split to evaluate (default: val)")
    parser.add_argument(
        "--min-frames-per-char",
        type=float,
        default=0.0,
        help="Must match the value used at training time: prunes the same "
        "unlearnable lines before the split so the seeded train/val "
        "partition stays identical. 0.0 = disabled (default).",
    )
    parser.add_argument(
        "--beam-search", action="store_true",
        help="Use CTC beam search instead of greedy decoding.",
    )
    parser.add_argument(
        "--lm-path", default=None,
        help="Path to a trained char n-gram .pkl (from train_lm.py). "
        "If provided with --beam-search, biases the beam search.",
    )
    parser.add_argument(
        "--beam-width", type=int, default=20,
        help="Beam width (default: 20).",
    )
    parser.add_argument(
        "--lm-weight", type=float, default=0.3,
        help="LM interpolation weight (default: 0.3). "
        "0 = pure CTC beam search, 1 = LM-dominated.",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run both greedy AND beam search, print both CERs for comparison.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    saved_config = ckpt.get("config", {})

    # Use alphabet from checkpoint if available, else build from data
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
    # The builder consumes a train-style Namespace; we synthesise the
    # argparse defaults since training-time knobs (lambdas, SSL flags) do
    # not affect ``forward()`` — only the architecture matters for the
    # state-dict load below.
    model = spec.builder(default_train_args(), num_classes).to(device)

    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if result.missing_keys:
        print(f"  Warning: missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Warning: unexpected keys: {result.unexpected_keys}")
    model.eval()
    print(f"Model {ver}: {model.count_parameters():,} params")

    img_h = saved_config.get("img_height", spec.img_height)
    dataset = AltoLineDataset(args.alto_dirs, img_height=img_h)

    # Mirror train.py's pre-split pruning so the seeded partition matches.
    if args.min_frames_per_char > 0.0:
        removed, kept = dataset.filter_unlearnable(
            char_to_idx,
            width_stride=spec.cnn_width_stride,
            min_frames_per_char=args.min_frames_per_char,
        )
        print(f"Filtered {removed} unlearnable lines; {kept} remain")

    # Same split as train.py (seed=42, 80/20)
    if args.split != "all":
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        eval_ds = val_ds if args.split == "val" else train_ds
        print(f"Evaluating on {args.split} split: {len(eval_ds)} lines")
    else:
        eval_ds = dataset
        print(f"Evaluating on all data: {len(eval_ds)} lines")

    if spec.collate_style == "v5":
        collate = partial(collate_alto_v5_fn, char_to_idx=char_to_idx)
    else:
        collate = partial(
            collate_alto_fn,
            window_size=spec.window_size,
            stride=spec.stride,
            char_to_idx=char_to_idx,
        )
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    # --- Load LM if requested ---
    lm = None
    if args.lm_path:
        from beam_decode import CharNgramLM
        lm = CharNgramLM(args.lm_path)
        print(
            f"Loaded char {lm.order}-gram LM from {args.lm_path} "
            f"({lm.total_chars:,} tokens, {lm.vocab_size} chars)"
        )

    # --- Evaluate ---
    if args.compare:
        print("\n=== Greedy ===")
        cer_greedy = evaluate_cer(
            model, loader, device, idx_to_char, verbose=False,
            max_samples=args.max_samples,
        )
        print(f"Greedy CER: {cer_greedy:.1%}")

        print("\n=== Beam search ===")
        cer_beam = evaluate_cer(
            model, loader, device, idx_to_char, verbose=False,
            max_samples=args.max_samples,
            use_beam=True, lm=lm,
            beam_width=args.beam_width, lm_weight=args.lm_weight,
        )
        delta = cer_greedy - cer_beam
        pct = (delta / cer_greedy * 100) if cer_greedy > 0 else 0
        print(f"Beam CER:   {cer_beam:.1%}")
        print(f"\nDelta: {delta:+.1%} ({pct:+.1f}% relative)")
    elif args.beam_search:
        mode = f"beam (w={args.beam_width}, lm_w={args.lm_weight})" if lm \
            else f"beam (w={args.beam_width}, no LM)"
        print(f"\nDecoding: {mode}")
        cer = evaluate_cer(
            model, loader, device, idx_to_char,
            max_samples=args.max_samples,
            use_beam=True, lm=lm,
            beam_width=args.beam_width, lm_weight=args.lm_weight,
        )
        print(f"\nCER: {cer:.1%}")
    else:
        cer = evaluate_cer(
            model, loader, device, idx_to_char,
            max_samples=args.max_samples,
        )
        print(f"\nCER: {cer:.1%}")
