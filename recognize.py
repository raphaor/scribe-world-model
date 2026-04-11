"""
CTC Recognition Evaluation for HWM-v2
Greedy CTC decoding and Character Error Rate (CER) evaluation.
"""

import sys
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


def evaluate_cer(model, loader, device, idx_to_char, max_samples=None, verbose=True):
    """Run full CTC evaluation on a DataLoader."""
    model.eval()
    all_preds = []
    all_gts = []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            img_seqs, targets, input_lengths, target_lengths, raw_texts = batch
            img_seqs = img_seqs.to(device, non_blocking=True)
            input_lengths_cpu = input_lengths.clone()

            with torch.amp.autocast("cuda", enabled=use_amp):
                _, z_seq, ctc_logits = model(img_seqs)

            decoded = ctc_greedy_decode(
                ctc_logits.cpu(), input_lengths_cpu, idx_to_char
            )
            all_preds.extend(decoded)
            all_gts.extend(raw_texts)

            if max_samples and len(all_preds) >= max_samples:
                break

    cer = compute_cer(all_preds, all_gts)

    if verbose:
        print("\nExamples (first 10):")
        for pred, gt in zip(all_preds[:10], all_gts[:10]):
            mark = "OK" if pred == gt else "ERR"
            print(f"  [{mark}] GT:   {_safe_print(gt)}")
            print(f"        PRED: {_safe_print(pred)}")

    torch.cuda.empty_cache()
    return cer


if __name__ == "__main__":
    import argparse
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import config
    from model import HWMv2, HWMv3, HWMv4, HWMv5
    from data_alto import AltoLineDataset, build_alphabet, collate_alto_fn, collate_alto_v5_fn
    from functools import partial
    from torch.utils.data import DataLoader, random_split

    parser = argparse.ArgumentParser(description="Evaluate HWM CER")
    parser.add_argument("--model", default="hwm_v4.pt", help="Model checkpoint")
    parser.add_argument("--model-version", choices=["v2", "v3", "v4", "v5"], default="v5")
    parser.add_argument("--alto-dirs", nargs="+", default=config.ALTO_DIRS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", choices=["all", "val", "train"], default="val",
                        help="Which split to evaluate (default: val)")
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
    if ver == "v5":
        model = HWMv5(
            img_height=saved_config.get("img_height", config.IMG_HEIGHT_V5),
            embedding_dim=saved_config.get("embedding_dim", config.EMBEDDING_DIM_V5),
            num_layers=config.NUM_LAYERS_V5,
            num_heads=config.NUM_HEADS_V5,
            ff_dim=config.FF_DIM_V5,
            num_classes=num_classes,
            ctc_hidden=config.CTC_HIDDEN_V5,
            ctc_num_lstm=config.CTC_NUM_LSTM_V5,
        ).to(device)
    elif ver == "v4":
        model = HWMv4(
            img_height=saved_config.get("img_height", config.IMG_HEIGHT_V4),
            window_size=saved_config.get("window_size", config.WINDOW_SIZE_V4),
            embedding_dim=saved_config.get("embedding_dim", config.EMBEDDING_DIM_V4),
            num_layers=config.NUM_LAYERS_V4,
            num_heads=config.NUM_HEADS_V4,
            ff_dim=config.FF_DIM_V4,
            num_classes=num_classes,
            ctc_hidden=config.CTC_HIDDEN_V4,
        ).to(device)
        ws = config.WINDOW_SIZE_V4
        stride = config.STRIDE_V4
    elif ver == "v3":
        model = HWMv3(
            img_height=saved_config.get("img_height", config.IMG_HEIGHT_V3),
            window_size=saved_config.get("window_size", config.WINDOW_SIZE_V3),
            embedding_dim=saved_config.get("embedding_dim", config.EMBEDDING_DIM_V3),
            num_layers=config.NUM_LAYERS_V3,
            num_heads=config.NUM_HEADS_V3,
            ff_dim=config.FF_DIM_V3,
            num_classes=num_classes,
        ).to(device)
        ws = config.WINDOW_SIZE_V3
        stride = config.STRIDE_V3
    else:
        model = HWMv2(
            img_height=saved_config.get("img_height", config.IMG_HEIGHT_V2),
            window_size=saved_config.get("window_size", config.WINDOW_SIZE),
            embedding_dim=saved_config.get("embedding_dim", config.EMBEDDING_DIM_V2),
            num_layers=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS,
            ff_dim=saved_config.get("ff_dim", config.FF_DIM_V2),
            num_classes=num_classes,
        ).to(device)
        ws = config.WINDOW_SIZE
        stride = config.STRIDE

    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if result.missing_keys:
        print(f"  Warning: missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Warning: unexpected keys: {result.unexpected_keys}")
    model.eval()
    print(f"Model {ver}: {model.count_parameters():,} params")

    img_h = saved_config.get("img_height", config.IMG_HEIGHT_V5 if ver == "v5" else config.IMG_HEIGHT_V4)
    dataset = AltoLineDataset(args.alto_dirs, img_height=img_h)

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

    if ver == "v5":
        collate = partial(collate_alto_v5_fn, char_to_idx=char_to_idx)
    else:
        collate = partial(collate_alto_fn, window_size=ws, stride=stride, char_to_idx=char_to_idx)
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    cer = evaluate_cer(model, loader, device, idx_to_char)
    print(f"\nCER: {cer:.1%}")
