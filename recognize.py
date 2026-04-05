"""
CTC Recognition Evaluation for HWM-v2
Greedy CTC decoding and Character Error Rate (CER) evaluation.
"""

import torch


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


def evaluate_cer(model, loader, device, idx_to_char, max_samples=None):
    """Run full CTC evaluation on a DataLoader."""
    model.eval()
    all_preds = []
    all_gts = []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            img_seqs, targets, input_lengths, target_lengths = batch
            img_seqs = img_seqs.to(device, non_blocking=True)
            input_lengths_cpu = input_lengths.clone()

            with torch.amp.autocast("cuda", enabled=use_amp):
                _, z_seq, ctc_logits = model(img_seqs)

            decoded = ctc_greedy_decode(
                ctc_logits.cpu(), input_lengths_cpu, idx_to_char
            )
            all_preds.extend(decoded)

            offset = 0
            for tlen in target_lengths:
                gt_indices = targets[offset : offset + tlen].tolist()
                gt_text = "".join(idx_to_char.get(i, "?") for i in gt_indices)
                all_gts.append(gt_text)
                offset += tlen

            if max_samples and len(all_preds) >= max_samples:
                break

    cer = compute_cer(all_preds, all_gts)

    print("\nExamples (first 10):")
    for pred, gt in zip(all_preds[:10], all_gts[:10]):
        mark = "OK" if pred == gt else "ERR"
        print(f"  [{mark}] GT:   {gt}")
        print(f"        PRED: {pred}")

    return cer


if __name__ == "__main__":
    import argparse
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import config
    from model import HWMv2
    from data_alto import AltoLineDataset, build_alphabet, collate_alto_fn
    from functools import partial
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Evaluate HWM-v2 CER")
    parser.add_argument("--model", default="hwm_v2.pt", help="Model checkpoint")
    parser.add_argument("--alto-dirs", nargs="+", default=config.ALTO_DIRS)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    saved_config = ckpt.get("config", {})

    char_to_idx, idx_to_char = build_alphabet(args.alto_dirs)
    num_classes = len(char_to_idx) + 1

    model = HWMv2(
        img_height=saved_config.get("img_height", config.IMG_HEIGHT_V2),
        window_size=saved_config.get("window_size", config.WINDOW_SIZE),
        embedding_dim=saved_config.get("embedding_dim", config.EMBEDDING_DIM_V2),
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        ff_dim=saved_config.get("ff_dim", config.FF_DIM_V2),
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = AltoLineDataset(
        args.alto_dirs, img_height=saved_config.get("img_height", config.IMG_HEIGHT_V2)
    )
    collate = partial(collate_alto_fn, char_to_idx=char_to_idx)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    cer = evaluate_cer(model, loader, device, idx_to_char)
    print(f"\nCER: {cer:.1%}")
