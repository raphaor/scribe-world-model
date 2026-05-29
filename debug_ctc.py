"""
Diagnostic: load a batch, run through model, inspect CTC behavior.
Usage: python debug_ctc.py
"""
import torch
import numpy as np
from data_alto import AltoLineDataset, collate_alto_v5_fn
from torch.utils.data import DataLoader
from model import LectaurepClone

def main():
    device = torch.device("cpu")

    # Load dataset
    ds = AltoLineDataset(
        alto_dirs=[
            "D:/OCR_genealogie/Alto/lectaurep_bronod_notaire_paris_18e",
        ],
        img_height=120,
    )
    print(f"Dataset: {len(ds)} lines")
    print(f"Alphabet: {len(ds.get_alphabet()[0])} chars")

    char_to_idx, idx_to_char = ds.get_alphabet()

    # Load model
    model = LectaurepClone(
        num_classes=len(char_to_idx) + 1,  # +1 for blank
        img_height=120,
        hidden=200,
        num_lstm_layers=3,
        dropout=0.1,
    )
    model.to(device)

    # Try loading checkpoint if available
    import os
    ckpt_path = "hwm_lectaurep.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    else:
        print("No checkpoint found, using random init")

    model.eval()

    # Get a batch with known text
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_alto_v5_fn, num_workers=0)
    batch = next(iter(loader))
    img_seqs, targets, input_lengths, target_lengths, raw_texts = batch

    print(f"\n=== BATCH INFO ===")
    print(f"Image shape: {img_seqs.shape}")
    print(f"Input lengths (W//8): {input_lengths.tolist()}")
    print(f"Target lengths: {target_lengths.tolist()}")
    for i, (text, ilen, tlen) in enumerate(zip(raw_texts, input_lengths, target_lengths)):
        print(f"  [{i}] len={len(text):3d} chars, W//8={ilen.item()}, target_len={tlen.item()}, ratio={ilen.item()/max(tlen.item(),1):.1f}")

    # Forward pass
    with torch.no_grad():
        _, z_seq, ctc_logits = model(img_seqs, input_lengths=input_lengths)

    print(f"\n=== MODEL OUTPUT ===")
    print(f"Encoder output shape: {z_seq.shape}")
    print(f"CTC logits shape: {ctc_logits.shape}")  # (B, T, num_classes)

    # Check what the model predicts (greedy CTC decode)
    probs = ctc_logits.exp()  # undo log_softmax
    blank_probs = probs[:, :, 0]  # (B, T)
    print(f"\n=== BLANK PROBABILITIES ===")
    for i in range(min(4, len(raw_texts))):
        bp = blank_probs[i]
        valid_len = input_lengths[i].item()
        print(f"  [{i}] min={bp[:valid_len].min():.4f}, max={bp[:valid_len].max():.4f}, mean={bp[:valid_len].mean():.4f}")

    # Top predictions at each time step
    print(f"\n=== TOP PREDICTIONS PER STEP ===")
    for i in range(min(4, len(raw_texts))):
        valid_len = input_lengths[i].item()
        top_vals, top_idx = ctc_logits[i, :valid_len].max(dim=-1)
        # Decode CTC
        decoded = []
        prev = None
        for t in range(valid_len):
            idx = top_idx[t].item()
            if idx != 0 and idx != prev:
                decoded.append(idx_to_char.get(idx, f"<{idx}>"))
            prev = idx
        
        pred_str = "".join(decoded)
        print(f"  [{i}] GT: '{raw_texts[i][:60]}...' ({len(raw_texts[i])} chars)")
        print(f"      PRED: '{pred_str}' ({len(pred_str)} chars)")
        print(f"      Non-blank steps: {(top_idx[:valid_len] != 0).sum().item()}/{valid_len}")

    # Distribution analysis: how spread are the logits?
    print(f"\n=== LOGIT DISTRIBUTION ===")
    for i in range(min(2, len(raw_texts))):
        valid_len = input_lengths[i].item()
        logits = ctc_logits[i, :valid_len]  # (T, num_classes)
        # Average top-5 classes at each step
        top5_vals, top5_idx = logits.topk(5, dim=-1)
        print(f"  [{i}] Top-5 avg probs: {top5_vals[0].exp().tolist()}")
        print(f"      Top-5 classes: {top5_idx[0].tolist()}")
        print(f"      Blank logit avg: {logits[:, 0].mean():.2f}")
        print(f"      Non-blank max logit avg: {logits[:, 1:].max(dim=-1)[0].mean():.2f}")

    # Check if the model's features are dead
    print(f"\n=== FEATURE ACTIVATION STATS ===")
    with torch.no_grad():
        z = z_seq[0, :input_lengths[0]]  # first sample, valid steps
        print(f"  Encoder features: min={z.min():.4f}, max={z.max():.4f}, mean={z.mean():.4f}, std={z.std():.4f}")
        dead = (z.abs() < 1e-6).float().mean()
        print(f"  Dead features (<1e-6): {dead*100:.1f}%")

if __name__ == "__main__":
    main()
