"""
Multi-task training for HWM-v2.
Supports:
  - Supervised: prediction + SIGReg + CTC (with ALTO data)
  - Self-supervised: prediction + SIGReg only (adapt mode)
"""

import sys
import os
import argparse
import json
from collections import defaultdict
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from model import HWMv2
from data_alto import AltoLineDataset, build_alphabet, collate_alto_fn


def train_epoch(model, loader, optimizer, device, epoch, mode="full"):
    model.train()
    totals = defaultdict(float)
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        if mode == "full":
            img_seqs, targets, input_lengths, target_lengths = batch
            img_seqs = img_seqs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            if img_seqs.shape[1] < 2:
                continue

            optimizer.zero_grad()
            loss, losses = model.compute_loss(
                img_seqs, targets, input_lengths, target_lengths
            )

        elif mode == "adapt":
            img_seqs = batch[0].to(device)
            if img_seqs.shape[1] < 2:
                continue

            optimizer.zero_grad()
            loss, losses = model.adapt(img_seqs)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in losses.items():
            totals[k] += v
        num_batches += 1

    return {k: v / num_batches for k, v in totals.items()} if num_batches > 0 else {}


def train(
    model,
    train_loader,
    val_loader=None,
    num_epochs=30,
    lr=1e-3,
    device="cpu",
    mode="full",
    save_path="hwm_v2.pt",
    idx_to_char=None,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    for epoch in range(1, num_epochs + 1):
        losses = train_epoch(model, train_loader, optimizer, device, epoch, mode=mode)
        scheduler.step()

        loss_str = " | ".join(f"{k}={v:.4f}" for k, v in losses.items())
        print(f"Epoch {epoch}/{num_epochs} - {loss_str}")

        if val_loader and model.ctc_head and idx_to_char:
            from recognize import evaluate_cer

            cer = evaluate_cer(model, val_loader, device, idx_to_char)
            print(f"  Val CER: {cer:.1%}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "img_height": model.img_height,
                "window_size": model.window_size,
                "embedding_dim": model.embedding_dim,
                "num_classes": model.ctc_head.proj.out_features
                if model.ctc_head
                else None,
            },
        },
        save_path,
    )
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HWM-v2")
    parser.add_argument("--mode", choices=["full", "adapt"], default="full")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data", default="alto", choices=["alto", "synthetic"])
    parser.add_argument("--alto-dirs", nargs="+", default=config.ALTO_DIRS)
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    char_to_idx = None
    idx_to_char = None
    num_classes = None

    if args.data == "alto":
        char_to_idx, idx_to_char = build_alphabet(args.alto_dirs)
        num_classes = len(char_to_idx) + 1
        dataset = AltoLineDataset(args.alto_dirs, img_height=config.IMG_HEIGHT_V2)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        collate = partial(collate_alto_fn, char_to_idx=char_to_idx)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)
    else:
        raise ValueError("Only 'alto' data mode is supported in v2")

    model = HWMv2(
        img_height=config.IMG_HEIGHT_V2,
        window_size=config.WINDOW_SIZE,
        embedding_dim=config.EMBEDDING_DIM_V2,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        ff_dim=config.FF_DIM_V2,
        dropout=config.DROPOUT,
        num_classes=num_classes if args.mode == "full" else None,
    ).to(device)

    print(f"Model params: {model.count_parameters():,}")

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    train(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        mode=args.mode,
        device=device,
        idx_to_char=idx_to_char,
    )
