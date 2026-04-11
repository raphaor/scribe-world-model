"""
Multi-task training for HWM-v2/v3.
Supports:
  - full: prediction + SIGReg + CTC (with ALTO data)
  - adapt: prediction + SIGReg only (self-supervised)
  - mixed (default): alternates full and adapt batches each step
  - Resume from checkpoint (optimizer, scheduler, epoch state restored)
"""

import sys
import os
import gc
import argparse
import time
from collections import defaultdict
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from model import HWMv2, HWMv3, HWMv4, HWMv5
from data_alto import (
    AltoLineDataset,
    UnannotatedLineDataset,
    collate_alto_fn,
    collate_unannotated_fn,
    collate_alto_v5_fn,
    collate_unannotated_v5_fn,
)


def _progress_bar(epoch, batch_idx, total_batches, losses, elapsed, bar_width=30):
    pct = (batch_idx + 1) / total_batches
    filled = int(bar_width * pct)
    bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
    loss_str = " | ".join(f"{k}={v:.4f}" for k, v in losses.items())
    eta = elapsed / pct - elapsed if pct > 0 else 0
    sys.stdout.write(
        f"\r  Epoch {epoch} [{bar}] {batch_idx + 1}/{total_batches} "
        f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left) - {loss_str}"
    )
    sys.stdout.flush()


def _step_full(model, batch, optimizer, device, use_amp):
    """One supervised training step (prediction + SIGReg + CTC)."""
    img_seqs, targets, input_lengths, target_lengths, _raw = batch
    img_seqs = img_seqs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    input_lengths = input_lengths.to(device, non_blocking=True)
    target_lengths = target_lengths.to(device, non_blocking=True)

    if img_seqs.shape[1] < 2:
        return None, None

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=use_amp):
        loss, losses = model.compute_loss(
            img_seqs, targets, input_lengths, target_lengths
        )
    del img_seqs, targets, input_lengths, target_lengths
    return loss, losses


def _step_adapt(model, batch, optimizer, device, use_amp):
    """One self-supervised training step (prediction + SIGReg only)."""
    img_seqs = batch[0].to(device, non_blocking=True)

    if img_seqs.shape[1] < 2:
        return None, None

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=use_amp):
        loss, losses = model.adapt(img_seqs)
    del img_seqs
    return loss, losses


def train_epoch(
    model, loader, optimizer, device, epoch, mode="full", scaler=None,
    adapt_loader=None,
):
    model.train()
    totals = defaultdict(float)
    num_batches = 0
    use_amp = scaler is not None
    t0 = time.time()

    if mode == "mixed" and adapt_loader is not None:
        # Interleave full and adapt batches: full, adapt, full, adapt, ...
        # NOTE: do NOT use itertools.cycle here — it caches every batch
        # in memory, causing progressive memory growth across the epoch.
        adapt_iter = iter(adapt_loader)
        total_batches = len(loader) * 2
        for batch_idx, full_batch in enumerate(loader):
            # --- supervised step ---
            loss, losses = _step_full(model, full_batch, optimizer, device, use_amp)
            del full_batch
            if loss is not None:
                _backward(loss, optimizer, model, scaler, use_amp)
                for k, v in losses.items():
                    totals[k] += v
                num_batches += 1
                del loss, losses

            # --- self-supervised step (restart iter if exhausted) ---
            try:
                adapt_batch = next(adapt_iter)
            except StopIteration:
                adapt_iter = iter(adapt_loader)
                adapt_batch = next(adapt_iter)
            loss, losses = _step_adapt(model, adapt_batch, optimizer, device, use_amp)
            del adapt_batch
            if loss is not None:
                _backward(loss, optimizer, model, scaler, use_amp)
                for k, v in losses.items():
                    totals[f"a_{k}"] += v
                num_batches += 1
                del loss, losses

            if batch_idx % 25 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()

            running = {k: v / max(1, num_batches // 2) for k, v in totals.items()}
            _progress_bar(epoch, batch_idx * 2 + 1, total_batches, running, time.time() - t0)
    else:
        # Pure full or pure adapt mode
        step_fn = _step_full if mode == "full" else _step_adapt
        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader):
            loss, losses = step_fn(model, batch, optimizer, device, use_amp)
            del batch
            if loss is None:
                continue

            _backward(loss, optimizer, model, scaler, use_amp)

            for k, v in losses.items():
                totals[k] += v
            num_batches += 1
            del loss, losses

            if batch_idx % 50 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()

            running = {k: v / num_batches for k, v in totals.items()}
            _progress_bar(epoch, batch_idx, total_batches, running, time.time() - t0)

    sys.stdout.write("\n")
    divisor = max(1, num_batches // 2) if mode == "mixed" else max(1, num_batches)
    return {k: v / divisor for k, v in totals.items()}


def _backward(loss, optimizer, model, scaler, use_amp):
    """Backward pass + gradient clipping."""
    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def train(
    model,
    train_loader,
    val_loader=None,
    adapt_loader=None,
    num_epochs=30,
    lr=1e-3,
    device="cpu",
    mode="full",
    save_path="hwm_v2.pt",
    idx_to_char=None,
    char_to_idx=None,
    start_epoch=1,
    optimizer_state=None,
    scheduler_state=None,
    scaler_state=None,
):
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Always create scheduler with last_epoch=-1 so it initializes
    # initial_lr in the optimizer. Then restore or adjust state after.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6,
    )

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    elif start_epoch > 1:
        # No saved scheduler (e.g. mode switch full→adapt): advance the
        # scheduler to the correct position on the cosine curve so the
        # LR matches where training left off, not the initial value.
        for _ in range(start_epoch - 1):
            scheduler.step()
    if scaler_state is not None and use_amp:
        scaler.load_state_dict(scaler_state)

    best_loss = float("inf")

    for epoch in range(start_epoch, num_epochs + 1):
        lr_current = optimizer.param_groups[0]["lr"]
        losses = train_epoch(
            model, train_loader, optimizer, device, epoch,
            mode=mode, scaler=scaler, adapt_loader=adapt_loader,
        )
        scheduler.step()

        loss_str = " | ".join(f"{k}={v:.4f}" for k, v in losses.items())
        print(f"Epoch {epoch}/{num_epochs} (lr={lr_current:.2e}) - {loss_str}")

        current_loss = losses.get("total", float("inf"))
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch": epoch,
                    "loss": current_loss,
                    "config": {
                        "img_height": model.img_height,
                        "window_size": getattr(model, "window_size", None),
                        "embedding_dim": model.embedding_dim,
                        "num_classes": model.ctc_head.proj.out_features
                        if model.ctc_head
                        else None,
                    },
                    "char_to_idx": char_to_idx,
                },
                save_path,
            )

        if model.ctc_head and idx_to_char:
            from recognize import evaluate_cer

            cer_samples = min(100, len(train_loader.dataset) // 10)
            train_cer = evaluate_cer(
                model, train_loader, device, idx_to_char,
                max_samples=cer_samples, verbose=False,
            )
            if val_loader:
                val_cer = evaluate_cer(
                    model, val_loader, device, idx_to_char,
                    max_samples=cer_samples, verbose=True,
                )
                print(f"  Train CER: {train_cer:.1%} | Val CER: {val_cer:.1%} ({cer_samples} samples)")
            else:
                print(f"  Train CER: {train_cer:.1%} ({cer_samples} samples)")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Training complete. Best loss: {best_loss:.4f}. Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HWM")
    parser.add_argument("--mode", choices=["mixed", "full", "adapt"], default="mixed")
    parser.add_argument("--model-version", choices=["v2", "v3", "v4", "v5"], default="v5")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data", default="alto", choices=["alto", "synthetic"])
    parser.add_argument("--alto-dirs", nargs="+", default=config.ALTO_DIRS)
    parser.add_argument(
        "--unannotated-dirs",
        nargs="+",
        default=None,
        help="Extra dirs with ALTO pages used without their text (adapt data). "
        "In mixed mode, defaults to --alto-dirs if not specified.",
    )
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers (0=main thread)",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    start_epoch = 1
    optimizer_state = None
    scheduler_state = None
    scaler_state = None

    if args.data == "alto":
        ver = args.model_version
        if ver == "v5":
            img_h = config.IMG_HEIGHT_V5
            ws = None
            stride = None
        elif ver == "v4":
            img_h = config.IMG_HEIGHT_V4
            ws = config.WINDOW_SIZE_V4
            stride = config.STRIDE_V4
        elif ver == "v3":
            img_h = config.IMG_HEIGHT_V3
            ws = config.WINDOW_SIZE_V3
            stride = config.STRIDE_V3
        else:
            img_h = config.IMG_HEIGHT_V2
            ws = config.WINDOW_SIZE
            stride = config.STRIDE

        dataset = AltoLineDataset(args.alto_dirs, img_height=img_h, augment=True)
        char_to_idx, idx_to_char = dataset.get_alphabet()
        print(f"Alphabet: {len(char_to_idx)} characters")
        num_classes = len(char_to_idx) + 1

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        if ver == "v5":
            collate = partial(collate_alto_v5_fn, char_to_idx=char_to_idx)
        else:
            collate = partial(
                collate_alto_fn, window_size=ws, stride=stride, char_to_idx=char_to_idx
            )
        pin_mem = device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.num_workers,
            pin_memory=pin_mem,
            persistent_workers=args.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            collate_fn=collate,
            num_workers=args.num_workers,
            pin_memory=pin_mem,
            persistent_workers=args.num_workers > 0,
        )

        # --- Build adapt_loader for mixed mode ---
        # NOTE: when --unannotated-dirs is not provided, the same ALTO dirs
        # are used for both supervised and self-supervised steps. Each image
        # is then seen twice per epoch (once with CTC, once without). This is
        # acceptable (different objectives, random augmentation) but for a
        # true semi-supervised benefit, provide separate unannotated scans
        # via --unannotated-dirs.
        adapt_loader = None
        if args.mode == "mixed":
            unannotated_dirs = args.unannotated_dirs or args.alto_dirs
            adapt_ds = UnannotatedLineDataset(
                unannotated_dirs, img_height=img_h, augment=True
            )
            if ver == "v5":
                adapt_collate = collate_unannotated_v5_fn
            else:
                adapt_collate = partial(
                    collate_unannotated_fn, window_size=ws, stride=stride
                )
            adapt_loader = DataLoader(
                adapt_ds,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=adapt_collate,
                num_workers=args.num_workers,
                pin_memory=pin_mem,
                persistent_workers=args.num_workers > 0,
            )
            print(f"Mixed mode: {len(train_ds)} annotated + {len(adapt_ds)} unannotated lines")
    else:
        raise ValueError("Only 'alto' data mode is supported")

    # CTC head is needed in full and mixed modes, or if checkpoint had one
    need_ctc = args.mode in ("full", "mixed")
    ckpt = None
    ckpt_ctc_classes = None

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ckpt_ctc_classes = ckpt.get("config", {}).get("num_classes")
        # Restore char_to_idx from checkpoint if available
        ckpt_char_to_idx = ckpt.get("char_to_idx")
        if ckpt_char_to_idx:
            char_to_idx = ckpt_char_to_idx
            idx_to_char = {v: k for k, v in char_to_idx.items()}

    # Preserve CTC head from checkpoint even in adapt mode
    if need_ctc:
        model_num_classes = num_classes
    elif ckpt_ctc_classes:
        model_num_classes = ckpt_ctc_classes
    else:
        model_num_classes = None

    if ver == "v5":
        model = HWMv5(
            img_height=config.IMG_HEIGHT_V5,
            embedding_dim=config.EMBEDDING_DIM_V5,
            num_layers=config.NUM_LAYERS_V5,
            num_heads=config.NUM_HEADS_V5,
            ff_dim=config.FF_DIM_V5,
            dropout=config.DROPOUT,
            num_classes=model_num_classes,
            lambda_ctc=config.LAMBDA_CTC_V5,
            ctc_hidden=config.CTC_HIDDEN_V5,
        ).to(device)
        save_path = "hwm_v5.pt"
    elif ver == "v4":
        model = HWMv4(
            img_height=config.IMG_HEIGHT_V4,
            window_size=config.WINDOW_SIZE_V4,
            embedding_dim=config.EMBEDDING_DIM_V4,
            num_layers=config.NUM_LAYERS_V4,
            num_heads=config.NUM_HEADS_V4,
            ff_dim=config.FF_DIM_V4,
            dropout=config.DROPOUT,
            num_classes=model_num_classes,
            lambda_ctc=config.LAMBDA_CTC_V4,
            ctc_hidden=config.CTC_HIDDEN_V4,
        ).to(device)
        save_path = "hwm_v4.pt"
    elif ver == "v3":
        model = HWMv3(
            img_height=config.IMG_HEIGHT_V3,
            window_size=config.WINDOW_SIZE_V3,
            embedding_dim=config.EMBEDDING_DIM_V3,
            num_layers=config.NUM_LAYERS_V3,
            num_heads=config.NUM_HEADS_V3,
            ff_dim=config.FF_DIM_V3,
            dropout=config.DROPOUT,
            num_classes=model_num_classes,
            lambda_ctc=config.LAMBDA_CTC_V3,
        ).to(device)
        save_path = "hwm_v3.pt"
    else:
        model = HWMv2(
            img_height=config.IMG_HEIGHT_V2,
            window_size=config.WINDOW_SIZE,
            embedding_dim=config.EMBEDDING_DIM_V2,
            num_layers=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS,
            ff_dim=config.FF_DIM_V2,
            dropout=config.DROPOUT,
            num_classes=model_num_classes,
        ).to(device)
        save_path = "hwm_v2.pt"

    print(f"Model params: {model.count_parameters():,}")

    if ckpt is not None:
        state_dict = ckpt["model_state_dict"]
        model_state = model.state_dict()
        filtered = {
            k: v
            for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped = set(state_dict.keys()) - set(filtered.keys())
        new_keys = set(model_state.keys()) - set(state_dict.keys())
        if skipped:
            print(f"  Warning: skipped layers (shape mismatch): {skipped}")
        if new_keys:
            print(f"  Warning: new layers not in checkpoint: {new_keys}")
        model.load_state_dict(filtered, strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1

        # Restore optimizer/scheduler only if model architecture matches exactly
        if not skipped and not new_keys:
            optimizer_state = ckpt.get("optimizer_state_dict")
            scheduler_state = ckpt.get("scheduler_state_dict")
            scaler_state = ckpt.get("scaler_state_dict")
        else:
            optimizer_state = None
            scheduler_state = None
            scaler_state = None

        prev_loss = ckpt.get("loss", "N/A")
        print(
            f"Resuming from {args.checkpoint} (epoch {start_epoch - 1}, loss {prev_loss})"
        )

    train(
        model,
        train_loader,
        val_loader,
        adapt_loader=adapt_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        mode=args.mode,
        device=device,
        save_path=save_path,
        idx_to_char=idx_to_char,
        char_to_idx=char_to_idx,
        start_epoch=start_epoch,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        scaler_state=scaler_state,
    )
