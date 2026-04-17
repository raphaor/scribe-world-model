"""
Multi-task training for HWM-v2/v3/v4/v5.
Supports:
  - full: prediction + SIGReg + CTC (with ALTO data)
  - adapt: prediction + SIGReg only (self-supervised)
  - mixed (default): alternates full and adapt batches each step
  - Resume from checkpoint (optimizer, scheduler, epoch state restored)

Curriculum / fine-tuning (pretrain adapt → fine-tune full):
  - Param groups: encoder/predictor get a lower LR than the CTC head
    (see --encoder-lr-mult). Standard fine-tuning recipe: the head is fresh
    and needs a stronger push than the pretrained trunk.
  - Linear warmup (--warmup-epochs) avoids wrecking pretrained features when
    CTC gradients first start flowing.
  - Optional encoder freeze (--freeze-encoder-epochs) lets the head stabilize
    on top of frozen features before unfreezing the trunk (ULMFiT style).
  - Phase switch (different --mode than the checkpoint) auto-resets the
    epoch counter, optimizer and scheduler, so the LR budget of the new
    phase is not squashed by the previous phase's cosine decay.
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


def _set_encoder_frozen(model, frozen):
    """Freeze/unfreeze encoder + predictor trunk (CTC head stays trainable)."""
    for p in model.encoder.parameters():
        p.requires_grad_(not frozen)
    for p in model.predictor.parameters():
        p.requires_grad_(not frozen)


def _build_param_groups(model, lr, encoder_lr_mult):
    """
    Discriminative LR: the trunk (encoder + predictor) is typically
    pretrained and moves at a fraction of the base LR. The CTC head is
    fresh after a phase switch and uses the full LR.
    """
    trunk_params = (
        list(model.encoder.parameters()) + list(model.predictor.parameters())
    )
    groups = [{"params": trunk_params, "lr": lr * encoder_lr_mult, "name": "trunk"}]
    if model.ctc_head is not None:
        groups.append(
            {"params": list(model.ctc_head.parameters()), "lr": lr, "name": "head"}
        )
    return groups


def _build_scheduler(optimizer, remaining_epochs, warmup_epochs):
    """Linear warmup (if any) followed by cosine decay over remaining epochs."""
    if warmup_epochs > 0 and remaining_epochs > warmup_epochs:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining_epochs - warmup_epochs, eta_min=1e-6,
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
        )
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, remaining_epochs), eta_min=1e-6,
    )


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
    encoder_lr_mult=0.1,
    warmup_epochs=0,
    freeze_encoder_epochs=0,
):
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    param_groups = _build_param_groups(model, lr, encoder_lr_mult)
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

    # T_max covers only the epochs of this invocation. On a phase switch
    # the caller resets start_epoch to 1 so the new phase gets a full
    # LR budget instead of inheriting the previous phase's cosine decay.
    remaining_epochs = max(1, num_epochs - start_epoch + 1)
    # Warmup only makes sense when we're not resuming mid-schedule.
    effective_warmup = warmup_epochs if scheduler_state is None else 0
    scheduler = _build_scheduler(optimizer, remaining_epochs, effective_warmup)

    # Resuming an old checkpoint (pre param-groups) would mismatch the new
    # optimizer structure; fall back to a fresh state rather than crashing.
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except (ValueError, KeyError) as e:
            print(f"  Could not restore optimizer state ({e}); using fresh optimizer.")
            optimizer_state = None
            scheduler_state = None
    if scheduler_state is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
        except (ValueError, KeyError) as e:
            print(f"  Could not restore scheduler state ({e}); using fresh scheduler.")
    if scaler_state is not None and use_amp:
        try:
            scaler.load_state_dict(scaler_state)
        except (ValueError, KeyError):
            pass

    group_summary = ", ".join(
        f"{g.get('name', i)}={g['lr']:.2e}"
        for i, g in enumerate(optimizer.param_groups)
    )
    print(
        f"Optimizer: AdamW, {len(optimizer.param_groups)} group(s) [{group_summary}], "
        f"warmup={effective_warmup}ep, cosine over {remaining_epochs}ep"
    )
    if freeze_encoder_epochs > 0:
        print(f"  Encoder+predictor frozen for {freeze_encoder_epochs} epoch(s).")

    best_loss = float("inf")

    for epoch in range(start_epoch, num_epochs + 1):
        if freeze_encoder_epochs > 0:
            frozen = (epoch - start_epoch) < freeze_encoder_epochs
            _set_encoder_frozen(model, frozen)

        lr_str = " ".join(
            f"{g.get('name', i)}={g['lr']:.2e}"
            for i, g in enumerate(optimizer.param_groups)
        )
        losses = train_epoch(
            model, train_loader, optimizer, device, epoch,
            mode=mode, scaler=scaler, adapt_loader=adapt_loader,
        )
        scheduler.step()

        loss_str = " | ".join(f"{k}={v:.4f}" for k, v in losses.items())
        print(f"Epoch {epoch}/{num_epochs} (lr {lr_str}) - {loss_str}")

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
                    "mode": mode,
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
    parser.add_argument(
        "--encoder-lr-mult",
        type=float,
        default=0.1,
        help="LR multiplier for the encoder+predictor trunk vs the CTC head "
        "(default 0.1). Use 1.0 to disable discriminative LR.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Linear LR warmup epochs at the start of this training run. "
        "Auto-enabled to 2 on phase switch unless explicitly set.",
    )
    parser.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=0,
        help="Freeze encoder+predictor for the first N epochs so the CTC "
        "head can stabilize on top of frozen features (ULMFiT style).",
    )
    parser.add_argument(
        "--phase-restart",
        action="store_true",
        help="Force phase-switch behavior: reset epoch counter, optimizer, "
        "scheduler even when the mode matches the checkpoint.",
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
            ctc_num_lstm=config.CTC_NUM_LSTM_V5,
            jepa_num_targets=config.JEPA_NUM_TARGETS_V5,
            jepa_min_size=config.JEPA_MIN_SIZE_V5,
            jepa_max_size=config.JEPA_MAX_SIZE_V5,
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

        # Phase switch: if the training mode changed (e.g. adapt → full for
        # curriculum fine-tuning), the previous optimizer/scheduler state is
        # no longer meaningful — Adam moments were tuned for different loss
        # scales and the cosine LR has already decayed. Start fresh.
        ckpt_mode = ckpt.get("mode")
        phase_switch = args.phase_restart or (
            ckpt_mode is not None and ckpt_mode != args.mode
        )
        if phase_switch:
            label = f"{ckpt_mode} -> {args.mode}" if ckpt_mode else "explicit"
            print(
                f"Phase switch ({label}): resetting epoch counter, optimizer, scheduler."
            )
            start_epoch = 1
            optimizer_state = None
            scheduler_state = None
            scaler_state = None
            if args.warmup_epochs == 0:
                args.warmup_epochs = 2
                print(
                    "  Auto-enabling 2-epoch linear warmup (override with --warmup-epochs)."
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
        encoder_lr_mult=args.encoder_lr_mult,
        warmup_epochs=args.warmup_epochs,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
    )
