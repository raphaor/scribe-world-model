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
import copy
import argparse
import time
from collections import defaultdict
from functools import partial

# Must be set before the CUDA caching allocator initialises (i.e. before
# `import torch`). expandable_segments lets the allocator grow/shrink
# segments instead of fragmenting — with variable-width batches this
# avoids spilling into slow shared GPU memory.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from model import (
    HWMv2,
    HWMv3,
    HWMv4,
    HWMv5,
    HWMv6,
    HWMv7,
    HWMv8,
    HWMv9,
    HWMv10,
    HWMv11,
    HWMv12,
    LectaurepClone,
)
from data_alto import (
    AltoLineDataset,
    UnannotatedLineDataset,
    collate_alto_fn,
    collate_unannotated_fn,
    collate_alto_v5_fn,
    collate_unannotated_v5_fn,
    line_widths,
    LengthBucketBatchSampler,
)


def _progress_bar(epoch, batch_idx, total_batches, losses, elapsed, bar_width=30):
    pct = (batch_idx + 1) / total_batches
    filled = int(bar_width * pct)
    bar = "#" * filled + "-" * (bar_width - filled)
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
    """One self-supervised training step (prediction + SIGReg only).

    In pure ``--mode adapt``, the batch comes from the supervised
    train_loader (collate_alto_v5_fn): ``(padded, targets, input_lengths,
    target_lengths, raw_texts)``. In mixed mode's adapt branch the batch
    comes from collate_unannotated_v5_fn: ``(padded, input_lengths)``.
    Older v2-v4 collate_unannotated_fn returns ``(padded,)`` only. We
    pick the right index by batch length to pass lengths to the JEPA
    mask sampler so it stays inside the valid (non-padding) region.
    """
    img_seqs = batch[0].to(device, non_blocking=True)
    if len(batch) >= 3:
        # Supervised v5 collate: input_lengths is batch[2] (batch[1] is
        # the flattened targets tensor, which is NOT per-sample lengths).
        input_lengths = batch[2].to(device, non_blocking=True)
    elif len(batch) == 2:
        # Unannotated v5 collate: (padded, input_lengths).
        input_lengths = batch[1].to(device, non_blocking=True)
    else:
        input_lengths = None

    if img_seqs.shape[1] < 2:
        return None, None

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=use_amp):
        if input_lengths is not None:
            loss, losses = model.adapt(img_seqs, input_lengths=input_lengths)
        else:
            loss, losses = model.adapt(img_seqs)
    del img_seqs
    if input_lengths is not None:
        del input_lengths
    return loss, losses


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    mode="full",
    scaler=None,
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
            _progress_bar(
                epoch, batch_idx * 2 + 1, total_batches, running, time.time() - t0
            )
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
    """Freeze/unfreeze encoder + SSL trunk (CTC head stays trainable).

    v5-v7 expose ``predictor``; v8 exposes ``decoder`` instead (MAE pixel
    head). v10 exposes ``context_transformer``, ``jepa_predictor``, and
    ``proj_head``.  Only toggle those that actually exist.
    """
    for p in model.encoder.parameters():
        p.requires_grad_(not frozen)
    for name in (
        "predictor",
        "decoder",
        "jepa_predictor",
        "proj_head",
        "context_transformer",
    ):
        sub = getattr(model, name, None)
        if sub is not None:
            for p in sub.parameters():
                p.requires_grad_(not frozen)


def _build_param_groups(model, lr, encoder_lr_mult):
    """
    Discriminative LR: the trunk (encoder + SSL modules) is typically
    pretrained and moves at a fraction of the base LR. The CTC head is
    fresh after a phase switch and uses the full LR.

    ``SSL modules`` covers the predictor (v5-v7), the MAE decoder (v8),
    cross-attn predictor (v7), and the projection head (v6/v7). We
    collect whichever attributes exist on the model.
    """
    trunk_params = list(model.encoder.parameters())
    for name in (
        "predictor",
        "decoder",
        "jepa_predictor",
        "proj_head",
        "context_transformer",
    ):
        sub = getattr(model, name, None)
        if sub is not None:
            trunk_params.extend(list(sub.parameters()))
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
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=remaining_epochs - warmup_epochs,
            eta_min=1e-6,
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, remaining_epochs),
        eta_min=1e-6,
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
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            mode=mode,
            scaler=scaler,
            adapt_loader=adapt_loader,
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

            # Train + Val CER: each on a fixed random 500-line subset
            # (seeded -> representative AND stable epoch-to-epoch). A
            # full-val pass every epoch was too slow. The protocol's
            # "official" CER stays a separate full recognize.py run.
            train_samples = min(500, len(train_loader.dataset) // 10)
            train_cer = evaluate_cer(
                model,
                train_loader,
                device,
                idx_to_char,
                max_samples=train_samples,
                verbose=False,
            )
            if val_loader:
                val_samples = min(500, len(val_loader.dataset) // 10)
                val_cer = evaluate_cer(
                    model,
                    val_loader,
                    device,
                    idx_to_char,
                    max_samples=val_samples,
                    verbose=True,
                )
                print(
                    f"  Train CER: {train_cer:.1%} ({train_samples} samp) | "
                    f"Val CER: {val_cer:.1%} ({val_samples} samp)"
                )
            else:
                print(f"  Train CER: {train_cer:.1%} ({train_samples} samp)")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Training complete. Best loss: {best_loss:.4f}. Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HWM")
    parser.add_argument("--mode", choices=["mixed", "full", "adapt"], default="mixed")
    parser.add_argument(
        "--model-version",
        choices=[
            "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        ],
        default="v5",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable image augmentation (elastic deformations).")
    parser.add_argument("--no-bucket", action="store_true",
                        help="Disable width bucketing (batch lines of similar width).")
    parser.add_argument("--data", default="alto", choices=["alto", "synthetic"])
    parser.add_argument("--alto-dirs", nargs="+", default=config.ALTO_DIRS)
    parser.add_argument(
        "--exclude-dirs",
        nargs="+",
        default=[],
        help="Substrings to exclude from --alto-dirs (case-sensitive substring "
        "match). For each dir in --alto-dirs, if any --exclude-dirs entry "
        "appears in its path, the dir is dropped. Useful for the held-out-"
        "scribe transfer protocol: pre-train SSL on all dirs except one, "
        "then fine-tune CTC on the held-out dir only.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Override the default checkpoint save path (e.g. hwm_v11.pt). "
        "Use to keep adapt/fine-tune/baseline runs from overwriting each "
        "other (e.g. --save-path hwm_v11_adapt.pt).",
    )
    parser.add_argument(
        "--unannotated-dirs",
        nargs="+",
        default=None,
        help="Extra dirs with ALTO pages used without their text (adapt data). "
        "In mixed mode, defaults to --alto-dirs if not specified.",
    )
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--oversample-factor",
        type=float,
        default=1.0,
        help="Oversample batches containing long lines (>= --long-threshold-px). "
        "2.0 = each long batch appears twice per epoch. 1.0 = disabled.",
    )
    parser.add_argument(
        "--long-threshold-px",
        type=int,
        default=800,
        help="Width threshold (px) for oversampling long-line batches.",
    )
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
    parser.add_argument(
        "--no-jepa",
        action="store_true",
        help="Disable the JEPA / world-model branch entirely (CTC-only "
        "baseline). Equivalent to --lambda-pred 0 and skipping the "
        "predictor at training time. Use this to ablate whether the "
        "self-supervised pretext task actually helps CTC.",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="(v12) Gradient-checkpoint the encoder conv stem: drop its "
        "activations and recompute them in the backward pass. ~30%% more "
        "compute for a large peak-VRAM cut — lets a bigger batch fit "
        "without spilling into shared GPU memory.",
    )
    parser.add_argument(
        "--target-norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="LayerNorm pred + stop-grad target before MSE (wav2vec2 / "
        "I-JEPA trick). Defaults to config.TARGET_NORM_V5 for v5.",
    )
    parser.add_argument(
        "--lambda-pred",
        type=float,
        default=None,
        help="Override the JEPA prediction-loss weight. Defaults to "
        "config.LAMBDA_PRED_V5 for v5. Set to 0 with --no-jepa for the "
        "CTC-only baseline.",
    )
    parser.add_argument(
        "--lambda-sigreg",
        type=float,
        default=None,
        help="Override the SIGReg / VICReg / SIGRegV2 anti-collapse weight. "
        "Defaults to the per-model config constant. Useful in pure --mode "
        "adapt to push the anti-collapse pressure when CTC isn't there to "
        "prevent shortcuts.",
    )
    parser.add_argument(
        "--pred-loss",
        choices=["mse", "infonce"],
        default=None,
        help="Prediction loss type for v5. 'mse' (legacy regression) "
        "vs 'infonce' (contrastive, default). MSE has a trivial-mean "
        "minimum that breaks self-supervised training.",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        # benchmark=True only pays off with STABLE input shapes; here the
        # batch width changes every step, so it would re-benchmark and
        # cache a conv workspace per shape — fragmentation, not speed.
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    start_epoch = 1
    optimizer_state = None
    scheduler_state = None
    scaler_state = None

    # Apply --exclude-dirs filter against --alto-dirs (substring match).
    if args.exclude_dirs:
        original_dirs = list(args.alto_dirs)
        args.alto_dirs = [
            d for d in args.alto_dirs
            if not any(ex in d for ex in args.exclude_dirs)
        ]
        excluded = [d for d in original_dirs if d not in args.alto_dirs]
        print(f"Excluded {len(excluded)} dir(s) matching {args.exclude_dirs}:")
        for d in excluded:
            print(f"  - {d}")
        print(f"Using {len(args.alto_dirs)} dir(s):")
        for d in args.alto_dirs:
            print(f"  + {d}")
        if not args.alto_dirs:
            raise ValueError(
                "All --alto-dirs were excluded; nothing left to train on."
            )

    if args.data == "alto":
        ver = args.model_version
        if ver in ("v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"):
            if ver in ("v12", "v13", "v14", "v15"):
                img_h = config.IMG_HEIGHT_V12
            elif ver == "v11":
                img_h = config.IMG_HEIGHT_V11
            elif ver in ("v9", "v10"):
                img_h = config.IMG_HEIGHT_V9
            elif ver == "v8":
                img_h = config.IMG_HEIGHT_V8
            else:
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

        dataset = AltoLineDataset(args.alto_dirs, img_height=img_h,
                                  augment=not args.no_augment)
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
        # Validation must be measured on CLEAN images. train_ds and
        # val_ds both wrap the same AltoLineDataset, so `augment` is
        # shared — rebind val to a shallow copy that shares the `samples`
        # list (no RAM duplication) but carries its own augment=False.
        # The split indices are reused, so the partition is unchanged.
        _val_base = copy.copy(dataset)
        _val_base.augment = False
        val_ds = Subset(_val_base, val_ds.indices)

        if ver in ("v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"):
            collate = partial(collate_alto_v5_fn, char_to_idx=char_to_idx)
        else:
            collate = partial(
                collate_alto_fn, window_size=ws, stride=stride, char_to_idx=char_to_idx
            )
        pin_mem = device.type == "cuda"
        # v5+ feeds full-line images (variable width) to the loader, so
        # bucket by width to bound peak VRAM and kill padding waste.
        # v2-v4 pre-extract fixed-size frame columns — plain batching.
        use_bucketing = ver in ("v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15") \
            and not args.no_bucket

        def _make_loader(ds, collate_fn, shuffle):
            common = dict(
                collate_fn=collate_fn,
                num_workers=args.num_workers,
                pin_memory=pin_mem,
                persistent_workers=args.num_workers > 0,
            )
            if use_bucketing:
                return DataLoader(
                    ds,
                    batch_sampler=LengthBucketBatchSampler(
                        line_widths(ds), args.batch_size, shuffle=shuffle,
                        oversample_factor=args.oversample_factor,
                        long_threshold_px=args.long_threshold_px,
                    ),
                    **common,
                )
            return DataLoader(
                ds, batch_size=args.batch_size, shuffle=shuffle, **common
            )

        train_loader = _make_loader(train_ds, collate, shuffle=True)
        val_loader = _make_loader(val_ds, collate, shuffle=False)

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
            if ver in ("v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"):
                adapt_collate = collate_unannotated_v5_fn
            else:
                adapt_collate = partial(
                    collate_unannotated_fn, window_size=ws, stride=stride
                )
            adapt_loader = _make_loader(adapt_ds, adapt_collate, shuffle=True)
            print(
                f"Mixed mode: {len(train_ds)} annotated + {len(adapt_ds)} unannotated lines"
            )
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

    if ver in ("v5", "v6", "v7"):
        target_norm = (
            args.target_norm if args.target_norm is not None else config.TARGET_NORM_V5
        )
        pred_loss_type = args.pred_loss or config.PRED_LOSS_V5
        if args.no_jepa:
            lambda_pred = 0.0
            use_jepa = False
        else:
            lambda_pred = (
                args.lambda_pred
                if args.lambda_pred is not None
                else config.LAMBDA_PRED_V5
            )
            use_jepa = lambda_pred > 0
        print(
            f"{ver} JEPA config: use_jepa={use_jepa} lambda_pred={lambda_pred} "
            f"pred_loss={pred_loss_type} target_norm={target_norm} "
            f"num_targets={config.JEPA_NUM_TARGETS_V5} "
            f"size=[{config.JEPA_MIN_SIZE_V5},{config.JEPA_MAX_SIZE_V5}] "
            f"embed_dim={config.EMBEDDING_DIM_V5}"
        )
        model_kwargs = dict(
            img_height=config.IMG_HEIGHT_V5,
            embedding_dim=config.EMBEDDING_DIM_V5,
            num_layers=config.NUM_LAYERS_V5,
            num_heads=config.NUM_HEADS_V5,
            ff_dim=config.FF_DIM_V5,
            dropout=config.DROPOUT,
            num_classes=model_num_classes,
            lambda_ctc=config.LAMBDA_CTC_V5,
            lambda_pred=lambda_pred,
            ctc_hidden=config.CTC_HIDDEN_V5,
            ctc_num_lstm=config.CTC_NUM_LSTM_V5,
            jepa_num_targets=config.JEPA_NUM_TARGETS_V5,
            jepa_min_size=config.JEPA_MIN_SIZE_V5,
            jepa_max_size=config.JEPA_MAX_SIZE_V5,
            use_jepa=use_jepa,
            target_norm=target_norm,
            pred_loss_type=pred_loss_type,
            infonce_temp=config.INFONCE_TEMP_V5,
        )
        if ver == "v5":
            model = HWMv5(**model_kwargs).to(device)
            save_path = "hwm_v5.pt"
        elif ver == "v6":
            print(
                f"v6 projection head: in={config.EMBEDDING_DIM_V6} "
                f"hidden={config.PROJ_HIDDEN_V6} out={config.PROJ_DIM_V6}"
            )
            model = HWMv6(
                proj_dim=config.PROJ_DIM_V6,
                proj_hidden=config.PROJ_HIDDEN_V6,
                **model_kwargs,
            ).to(device)
            save_path = "hwm_v6.pt"
        else:
            print(
                f"v7 projection head: in={config.EMBEDDING_DIM_V7} "
                f"hidden={config.PROJ_HIDDEN_V7} out={config.PROJ_DIM_V7} | "
                f"cross-attn predictor: {config.JEPA_PRED_LAYERS_V7} layers"
            )
            model = HWMv7(
                proj_dim=config.PROJ_DIM_V7,
                proj_hidden=config.PROJ_HIDDEN_V7,
                jepa_pred_layers=config.JEPA_PRED_LAYERS_V7,
                **model_kwargs,
            ).to(device)
            save_path = "hwm_v7.pt"
    elif ver == "v8":
        # v8: ViT + MAE. The ``--no-jepa`` flag doubles as "disable the
        # SSL branch" here — it zeros lambda_mae and skips the decoder.
        if args.no_jepa:
            lambda_mae = 0.0
            use_mae = False
        else:
            lambda_mae = (
                args.lambda_pred
                if args.lambda_pred is not None
                else config.LAMBDA_MAE_V8
            )
            use_mae = lambda_mae > 0
        print(
            f"v8 MAE config: use_mae={use_mae} lambda_mae={lambda_mae} "
            f"lambda_ctc={config.LAMBDA_CTC_V8} "
            f"patches={config.PATCH_H_V8}x{config.PATCH_W_V8} "
            f"embed_dim={config.EMBEDDING_DIM_V8} "
            f"dec_dim={config.DEC_DIM_V8} dec_layers={config.DEC_LAYERS_V8} "
            f"mask_blocks={config.MASK_NUM_BLOCKS_V8}"
        )
        model = HWMv8(
            img_height=config.IMG_HEIGHT_V8,
            patch_h=config.PATCH_H_V8,
            patch_w=config.PATCH_W_V8,
            embedding_dim=config.EMBEDDING_DIM_V8,
            num_layers=config.NUM_LAYERS_V8,
            num_heads=config.NUM_HEADS_V8,
            ff_dim=config.FF_DIM_V8,
            dropout=config.DROPOUT,
            num_classes=model_num_classes,
            lambda_mae=lambda_mae,
            lambda_ctc=config.LAMBDA_CTC_V8,
            ctc_hidden=config.CTC_HIDDEN_V8,
            ctc_num_lstm=config.CTC_NUM_LSTM_V8,
            dec_dim=config.DEC_DIM_V8,
            dec_layers=config.DEC_LAYERS_V8,
            dec_heads=config.DEC_HEADS_V8,
            dec_ff=config.DEC_FF_V8,
            mask_num_blocks=config.MASK_NUM_BLOCKS_V8,
            mask_min_h=config.MASK_MIN_H_V8,
            mask_max_h=config.MASK_MAX_H_V8,
            mask_min_w=config.MASK_MIN_W_V8,
            mask_max_w=config.MASK_MAX_W_V8,
            max_n_h=config.MAX_N_H_V8,
            use_mae=use_mae,
        ).to(device)
        save_path = "hwm_v8.pt"
    elif ver == "v9":
        # v9: hybrid CNN+ViT + MSN (image-masked consistency) + SIGReg
        # + linear CTC. ``--no-jepa`` zeroes the MSN term (CTC-only baseline).
        if args.no_jepa:
            lambda_msn = 0.0
            use_msn = False
        else:
            lambda_msn = (
                args.lambda_pred
                if args.lambda_pred is not None
                else config.LAMBDA_MSN_V9
            )
            use_msn = lambda_msn > 0
        print(
            f"v9 MSN config: use_msn={use_msn} lambda_msn={lambda_msn} "
            f"lambda_sigreg={config.LAMBDA_SIGREG_V9} "
            f"lambda_ctc={config.LAMBDA_CTC_V9} "
            f"stem_ch={config.STEM_CHANNELS_V9} "
            f"patch={config.PATCH_H_V9}x{config.PATCH_W_V9} "
            f"embed_dim={config.EMBEDDING_DIM_V9} "
            f"mask_blocks={config.MASK_NUM_BLOCKS_V9}"
        )
        model = HWMv9(
            img_height=config.IMG_HEIGHT_V9,
            stem_channels=config.STEM_CHANNELS_V9,
            patch_h=config.PATCH_H_V9,
            patch_w=config.PATCH_W_V9,
            embedding_dim=config.EMBEDDING_DIM_V9,
            num_layers=config.NUM_LAYERS_V9,
            num_heads=config.NUM_HEADS_V9,
            ff_dim=config.FF_DIM_V9,
            dropout=config.DROPOUT,
            num_classes=model_num_classes,
            lambda_msn=lambda_msn,
            lambda_sigreg=config.LAMBDA_SIGREG_V9,
            lambda_ctc=config.LAMBDA_CTC_V9,
            mask_num_blocks=config.MASK_NUM_BLOCKS_V9,
            mask_min_h=config.MASK_MIN_H_V9,
            mask_max_h=config.MASK_MAX_H_V9,
            mask_min_w=config.MASK_MIN_W_V9,
            mask_max_w=config.MASK_MAX_W_V9,
            max_n_h=config.MAX_N_H_V9,
            use_msn=use_msn,
        ).to(device)
        save_path = "hwm_v9.pt"
    elif ver == "v10":
        if args.no_jepa:
            lambda_pred = 0.0
            use_jepa = False
        else:
            lambda_pred = (
                args.lambda_pred
                if args.lambda_pred is not None
                else config.LAMBDA_PRED_V10
            )
            use_jepa = lambda_pred > 0
        print(
            f"v10 JEPA config: use_jepa={use_jepa} lambda_pred={lambda_pred} "
            f"lambda_sigreg={config.LAMBDA_SIGREG_V10} "
            f"lambda_ctc={config.LAMBDA_CTC_V10} "
            f"stem_ch={config.STEM_CHANNELS_V10} "
            f"patch={config.PATCH_H_V10}x{config.PATCH_W_V10} "
            f"embed_dim={config.EMBEDDING_DIM_V10} "
            f"pred_layers={config.PRED_NUM_LAYERS_V10} "
            f"mask_blocks={config.MASK_NUM_BLOCKS_V10}"
        )
        model = HWMv10(
            img_height=config.IMG_HEIGHT_V10,
            stem_channels=config.STEM_CHANNELS_V10,
            patch_h=config.PATCH_H_V10,
            patch_w=config.PATCH_W_V10,
            embedding_dim=config.EMBEDDING_DIM_V10,
            num_layers=config.NUM_LAYERS_V10,
            num_heads=config.NUM_HEADS_V10,
            ff_dim=config.FF_DIM_V10,
            pred_num_layers=config.PRED_NUM_LAYERS_V10,
            pred_ff_dim=config.PRED_FF_DIM_V10,
            dropout=config.DROPOUT,
            num_classes=model_num_classes,
            lambda_pred=lambda_pred,
            lambda_sigreg=config.LAMBDA_SIGREG_V10,
            lambda_ctc=config.LAMBDA_CTC_V10,
            sigreg_var=config.SIGREG_VAR_V10,
            sigreg_cov=config.SIGREG_COV_V10,
            sigreg_gamma=config.SIGREG_GAMMA_V10,
            ctc_hidden=config.CTC_HIDDEN_V10,
            ctc_num_lstm=config.CTC_NUM_LSTM_V10,
            mask_num_blocks=config.MASK_NUM_BLOCKS_V10,
            mask_min_h=config.MASK_MIN_H_V10,
            mask_max_h=config.MASK_MAX_H_V10,
            mask_min_w=config.MASK_MIN_W_V10,
            mask_max_w=config.MASK_MAX_W_V10,
            max_n_h=config.MAX_N_H_V10,
            use_jepa=use_jepa,
        ).to(device)
        save_path = "hwm_v10.pt"
    elif ver == "v11":
        # v11: Kraken 1D encoder + SimSiam consistency on perturbed view.
        # ``--no-jepa`` disables the SSL pretext (CTC-only baseline).
        if args.no_jepa:
            lambda_cons = 0.0
            use_pretext = False
        else:
            lambda_cons = (
                args.lambda_pred
                if args.lambda_pred is not None
                else config.LAMBDA_CONS_V11
            )
            use_pretext = lambda_cons > 0
        lambda_sigreg_v11 = (
            args.lambda_sigreg
            if args.lambda_sigreg is not None
            else config.LAMBDA_SIGREG_V11
        )
        print(
            f"v11 SimSiam config: use_pretext={use_pretext} "
            f"lambda_cons={lambda_cons} "
            f"lambda_sigreg={lambda_sigreg_v11} "
            f"lambda_ctc={config.LAMBDA_CTC_V11} "
            f"embed_dim={config.EMBEDDING_DIM_V11} "
            f"pred_hidden={config.PRED_HIDDEN_V11} | "
            f"pert: shift=±{config.PERT_V11_SHIFT_X}px "
            f"shear=±{config.PERT_V11_SHEAR_DEG}° "
            f"mask={config.PERT_V11_MASK_BLOCKS} blocks "
            f"({config.PERT_V11_MASK_W_MIN}-{config.PERT_V11_MASK_W_MAX}px)"
        )
        model = HWMv11(
            img_height=config.IMG_HEIGHT_V11,
            embedding_dim=config.EMBEDDING_DIM_V11,
            pred_hidden=config.PRED_HIDDEN_V11,
            num_classes=model_num_classes,
            lambda_cons=lambda_cons,
            lambda_sigreg=lambda_sigreg_v11,
            lambda_ctc=config.LAMBDA_CTC_V11,
            sigreg_var=config.SIGREG_VAR_V11,
            sigreg_cov=config.SIGREG_COV_V11,
            sigreg_gamma=config.SIGREG_GAMMA_V11,
            ctc_hidden=config.CTC_HIDDEN_V11,
            ctc_num_lstm=config.CTC_NUM_LSTM_V11,
            pert_shift_x=config.PERT_V11_SHIFT_X,
            pert_shear_deg=config.PERT_V11_SHEAR_DEG,
            pert_mask_blocks=config.PERT_V11_MASK_BLOCKS,
            pert_mask_w_min=config.PERT_V11_MASK_W_MIN,
            pert_mask_w_max=config.PERT_V11_MASK_W_MAX,
            pert_contrast_min=config.PERT_V11_CONTRAST_MIN,
            pert_contrast_max=config.PERT_V11_CONTRAST_MAX,
            pert_brightness=config.PERT_V11_BRIGHTNESS,
            pert_noise_std=config.PERT_V11_NOISE_STD,
            use_pretext=use_pretext,
        ).to(device)
        save_path = "hwm_v11.pt"
    elif ver == "v12":
        # v12: Kraken conv + Transformer encoder (Option B, no final LN),
        # masked-segment InfoNCE + Epps-Pulley SIGReg + CTC. ``--no-jepa``
        # disables the SSL pretext (CTC + SIGReg baseline).
        if args.no_jepa:
            lambda_jepa = 0.0
            use_pretext = False
        else:
            lambda_jepa = (
                args.lambda_pred
                if args.lambda_pred is not None
                else config.LAMBDA_JEPA_V12
            )
            use_pretext = lambda_jepa > 0
        lambda_sigreg_v12 = (
            args.lambda_sigreg
            if args.lambda_sigreg is not None
            else config.LAMBDA_SIGREG_V12
        )
        print(
            f"v12 config: use_pretext={use_pretext} lambda_jepa={lambda_jepa} "
            f"lambda_sigreg={lambda_sigreg_v12} lambda_ctc={config.LAMBDA_CTC_V12} "
            f"writer_contrastive={config.USE_WRITER_CONTRASTIVE_V12} "
            f"embed_dim={config.EMBEDDING_DIM_V12} layers={config.NUM_LAYERS_V12} | "
            f"mask: {config.JEPA_NUM_TARGETS_V12} blocks "
            f"[{config.JEPA_MIN_SIZE_V12},{config.JEPA_MAX_SIZE_V12}] frames | "
            f"sigreg: {config.SIGREG_PROJECTIONS_V12} proj, "
            f"{config.SIGREG_KNOTS_V12} knots"
        )
        model = HWMv12(
            img_height=config.IMG_HEIGHT_V12,
            embedding_dim=config.EMBEDDING_DIM_V12,
            num_layers=config.NUM_LAYERS_V12,
            num_heads=config.NUM_HEADS_V12,
            ff_dim=config.FF_DIM_V12,
            dropout=config.DROPOUT,
            num_classes=model_num_classes,
            lambda_ctc=config.LAMBDA_CTC_V12,
            lambda_jepa=lambda_jepa,
            lambda_sigreg=lambda_sigreg_v12,
            lambda_wc=config.LAMBDA_WC_V12,
            ctc_hidden=config.CTC_HIDDEN_V12,
            ctc_num_lstm=config.CTC_NUM_LSTM_V12,
            proj_dim=config.PROJ_DIM_V12,
            proj_hidden=config.PROJ_HIDDEN_V12,
            jepa_num_targets=config.JEPA_NUM_TARGETS_V12,
            jepa_min_size=config.JEPA_MIN_SIZE_V12,
            jepa_max_size=config.JEPA_MAX_SIZE_V12,
            sigreg_projections=config.SIGREG_PROJECTIONS_V12,
            sigreg_knots=config.SIGREG_KNOTS_V12,
            infonce_temp=config.INFONCE_TEMP_V12,
            supcon_temp=config.SUPCON_TEMP_V12,
            use_pretext=use_pretext,
            use_writer_contrastive=config.USE_WRITER_CONTRASTIVE_V12,
            use_checkpoint=args.grad_checkpoint,
        ).to(device)
        save_path = "hwm_v12.pt"
    elif ver == "v13":
        if args.lambda_pred is not None and args.lambda_pred == 0:
            lambda_jepa = 0.0
            use_pretext = False
        else:
            lambda_jepa = (
                args.lambda_pred
                if args.lambda_pred is not None
                else config.LAMBDA_JEPA_V13
            )
            use_pretext = lambda_jepa > 0
        lambda_sigreg_v13 = (
            args.lambda_sigreg
            if args.lambda_sigreg is not None
            else config.LAMBDA_SIGREG_V13
        )
        print(
            f"v13 config: use_pretext={use_pretext} lambda_jepa={lambda_jepa} "
            f"lambda_sigreg={lambda_sigreg_v13} lambda_ctc={config.LAMBDA_CTC_V13} "
            f"writer_contrastive={config.USE_WRITER_CONTRASTIVE_V13} "
            f"embed_dim={config.EMBEDDING_DIM_V13} layers={config.NUM_LAYERS_V13} "
            f"ctc_lstm={config.CTC_NUM_LSTM_V13} | "
            f"mask: {config.JEPA_NUM_TARGETS_V13} blocks "
            f"[{config.JEPA_MIN_SIZE_V13},{config.JEPA_MAX_SIZE_V13}] frames | "
            f"sigreg: {config.SIGREG_PROJECTIONS_V13} proj, "
            f"{config.SIGREG_KNOTS_V13} knots"
        )
        model = HWMv12(
            img_height=config.IMG_HEIGHT_V12,
            embedding_dim=config.EMBEDDING_DIM_V13,
            num_layers=config.NUM_LAYERS_V13,
            num_heads=config.NUM_HEADS_V13,
            ff_dim=config.FF_DIM_V13,
            dropout=config.DROPOUT,
            num_classes=model_num_classes,
            lambda_ctc=config.LAMBDA_CTC_V13,
            lambda_jepa=lambda_jepa,
            lambda_sigreg=lambda_sigreg_v13,
            lambda_wc=config.LAMBDA_WC_V13,
            ctc_hidden=config.CTC_HIDDEN_V13,
            ctc_num_lstm=config.CTC_NUM_LSTM_V13,
            proj_dim=config.PROJ_DIM_V13,
            proj_hidden=config.PROJ_HIDDEN_V13,
            jepa_num_targets=config.JEPA_NUM_TARGETS_V13,
            jepa_min_size=config.JEPA_MIN_SIZE_V13,
            jepa_max_size=config.JEPA_MAX_SIZE_V13,
            sigreg_projections=config.SIGREG_PROJECTIONS_V13,
            sigreg_knots=config.SIGREG_KNOTS_V13,
            infonce_temp=config.INFONCE_TEMP_V13,
            supcon_temp=config.SUPCON_TEMP_V13,
            use_pretext=use_pretext,
            use_writer_contrastive=config.USE_WRITER_CONTRASTIVE_V13,
            use_checkpoint=args.grad_checkpoint,
        ).to(device)
        save_path = "hwm_v13.pt"
    elif ver == "v14":
        # v14: compromise capacity (embed_dim=256, 3 BiLSTM CTC layers)
        # with unified SIGReg (no shape/scale split). Full training only.
        if args.lambda_pred is not None and args.lambda_pred == 0:
            lambda_jepa = 0.0
            use_pretext = False
        else:
            lambda_jepa = (
                args.lambda_pred
                if args.lambda_pred is not None
                else config.LAMBDA_JEPA_V14
            )
            use_pretext = lambda_jepa > 0
        lambda_sigreg_v14 = (
            args.lambda_sigreg
            if args.lambda_sigreg is not None
            else config.LAMBDA_SIGREG_V14
        )
        print(
            f"v14 config: use_pretext={use_pretext} lambda_jepa={lambda_jepa} "
            f"lambda_sigreg={lambda_sigreg_v14} lambda_ctc={config.LAMBDA_CTC_V14} "
            f"writer_contrastive={config.USE_WRITER_CONTRASTIVE_V14} "
            f"embed_dim={config.EMBEDDING_DIM_V14} layers={config.NUM_LAYERS_V14} "
            f"ctc_lstm={config.CTC_NUM_LSTM_V14} | "
            f"mask: {config.JEPA_NUM_TARGETS_V14} blocks "
            f"[{config.JEPA_MIN_SIZE_V14},{config.JEPA_MAX_SIZE_V14}] frames | "
            f"sigreg: {config.SIGREG_PROJECTIONS_V14} proj, "
            f"{config.SIGREG_KNOTS_V14} knots"
        )
        model = HWMv12(
            img_height=config.IMG_HEIGHT_V12,
            embedding_dim=config.EMBEDDING_DIM_V14,
            num_layers=config.NUM_LAYERS_V14,
            num_heads=config.NUM_HEADS_V14,
            ff_dim=config.FF_DIM_V14,
            dropout=config.DROPOUT,
            num_classes=model_num_classes,
            lambda_ctc=config.LAMBDA_CTC_V14,
            lambda_jepa=lambda_jepa,
            lambda_sigreg=lambda_sigreg_v14,
            lambda_wc=config.LAMBDA_WC_V14,
            ctc_hidden=config.CTC_HIDDEN_V14,
            ctc_num_lstm=config.CTC_NUM_LSTM_V14,
            proj_dim=config.PROJ_DIM_V14,
            proj_hidden=config.PROJ_HIDDEN_V14,
            jepa_num_targets=config.JEPA_NUM_TARGETS_V14,
            jepa_min_size=config.JEPA_MIN_SIZE_V14,
            jepa_max_size=config.JEPA_MAX_SIZE_V14,
            sigreg_projections=config.SIGREG_PROJECTIONS_V14,
            sigreg_knots=config.SIGREG_KNOTS_V14,
            infonce_temp=config.INFONCE_TEMP_V14,
            supcon_temp=config.SUPCON_TEMP_V14,
            use_pretext=use_pretext,
            use_writer_contrastive=config.USE_WRITER_CONTRASTIVE_V14,
            use_checkpoint=args.grad_checkpoint,
        ).to(device)
        save_path = "hwm_v14.pt"
    elif ver == "v15":
        # Lectaurep clone: pure CTC, no Transformer/JEPA/SIGReg.
        # CNN → 960-dim → 3×BiLSTM(200) → Linear → CTC
        # Exact reproduction of the official lectaurep_base architecture.
        model = LectaurepClone(
            img_height=config.LECTAUREP_IMG_HEIGHT,
            num_classes=model_num_classes,
            hidden=config.LECTAUREP_HIDDEN,
            num_lstm_layers=config.LECTAUREP_NUM_LSTM,
            dropout=config.LECTAUREP_DROPOUT,
        ).to(device)
        print(
            f"Lectaurep clone: hidden={config.LECTAUREP_HIDDEN} "
            f"lstm_layers={config.LECTAUREP_NUM_LSTM} "
            f"dropout={config.LECTAUREP_DROPOUT}"
        )
        save_path = "hwm_lectaurep.pt"
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

    # User override (e.g. to keep adapt vs fine-tune vs baseline runs separate).
    if args.save_path:
        save_path = args.save_path

    print(f"Model params: {model.count_parameters():,}")
    print(f"Save path: {save_path}")

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
