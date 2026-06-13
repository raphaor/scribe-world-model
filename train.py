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
from model_registry import get_spec, known_versions
from data_alto import (
    AltoLineDataset,
    UnannotatedLineDataset,
    collate_alto_fn,
    collate_unannotated_fn,
    collate_alto_v5_fn,
    collate_unannotated_v5_fn,
    line_widths,
    line_lengths,
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
        "lstm_layers",      # v15 (LectaurepClone)
        "lstm_dropouts",    # v15 (LectaurepClone)
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
        "lstm_layers",      # v15 (LectaurepClone)
        "lstm_dropouts",    # v15 (LectaurepClone)
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


def _build_scheduler(optimizer, remaining_epochs, warmup_epochs, constant_lr=False):
    """Linear warmup (if any) followed by cosine decay over remaining epochs.

    If constant_lr=True, skip the cosine entirely (matches ketos recipe).
    """
    if constant_lr and warmup_epochs == 0:
        # No scheduler at all — constant LR, exactly like ketos.
        return None
    if constant_lr and warmup_epochs > 0:
        # Warmup only, then constant.
        return optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
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
    no_amp=False,
    constant_lr=False,
):
    use_amp = device.type == "cuda" and not no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    param_groups = _build_param_groups(model, lr, encoder_lr_mult)
    optimizer = optim.Adam(param_groups)

    # T_max covers only the epochs of this invocation. On a phase switch
    # the caller resets start_epoch to 1 so the new phase gets a full
    # LR budget instead of inheriting the previous phase's cosine decay.
    remaining_epochs = max(1, num_epochs - start_epoch + 1)
    # Warmup only makes sense when we're not resuming mid-schedule.
    effective_warmup = warmup_epochs if scheduler_state is None else 0
    scheduler = _build_scheduler(
        optimizer, remaining_epochs, effective_warmup, constant_lr=constant_lr
    )

    # Resuming an old checkpoint (pre param-groups) would mismatch the new
    # optimizer structure; fall back to a fresh state rather than crashing.
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except (ValueError, KeyError) as e:
            print(f"  Could not restore optimizer state ({e}); using fresh optimizer.")
            optimizer_state = None
            scheduler_state = None
    if scheduler is not None and scheduler_state is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
        except (ValueError, KeyError) as e:
            print(f"  Could not restore scheduler state ({e}); using fresh scheduler.")
    if scaler_state is not None and use_amp:
        try:
            scaler.load_state_dict(scaler_state)
        except (ValueError, KeyError):
            pass

    schedule_str = "constant" if constant_lr else f"cosine over {remaining_epochs}ep"
    group_summary = ", ".join(
        f"{g.get('name', i)}={g['lr']:.2e}"
        for i, g in enumerate(optimizer.param_groups)
    )
    print(
        f"Optimizer: Adam, {len(optimizer.param_groups)} group(s) [{group_summary}], "
        f"warmup={effective_warmup}ep, {schedule_str}"
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
        if scheduler is not None:
            scheduler.step()

        loss_str = " | ".join(f"{k}={v:.4f}" for k, v in losses.items())
        print(f"Epoch {epoch}/{num_epochs} (lr {lr_str}) - {loss_str}")
        parts = []
        dropped = getattr(model, "_ctc_dropped_samples", 0)
        total_b = getattr(model, "_ctc_total_batches", 0)
        if dropped > 0:
            parts.append(f"[CTC] {dropped}/{total_b} samples dropped")
        if hasattr(model, "_ctc_dropped_samples"):
            model._ctc_dropped_samples = 0
            model._ctc_total_batches = 0
        from data_alto import collate_alto_v5_fn as _collate

        collate_dropped = getattr(_collate, "_total_dropped", 0)
        if collate_dropped > 0:
            parts.append(f"[collate] {collate_dropped} sample(s) dropped")
            _collate._total_dropped = 0
        if parts:
            print("  " + " | ".join(parts))

        current_loss = losses.get("total", float("inf"))
        if current_loss < best_loss:
            best_loss = current_loss
            ckpt_data = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
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
            }
            if scheduler is not None:
                ckpt_data["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(ckpt_data, save_path)

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
        choices=known_versions(),
        default="v5",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable image augmentation (rotation, shear, h-stretch, "
        "stroke width, blur, contrast, elastic deformation, noise).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP (float16). Needed for models with large "
        "intermediate dims (e.g. Lectaurep clone, 960-dim BiLSTM).",
    )
    parser.add_argument(
        "--no-bucket",
        action="store_true",
        help="Disable width bucketing (batch lines of similar width).",
    )
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
        "--min-frames-per-char",
        type=float,
        default=0.0,
        help="If >0, prune training/val lines where T=W//cnn_stride < this * "
        "encoded_length before the split. 1.0 = remove only CTC-impossible "
        "lines (restores uniform batch counts the collate otherwise erodes); "
        ">1.0 also drops cramped lines. 0.0 = disabled (default).",
    )
    parser.add_argument(
        "--length-weight-power",
        type=float,
        default=0.0,
        help="If >0, the bucket sampler draws lines with probability ∝ "
        "len(text)**power, realigning per-line sampling with the "
        "character-weighted CER. 0.5 (∝√L) is prudent, 1.0 matches CER "
        "exactly. 0.0 = uniform (default).",
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
        "--constant-lr",
        action="store_true",
        help="Disable cosine LR decay: use a constant learning rate "
        "(matches ketos/Lectaurep training recipe).",
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
            d for d in args.alto_dirs if not any(ex in d for ex in args.exclude_dirs)
        ]
        excluded = [d for d in original_dirs if d not in args.alto_dirs]
        print(f"Excluded {len(excluded)} dir(s) matching {args.exclude_dirs}:")
        for d in excluded:
            print(f"  - {d}")
        print(f"Using {len(args.alto_dirs)} dir(s):")
        for d in args.alto_dirs:
            print(f"  + {d}")
        if not args.alto_dirs:
            raise ValueError("All --alto-dirs were excluded; nothing left to train on.")

    # --- Load checkpoint metadata BEFORE dataset construction ---
    # Same logic as recognize.py: when fine-tuning from a checkpoint, its
    # alphabet (char_to_idx) and num_classes must take precedence over the
    # new dataset's. Without this, the collate encodes targets with the new
    # dataset's alphabet while the model's CTC head uses the checkpoint's —
    # the CTC head gets skipped (shape mismatch) and predictions are garbage.
    ckpt = None
    ckpt_ctc_classes = None
    ckpt_char_to_idx = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ckpt_ctc_classes = ckpt.get("config", {}).get("num_classes")
        ckpt_char_to_idx = ckpt.get("char_to_idx")
        if ckpt_char_to_idx:
            print(
                f"Checkpoint alphabet: {len(ckpt_char_to_idx)} characters "
                f"(num_classes={ckpt_ctc_classes})"
            )
        else:
            print("WARNING: checkpoint has no char_to_idx; building alphabet from data")

    if args.data == "alto":
        ver = args.model_version
        spec = get_spec(ver)
        img_h = spec.img_height
        ws = spec.window_size
        stride = spec.stride

        dataset = AltoLineDataset(
            args.alto_dirs, img_height=img_h, augment=not args.no_augment
        )
        # Build the dataset alphabet to discover new characters.
        ds_char_to_idx, _ = dataset.get_alphabet()
        if ckpt_char_to_idx:
            # Merge: preserve checkpoint char→idx mappings (so pretrained CTC
            # head weights stay valid), append new characters from the dataset
            # at the end of the index range.
            new_chars = sorted(set(ds_char_to_idx) - set(ckpt_char_to_idx))
            char_to_idx = dict(ckpt_char_to_idx)
            if new_chars:
                next_idx = max(char_to_idx.values()) + 1
                for c in new_chars:
                    char_to_idx[c] = next_idx
                    next_idx += 1
                print(
                    f"Checkpoint alphabet extended: {len(ckpt_char_to_idx)} "
                    f"+ {len(new_chars)} new = {len(char_to_idx)} chars"
                )
                print(f"  New characters: {repr(''.join(new_chars[:50]))}")
            else:
                print(f"Checkpoint alphabet: {len(char_to_idx)} characters")
            idx_to_char = {v: k for k, v in char_to_idx.items()}
        else:
            char_to_idx = ds_char_to_idx
            idx_to_char = {v: k for k, v in char_to_idx.items()}
            print(f"Alphabet from data: {len(char_to_idx)} characters")
        num_classes = len(char_to_idx) + 1  # +1 for CTC blank at index 0

        # Prune CTC-unalignable lines before the split (opt-in). The alphabet
        # is built from the full set first, so a char that only appears on a
        # pruned line keeps a stable index (just unused).
        if args.min_frames_per_char > 0.0:
            removed, kept = dataset.filter_unlearnable(
                char_to_idx,
                width_stride=spec.cnn_width_stride,
                min_frames_per_char=args.min_frames_per_char,
            )
            print(
                f"Filtered {removed} unlearnable lines "
                f"(T < {args.min_frames_per_char:g}*L, stride={spec.cnn_width_stride}); "
                f"{kept} remain"
            )

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

        if spec.collate_style == "v5":
            collate = partial(collate_alto_v5_fn, char_to_idx=char_to_idx)
        else:
            collate = partial(
                collate_alto_fn, window_size=ws, stride=stride, char_to_idx=char_to_idx
            )
        pin_mem = device.type == "cuda"
        # v5+ feeds full-line images (variable width) to the loader, so
        # bucket by width to bound peak VRAM and kill padding waste.
        # v2-v4 pre-extract fixed-size frame columns — plain batching.
        use_bucketing = spec.use_bucketing and not args.no_bucket

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
                        line_widths(ds),
                        args.batch_size,
                        shuffle=shuffle,
                        oversample_factor=args.oversample_factor,
                        long_threshold_px=args.long_threshold_px,
                        lengths=line_lengths(ds),
                        length_weight_power=args.length_weight_power,
                    ),
                    **common,
                )
            return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, **common)

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
            if spec.collate_style == "v5":
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

    # CTC head is needed in full and mixed modes, or if checkpoint had one.
    # num_classes already set correctly above (checkpoint takes precedence).
    need_ctc = args.mode in ("full", "mixed")

    # ckpt and ckpt_ctc_classes already loaded early (before dataset).

    # Preserve CTC head from checkpoint even in adapt mode
    if need_ctc or ckpt is not None:
        model_num_classes = num_classes
    else:
        model_num_classes = None

    # Build the model from the per-version spec (see model_registry.py).
    # spec.builder reads --no-jepa / --lambda-pred / --lambda-sigreg etc.
    # and prints a version-specific config summary.
    model = spec.builder(args, model_num_classes).to(device)
    save_path = spec.save_path

    # Version-specific training-loop overrides (applied only if the user is
    # on the defaults; explicit CLI overrides win).
    if spec.force_no_amp and not args.no_amp:
        print(f"  NOTE: {ver} forces --no-amp for training stability.")
        args.no_amp = True
    if spec.force_encoder_lr_mult is not None and args.encoder_lr_mult == 0.1:
        print(
            f"  NOTE: {ver} forces --encoder-lr-mult "
            f"{spec.force_encoder_lr_mult} (no discriminative LR)."
        )
        args.encoder_lr_mult = spec.force_encoder_lr_mult

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

        # Expand CTC head if the alphabet was extended beyond the checkpoint.
        # The proj weight/bias have shape (num_classes, D) / (num_classes,).
        # Old rows are copied from the checkpoint; new rows are Xavier-init.
        if ckpt_ctc_classes is not None and num_classes > ckpt_ctc_classes:
            ckpt_state = ckpt["model_state_dict"]
            ckpt_w = ckpt_state.get("ctc_head.proj.weight")
            ckpt_b = ckpt_state.get("ctc_head.proj.bias")
            if ckpt_w is not None and ckpt_b is not None:
                old_n = ckpt_w.shape[0]
                with torch.no_grad():
                    model.ctc_head.proj.weight.data[:old_n] = ckpt_w
                    model.ctc_head.proj.bias.data[:old_n] = ckpt_b
                    # New output neurons: Xavier for reasonable starting logits,
                    # zero bias so new classes start uninformative.
                    torch.nn.init.xavier_uniform_(
                        model.ctc_head.proj.weight.data[old_n:]
                    )
                    torch.nn.init.zeros_(model.ctc_head.proj.bias.data[old_n:])
                print(
                    f"  Expanded CTC head: {old_n} → {num_classes} classes "
                    f"(pretrained weights preserved, "
                    f"{num_classes - old_n} new rows Xavier-init)"
                )

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
        no_amp=args.no_amp,
        constant_lr=args.constant_lr,
    )
