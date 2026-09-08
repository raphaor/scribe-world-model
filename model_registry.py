"""Per-model-version configuration registry.

Single source of truth for everything that varies by ``--model-version``:
    * data loading: img_height, collate style, window/stride, bucketing
    * model construction: a ``builder`` callable
    * training overrides: force_no_amp, force_encoder_lr_mult
    * checkpoint save_path

Adding a new version
--------------------
1. Add the model class in ``model.py``.
2. Write a ``_build_vN(args, num_classes)`` function below.
3. Add a ``ModelSpec`` entry in ``REGISTRY``.

Nothing in ``train.py`` should need to learn the new version name. The
``--model-version`` argparse choices, the dataset height, the collate, the
bucketing flag and the model construction all read from this registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch.nn as nn

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
    HWMv16,
    HWMv17,
    HWMv18,
    HWMv19,
    HWMv20,
    LectaurepClone,
)


@dataclass(frozen=True)
class ModelSpec:
    """Routing data for one ``--model-version``."""

    # --- data ---
    img_height: int
    collate_style: str                          # "v5" (full-line) | "windowed"
    window_size: Optional[int] = None           # required when collate_style == "windowed"
    stride: Optional[int] = None
    use_bucketing: bool = True
    cnn_width_stride: int = 8                    # CNN horizontal downsample (3x MaxPool(2)=8 for v5+); used by --min-frames-per-char

    # --- checkpoint ---
    save_path: str = ""

    # --- training overrides applied after CLI parsing ---
    force_no_amp: bool = False
    force_encoder_lr_mult: Optional[float] = None

    # --- model construction ---
    # builder(args, num_classes) -> nn.Module on CPU. train.py calls .to(device).
    builder: Callable[..., nn.Module] = None


# =============================================================================
# Builders -- one per version. Bodies are copied verbatim from the historical
# train.py dispatch; lambda/config logic is preserved bit-for-bit so that
# checkpoints stay reproducible.
# =============================================================================


def _build_v2(args, num_classes):
    return HWMv2(
        img_height=config.IMG_HEIGHT_V2,
        window_size=config.WINDOW_SIZE,
        embedding_dim=config.EMBEDDING_DIM_V2,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        ff_dim=config.FF_DIM_V2,
        dropout=config.DROPOUT,
        num_classes=num_classes,
    )


def _build_v3(args, num_classes):
    return HWMv3(
        img_height=config.IMG_HEIGHT_V3,
        window_size=config.WINDOW_SIZE_V3,
        embedding_dim=config.EMBEDDING_DIM_V3,
        num_layers=config.NUM_LAYERS_V3,
        num_heads=config.NUM_HEADS_V3,
        ff_dim=config.FF_DIM_V3,
        dropout=config.DROPOUT,
        num_classes=num_classes,
        lambda_ctc=config.LAMBDA_CTC_V3,
    )


def _build_v4(args, num_classes):
    return HWMv4(
        img_height=config.IMG_HEIGHT_V4,
        window_size=config.WINDOW_SIZE_V4,
        embedding_dim=config.EMBEDDING_DIM_V4,
        num_layers=config.NUM_LAYERS_V4,
        num_heads=config.NUM_HEADS_V4,
        ff_dim=config.FF_DIM_V4,
        dropout=config.DROPOUT,
        num_classes=num_classes,
        lambda_ctc=config.LAMBDA_CTC_V4,
        ctc_hidden=config.CTC_HIDDEN_V4,
    )


def _v5_family_kwargs(args, num_classes):
    """Shared kwargs for v5/v6/v7 (same CNN+Transformer+JEPA skeleton)."""
    target_norm = (
        args.target_norm if args.target_norm is not None else config.TARGET_NORM_V5
    )
    pred_loss_type = args.pred_loss or config.PRED_LOSS_V5
    if args.no_jepa:
        lambda_pred = 0.0
        use_jepa = False
    else:
        lambda_pred = (
            args.lambda_pred if args.lambda_pred is not None else config.LAMBDA_PRED_V5
        )
        use_jepa = lambda_pred > 0
    print(
        f"JEPA config: use_jepa={use_jepa} lambda_pred={lambda_pred} "
        f"pred_loss={pred_loss_type} target_norm={target_norm} "
        f"num_targets={config.JEPA_NUM_TARGETS_V5} "
        f"size=[{config.JEPA_MIN_SIZE_V5},{config.JEPA_MAX_SIZE_V5}] "
        f"embed_dim={config.EMBEDDING_DIM_V5}"
    )
    return dict(
        img_height=config.IMG_HEIGHT_V5,
        embedding_dim=config.EMBEDDING_DIM_V5,
        num_layers=config.NUM_LAYERS_V5,
        num_heads=config.NUM_HEADS_V5,
        ff_dim=config.FF_DIM_V5,
        dropout=config.DROPOUT,
        num_classes=num_classes,
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


def _build_v5(args, num_classes):
    return HWMv5(**_v5_family_kwargs(args, num_classes))


def _build_v6(args, num_classes):
    print(
        f"v6 projection head: in={config.EMBEDDING_DIM_V6} "
        f"hidden={config.PROJ_HIDDEN_V6} out={config.PROJ_DIM_V6}"
    )
    return HWMv6(
        proj_dim=config.PROJ_DIM_V6,
        proj_hidden=config.PROJ_HIDDEN_V6,
        **_v5_family_kwargs(args, num_classes),
    )


def _build_v7(args, num_classes):
    print(
        f"v7 projection head: in={config.EMBEDDING_DIM_V7} "
        f"hidden={config.PROJ_HIDDEN_V7} out={config.PROJ_DIM_V7} | "
        f"cross-attn predictor: {config.JEPA_PRED_LAYERS_V7} layers"
    )
    return HWMv7(
        proj_dim=config.PROJ_DIM_V7,
        proj_hidden=config.PROJ_HIDDEN_V7,
        jepa_pred_layers=config.JEPA_PRED_LAYERS_V7,
        **_v5_family_kwargs(args, num_classes),
    )


def _build_v8(args, num_classes):
    # v8: ViT + MAE. ``--no-jepa`` doubles as "disable the SSL branch".
    if args.no_jepa:
        lambda_mae = 0.0
        use_mae = False
    else:
        lambda_mae = (
            args.lambda_pred if args.lambda_pred is not None else config.LAMBDA_MAE_V8
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
    return HWMv8(
        img_height=config.IMG_HEIGHT_V8,
        patch_h=config.PATCH_H_V8,
        patch_w=config.PATCH_W_V8,
        embedding_dim=config.EMBEDDING_DIM_V8,
        num_layers=config.NUM_LAYERS_V8,
        num_heads=config.NUM_HEADS_V8,
        ff_dim=config.FF_DIM_V8,
        dropout=config.DROPOUT,
        num_classes=num_classes,
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
    )


def _build_v9(args, num_classes):
    # v9: hybrid CNN+ViT + MSN + SIGReg + linear CTC. ``--no-jepa`` zeroes MSN.
    if args.no_jepa:
        lambda_msn = 0.0
        use_msn = False
    else:
        lambda_msn = (
            args.lambda_pred if args.lambda_pred is not None else config.LAMBDA_MSN_V9
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
    return HWMv9(
        img_height=config.IMG_HEIGHT_V9,
        stem_channels=config.STEM_CHANNELS_V9,
        patch_h=config.PATCH_H_V9,
        patch_w=config.PATCH_W_V9,
        embedding_dim=config.EMBEDDING_DIM_V9,
        num_layers=config.NUM_LAYERS_V9,
        num_heads=config.NUM_HEADS_V9,
        ff_dim=config.FF_DIM_V9,
        dropout=config.DROPOUT,
        num_classes=num_classes,
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
    )


def _build_v10(args, num_classes):
    if args.no_jepa:
        lambda_pred = 0.0
        use_jepa = False
    else:
        lambda_pred = (
            args.lambda_pred if args.lambda_pred is not None else config.LAMBDA_PRED_V10
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
    return HWMv10(
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
        num_classes=num_classes,
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
    )


def _build_v11(args, num_classes):
    # v11: Kraken 1D encoder + SimSiam consistency. ``--no-jepa`` disables SSL.
    if args.no_jepa:
        lambda_cons = 0.0
        use_pretext = False
    else:
        lambda_cons = (
            args.lambda_pred if args.lambda_pred is not None else config.LAMBDA_CONS_V11
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
    return HWMv11(
        img_height=config.IMG_HEIGHT_V11,
        embedding_dim=config.EMBEDDING_DIM_V11,
        pred_hidden=config.PRED_HIDDEN_V11,
        num_classes=num_classes,
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
    )


def _build_v12(args, num_classes):
    # v12: Kraken conv + Transformer + InfoNCE + SIGReg + CTC.
    if args.no_jepa:
        lambda_jepa = 0.0
        use_pretext = False
    else:
        lambda_jepa = (
            args.lambda_pred if args.lambda_pred is not None else config.LAMBDA_JEPA_V12
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
    return HWMv12(
        img_height=config.IMG_HEIGHT_V12,
        embedding_dim=config.EMBEDDING_DIM_V12,
        num_layers=config.NUM_LAYERS_V12,
        num_heads=config.NUM_HEADS_V12,
        ff_dim=config.FF_DIM_V12,
        dropout=config.DROPOUT,
        num_classes=num_classes,
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
    )


def _build_v13(args, num_classes):
    if args.lambda_pred is not None and args.lambda_pred == 0:
        lambda_jepa = 0.0
        use_pretext = False
    else:
        lambda_jepa = (
            args.lambda_pred if args.lambda_pred is not None else config.LAMBDA_JEPA_V13
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
    return HWMv12(
        img_height=config.IMG_HEIGHT_V12,
        embedding_dim=config.EMBEDDING_DIM_V13,
        num_layers=config.NUM_LAYERS_V13,
        num_heads=config.NUM_HEADS_V13,
        ff_dim=config.FF_DIM_V13,
        dropout=config.DROPOUT,
        num_classes=num_classes,
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
    )


def _build_v14(args, num_classes):
    # v14: compromise capacity + unified SIGReg. Full training only.
    if args.lambda_pred is not None and args.lambda_pred == 0:
        lambda_jepa = 0.0
        use_pretext = False
    else:
        lambda_jepa = (
            args.lambda_pred if args.lambda_pred is not None else config.LAMBDA_JEPA_V14
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
    return HWMv12(
        img_height=config.IMG_HEIGHT_V12,
        embedding_dim=config.EMBEDDING_DIM_V14,
        num_layers=config.NUM_LAYERS_V14,
        num_heads=config.NUM_HEADS_V14,
        ff_dim=config.FF_DIM_V14,
        dropout=config.DROPOUT,
        num_classes=num_classes,
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
    )


def _build_v15(args, num_classes):
    # Lectaurep clone: pure CTC, no Transformer/JEPA/SIGReg.
    print(
        f"Lectaurep clone: hidden={config.LECTAUREP_HIDDEN} "
        f"lstm_layers={config.LECTAUREP_NUM_LSTM} "
        f"dropout={config.LECTAUREP_DROPOUT}"
    )
    return LectaurepClone(
        img_height=config.LECTAUREP_IMG_HEIGHT,
        num_classes=num_classes,
        hidden=config.LECTAUREP_HIDDEN,
        num_lstm_layers=config.LECTAUREP_NUM_LSTM,
        dropout=config.LECTAUREP_DROPOUT,
    )


def _build_v16(args, num_classes):
    # v16: v15 training recipe + v14 encoder (Transformer + JEPA + SIGReg).
    if args.lambda_pred is not None and args.lambda_pred == 0:
        lambda_jepa = 0.0
        use_pretext = False
    else:
        lambda_jepa = (
            args.lambda_pred if args.lambda_pred is not None else config.LAMBDA_JEPA_V16
        )
        use_pretext = lambda_jepa > 0
    lambda_sigreg_v16 = (
        args.lambda_sigreg
        if args.lambda_sigreg is not None
        else config.LAMBDA_SIGREG_V16
    )
    print(
        f"v16 config: use_pretext={use_pretext} lambda_jepa={lambda_jepa} "
        f"lambda_sigreg={lambda_sigreg_v16} lambda_ctc={config.LAMBDA_CTC_V16} "
        f"embed_dim={config.EMBEDDING_DIM_V16} layers={config.NUM_LAYERS_V16} "
        f"heads={config.NUM_HEADS_V16} ff={config.FF_DIM_V16} | "
        f"lstm: {config.NUM_LSTM_V16}×BiLSTM({config.LSTM_HIDDEN_V16}) "
        f"drop_mid={config.LSTM_DROPOUT_MID_V16} "
        f"drop_last={config.LSTM_DROPOUT_LAST_V16} | "
        f"mask: {config.JEPA_NUM_TARGETS_V16} blocks "
        f"[{config.JEPA_MIN_SIZE_V16},{config.JEPA_MAX_SIZE_V16}] frames | "
        f"sigreg: {config.SIGREG_PROJECTIONS_V16} proj, "
        f"{config.SIGREG_KNOTS_V16} knots"
    )
    return HWMv16(
        img_height=config.LECTAUREP_IMG_HEIGHT,  # 120, same as v15
        embedding_dim=config.EMBEDDING_DIM_V16,
        num_layers=config.NUM_LAYERS_V16,
        num_heads=config.NUM_HEADS_V16,
        ff_dim=config.FF_DIM_V16,
        dropout=config.DROPOUT_V16,
        num_classes=num_classes,
        lambda_ctc=config.LAMBDA_CTC_V16,
        lambda_jepa=lambda_jepa,
        lambda_sigreg=lambda_sigreg_v16,
        lambda_wc=config.LAMBDA_WC_V16,
        lstm_hidden=config.LSTM_HIDDEN_V16,
        num_lstm_layers=config.NUM_LSTM_V16,
        lstm_dropout_mid=config.LSTM_DROPOUT_MID_V16,
        lstm_dropout_last=config.LSTM_DROPOUT_LAST_V16,
        proj_dim=config.PROJ_DIM_V16,
        proj_hidden=config.PROJ_HIDDEN_V16,
        jepa_num_targets=config.JEPA_NUM_TARGETS_V16,
        jepa_min_size=config.JEPA_MIN_SIZE_V16,
        jepa_max_size=config.JEPA_MAX_SIZE_V16,
        sigreg_projections=config.SIGREG_PROJECTIONS_V16,
        sigreg_knots=config.SIGREG_KNOTS_V16,
        infonce_temp=config.INFONCE_TEMP_V16,
        supcon_temp=config.SUPCON_TEMP_V16,
        use_pretext=use_pretext,
        use_writer_contrastive=config.USE_WRITER_CONTRASTIVE_V16,
        use_checkpoint=args.grad_checkpoint,
    )


def _build_v17(args, num_classes):
    # v17: retour BiLSTM + JEPA + SIGReg. Pas de Transformer.
    if args.no_jepa or (args.lambda_pred is not None and args.lambda_pred == 0):
        lambda_jepa = 0.0
        use_pretext = False
    else:
        lambda_jepa = (
            args.lambda_pred if args.lambda_pred is not None else config.LAMBDA_JEPA_V17
        )
        use_pretext = lambda_jepa > 0
    lambda_sigreg_v17 = (
        args.lambda_sigreg
        if args.lambda_sigreg is not None
        else config.LAMBDA_SIGREG_V17
    )
    print(
        f"v17 config: use_pretext={use_pretext} lambda_jepa={lambda_jepa} "
        f"lambda_sigreg={lambda_sigreg_v17} lambda_ctc={config.LAMBDA_CTC_V17} "
        f"lstm: {config.NUM_LSTM_V17}×BiLSTM({config.LSTM_HIDDEN_V17}) "
        f"drop_mid={config.LSTM_DROPOUT_MID_V17} "
        f"drop_last={config.LSTM_DROPOUT_LAST_V17} | "
        f"mask: {config.JEPA_NUM_TARGETS_V17} blocks "
        f"[{config.JEPA_MIN_SIZE_V17},{config.JEPA_MAX_SIZE_V17}] frames | "
        f"sigreg: {config.SIGREG_PROJECTIONS_V17} proj, "
        f"{config.SIGREG_KNOTS_V17} knots"
    )
    return HWMv17(
        img_height=config.LECTAUREP_IMG_HEIGHT,  # 120, meme CNN que v15
        num_classes=num_classes,
        lambda_ctc=config.LAMBDA_CTC_V17,
        lambda_jepa=lambda_jepa,
        lambda_sigreg=lambda_sigreg_v17,
        lambda_wc=config.LAMBDA_WC_V17,
        lstm_hidden=config.LSTM_HIDDEN_V17,
        num_lstm_layers=config.NUM_LSTM_V17,
        lstm_dropout_mid=config.LSTM_DROPOUT_MID_V17,
        lstm_dropout_last=config.LSTM_DROPOUT_LAST_V17,
        proj_dim=config.PROJ_DIM_V17,
        proj_hidden=config.PROJ_HIDDEN_V17,
        jepa_num_targets=config.JEPA_NUM_TARGETS_V17,
        jepa_min_size=config.JEPA_MIN_SIZE_V17,
        jepa_max_size=config.JEPA_MAX_SIZE_V17,
        sigreg_projections=config.SIGREG_PROJECTIONS_V17,
        sigreg_knots=config.SIGREG_KNOTS_V17,
        infonce_temp=config.INFONCE_TEMP_V17,
        supcon_temp=config.SUPCON_TEMP_V17,
        use_pretext=use_pretext,
        use_writer_contrastive=config.USE_WRITER_CONTRASTIVE_V17,
    )


def _build_v18(args, num_classes):
    # v18: branches CTC / JEPA decouplees apres le CNN partage.
    # BiLSTM ne voit que CTC; JEPA passe par Linear(960->384) sans LayerNorm;
    # SIGReg sur la sortie JEPA, pas sur z_seq.
    if args.no_jepa or (args.lambda_pred is not None and args.lambda_pred == 0):
        lambda_jepa = 0.0
        use_pretext = False
        lambda_sigreg_v18 = 0.0
    else:
        lambda_jepa = (
            args.lambda_pred if args.lambda_pred is not None else config.LAMBDA_JEPA_V18
        )
        use_pretext = lambda_jepa > 0
        lambda_sigreg_v18 = (
            args.lambda_sigreg
            if args.lambda_sigreg is not None
            else config.LAMBDA_SIGREG_V18
        )
    print(
        f"v18 config: use_pretext={use_pretext} lambda_jepa={lambda_jepa} "
        f"lambda_sigreg={lambda_sigreg_v18} lambda_ctc={config.LAMBDA_CTC_V18} "
        f"jepa_dim={config.JEPA_DIM_V18} (Linear 960->{config.JEPA_DIM_V18}, no LN) | "
        f"jepa_predictor: {config.JEPA_PRED_NUM_LAYERS_V18}L Transformer("
        f"{config.JEPA_PRED_NUM_HEADS_V18}h, ff={config.JEPA_PRED_DIM_FF_V18}, "
        f"drop={config.JEPA_PRED_DROPOUT_V18}) | "
        f"lstm: {config.NUM_LSTM_V18}×BiLSTM({config.LSTM_HIDDEN_V18}) "
        f"drop_mid={config.LSTM_DROPOUT_MID_V18} "
        f"drop_last={config.LSTM_DROPOUT_LAST_V18} | "
        f"mask: {config.JEPA_NUM_TARGETS_V18} blocks "
        f"[{config.JEPA_MIN_SIZE_V18},{config.JEPA_MAX_SIZE_V18}] frames | "
        f"sigreg(on z_jepa): {config.SIGREG_PROJECTIONS_V18} proj, "
        f"{config.SIGREG_KNOTS_V18} knots"
    )
    return HWMv18(
        img_height=config.LECTAUREP_IMG_HEIGHT,  # 120
        num_classes=num_classes,
        lambda_ctc=config.LAMBDA_CTC_V18,
        lambda_jepa=lambda_jepa,
        lambda_sigreg=lambda_sigreg_v18,
        lambda_wc=config.LAMBDA_WC_V18,
        jepa_dim=config.JEPA_DIM_V18,
        lstm_hidden=config.LSTM_HIDDEN_V18,
        num_lstm_layers=config.NUM_LSTM_V18,
        lstm_dropout_mid=config.LSTM_DROPOUT_MID_V18,
        lstm_dropout_last=config.LSTM_DROPOUT_LAST_V18,
        proj_dim=config.PROJ_DIM_V18,
        proj_hidden=config.PROJ_HIDDEN_V18,
        jepa_num_targets=config.JEPA_NUM_TARGETS_V18,
        jepa_min_size=config.JEPA_MIN_SIZE_V18,
        jepa_max_size=config.JEPA_MAX_SIZE_V18,
        sigreg_projections=config.SIGREG_PROJECTIONS_V18,
        sigreg_knots=config.SIGREG_KNOTS_V18,
        infonce_temp=config.INFONCE_TEMP_V18,
        supcon_temp=config.SUPCON_TEMP_V18,
        use_pretext=use_pretext,
        use_writer_contrastive=config.USE_WRITER_CONTRASTIVE_V18,
        jepa_pred_num_layers=config.JEPA_PRED_NUM_LAYERS_V18,
        jepa_pred_num_heads=config.JEPA_PRED_NUM_HEADS_V18,
        jepa_pred_dim_ff=config.JEPA_PRED_DIM_FF_V18,
        jepa_pred_dropout=config.JEPA_PRED_DROPOUT_V18,
    )


def _build_v19(args, num_classes):
    # v19: LeVJEPA transpose aux lignes manuscrites. Invariance MSE
    # globale <- locales + SIGReg Epps-Pulley sur les [cls]. Debrayable
    # via --no-jepa / --lambda-pred 0 ; lambda_sigreg = seul hp SSL.
    if args.no_jepa or (args.lambda_pred is not None and args.lambda_pred == 0):
        lambda_inv = 0.0
        use_pretext = False
        lambda_sigreg_v19 = 0.0
    else:
        lambda_inv = (
            args.lambda_pred if args.lambda_pred is not None else config.LAMBDA_INV_V19
        )
        use_pretext = lambda_inv > 0
        lambda_sigreg_v19 = (
            args.lambda_sigreg
            if args.lambda_sigreg is not None
            else config.LAMBDA_SIGREG_V19
        )
    print(
        f"v19 LeVJEPA config: use_pretext={use_pretext} "
        f"lambda_inv={lambda_inv} lambda_sigreg={lambda_sigreg_v19} "
        f"lambda_ctc={config.LAMBDA_CTC_V19} | "
        f"patch={config.PATCH_V19} (1 token = "
        f"{4 * config.PATCH_V19}px) "
        f"fenetres K={config.WINDOW_PATCHES_V19} "
        f"s={config.WINDOW_STRIDE_V19} (50% chevauchement) | "
        f"transformer block-causal {config.NUM_LAYERS_V19}x "
        f"d={config.EMBEDDING_DIM_V19} ff={config.FF_DIM_V19} RoPE | "
        f"ctc: {config.NUM_LSTM_V19}xBiLSTM({config.LSTM_HIDDEN_V19}) | "
        f"ssl: V={config.NUM_LOCAL_VIEWS_V19} vues locales "
        f"crop=[{config.LOCAL_CROP_MIN_V19},{config.LOCAL_CROP_MAX_V19}] "
        f"drop={config.TOKEN_DROP_V19} "
        f"proj={config.EMBEDDING_DIM_V19}->{config.PROJ_HIDDEN_V19}"
        f"->{config.PROJ_DIM_V19}"
    )
    return HWMv19(
        img_height=config.IMG_HEIGHT_V19,
        stem_channels=config.STEM_CHANNELS_V19,
        patch=config.PATCH_V19,
        window_patches=config.WINDOW_PATCHES_V19,
        window_stride=config.WINDOW_STRIDE_V19,
        embedding_dim=config.EMBEDDING_DIM_V19,
        num_layers=config.NUM_LAYERS_V19,
        num_heads=config.NUM_HEADS_V19,
        ff_dim=config.FF_DIM_V19,
        dropout=config.DROPOUT_V19,
        num_classes=num_classes,
        lambda_ctc=config.LAMBDA_CTC_V19,
        lambda_inv=lambda_inv,
        lambda_sigreg=lambda_sigreg_v19,
        proj_hidden=config.PROJ_HIDDEN_V19,
        proj_dim=config.PROJ_DIM_V19,
        lstm_hidden=config.LSTM_HIDDEN_V19,
        num_lstm_layers=config.NUM_LSTM_V19,
        lstm_dropout_mid=config.LSTM_DROPOUT_MID_V19,
        lstm_dropout_last=config.LSTM_DROPOUT_LAST_V19,
        num_local_views=config.NUM_LOCAL_VIEWS_V19,
        token_drop=(
            args.token_drop
            if getattr(args, "token_drop", None) is not None
            else config.TOKEN_DROP_V19
        ),
        local_crop_min=config.LOCAL_CROP_MIN_V19,
        local_crop_max=config.LOCAL_CROP_MAX_V19,
        photo_contrast=config.PHOTO_CONTRAST_V19,
        photo_brightness=config.PHOTO_BRIGHTNESS_V19,
        photo_noise_std=config.PHOTO_NOISE_STD_V19,
        sigreg_projections=config.SIGREG_PROJECTIONS_V19,
        sigreg_knots=config.SIGREG_KNOTS_V19,
        use_pretext=use_pretext,
    )


def _build_v20(args, num_classes):
    # v20 : construction v19 (encodeur CNN renforce + [cls] + SIGReg) portee
    # sur un comparateur choisi (cls vs all-tokens) et un LSTM leger.
    # Debrayable via --no-jepa / --lambda-pred 0 ; lambda_sigreg = seul hp SSL.
    if args.no_jepa or (args.lambda_pred is not None and args.lambda_pred == 0):
        lambda_inv = 0.0
        use_pretext = False
        lambda_sigreg_v20 = 0.0
    else:
        lambda_inv = (
            args.lambda_pred if args.lambda_pred is not None else config.LAMBDA_INV_V20
        )
        use_pretext = lambda_inv > 0
        lambda_sigreg_v20 = (
            args.lambda_sigreg
            if args.lambda_sigreg is not None
            else config.LAMBDA_SIGREG_V20
        )
    print(
        f"v20 config: use_pretext={use_pretext} "
        f"lambda_inv={lambda_inv} lambda_sigreg={lambda_sigreg_v20} "
        f"lambda_ctc={config.LAMBDA_CTC_V20} | "
        f"stem CNN renforce channels={config.STEM_CHANNELS_V20} | "
        f"compare_mode={config.COMPARE_MODE_V20} "
        f"(--compare-mode pour basculer cls/all) | "
        f"patch={config.PATCH_V20} (1 token = "
        f"{4 * config.PATCH_V20}px) "
        f"fenetres K={config.WINDOW_PATCHES_V20} "
        f"s={config.WINDOW_STRIDE_V20} | "
        f"transformer block-causal {config.NUM_LAYERS_V20}x "
        f"d={config.EMBEDDING_DIM_V20} ff={config.FF_DIM_V20} RoPE | "
        f"ctc: {config.NUM_LSTM_V20}xBiLSTM({config.LSTM_HIDDEN_V20}) leger | "
        f"ssl: V={config.NUM_LOCAL_VIEWS_V20} vues locales "
        f"crop=[{config.LOCAL_CROP_MIN_V20},{config.LOCAL_CROP_MAX_V20}] "
        f"drop={config.TOKEN_DROP_V20} (--token-drop) "
        f"proj={config.EMBEDDING_DIM_V20}->{config.PROJ_HIDDEN_V20}"
        f"->{config.PROJ_DIM_V20}"
    )
    return HWMv20(
        img_height=config.IMG_HEIGHT_V20,
        stem_channels=config.STEM_CHANNELS_V20,
        patch=config.PATCH_V20,
        window_patches=config.WINDOW_PATCHES_V20,
        window_stride=config.WINDOW_STRIDE_V20,
        embedding_dim=config.EMBEDDING_DIM_V20,
        num_layers=config.NUM_LAYERS_V20,
        num_heads=config.NUM_HEADS_V20,
        ff_dim=config.FF_DIM_V20,
        dropout=config.DROPOUT_V20,
        num_classes=num_classes,
        lambda_ctc=config.LAMBDA_CTC_V20,
        lambda_inv=lambda_inv,
        lambda_sigreg=lambda_sigreg_v20,
        proj_hidden=config.PROJ_HIDDEN_V20,
        proj_dim=config.PROJ_DIM_V20,
        compare_mode=(
            args.compare_mode
            if getattr(args, "compare_mode", None) is not None
            else config.COMPARE_MODE_V20
        ),
        lstm_hidden=config.LSTM_HIDDEN_V20,
        num_lstm_layers=config.NUM_LSTM_V20,
        lstm_dropout_mid=config.LSTM_DROPOUT_MID_V20,
        lstm_dropout_last=config.LSTM_DROPOUT_LAST_V20,
        num_local_views=config.NUM_LOCAL_VIEWS_V20,
        token_drop=(
            args.token_drop
            if getattr(args, "token_drop", None) is not None
            else config.TOKEN_DROP_V20
        ),
        local_crop_min=config.LOCAL_CROP_MIN_V20,
        local_crop_max=config.LOCAL_CROP_MAX_V20,
        photo_contrast=config.PHOTO_CONTRAST_V20,
        photo_brightness=config.PHOTO_BRIGHTNESS_V20,
        photo_noise_std=config.PHOTO_NOISE_STD_V20,
        sigreg_projections=config.SIGREG_PROJECTIONS_V20,
        sigreg_knots=config.SIGREG_KNOTS_V20,
        use_pretext=use_pretext,
    )


# =============================================================================
# Registry. Insertion order = --model-version --help display order.
# =============================================================================

REGISTRY: dict[str, ModelSpec] = {
    "v2": ModelSpec(
        img_height=config.IMG_HEIGHT_V2,
        collate_style="windowed",
        window_size=config.WINDOW_SIZE,
        stride=config.STRIDE,
        use_bucketing=False,
        save_path="hwm_v2.pt",
        builder=_build_v2,
    ),
    "v3": ModelSpec(
        img_height=config.IMG_HEIGHT_V3,
        collate_style="windowed",
        window_size=config.WINDOW_SIZE_V3,
        stride=config.STRIDE_V3,
        use_bucketing=False,
        save_path="hwm_v3.pt",
        builder=_build_v3,
    ),
    "v4": ModelSpec(
        img_height=config.IMG_HEIGHT_V4,
        collate_style="windowed",
        window_size=config.WINDOW_SIZE_V4,
        stride=config.STRIDE_V4,
        use_bucketing=False,
        save_path="hwm_v4.pt",
        builder=_build_v4,
    ),
    "v5": ModelSpec(
        img_height=config.IMG_HEIGHT_V5,
        collate_style="v5",
        save_path="hwm_v5.pt",
        builder=_build_v5,
    ),
    "v6": ModelSpec(
        img_height=config.IMG_HEIGHT_V5,
        collate_style="v5",
        save_path="hwm_v6.pt",
        builder=_build_v6,
    ),
    "v7": ModelSpec(
        img_height=config.IMG_HEIGHT_V5,
        collate_style="v5",
        save_path="hwm_v7.pt",
        builder=_build_v7,
    ),
    "v8": ModelSpec(
        img_height=config.IMG_HEIGHT_V8,
        collate_style="v5",
        save_path="hwm_v8.pt",
        builder=_build_v8,
    ),
    "v9": ModelSpec(
        img_height=config.IMG_HEIGHT_V9,
        collate_style="v5",
        save_path="hwm_v9.pt",
        builder=_build_v9,
    ),
    "v10": ModelSpec(
        img_height=config.IMG_HEIGHT_V9,
        collate_style="v5",
        save_path="hwm_v10.pt",
        builder=_build_v10,
    ),
    "v11": ModelSpec(
        img_height=config.IMG_HEIGHT_V11,
        collate_style="v5",
        save_path="hwm_v11.pt",
        builder=_build_v11,
    ),
    "v12": ModelSpec(
        img_height=config.IMG_HEIGHT_V12,
        collate_style="v5",
        save_path="hwm_v12.pt",
        builder=_build_v12,
    ),
    "v13": ModelSpec(
        img_height=config.IMG_HEIGHT_V12,
        collate_style="v5",
        save_path="hwm_v13.pt",
        builder=_build_v13,
    ),
    "v14": ModelSpec(
        img_height=config.IMG_HEIGHT_V12,
        collate_style="v5",
        save_path="hwm_v14.pt",
        builder=_build_v14,
    ),
    "v15": ModelSpec(
        img_height=config.IMG_HEIGHT_V12,  # 120, == LECTAUREP_IMG_HEIGHT
        collate_style="v5",
        save_path="hwm_lectaurep.pt",
        force_no_amp=True,
        force_encoder_lr_mult=1.0,
        builder=_build_v15,
    ),
    "v16": ModelSpec(
        img_height=config.IMG_HEIGHT_V12,
        collate_style="v5",
        save_path="hwm_v16.pt",
        force_no_amp=True,
        force_encoder_lr_mult=1.0,
        builder=_build_v16,
    ),
    "v17": ModelSpec(
        img_height=config.IMG_HEIGHT_V12,  # 120
        collate_style="v5",
        save_path="hwm_v17.pt",
        force_no_amp=True,
        force_encoder_lr_mult=1.0,
        builder=_build_v17,
    ),
    "v18": ModelSpec(
        img_height=config.IMG_HEIGHT_V12,  # 120
        collate_style="v5",
        save_path="hwm_v18.pt",
        force_no_amp=True,
        force_encoder_lr_mult=None,
        builder=_build_v18,
    ),
    "v19": ModelSpec(
        img_height=config.IMG_HEIGHT_V19,   # 64
        collate_style="v5",
        cnn_width_stride=64,                # 1 token = 64 px (stem /4 x patch 16)
        save_path="hwm_v19.pt",
        force_no_amp=True,
        force_encoder_lr_mult=1.0,
        builder=_build_v19,
    ),
    "v20": ModelSpec(
        img_height=config.IMG_HEIGHT_V20,   # 64
        collate_style="v5",
        cnn_width_stride=64,                # 1 token = 64 px (stem /4 x patch 16)
        save_path="hwm_v20.pt",
        force_no_amp=True,
        force_encoder_lr_mult=1.0,
        builder=_build_v20,
    ),
}


def get_spec(version: str) -> ModelSpec:
    if version not in REGISTRY:
        raise ValueError(
            f"Unknown --model-version {version!r}. "
            f"Known: {sorted(REGISTRY)}. "
            f"Register new versions in model_registry.py."
        )
    return REGISTRY[version]


def known_versions() -> list[str]:
    """Insertion-order list, used by argparse choices."""
    return list(REGISTRY)


def default_train_args():
    """Namespace with the same defaults as ``train.py`` argparse.

    Builders read these attributes to drive lambda weights and SSL flags.
    Inference scripts only need them to satisfy the builder signature —
    the training-time options don't affect ``forward()``.
    """
    import argparse
    return argparse.Namespace(
        no_jepa=False,
        no_amp=False,
        no_augment=False,
        no_bucket=False,
        lambda_pred=None,
        lambda_sigreg=None,
        target_norm=None,
        pred_loss=None,
        encoder_lr_mult=0.1,
        grad_checkpoint=False,
    )
