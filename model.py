"""
HWM-v1 Model - Handwriting World Model
Complete architecture combining encoder and predictor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import (
    CNNEncoder, Conv2DEncoder, Conv2DEncoderV2, Conv2DEncoderV3,
    KrakenEncoder, ViTEncoder,
)
from predictor import TransformerPredictor, JEPACrossAttnPredictor, MAEDecoder
from loss import HWMLoss, HybridLoss, MAEHybridLoss
from ctc_head import CTCHead, CTCHeadBiLSTM
from jepa import sample_jepa_mask, sample_2d_block_mask
import config


class HWMv1(nn.Module):
    """
    Handwriting World Model v1

    Architecture:
        Image columns → [Encoder] → Embeddings → [Predictor] → Next embedding

    Training objective:
        Predict z_{t+1} from z_{0:t}
        Loss: MSE(z_{t+1}, ż_{t+1}) + λ * SIGReg(Z)
    """

    def __init__(
        self,
        img_height=32,
        window_size=10,
        embedding_dim=64,
        num_layers=2,
        num_heads=2,
        ff_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.img_height = img_height
        self.window_size = window_size
        self.embedding_dim = embedding_dim

        # Encoder: columns → embeddings
        self.encoder = CNNEncoder(
            img_height=img_height, window_size=window_size, embedding_dim=embedding_dim
        )

        # Predictor: embedding sequence → next embedding
        self.predictor = TransformerPredictor(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # Loss function
        self.criterion = HWMLoss(lambda_sigreg=config.SIGREG_LAMBDA)

    def encode_sequence(self, img_columns):
        """
        Encode image columns to embeddings
        Args:
            img_columns: (B, T, H, W) or (B, T, H*W)
        Returns:
            z_seq: (B, T, D) embeddings
        """
        B, T = img_columns.shape[:2]

        # Flatten batch and time
        if img_columns.dim() == 4:
            H, W = img_columns.shape[2], img_columns.shape[3]
            img_columns = img_columns.view(B, T, H, W)
            img_flat = img_columns.view(B * T, H, W)
        else:
            img_flat = img_columns.view(B * T, -1)

        # Encode
        z_flat = self.encoder(img_flat)  # (B*T, D)

        # Reshape to sequence
        z_seq = z_flat.view(B, T, -1)  # (B, T, D)

        return z_seq

    def forward(self, img_columns):
        """
        Forward pass: encode columns and predict next embeddings (dense)
        Args:
            img_columns: (B, T, H, W) where T >= 2
        Returns:
            z_pred: (B, T-1, D) predicted next embedding at each position
            z_seq: (B, T, D) all embeddings
        """
        z_seq = self.encode_sequence(img_columns)       # (B, T, D)
        z_pred = self.predictor(z_seq[:, :-1, :])       # (B, T-1, D)
        return z_pred, z_seq

    def compute_loss(self, img_columns):
        """
        Compute training loss
        Args:
            img_columns: (B, T, H, W) where T >= 2
        Returns:
            total_loss, losses_dict
        """
        z_pred, z_seq = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()             # (B, T-1, D)
        total_loss, losses_dict = self.criterion(z_pred, z_target, z_seq)
        return total_loss, losses_dict

    def predict_future(self, img_columns, steps=1):
        """
        Predict multiple future embeddings
        Args:
            img_columns: (B, T, H, W)
            steps: number of future steps
        Returns:
            z_future: (B, steps, D)
        """
        # Encode
        z_seq = self.encode_sequence(img_columns)

        # Predict
        z_future = self.predictor.predict_sequence(z_seq, steps=steps)

        return z_future

    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HWMv2(nn.Module):
    """
    Handwriting World Model v2

    Architecture:
        Image columns -> [Conv2D Encoder] -> Embeddings
                                               |
                              +----------------+----------------+
                              |                                 |
                   [Transformer Predictor]          [CTC Recognition Head]
                   (next embedding, causal)         (character logits)
    """

    def __init__(
        self,
        img_height=48,
        window_size=10,
        embedding_dim=96,
        num_layers=2,
        num_heads=2,
        ff_dim=192,
        dropout=0.1,
        num_classes=None,
        lambda_sigreg=0.1,
        lambda_ctc=1.0,
    ):
        super().__init__()

        self.img_height = img_height
        self.window_size = window_size
        self.embedding_dim = embedding_dim

        self.encoder = Conv2DEncoder(img_height, window_size, embedding_dim)
        self.predictor = TransformerPredictor(
            embedding_dim, num_layers, num_heads, ff_dim, dropout
        )

        self.ctc_head = CTCHead(embedding_dim, num_classes) if num_classes else None
        self.criterion = HybridLoss(lambda_sigreg, lambda_ctc)

    def encode_sequence(self, img_columns):
        B, T = img_columns.shape[:2]
        z_seq = self.encoder(
            img_columns.reshape(B * T, img_columns.shape[2], img_columns.shape[3])
        ).view(B, T, -1)
        return z_seq

    def forward(self, img_columns):
        z_seq = self.encode_sequence(img_columns)       # (B, T, D)
        z_pred = self.predictor(z_seq[:, :-1, :])       # (B, T-1, D)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head else None
        return z_pred, z_seq, ctc_logits

    def compute_loss(
        self, img_columns, targets=None, input_lengths=None, target_lengths=None
    ):
        z_pred, z_seq, ctc_logits = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()             # (B, T-1, D)
        return self.criterion(
            z_pred, z_target, z_seq, ctc_logits, targets, input_lengths, target_lengths
        )

    def adapt(self, img_columns):
        z_pred, z_seq, _ = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()             # (B, T-1, D)
        return self.criterion(z_pred, z_target, z_seq)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HWMv3(nn.Module):
    """
    Handwriting World Model v3
    Deeper encoder, wider window, larger transformer.
    """

    def __init__(
        self,
        img_height=48,
        window_size=32,
        embedding_dim=128,
        num_layers=4,
        num_heads=4,
        ff_dim=384,
        dropout=0.1,
        num_classes=None,
        lambda_sigreg=0.1,
        lambda_ctc=2.0,
    ):
        super().__init__()
        self.img_height = img_height
        self.window_size = window_size
        self.embedding_dim = embedding_dim

        self.encoder = Conv2DEncoderV2(img_height, window_size, embedding_dim)
        self.predictor = TransformerPredictor(
            embedding_dim, num_layers, num_heads, ff_dim, dropout
        )
        self.ctc_head = CTCHead(embedding_dim, num_classes) if num_classes else None
        self.criterion = HybridLoss(lambda_sigreg, lambda_ctc)

    def encode_sequence(self, img_columns):
        B, T = img_columns.shape[:2]
        z_seq = self.encoder(
            img_columns.reshape(B * T, img_columns.shape[2], img_columns.shape[3])
        ).view(B, T, -1)
        return z_seq

    def forward(self, img_columns):
        z_seq = self.encode_sequence(img_columns)       # (B, T, D)
        z_pred = self.predictor(z_seq[:, :-1, :])       # (B, T-1, D)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head else None
        return z_pred, z_seq, ctc_logits

    def compute_loss(
        self, img_columns, targets=None, input_lengths=None, target_lengths=None
    ):
        z_pred, z_seq, ctc_logits = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()             # (B, T-1, D)
        return self.criterion(
            z_pred, z_target, z_seq, ctc_logits, targets, input_lengths, target_lengths
        )

    def adapt(self, img_columns):
        z_pred, z_seq, _ = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()             # (B, T-1, D)
        return self.criterion(z_pred, z_target, z_seq)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model():
    """Create HWM model with config defaults"""
    model = HWMv1(
        img_height=config.IMG_HEIGHT,
        window_size=config.WINDOW_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        ff_dim=config.FF_DIM,
        dropout=config.DROPOUT,
    )
    return model


class HWMv4(nn.Module):
    """
    Handwriting World Model v4
    Wider encoder (preserved horizontal resolution), larger transformer,
    BiLSTM CTC head for inter-frame context.
    """

    def __init__(
        self,
        img_height=48,
        window_size=32,
        embedding_dim=256,
        num_layers=4,
        num_heads=8,
        ff_dim=512,
        dropout=0.1,
        num_classes=None,
        lambda_sigreg=0.1,
        lambda_ctc=2.0,
        ctc_hidden=256,
    ):
        super().__init__()
        self.img_height = img_height
        self.window_size = window_size
        self.embedding_dim = embedding_dim

        self.encoder = Conv2DEncoderV3(img_height, window_size, embedding_dim)
        self.predictor = TransformerPredictor(
            embedding_dim, num_layers, num_heads, ff_dim, dropout
        )
        self.ctc_head = (
            CTCHeadBiLSTM(embedding_dim, num_classes, hidden_dim=ctc_hidden)
            if num_classes else None
        )
        self.criterion = HybridLoss(lambda_sigreg, lambda_ctc)

    def encode_sequence(self, img_columns):
        B, T = img_columns.shape[:2]
        z_seq = self.encoder(
            img_columns.reshape(B * T, img_columns.shape[2], img_columns.shape[3])
        ).view(B, T, -1)
        return z_seq

    def forward(self, img_columns):
        z_seq = self.encode_sequence(img_columns)       # (B, T, D)
        z_pred = self.predictor(z_seq[:, :-1, :])       # (B, T-1, D)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head else None
        return z_pred, z_seq, ctc_logits

    def compute_loss(
        self, img_columns, targets=None, input_lengths=None, target_lengths=None
    ):
        z_pred, z_seq, ctc_logits = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()             # (B, T-1, D)
        return self.criterion(
            z_pred, z_target, z_seq, ctc_logits, targets, input_lengths, target_lengths
        )

    def adapt(self, img_columns):
        z_pred, z_seq, _ = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()             # (B, T-1, D)
        return self.criterion(z_pred, z_target, z_seq)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HWMv5(nn.Module):
    """
    Handwriting World Model v5
    Kraken-style conv encoder on full line images + world model.
    No frame extraction — conv layers produce the embedding sequence directly.
    """

    def __init__(
        self,
        img_height=120,
        embedding_dim=256,
        num_layers=4,
        num_heads=8,
        ff_dim=512,
        dropout=0.1,
        num_classes=None,
        lambda_sigreg=0.1,
        lambda_ctc=0.5,
        lambda_pred=1.0,
        ctc_hidden=256,
        ctc_num_lstm=1,
        jepa_num_targets=4,
        jepa_min_size=4,
        jepa_max_size=10,
        use_jepa=True,
        target_norm=False,
        pred_loss_type="infonce",
        infonce_temp=0.1,
    ):
        super().__init__()
        self.img_height = img_height
        self.embedding_dim = embedding_dim
        self.jepa_num_targets = jepa_num_targets
        self.jepa_min_size = jepa_min_size
        self.jepa_max_size = jepa_max_size
        self.use_jepa = use_jepa
        self.target_norm = target_norm
        self.lambda_pred = lambda_pred
        self.pred_loss_type = pred_loss_type

        self.encoder = KrakenEncoder(img_height, embedding_dim)
        # Non-causal predictor: bidirectional attention over the latent
        # sequence, used to reconstruct masked target blocks from context.
        self.predictor = TransformerPredictor(
            embedding_dim, num_layers, num_heads, ff_dim, dropout, causal=False,
        )
        self.ctc_head = (
            CTCHeadBiLSTM(embedding_dim, num_classes, hidden_dim=ctc_hidden,
                          num_lstm_layers=ctc_num_lstm)
            if num_classes else None
        )
        # Learnable [MASK] token swapped in at target positions.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.criterion = HybridLoss(
            lambda_sigreg=lambda_sigreg,
            lambda_ctc=lambda_ctc,
            lambda_pred=lambda_pred,
            target_norm=target_norm,
            pred_loss_type=pred_loss_type,
            infonce_temp=infonce_temp,
        )

    def forward(self, img):
        """
        Inference-time forward: encoder + CTC head only. The predictor is
        only used during training (JEPA pretext task), so we skip it here
        to save compute during eval / recognize.
        Returns:
            (None, z_seq, ctc_logits) — tuple kept for call-site compat.
        """
        z_seq = self.encoder(img)                        # (B, T, D)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head is not None else None
        return None, z_seq, ctc_logits

    def _jepa_predict(self, z_seq, input_lengths=None):
        """
        I-JEPA style: mask several blocks of frames with a learnable
        [MASK] token, run the non-causal predictor on the resulting
        context, and return (pred@targets, stopgrad_z@targets). The
        encoder receives gradients only via the context positions; the
        target side is detached.
        """
        B, T, D = z_seq.shape
        mask = sample_jepa_mask(
            B, T,
            num_targets=self.jepa_num_targets,
            min_size=self.jepa_min_size,
            max_size=self.jepa_max_size,
            valid_lengths=input_lengths,
            device=z_seq.device,
        )
        mask_tok = self.mask_token.expand(B, T, D)
        z_ctx = torch.where(mask.unsqueeze(-1), mask_tok, z_seq)
        z_pred_full = self.predictor(z_ctx)               # (B, T, D), non-causal

        if mask.any():
            z_pred_t = z_pred_full[mask]                  # (N, D)
            z_tgt_t = z_seq.detach()[mask]                # (N, D)
        else:
            # Degenerate case (sequence shorter than min block size):
            # no targets sampled. Use the first position so the loss is
            # well-defined; gradient magnitude is tiny.
            z_pred_t = z_pred_full[:, 0, :]
            z_tgt_t = z_seq.detach()[:, 0, :]

        return z_pred_t, z_tgt_t

    def compute_loss(
        self, img, targets=None, input_lengths=None, target_lengths=None
    ):
        z_seq = self.encoder(img)
        if self.use_jepa:
            z_pred_t, z_tgt_t = self._jepa_predict(z_seq, input_lengths)
        else:
            z_pred_t, z_tgt_t = None, None
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head is not None else None
        return self.criterion(
            z_pred_t, z_tgt_t, z_seq,
            ctc_logits, targets, input_lengths, target_lengths,
        )

    def adapt(self, img, input_lengths=None):
        z_seq = self.encoder(img)
        if self.use_jepa:
            z_pred_t, z_tgt_t = self._jepa_predict(z_seq, input_lengths)
        else:
            z_pred_t, z_tgt_t = None, None
        return self.criterion(z_pred_t, z_tgt_t, z_seq)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HWMv6(HWMv5):
    """
    Handwriting World Model v6

    v5 + projection head on the JEPA branch (SimCLR / VICReg style).

    The motivation: the v5 encoder is shared between two objectives
    that pull in different directions.
      - CTC wants z_seq to be character-discriminative: each frame
        should uniquely identify the glyph it covers.
      - JEPA wants z_seq to be mutually predictable: each frame should
        be easy to infer from its neighbours.

    These goals are not identical. A 2-layer MLP inserted on the JEPA
    side absorbs the SSL-specific transformation: raw z_seq stays
    aligned with the CTC objective, and the projector weights learn
    whatever extra shaping InfoNCE / VICReg want on top. The projector
    is discarded at inference — nothing changes in the CTC path.

    Concretely:
        z_seq ─┬─► CTC head                          (unchanged)
               │
               └─► _jepa_predict:
                     predictor(z_seq w/ mask tokens) ─► pred
                     z_seq.detach()                  ─► target
                     proj_head(pred), proj_head(target) ─► InfoNCE

    VICReg still operates on raw z_seq (anti-collapse on the encoder,
    not on the projector).
    """

    def __init__(self, proj_dim=None, proj_hidden=None, **kwargs):
        super().__init__(**kwargs)
        proj_dim = proj_dim or self.embedding_dim
        proj_hidden = proj_hidden or self.embedding_dim
        self.proj_head = nn.Sequential(
            nn.Linear(self.embedding_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_dim),
        )

    def _jepa_predict(self, z_seq, input_lengths=None):
        B, T, D = z_seq.shape
        mask = sample_jepa_mask(
            B, T,
            num_targets=self.jepa_num_targets,
            min_size=self.jepa_min_size,
            max_size=self.jepa_max_size,
            valid_lengths=input_lengths,
            device=z_seq.device,
        )
        mask_tok = self.mask_token.expand(B, T, D)
        z_ctx = torch.where(mask.unsqueeze(-1), mask_tok, z_seq)
        z_pred_full = self.predictor(z_ctx)

        if mask.any():
            z_pred_t = z_pred_full[mask]
            z_tgt_t = z_seq.detach()[mask]
        else:
            z_pred_t = z_pred_full[:, 0, :]
            z_tgt_t = z_seq.detach()[:, 0, :]

        # Route both pred and target through the same projection head
        # before the loss. JEPA gradients now reach the encoder only
        # after passing through the projector.
        p_pred = self.proj_head(z_pred_t)
        p_tgt = self.proj_head(z_tgt_t)
        return p_pred, p_tgt


class HWMv7(HWMv6):
    """
    Handwriting World Model v7

    v6 + true I-JEPA cross-attention predictor.

    In v5/v6, the predictor did "in-place mask-token filling": target
    positions in z_seq were replaced by a [MASK] token and the whole
    sequence went through a bidirectional self-attention transformer.
    The predictions were read out at target positions. Two problems:

      - Context positions can attend to [MASK] tokens. The context
        representation is subtly contaminated by the mask pattern.
      - Self-attention among mask tokens at different target positions
        lets them share info in ways that aren't a pure "predict from
        context" operation.

    v7 splits the two roles explicitly:

      context_encoder = self.predictor (reused) run with
          src_key_padding_mask hiding target + padding positions. Output
          at context positions is a clean "context-aware" embedding.

      jepa_predictor = new module (cross-attention decoder). Queries =
          mask_token + positional encoding at every sequence position.
          K/V = context encoder output. memory_key_padding_mask hides
          target + padding from cross-attention. Mask tokens self-attend
          among themselves and cross-attend to context; context never
          sees the mask tokens.

    Rest (projection head from v6, VICReg on raw z_seq, CTC on raw
    z_seq) is unchanged.
    """

    def __init__(self, jepa_pred_layers=None, **kwargs):
        super().__init__(**kwargs)
        pred_layers = jepa_pred_layers or config.JEPA_PRED_LAYERS_V7
        self.jepa_predictor = JEPACrossAttnPredictor(
            embedding_dim=self.embedding_dim,
            num_layers=pred_layers,
            # Reuse the v5/v6 transformer head count and FFN width so
            # the cross-attn predictor has a similar capacity profile.
            num_heads=config.NUM_HEADS_V5,
            ff_dim=config.FF_DIM_V5,
        )

    def _jepa_predict(self, z_seq, input_lengths=None):
        B, T, D = z_seq.shape
        target_mask = sample_jepa_mask(
            B, T,
            num_targets=self.jepa_num_targets,
            min_size=self.jepa_min_size,
            max_size=self.jepa_max_size,
            valid_lengths=input_lengths,
            device=z_seq.device,
        )

        # Positions to ignore as context: target blocks + padding beyond
        # input_lengths. We pass this as src_key_padding_mask to the
        # context encoder and as memory_key_padding_mask to the cross-
        # attention predictor.
        ctx_kpm = target_mask
        if input_lengths is not None:
            ar = torch.arange(T, device=z_seq.device)
            pad_mask = ar[None, :] >= input_lengths[:, None]
            ctx_kpm = target_mask | pad_mask

        # Safety: if a row has all context positions hidden (very short
        # line fully masked), unmask position 0 so attention softmax
        # stays defined. This is extremely rare in practice.
        all_masked = ctx_kpm.all(dim=1)
        if all_masked.any():
            ctx_kpm = ctx_kpm.clone()
            ctx_kpm[all_masked, 0] = False

        # Context encoder: bidirectional self-attention over z_seq with
        # target+padding positions hidden from attention. Context output
        # at non-masked positions is used as memory.
        z_ctx_enc = self.predictor(z_seq, src_key_padding_mask=ctx_kpm)

        # Cross-attention predictor: mask-token queries at every position
        # attend to z_ctx_enc (with ctx_kpm hiding target+padding keys).
        pred_full = self.jepa_predictor(
            context=z_ctx_enc,
            memory_key_padding_mask=ctx_kpm,
            mask_token=self.mask_token,
            seq_len=T,
        )

        if target_mask.any():
            z_pred_t = pred_full[target_mask]
            z_tgt_t = z_seq.detach()[target_mask]
        else:
            z_pred_t = pred_full[:, 0, :]
            z_tgt_t = z_seq.detach()[:, 0, :]

        # v6 projection head kept.
        p_pred = self.proj_head(z_pred_t)
        p_tgt = self.proj_head(z_tgt_t)
        return p_pred, p_tgt


class HWMv8(nn.Module):
    """
    Handwriting World Model v8

    Two structural changes vs v5-v7:

      1. 2D patch encoder (ViT) instead of a 1D Kraken CNN. Handwriting
         has genuine 2D content (ascenders, descenders, diacritics)
         that a tall-thin-strip encoder compresses prematurely.
      2. MAE pretext task: predict the RAW PIXEL values of masked 2D
         patches via a small decoder. Pixels are external, fixed
         targets — the scale-collapse and trivial-mean failure modes
         that plagued v5 (SIGReg/VICReg, MSE-then-InfoNCE migrations)
         cannot apply.

    Flow:

      Image (B, H, W)
        → ViTEncoder         (2D patches → self-attention grid)
        → tokens (B, N_v*N_h, D)
               │
               ├─► mean over N_v rows → (B, N_h, D) → CTC head (BiLSTM)
               │
               └─► MAE decoder:
                     sample 2D block mask
                     mask_token substitution at masked positions
                     shallow transformer + pixel head
                     MSE(pred_pixels, true_pixels) at masked & valid
    """

    def __init__(
        self,
        img_height=120,
        patch_h=15,
        patch_w=16,
        embedding_dim=384,
        num_layers=4,
        num_heads=8,
        ff_dim=1536,
        dropout=0.1,
        num_classes=None,
        lambda_mae=1.0,
        lambda_ctc=1.0,
        ctc_hidden=256,
        ctc_num_lstm=1,
        dec_dim=256,
        dec_layers=2,
        dec_heads=8,
        dec_ff=1024,
        mask_num_blocks=4,
        mask_min_h=2,
        mask_max_h=4,
        mask_min_w=4,
        mask_max_w=16,
        max_n_h=400,
        use_mae=True,
    ):
        super().__init__()
        self.img_height = img_height
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.embedding_dim = embedding_dim
        self.n_v = img_height // patch_h
        self.use_mae = use_mae
        self.mask_num_blocks = mask_num_blocks
        self.mask_min_h = mask_min_h
        self.mask_max_h = mask_max_h
        self.mask_min_w = mask_min_w
        self.mask_max_w = mask_max_w
        self.max_n_h = max_n_h

        self.encoder = ViTEncoder(
            img_height=img_height,
            patch_h=patch_h,
            patch_w=patch_w,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_n_h=max_n_h,
        )
        self.decoder = (
            MAEDecoder(
                encoder_dim=embedding_dim,
                decoder_dim=dec_dim,
                num_layers=dec_layers,
                num_heads=dec_heads,
                ff_dim=dec_ff,
                dropout=dropout,
                patch_h=patch_h,
                patch_w=patch_w,
                max_n_h=max_n_h,
                n_v=self.n_v,
            )
            if use_mae
            else None
        )
        self.ctc_head = (
            CTCHeadBiLSTM(
                embedding_dim, num_classes,
                hidden_dim=ctc_hidden, num_lstm_layers=ctc_num_lstm,
            )
            if num_classes
            else None
        )
        self.criterion = MAEHybridLoss(
            lambda_mae=lambda_mae, lambda_ctc=lambda_ctc
        )

    def _round_up_width(self, img):
        """Right-pad image width to a multiple of patch_w."""
        W = img.shape[-1]
        pad = (self.patch_w - W % self.patch_w) % self.patch_w
        if pad > 0:
            img = F.pad(img, (0, pad))
        return img

    def _convert_lengths_to_patches(self, input_lengths):
        """
        Collate gives input_lengths in Kraken units (W // 8). We need
        n_h_valid in patch-grid units (W // patch_w). Factor = patch_w // 8.
        """
        factor = max(1, self.patch_w // 8)
        return input_lengths // factor

    def _padding_mask(self, n_h_valid, n_v, n_h, device):
        """(B, n_v*n_h) True = padding position (column >= n_h_valid[b])."""
        B = n_h_valid.size(0)
        ar = torch.arange(n_h, device=device)
        col_pad = ar[None, :] >= n_h_valid[:, None]     # (B, N_h)
        pad2d = col_pad.unsqueeze(1).expand(B, n_v, n_h)
        return pad2d.reshape(B, n_v * n_h)

    def _pool_vertical(self, tokens, n_v, n_h):
        B, _, D = tokens.shape
        return tokens.reshape(B, n_v, n_h, D).mean(dim=1)  # (B, N_h, D)

    def forward(self, img):
        """
        Inference-time forward: encoder + vertical pool + CTC head.
        Returned tuple matches the (pred, z_seq, ctc_logits) contract
        used by the evaluation pipeline.
        """
        img = self._round_up_width(img)
        tokens, (n_v, n_h) = self.encoder(img)
        z_seq = self._pool_vertical(tokens, n_v, n_h)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head is not None else None
        return None, z_seq, ctc_logits

    def compute_loss(
        self, img, targets=None, input_lengths=None, target_lengths=None
    ):
        img = self._round_up_width(img)
        B, H, W = img.shape
        n_v = self.n_v
        n_h = W // self.patch_w

        n_h_valid = None
        pad_mask_flat = None
        if input_lengths is not None:
            n_h_valid = torch.clamp(
                self._convert_lengths_to_patches(input_lengths), max=n_h
            )
            pad_mask_flat = self._padding_mask(n_h_valid, n_v, n_h, img.device)

        tokens, (n_v_g, n_h_g) = self.encoder(img, src_key_padding_mask=pad_mask_flat)

        z_seq = self._pool_vertical(tokens, n_v_g, n_h_g)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head is not None else None

        pred_pixels = None
        target_pixels = None
        scored_mask = None
        if self.use_mae and self.decoder is not None:
            mask2d = sample_2d_block_mask(
                B, n_v_g, n_h_g,
                num_blocks=self.mask_num_blocks,
                min_h=self.mask_min_h, max_h=self.mask_max_h,
                min_w=self.mask_min_w, max_w=self.mask_max_w,
                valid_h_lengths=n_h_valid, device=img.device,
            )
            mask_flat = mask2d.reshape(B, n_v_g * n_h_g)

            pred_pixels = self.decoder(
                enc_tokens=tokens,
                mask_flat=mask_flat,
                n_v=n_v_g, n_h=n_h_g,
                key_padding_mask=pad_mask_flat,
            )
            target_pixels = self.encoder.patchify_pixels(img).reshape(
                B, n_v_g * n_h_g, -1
            )
            valid = (
                ~pad_mask_flat if pad_mask_flat is not None
                else torch.ones_like(mask_flat)
            )
            scored_mask = mask_flat & valid

        return self.criterion(
            pred_pixels=pred_pixels,
            target_pixels=target_pixels,
            valid_mask=scored_mask,
            ctc_logits=ctc_logits,
            targets=targets,
            input_lengths=n_h_valid,
            target_lengths=target_lengths,
        )

    def adapt(self, img, input_lengths=None):
        return self.compute_loss(img, input_lengths=input_lengths)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """Test complete model"""
    print("\n" + "=" * 60)
    print("Testing HWM-v1 Model")
    print("=" * 60)

    # Create model
    model = create_model()

    # Count parameters
    total_params = model.count_parameters()
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    predictor_params = sum(p.numel() for p in model.predictor.parameters())

    print(f"\nParameter counts:")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Predictor: {predictor_params:,}")
    print(f"  Total: {total_params:,}")

    if total_params > config.MAX_PARAMS:
        print(f"  ⚠️  WARNING: Exceeds max params ({config.MAX_PARAMS:,})")
    else:
        print(f"  ✓ Under max params limit ({config.MAX_PARAMS:,})")

    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = config.BATCH_SIZE
    seq_len = 10
    img_columns = torch.randn(
        batch_size, seq_len, config.IMG_HEIGHT, config.WINDOW_SIZE
    )

    with torch.no_grad():
        z_pred, z_seq = model(img_columns)
        z_future = model.predict_future(img_columns, steps=3)

    print(f"  Input shape: {img_columns.shape}")
    print(f"  Embeddings shape: {z_seq.shape}")
    print(f"  Prediction shape: {z_pred.shape}")
    print(f"  Future predictions shape: {z_future.shape}")

    # Test loss computation
    print(f"\nTesting loss computation...")
    total_loss, losses_dict = model.compute_loss(img_columns)
    print(f"  Total loss: {losses_dict['total']:.4f}")
    print(f"  Pred loss: {losses_dict['pred']:.4f}")
    print(f"  SIGReg loss: {losses_dict['sigreg']:.4f}")

    print(f"\n✓ Model working correctly!")
    print("=" * 60)

    return model


if __name__ == "__main__":
    test_model()
