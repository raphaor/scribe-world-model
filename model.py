"""
HWM-v1 Model - Handwriting World Model
Complete architecture combining encoder and predictor
"""

import contextlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from encoder import (
    CNNEncoder,
    Conv2DEncoder,
    Conv2DEncoderV2,
    Conv2DEncoderV3,
    KrakenEncoder,
    KrakenEncoderV12,
    ViTEncoder,
    HybridCNNViTEncoder,
)
from predictor import TransformerPredictor, JEPACrossAttnPredictor, MAEDecoder
from loss import (
    HWMLoss,
    HybridLoss,
    MAEHybridLoss,
    MSNLoss,
    JEPALoss,
    SimSiamHybridLoss,
    V12Loss,
)
from ctc_head import CTCHead, CTCHeadBiLSTM
from jepa import sample_jepa_mask, sample_2d_block_mask
from loss_bundle import make_v12_bundle
import config


class HWMv16(nn.Module):
    """
    Handwriting World Model v16 — Kraken conv stem + Transformer encoder +
    2×BiLSTM(128) CTC + JEPA + SIGReg.

    Lessons from v15 applied to v14's architecture:
      - CNN uses Dropout2d (channel-level, like kraken)
      - BiLSTM dropout uses Dropout2d with p=0.3 on the last layer
      - All params collected in a single optimizer group (no _build_param_groups bug)
      - Adam (not AdamW), cosine + warmup schedule
      - 2 BiLSTM layers (Transformer already does sequential modeling)
      - 256-dim throughout (projection → Transformer → BiLSTM → CTC head)

    Flow
    ----
        line image (B, H, W)
          ├─► clean view ─► KrakenEncoderV12 ─► z  (B, T, 256)
          │       ├─ LayerNorm → 2×BiLSTM(128) → CTC
          │       ├─ SIGReg(z)
          │       └─ (optional writer contrastive)
          │
          └─► pixel-masked view ─► KrakenEncoderV12 ─► z_masked
                  InfoNCE( jepa_proj(z_masked@masked),
                           jepa_proj(sg z@masked) )

        L = λ_ctc·CTC + λ_jepa·InfoNCE + λ_sigreg·SIGReg

    The architecture reuses KrakenEncoderV12 from v12-v14 (CNN stem +
    projection + Transformer) but replaces the CTCHeadBiLSTM with
    explicit 2×BiLSTM(128) layers using Dropout2d between them.
    """

    def __init__(
        self,
        img_height=120,
        embedding_dim=256,
        num_layers=3,
        num_heads=4,
        ff_dim=1024,
        dropout=0.1,
        num_classes=None,
        lambda_ctc=1.0,
        lambda_jepa=0.5,
        lambda_sigreg=0.1,
        lambda_wc=0.2,
        lstm_hidden=128,
        num_lstm_layers=2,
        lstm_dropout_mid=0.1,
        lstm_dropout_last=0.3,
        proj_dim=128,
        proj_hidden=256,
        jepa_num_targets=4,
        jepa_min_size=8,
        jepa_max_size=20,
        sigreg_projections=256,
        sigreg_knots=17,
        infonce_temp=0.1,
        supcon_temp=0.1,
        use_pretext=True,
        use_writer_contrastive=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.img_height = img_height
        self.embedding_dim = embedding_dim
        self.use_pretext = use_pretext
        self.use_writer_contrastive = use_writer_contrastive
        self.jepa_num_targets = jepa_num_targets
        self.jepa_min_size = jepa_min_size
        self.jepa_max_size = jepa_max_size
        self.frame_stride = 8  # 3 × MaxPool(2)

        # --- Encoder: Kraken CNN stem + projection + Transformer ---
        # Same as v12-v14 but with Dropout2d in the conv stem
        self.encoder = KrakenEncoderV12(
            img_height=img_height,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
        )

        # Learnable scalar for pixel-space masking
        self.mask_pixel = nn.Parameter(torch.zeros(()))

        # JEPA projection head
        self.jepa_proj = nn.Sequential(
            nn.Linear(embedding_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_dim),
        )

        # Writer contrastive projection head (dormant unless enabled)
        self.style_proj = nn.Sequential(
            nn.Linear(embedding_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_dim),
        )

        # --- CTC path: LayerNorm → 2×BiLSTM(128) → Linear ---
        self.ctc_norm = nn.LayerNorm(embedding_dim)

        self.lstm_layers = nn.ModuleList()
        for i in range(num_lstm_layers):
            input_dim = embedding_dim if i == 0 else lstm_hidden * 2
            self.lstm_layers.append(
                nn.LSTM(
                    input_dim,
                    lstm_hidden,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )

        # Dropout2d between LSTM layers (channel-level, like kraken)
        self.lstm_dropouts = nn.ModuleList()
        for i in range(num_lstm_layers):
            p = lstm_dropout_last if i == num_lstm_layers - 1 else lstm_dropout_mid
            self.lstm_dropouts.append(nn.Dropout(p))

        # CTC output head
        self.ctc_head = CTCHead(lstm_hidden * 2, num_classes)

        # --- Loss ---
        # Composable bundle (see loss_bundle.py). Bit-for-bit equivalent
        # to the historical V12Loss aggregator but adding a new term means
        # registering a single LossTerm rather than editing the aggregator.
        self.criterion = make_v12_bundle(
            lambda_ctc=lambda_ctc,
            lambda_jepa=lambda_jepa,
            lambda_sigreg=lambda_sigreg,
            lambda_wc=lambda_wc,
            infonce_temp=infonce_temp,
            supcon_temp=supcon_temp,
            sigreg_projections=sigreg_projections,
            sigreg_knots=sigreg_knots,
        )

        # Initialize LSTM weights (ketos-style)
        self._init_lstm_weights()

    def _init_lstm_weights(self):
        """Initialize LSTM weights: orthogonal + forget gate bias 1.0."""
        for lstm in self.lstm_layers:
            for p in lstm.parameters():
                if p.data.dim() == 2:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.constant_(p.data, 0)
                    nn.init.constant_(p.data[len(p) // 4:len(p) // 2], 1.0)

    def _bilstm(self, z_seq, input_lengths=None):
        """2×BiLSTM(128): (B, T, 256) → (B, T, 256)."""
        for lstm, do in zip(self.lstm_layers, self.lstm_dropouts):
            if input_lengths is not None:
                packed = pack_padded_sequence(
                    z_seq, input_lengths.cpu(),
                    batch_first=True, enforce_sorted=False,
                )
                packed_out, _ = lstm(packed)
                z_seq, _ = pad_packed_sequence(packed_out, batch_first=True)
            else:
                z_seq, _ = lstm(z_seq)
            z_seq = do(z_seq)
        return z_seq

    def _pool_temporal(self, z_seq, input_lengths=None):
        """Padding-aware mean over the time axis → (B, D)."""
        B, T, D = z_seq.shape
        if input_lengths is None:
            return z_seq.mean(dim=1)
        ar = torch.arange(T, device=z_seq.device)
        valid = (ar[None, :] < input_lengths.clamp(max=T)[:, None]).float()
        z_masked = z_seq * valid.unsqueeze(-1)
        denom = input_lengths.clamp(min=1, max=T).float().unsqueeze(-1)
        return z_masked.sum(dim=1) / denom

    def _make_masks(self, img, T, input_lengths):
        """Sample frame-level block mask, return pixel-masked image + frame mask."""
        B, _, W = img.shape
        frame_mask = sample_jepa_mask(
            B, T,
            num_targets=self.jepa_num_targets,
            min_size=self.jepa_min_size,
            max_size=self.jepa_max_size,
            valid_lengths=input_lengths,
            device=img.device,
        )

        pixel_mask = frame_mask.repeat_interleave(self.frame_stride, dim=1)
        if pixel_mask.shape[1] < W:
            pixel_mask = F.pad(pixel_mask, (0, W - pixel_mask.shape[1]))
        else:
            pixel_mask = pixel_mask[:, :W]

        img_masked = torch.where(
            pixel_mask.unsqueeze(1),
            self.mask_pixel.to(img.dtype).expand_as(img),
            img,
        )
        return img_masked, frame_mask

    def forward(self, img, input_lengths=None):
        """Inference: encoder → LayerNorm → BiLSTM → CTC."""
        z_seq = self.encoder(img, input_lengths)
        z_seq = self._bilstm(self.ctc_norm(z_seq))
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head is not None else None
        return None, z_seq, ctc_logits

    def compute_loss(
        self,
        img,
        targets=None,
        input_lengths=None,
        target_lengths=None,
        writer_id=None,
    ):
        if input_lengths is not None:
            input_lengths = input_lengths.to(img.device)

        # 1. Encoder passes
        z_clean = self.encoder(img, input_lengths)  # (B, T, 256)
        B, T, D = z_clean.shape
        ctc_in = input_lengths.clamp(max=T) if input_lengths is not None else None

        # Valid frame mask
        valid_mask = None
        if ctc_in is not None:
            ar = torch.arange(T, device=img.device)
            valid_mask = ar[None, :] < ctc_in[:, None]

        z_masked = None
        frame_mask = None
        if self.use_pretext:
            img_masked, frame_mask = self._make_masks(img, T, ctc_in)
            if valid_mask is not None:
                frame_mask = frame_mask & valid_mask
            if frame_mask.any():
                z_masked = self.encoder(img_masked, input_lengths)
            else:
                frame_mask = None

        # 2. Heads + loss (float32 for SIGReg/InfoNCE precision)
        _f32 = (
            torch.amp.autocast("cuda", enabled=False)
            if img.is_cuda
            else contextlib.nullcontext()
        )
        with _f32:
            z_clean = z_clean.float()

            # CTC path: LayerNorm → BiLSTM → CTC
            ctc_logits = None
            if self.ctc_head is not None:
                lstm_out = self._bilstm(self.ctc_norm(z_clean))
                ctc_logits = self.ctc_head(lstm_out)

            # JEPA: masked-segment InfoNCE
            z_pred = z_target = None
            if z_masked is not None and frame_mask is not None:
                z_masked = z_masked.float()
                z_pred = self.jepa_proj(z_masked[frame_mask])
                z_target = self.jepa_proj(z_clean.detach()[frame_mask])

            # Writer contrastive (optional)
            line_vec = None
            if self.use_writer_contrastive and writer_id is not None:
                v = self._pool_temporal(z_clean, ctc_in)
                line_vec = self.style_proj(v)

            return self.criterion(
                z_pred=z_pred,
                z_target=z_target,
                z_seq=z_clean,
                valid_mask=valid_mask,
                ctc_logits=ctc_logits,
                targets=targets,
                input_lengths=ctc_in,
                target_lengths=target_lengths,
                line_vec=line_vec,
                writer_id=writer_id,
            )

    def adapt(self, img, input_lengths=None):
        """Self-supervised step: InfoNCE + SIGReg only."""
        return self.compute_loss(img, input_lengths=input_lengths)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HWMv17(nn.Module):
    """
    Handwriting World Model v17 — Kraken conv stem + 3×BiLSTM + JEPA + SIGReg.

    A return to BiLSTM simplicity with lessons from v15 (13.7% CER) and v16
    (16.5% CER, Transformer regression).

    Key design:
      - CNN stem: 4 conv layers, 3 MaxPool(2) → stride 8
      - No Linear(960→256) projection — the first BiLSTM eats 960 directly
        and contextualises while projecting to 256 (bidir hidden=128)
      - 3×BiLSTM(128) total: layer 1 does projection+context, layers 2-3 refine
      - JEPA InfoNCE on masked segments (debrayable via --no-jepa)
      - SIGReg Epps-Pulley (debrayable via --lambda-sigreg 0)
      - NO LayerNorm before SIGReg (LeWorldModel paper: normalisation before
        SIGReg prevents the Gaussian target from matching the learned distribution)

    Architecture
    ------------
        line image (B, H, W)
          ├─► clean view ─► CNN ─► reshape (B, T, 960)
          │     └─► 3×BiLSTM(128) ─► CTC head (Linear(256, C) + log_softmax)
          │         z (before CTC head) feeds SIGReg — raw, un-normalised
          │
          └─► pixel-masked view ─► CNN ─► reshape ─► 3×BiLSTM(128) ─► z_masked
                  InfoNCE( jepa_proj(z_masked@masked),
                           jepa_proj(sg z@masked) )

        L = λ_ctc·CTC + λ_jepa·InfoNCE + λ_sigreg·SIGReg

    Flow (detail)
    -------------
        CNN conv stem (same as LectaurepClone / KrakenEncoder):
            (B, 1, H, W) → Conv2d layers → (B, 64, H/8, W/8)
            reshape → (B, T, 960)  where T = W/8

        BiLSTM stack:
            Layer 0: BiLSTM(960, 128) → (B, T, 256)  — projection + context
            Layer 1: BiLSTM(256, 128) → (B, T, 256)  — refinement
            Layer 2: BiLSTM(256, 128) → (B, T, 256)  — refinement

        CTC path:
            lstm_out → Linear(256, num_classes) → log_softmax

        SIGReg path (directly on lstm_out, BEFORE any normalisation):
            lstm_out → SIGReg loss

    ~3.7M params. Adam, cosine, no AMP, single LR for all params.
    """

    def __init__(
        self,
        img_height=120,
        num_classes=None,
        lambda_ctc=1.0,
        lambda_jepa=0.5,
        lambda_sigreg=0.1,
        lambda_wc=0.2,
        lstm_hidden=128,
        num_lstm_layers=3,
        lstm_dropout_mid=0.1,
        lstm_dropout_last=0.3,
        proj_dim=128,
        proj_hidden=256,
        jepa_num_targets=4,
        jepa_min_size=8,
        jepa_max_size=20,
        sigreg_projections=256,
        sigreg_knots=17,
        infonce_temp=0.1,
        supcon_temp=0.1,
        use_pretext=True,
        use_writer_contrastive=False,
    ):
        super().__init__()
        self.img_height = img_height
        self.embedding_dim = lstm_hidden * 2  # 256 bidir
        self.use_pretext = use_pretext
        self.use_writer_contrastive = use_writer_contrastive
        self.jepa_num_targets = jepa_num_targets
        self.jepa_min_size = jepa_min_size
        self.jepa_max_size = jepa_max_size
        self.frame_stride = 8  # 3 × MaxPool(2)

        # --- CNN stem (identical to LectaurepClone / KrakenEncoder) ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 13), padding=(1, 6)),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=(3, 13), padding=(1, 6)),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=(3, 9), padding=(1, 4)),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=(3, 9), padding=(1, 4)),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        self.cnn_out_dim = 64 * (img_height // 8)  # 960 for h=120

        # Learnable scalar for pixel-space masking
        self.mask_pixel = nn.Parameter(torch.zeros(()))

        # JEPA projection head
        self.jepa_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_dim),
        )

        # Writer contrastive projection head (dormant unless enabled)
        self.style_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_dim),
        )

        # --- 3×BiLSTM(128) ---
        # Layer 0: input=960 (CNN output), hidden=128 → output 256 bidir
        # Layers 1-2: input=256, hidden=128 → output 256 bidir
        self.lstm_layers = nn.ModuleList()
        for i in range(num_lstm_layers):
            input_dim = self.cnn_out_dim if i == 0 else lstm_hidden * 2
            self.lstm_layers.append(
                nn.LSTM(
                    input_dim,
                    lstm_hidden,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )

        # Dropout between LSTM layers
        self.lstm_dropouts = nn.ModuleList()
        for i in range(num_lstm_layers):
            p = lstm_dropout_last if i == num_lstm_layers - 1 else lstm_dropout_mid
            self.lstm_dropouts.append(nn.Dropout(p))

        # CTC output head
        self.ctc_head = CTCHead(lstm_hidden * 2, num_classes)

        # No LayerNorm before SIGReg — deliberate (LeWorldModel paper sec 3.1):
        # a final LayerNorm constrains frames to a hypersphere, which prevents
        # the Gaussian target of SIGReg from matching the learned distribution.
        # LayerNorm is only used inside the CTC path, after BiLSTM output,
        # if needed. But for v17 we keep it clean: raw lstm_out → CTC + SIGReg.
        self.ctc_norm = nn.Identity()

        # --- Loss bundle (CTC + JEPA + SIGReg) ---
        self.criterion = make_v12_bundle(
            lambda_ctc=lambda_ctc,
            lambda_jepa=lambda_jepa,
            lambda_sigreg=lambda_sigreg,
            lambda_wc=lambda_wc,
            infonce_temp=infonce_temp,
            supcon_temp=supcon_temp,
            sigreg_projections=sigreg_projections,
            sigreg_knots=sigreg_knots,
        )

        # Initialize LSTM weights (ketos-style)
        self._init_lstm_weights()

    def _init_lstm_weights(self):
        """Initialize LSTM weights: orthogonal + forget gate bias 1.0."""
        for lstm in self.lstm_layers:
            for p in lstm.parameters():
                if p.data.dim() == 2:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.constant_(p.data, 0)
                    nn.init.constant_(p.data[len(p) // 4:len(p) // 2], 1.0)

    def _run_encoder(self, img):
        """CNN forward: (B, H, W) → (B, T, 960)."""
        x = img.unsqueeze(1)  # (B, 1, H, W)
        x = self.encoder(x)   # (B, 64, H/8, W/8)
        B, C, H, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * H)  # (B, T, 960)
        return x, T

    def _bilstm(self, z_seq, input_lengths=None):
        """3×BiLSTM(128): (B, T, 960) → (B, T, 256)."""
        for lstm, do in zip(self.lstm_layers, self.lstm_dropouts):
            if input_lengths is not None:
                packed = pack_padded_sequence(
                    z_seq, input_lengths.cpu(),
                    batch_first=True, enforce_sorted=False,
                )
                packed_out, _ = lstm(packed)
                z_seq, _ = pad_packed_sequence(packed_out, batch_first=True)
            else:
                z_seq, _ = lstm(z_seq)
            z_seq = do(z_seq)
        return z_seq

    def _pool_temporal(self, z_seq, input_lengths=None):
        """Padding-aware mean over the time axis → (B, D)."""
        B, T, D = z_seq.shape
        if input_lengths is None:
            return z_seq.mean(dim=1)
        ar = torch.arange(T, device=z_seq.device)
        valid = (ar[None, :] < input_lengths.clamp(max=T)[:, None]).float()
        z_masked = z_seq * valid.unsqueeze(-1)
        denom = input_lengths.clamp(min=1, max=T).float().unsqueeze(-1)
        return z_masked.sum(dim=1) / denom

    def _make_masks(self, img, T, input_lengths):
        """Sample frame-level block mask, return pixel-masked image + frame mask."""
        B, _, W = img.shape
        frame_mask = sample_jepa_mask(
            B, T,
            num_targets=self.jepa_num_targets,
            min_size=self.jepa_min_size,
            max_size=self.jepa_max_size,
            valid_lengths=input_lengths,
            device=img.device,
        )

        pixel_mask = frame_mask.repeat_interleave(self.frame_stride, dim=1)
        if pixel_mask.shape[1] < W:
            pixel_mask = F.pad(pixel_mask, (0, W - pixel_mask.shape[1]))
        else:
            pixel_mask = pixel_mask[:, :W]

        img_masked = torch.where(
            pixel_mask.unsqueeze(1),
            self.mask_pixel.to(img.dtype).expand_as(img),
            img,
        )
        return img_masked, frame_mask

    def forward(self, img, input_lengths=None):
        """Inference: CNN → 3×BiLSTM → CTC."""
        z_raw, T = self._run_encoder(img)
        z_seq = self._bilstm(z_raw, input_lengths)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head is not None else None
        return None, z_seq, ctc_logits

    def compute_loss(
        self,
        img,
        targets=None,
        input_lengths=None,
        target_lengths=None,
        writer_id=None,
    ):
        if input_lengths is not None:
            input_lengths = input_lengths.to(img.device)

        # 1. Clean encoder pass: CNN → (B, T, 960) → BiLSTM → (B, T, 256)
        z_raw, T = self._run_encoder(img)
        z_seq = self._bilstm(z_raw, input_lengths)
        B, T_seq, D = z_seq.shape
        ctc_in = input_lengths.clamp(max=T_seq) if input_lengths is not None else None

        # Valid frame mask
        valid_mask = None
        if ctc_in is not None:
            ar = torch.arange(T_seq, device=img.device)
            valid_mask = ar[None, :] < ctc_in[:, None]

        # 2. Masked encoder pass for JEPA
        z_masked = None
        frame_mask = None
        if self.use_pretext:
            img_masked, frame_mask = self._make_masks(img, T, ctc_in)
            if valid_mask is not None:
                frame_mask = frame_mask & valid_mask
            if frame_mask.any():
                z_raw_m, _ = self._run_encoder(img_masked)
                z_masked = self._bilstm(z_raw_m, input_lengths)
            else:
                frame_mask = None

        # 3. Heads + loss (float32 for SIGReg/InfoNCE precision)
        _f32 = (
            torch.amp.autocast("cuda", enabled=False)
            if img.is_cuda
            else contextlib.nullcontext()
        )
        with _f32:
            z_seq = z_seq.float()

            # CTC: BiLSTM output → CTC head
            ctc_logits = None
            if self.ctc_head is not None:
                ctc_logits = self.ctc_head(z_seq)

            # JEPA: masked-segment InfoNCE
            z_pred = z_target = None
            if z_masked is not None and frame_mask is not None:
                z_masked = z_masked.float()
                z_pred = self.jepa_proj(z_masked[frame_mask])
                z_target = self.jepa_proj(z_seq.detach()[frame_mask])

            # Writer contrastive (optional)
            line_vec = None
            if self.use_writer_contrastive and writer_id is not None:
                v = self._pool_temporal(z_seq, ctc_in)
                line_vec = self.style_proj(v)

            return self.criterion(
                z_pred=z_pred,
                z_target=z_target,
                z_seq=z_seq,
                valid_mask=valid_mask,
                ctc_logits=ctc_logits,
                targets=targets,
                input_lengths=ctc_in,
                target_lengths=target_lengths,
                line_vec=line_vec,
                writer_id=writer_id,
            )

    def adapt(self, img, input_lengths=None):
        """Self-supervised step: InfoNCE + SIGReg only."""
        return self.compute_loss(img, input_lengths=input_lengths)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HWMv18(nn.Module):
    """
    Handwriting World Model v18 — decoupled JEPA / CTC branches.

    Hypothesis after v17: the JEPA InfoNCE gradient was polluting the
    BiLSTM stack, starving CTC. v18 keeps the CNN shared (so JEPA still
    teaches useful low-level features) but routes JEPA through a separate
    Linear(960 → 384) head that the BiLSTM never sees. The BiLSTM stack
    only receives CTC gradient.

    Architecture
    ------------
        line image (B, H, W)
          │
          ▼
        CNN stem (shared) ─► z_cnn  (B, T, 960)
          │
          ├─► CTC branch
          │     3×BiLSTM(128) ─► z_seq (B, T, 256) ─► CTC head ─► CTC
          │
          └─► JEPA branch
                Linear(960 → 384) ─► z_jepa  (B, T, 384)
                  │
                  ├─► SIGReg(z_jepa_clean)   — anti-collapse on JEPA side
                  │
                  └─► clean / masked: InfoNCE(
                          jepa_proj(z_jepa_masked[mask]),
                          jepa_proj(sg z_jepa_clean[mask]))

    Notable choices
    ---------------
      - No LayerNorm on the JEPA branch: SIGReg is the regulariser, and
        LayerNorm would constrain the embedding to a sphere — the Gaussian
        target of SIGReg cannot match a hyperspherical distribution
        (LeWorldModel paper sec 3.1).
      - SIGReg is on z_jepa_clean (Linear output), NOT on z_seq (BiLSTM
        output). CTC's discriminative loss already prevents trivial
        collapse on the BiLSTM path.
      - The masked view skips the BiLSTM entirely: ~60% cheaper second
        forward pass.
    """

    def __init__(
        self,
        img_height=120,
        num_classes=None,
        lambda_ctc=1.0,
        lambda_jepa=0.2,
        lambda_sigreg=0.1,
        lambda_wc=0.2,
        jepa_dim=384,
        lstm_hidden=128,
        num_lstm_layers=3,
        lstm_dropout_mid=0.1,
        lstm_dropout_last=0.3,
        proj_dim=128,
        proj_hidden=256,
        jepa_num_targets=4,
        jepa_min_size=8,
        jepa_max_size=20,
        sigreg_projections=256,
        sigreg_knots=17,
        infonce_temp=0.1,
        supcon_temp=0.1,
        use_pretext=True,
        use_writer_contrastive=False,
    ):
        super().__init__()
        self.img_height = img_height
        self.embedding_dim = lstm_hidden * 2  # 256 bidir (CTC path)
        self.jepa_dim = jepa_dim
        self.use_pretext = use_pretext
        self.use_writer_contrastive = use_writer_contrastive
        self.jepa_num_targets = jepa_num_targets
        self.jepa_min_size = jepa_min_size
        self.jepa_max_size = jepa_max_size
        self.frame_stride = 8  # 3 × MaxPool(2)

        # --- CNN stem (identical to v17 / LectaurepClone) ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 13), padding=(1, 6)),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=(3, 13), padding=(1, 6)),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=(3, 9), padding=(1, 4)),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=(3, 9), padding=(1, 4)),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        self.cnn_out_dim = 64 * (img_height // 8)  # 960 for h=120

        # Learnable scalar for pixel-space masking
        self.mask_pixel = nn.Parameter(torch.zeros(()))

        # --- JEPA branch: Linear(960 → jepa_dim), no LayerNorm ---
        # SIGReg on this output is the anti-collapse regulariser; LayerNorm
        # here would prevent SIGReg's Gaussian target from being matchable.
        self.cnn_to_jepa = nn.Linear(self.cnn_out_dim, jepa_dim)

        # JEPA projection head — input dim = jepa_dim (NOT BiLSTM out).
        self.jepa_proj = nn.Sequential(
            nn.Linear(jepa_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_dim),
        )

        # Writer contrastive head (dormant unless enabled) — sourced from
        # BiLSTM output like v17, since that's the "recognition" feature.
        self.style_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_dim),
        )

        # --- 3×BiLSTM(128) — CTC path only ---
        self.lstm_layers = nn.ModuleList()
        for i in range(num_lstm_layers):
            input_dim = self.cnn_out_dim if i == 0 else lstm_hidden * 2
            self.lstm_layers.append(
                nn.LSTM(
                    input_dim,
                    lstm_hidden,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )
        self.lstm_dropouts = nn.ModuleList()
        for i in range(num_lstm_layers):
            p = lstm_dropout_last if i == num_lstm_layers - 1 else lstm_dropout_mid
            self.lstm_dropouts.append(nn.Dropout(p))

        # CTC output head
        self.ctc_head = CTCHead(lstm_hidden * 2, num_classes)

        # --- Loss bundle (CTC + JEPA + SIGReg) ---
        self.criterion = make_v12_bundle(
            lambda_ctc=lambda_ctc,
            lambda_jepa=lambda_jepa,
            lambda_sigreg=lambda_sigreg,
            lambda_wc=lambda_wc,
            infonce_temp=infonce_temp,
            supcon_temp=supcon_temp,
            sigreg_projections=sigreg_projections,
            sigreg_knots=sigreg_knots,
        )

        self._init_lstm_weights()

    def _init_lstm_weights(self):
        for lstm in self.lstm_layers:
            for p in lstm.parameters():
                if p.data.dim() == 2:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.constant_(p.data, 0)
                    nn.init.constant_(p.data[len(p) // 4:len(p) // 2], 1.0)

    def _run_encoder(self, img):
        """CNN forward: (B, H, W) → (B, T, 960)."""
        x = img.unsqueeze(1)
        x = self.encoder(x)
        B, C, H, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * H)
        return x, T

    def _bilstm(self, z_seq, input_lengths=None):
        """3×BiLSTM(128): (B, T, 960) → (B, T_seq, 256)."""
        for lstm, do in zip(self.lstm_layers, self.lstm_dropouts):
            if input_lengths is not None:
                packed = pack_padded_sequence(
                    z_seq, input_lengths.cpu(),
                    batch_first=True, enforce_sorted=False,
                )
                packed_out, _ = lstm(packed)
                z_seq, _ = pad_packed_sequence(packed_out, batch_first=True)
            else:
                z_seq, _ = lstm(z_seq)
            z_seq = do(z_seq)
        return z_seq

    def _pool_temporal(self, z_seq, input_lengths=None):
        B, T, D = z_seq.shape
        if input_lengths is None:
            return z_seq.mean(dim=1)
        ar = torch.arange(T, device=z_seq.device)
        valid = (ar[None, :] < input_lengths.clamp(max=T)[:, None]).float()
        z_masked = z_seq * valid.unsqueeze(-1)
        denom = input_lengths.clamp(min=1, max=T).float().unsqueeze(-1)
        return z_masked.sum(dim=1) / denom

    def _make_masks(self, img, T, input_lengths):
        B, _, W = img.shape
        frame_mask = sample_jepa_mask(
            B, T,
            num_targets=self.jepa_num_targets,
            min_size=self.jepa_min_size,
            max_size=self.jepa_max_size,
            valid_lengths=input_lengths,
            device=img.device,
        )
        pixel_mask = frame_mask.repeat_interleave(self.frame_stride, dim=1)
        if pixel_mask.shape[1] < W:
            pixel_mask = F.pad(pixel_mask, (0, W - pixel_mask.shape[1]))
        else:
            pixel_mask = pixel_mask[:, :W]
        img_masked = torch.where(
            pixel_mask.unsqueeze(1),
            self.mask_pixel.to(img.dtype).expand_as(img),
            img,
        )
        return img_masked, frame_mask

    def forward(self, img, input_lengths=None):
        """Inference: CNN → 3×BiLSTM → CTC. JEPA branch is training-only."""
        z_cnn, T = self._run_encoder(img)
        z_seq = self._bilstm(z_cnn, input_lengths)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head is not None else None
        return None, z_seq, ctc_logits

    def compute_loss(
        self,
        img,
        targets=None,
        input_lengths=None,
        target_lengths=None,
        writer_id=None,
    ):
        if input_lengths is not None:
            input_lengths = input_lengths.to(img.device)

        # 1. CNN forward (shared)
        z_cnn, T = self._run_encoder(img)                  # (B, T, 960)

        # 2. CTC branch: BiLSTM → CTC head (later, in float32)
        z_seq = self._bilstm(z_cnn, input_lengths)         # (B, T_seq, 256)
        B, T_seq, _ = z_seq.shape
        ctc_in = input_lengths.clamp(max=T_seq) if input_lengths is not None else None

        # 3. JEPA branch (clean view): Linear(960 → 384) on z_cnn
        z_jepa_clean = self.cnn_to_jepa(z_cnn)             # (B, T, 384)

        # Valid mask on the CNN time axis (for SIGReg + JEPA frame_mask).
        valid_mask_cnn = None
        cnn_lengths = None
        if input_lengths is not None:
            cnn_lengths = input_lengths.clamp(max=T)
            ar = torch.arange(T, device=img.device)
            valid_mask_cnn = ar[None, :] < cnn_lengths[:, None]

        # 4. Masked view: CNN + Linear ONLY (skip BiLSTM)
        z_pred = z_target = None
        if self.use_pretext:
            img_masked, frame_mask = self._make_masks(img, T, cnn_lengths)
            if valid_mask_cnn is not None:
                frame_mask = frame_mask & valid_mask_cnn
            if frame_mask.any():
                z_cnn_m, _ = self._run_encoder(img_masked)
                z_jepa_masked = self.cnn_to_jepa(z_cnn_m)  # (B, T, 384)
            else:
                frame_mask = None
                z_jepa_masked = None
        else:
            frame_mask = None
            z_jepa_masked = None

        # 5. Heads + loss (float32 for SIGReg / InfoNCE precision)
        _f32 = (
            torch.amp.autocast("cuda", enabled=False)
            if img.is_cuda
            else contextlib.nullcontext()
        )
        with _f32:
            z_seq = z_seq.float()
            z_jepa_clean = z_jepa_clean.float()

            ctc_logits = None
            if self.ctc_head is not None:
                ctc_logits = self.ctc_head(z_seq)

            if z_jepa_masked is not None and frame_mask is not None:
                z_jepa_masked = z_jepa_masked.float()
                z_pred = self.jepa_proj(z_jepa_masked[frame_mask])
                z_target = self.jepa_proj(z_jepa_clean.detach()[frame_mask])

            line_vec = None
            if self.use_writer_contrastive and writer_id is not None:
                v = self._pool_temporal(z_seq, ctc_in)
                line_vec = self.style_proj(v)

            # SIGReg on the JEPA branch (z_jepa_clean), NOT on z_seq:
            # CTC already protects the BiLSTM path from trivial collapse,
            # while the JEPA InfoNCE target needs an anti-collapse anchor.
            return self.criterion(
                z_pred=z_pred,
                z_target=z_target,
                z_seq=z_jepa_clean,
                valid_mask=valid_mask_cnn,
                ctc_logits=ctc_logits,
                targets=targets,
                input_lengths=ctc_in,
                target_lengths=target_lengths,
                line_vec=line_vec,
                writer_id=writer_id,
            )

    def adapt(self, img, input_lengths=None):
        """Self-supervised step: InfoNCE + SIGReg only (CTC term auto-skips)."""
        return self.compute_loss(img, input_lengths=input_lengths)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LectaurepClone(nn.Module):
    """
    Faithful reproduction of the lectaurep_base model architecture.

    From the official training command:
      ketos train ... -s '[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2
        Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3
        Lbx200 Do0.1,2 Lbx200 Do.1,2 Lbx200 Do]'

    Architecture:
      CNN (4 conv layers, identical to KrakenEncoder's conv stem)
        → reshape to (B, T, 960)  [64 * (120/8)]
        → 3 × BiLSTM(hidden=200, bidirectional)
        → Linear(400, num_classes) + log_softmax

    Total: ~4.0M params (matches the official report).

    This is a CTC-only model — no Transformer, no JEPA, no SIGReg.
    compute_loss() returns only CTC loss.  The interface (forward returning
    a 3-tuple, ctc_head attribute, encoder attribute, etc.) is compatible
    with the existing train.py / recognize.py infrastructure.
    """

    def __init__(
        self,
        img_height=120,
        num_classes=100,
        hidden=200,
        num_lstm_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.img_height = img_height
        self.embedding_dim = 64 * (img_height // 8)  # 960 for h=120
        self.num_classes = num_classes
        self.hidden = hidden
        self.num_lstm_layers = num_lstm_layers

        # --- CNN encoder (identical to KrakenEncoder's conv stem) ---
        # Kraken uses Dropout2d (drops entire channels) not Dropout (drops pixels)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 13), padding=(1, 6)),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=(3, 13), padding=(1, 6)),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=(3, 9), padding=(1, 4)),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=(3, 9), padding=(1, 4)),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        # --- 3 × BiLSTM(200) ---
        # Layer 1: input=960, hidden=200.  Layers 2-3: input=400, hidden=200.
        self.lstm_layers = nn.ModuleList()
        for i in range(num_lstm_layers):
            input_dim = self.embedding_dim if i == 0 else hidden * 2
            self.lstm_layers.append(
                nn.LSTM(
                    input_dim,
                    hidden,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )

        self.lstm_dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(num_lstm_layers - 1)]
            + [nn.Dropout(0.5)]  # Kraken's VGSL 'Do]' = default p=0.5 for last layer
        )

        # --- CTC output head ---
        # Matches Kraken's LinSoftmax: Linear(400, num_classes) + log_softmax.
        # We wrap in a thin module with a .proj attribute so that
        # train.py's checkpoint save (model.ctc_head.proj.out_features) works
        # without special-casing.
        self.ctc_head = CTCHead(hidden * 2, num_classes)

        # Placeholder attributes expected by _build_param_groups / _set_encoder_frozen
        # (encoder is the Sequential above, which is already an attribute)
        self.ctc_norm = nn.Identity()  # no-op, but train.py passes z through it
        self.window_size = None
        self._ctc_dropped_samples = 0
        self._ctc_total_batches = 0

        # Initialize weights exactly like ketos (vgsl/model.py init_weights)
        self.init_weights()

    def init_weights(self):
        """Weight initialization matching ketos VGSL init_weights().
        
        - Conv2d: uniform(-0.1, 0.1)
        - LSTM: orthogonal for weights, forget gate bias = 1.0 (jozefowicz 2015)
        - Linear: xavier_uniform + bias = 0
        """
        def _wi(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.LSTM):
                for p in m.parameters():
                    if p.data.dim() == 2:
                        nn.init.orthogonal_(p.data)
                    else:
                        # bias: set forget gate to 1.0 (jozefowicz 2015)
                        nn.init.constant_(p.data, 0)
                        nn.init.constant_(p.data[len(p) // 4:len(p) // 2], 1.0)
            elif isinstance(m, nn.Conv2d):
                for p in m.parameters():
                    nn.init.uniform_(p.data, -0.1, 0.1)
        self.apply(_wi)

    def _encode(self, img):
        """CNN forward: (B, H, W) → (B, T, 960)."""
        x = img.unsqueeze(1)  # (B, 1, H, W)
        x = self.encoder(x)  # (B, 64, H/8, W/8)
        B, C, H, T = x.shape
        x = x.permute(0, 3, 1, 2)  # (B, T, C, H)
        x = x.reshape(B, T, C * H)  # (B, T, 960)
        return x

    def _bilstm(self, z_seq, input_lengths=None):
        """3 × BiLSTM(200): (B, T, 960) → (B, T, 400).
        
        Uses pack_padded_sequence to ignore padding, like ketos.
        input_lengths: (B,) tensor of actual sequence lengths before padding.
        """
        for lstm, do in zip(self.lstm_layers, self.lstm_dropouts):
            if input_lengths is not None:
                # ketos packs sequences so LSTM ignores padding
                packed = pack_padded_sequence(
                    z_seq, input_lengths.cpu(),
                    batch_first=True, enforce_sorted=False
                )
                packed_out, _ = lstm(packed)
                z_seq, _ = pad_packed_sequence(packed_out, batch_first=True)
            else:
                z_seq, _ = lstm(z_seq)
            z_seq = do(z_seq)
        return z_seq

    def forward(self, img, input_lengths=None):
        """Inference forward.  Returns (None, z_seq, ctc_logits) for compatibility."""
        z_seq = self._encode(img)
        z_seq = self._bilstm(z_seq, input_lengths=input_lengths)
        ctc_logits = self.ctc_head(z_seq)  # CTCHead includes log_softmax
        return None, z_seq, ctc_logits

    def compute_loss(
        self, img, targets=None, input_lengths=None, target_lengths=None, writer_id=None
    ):
        """CTC-only loss.  Compatible with train.py's _step_full contract."""
        if input_lengths is not None:
            input_lengths = input_lengths.to(img.device)

        z_seq = self._encode(img)
        z_seq = self._bilstm(z_seq, input_lengths=input_lengths)
        ctc_logits = self.ctc_head(z_seq)  # (B, T, C) log-probs

        B, T, _ = ctc_logits.shape
        ctc_in = input_lengths.clamp(max=T) if input_lengths is not None else None

        if targets is None or ctc_in is None or target_lengths is None:
            return None, {}

        # Diagnostic: log how many samples would fail CTC alignment.
        # ketos filters these upstream; we clamp and warn instead.
        bad = (ctc_in < target_lengths).sum().item()
        self._ctc_dropped_samples += bad
        self._ctc_total_batches += B

        ctc_loss = F.ctc_loss(
            ctc_logits.permute(1, 0, 2),  # (T, B, C)
            targets,
            ctc_in,
            target_lengths,
            blank=0,
            reduction="sum",  # ketos: reduction='sum', NO division by B
            zero_infinity=True,  # ketos: zero_infinity=True
        )
        # ketos returns raw sum loss to Lightning's training_step.
        # Do NOT divide by B — matching ketos exactly so Adam sees the
        # same gradient scale.  The loss value will be batch-size dependent,
        # but Adam is adaptive and handles this fine.
        return ctc_loss, {"ctc": ctc_loss.item(), "total": ctc_loss.item()}

    def adapt(self, img_seqs, input_lengths=None):
        return None, None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
        z_seq = self.encode_sequence(img_columns)  # (B, T, D)
        z_pred = self.predictor(z_seq[:, :-1, :])  # (B, T-1, D)
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
        z_target = z_seq[:, 1:, :].detach()  # (B, T-1, D)
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
        z_seq = self.encode_sequence(img_columns)  # (B, T, D)
        z_pred = self.predictor(z_seq[:, :-1, :])  # (B, T-1, D)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head else None
        return z_pred, z_seq, ctc_logits

    def compute_loss(
        self, img_columns, targets=None, input_lengths=None, target_lengths=None
    ):
        z_pred, z_seq, ctc_logits = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()  # (B, T-1, D)
        return self.criterion(
            z_pred, z_target, z_seq, ctc_logits, targets, input_lengths, target_lengths
        )

    def adapt(self, img_columns):
        z_pred, z_seq, _ = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()  # (B, T-1, D)
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
        z_seq = self.encode_sequence(img_columns)  # (B, T, D)
        z_pred = self.predictor(z_seq[:, :-1, :])  # (B, T-1, D)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head else None
        return z_pred, z_seq, ctc_logits

    def compute_loss(
        self, img_columns, targets=None, input_lengths=None, target_lengths=None
    ):
        z_pred, z_seq, ctc_logits = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()  # (B, T-1, D)
        return self.criterion(
            z_pred, z_target, z_seq, ctc_logits, targets, input_lengths, target_lengths
        )

    def adapt(self, img_columns):
        z_pred, z_seq, _ = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()  # (B, T-1, D)
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
            if num_classes
            else None
        )
        self.criterion = HybridLoss(lambda_sigreg, lambda_ctc)

    def encode_sequence(self, img_columns):
        B, T = img_columns.shape[:2]
        z_seq = self.encoder(
            img_columns.reshape(B * T, img_columns.shape[2], img_columns.shape[3])
        ).view(B, T, -1)
        return z_seq

    def forward(self, img_columns):
        z_seq = self.encode_sequence(img_columns)  # (B, T, D)
        z_pred = self.predictor(z_seq[:, :-1, :])  # (B, T-1, D)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head else None
        return z_pred, z_seq, ctc_logits

    def compute_loss(
        self, img_columns, targets=None, input_lengths=None, target_lengths=None
    ):
        z_pred, z_seq, ctc_logits = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()  # (B, T-1, D)
        return self.criterion(
            z_pred, z_target, z_seq, ctc_logits, targets, input_lengths, target_lengths
        )

    def adapt(self, img_columns):
        z_pred, z_seq, _ = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()  # (B, T-1, D)
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
            embedding_dim,
            num_layers,
            num_heads,
            ff_dim,
            dropout,
            causal=False,
        )
        self.ctc_head = (
            CTCHeadBiLSTM(
                embedding_dim,
                num_classes,
                hidden_dim=ctc_hidden,
                num_lstm_layers=ctc_num_lstm,
            )
            if num_classes
            else None
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
        z_seq = self.encoder(img)  # (B, T, D)
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
            B,
            T,
            num_targets=self.jepa_num_targets,
            min_size=self.jepa_min_size,
            max_size=self.jepa_max_size,
            valid_lengths=input_lengths,
            device=z_seq.device,
        )
        mask_tok = self.mask_token.expand(B, T, D)
        z_ctx = torch.where(mask.unsqueeze(-1), mask_tok, z_seq)
        z_pred_full = self.predictor(z_ctx)  # (B, T, D), non-causal

        if mask.any():
            z_pred_t = z_pred_full[mask]  # (N, D)
            z_tgt_t = z_seq.detach()[mask]  # (N, D)
        else:
            # Degenerate case (sequence shorter than min block size):
            # no targets sampled. Use the first position so the loss is
            # well-defined; gradient magnitude is tiny.
            z_pred_t = z_pred_full[:, 0, :]
            z_tgt_t = z_seq.detach()[:, 0, :]

        return z_pred_t, z_tgt_t

    def compute_loss(self, img, targets=None, input_lengths=None, target_lengths=None):
        z_seq = self.encoder(img)
        if self.use_jepa:
            z_pred_t, z_tgt_t = self._jepa_predict(z_seq, input_lengths)
        else:
            z_pred_t, z_tgt_t = None, None
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head is not None else None
        return self.criterion(
            z_pred_t,
            z_tgt_t,
            z_seq,
            ctc_logits,
            targets,
            input_lengths,
            target_lengths,
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
            B,
            T,
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
            B,
            T,
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
                embedding_dim,
                num_classes,
                hidden_dim=ctc_hidden,
                num_lstm_layers=ctc_num_lstm,
            )
            if num_classes
            else None
        )
        self.criterion = MAEHybridLoss(lambda_mae=lambda_mae, lambda_ctc=lambda_ctc)

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
        col_pad = ar[None, :] >= n_h_valid[:, None]  # (B, N_h)
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

    def compute_loss(self, img, targets=None, input_lengths=None, target_lengths=None):
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
                B,
                n_v_g,
                n_h_g,
                num_blocks=self.mask_num_blocks,
                min_h=self.mask_min_h,
                max_h=self.mask_max_h,
                min_w=self.mask_min_w,
                max_w=self.mask_max_w,
                valid_h_lengths=n_h_valid,
                device=img.device,
            )
            mask_flat = mask2d.reshape(B, n_v_g * n_h_g)

            pred_pixels = self.decoder(
                enc_tokens=tokens,
                mask_flat=mask_flat,
                n_v=n_v_g,
                n_h=n_h_g,
                key_padding_mask=pad_mask_flat,
            )
            target_pixels = self.encoder.patchify_pixels(img).reshape(
                B, n_v_g * n_h_g, -1
            )
            valid = (
                ~pad_mask_flat
                if pad_mask_flat is not None
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


class HWMv9(nn.Module):
    """
    Handwriting World Model v9 — return to the LeWorldModel recipe,
    modernised with a hybrid CNN+ViT encoder and image-level masking.

    Motivation vs v5–v8:

      - v5–v7 masked the *embeddings* (``z_seq``) and used contrastive
        / predictor tricks to avoid scale collapse. v8 masked the
        *image* and reconstructed raw pixels. v9 keeps the image-level
         masking (2D blocks) but drops pixel reconstruction:
        "same representation with and without the mask" is enforced
        on the encoder output itself, as in MSN / data2vec.
      - Encoder: hybrid CNN stem + ViT transformer. Pure ViT (v8)
        needs lots of data to learn low-level features; a CNN stem
        provides the standard visual inductive biases cheaply.
      - Regulariser: SIGReg (LeWorldModel paper) instead of VICReg.
      - CTC head: plain Linear (no BiLSTM). The transformer already
        mixes context; another sequential head is redundant.

    Flow:

      image ─┬─► CNN stem + patch_embed ─► transformer ─► z_clean
             │                                              │
             │                            ┌─ vertical pool ─┴─► Linear CTC
             │                            └─ SIGReg
             │
             └─► sample 2D block mask + upsample to pixel resolution
                 ─► replace masked PIXELS with learnable scalar
                 ─► CNN stem + patch_embed + transformer ─► z_masked
                                                              │
                                                              ▼
                              MSE(z_masked, z_clean.detach()) at masked & valid

    Pixel-space masking (vs token-space) is the key fix: with token-
    space masking, the CNN stem's receptive field overlap let adjacent
    unmasked tokens leak masked-region info, collapsing the pretext
    task in a few hundred batches.
    """

    def __init__(
        self,
        img_height=120,
        stem_channels=64,
        patch_h=3,
        patch_w=4,
        embedding_dim=384,
        num_layers=4,
        num_heads=8,
        ff_dim=1536,
        dropout=0.1,
        num_classes=None,
        lambda_msn=1.0,
        lambda_sigreg=0.1,
        lambda_ctc=1.0,
        mask_num_blocks=4,
        mask_min_h=2,
        mask_max_h=4,
        mask_min_w=4,
        mask_max_w=16,
        max_n_h=400,
        use_msn=True,
    ):
        super().__init__()
        self.img_height = img_height
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.embedding_dim = embedding_dim
        self.use_msn = use_msn
        self.mask_num_blocks = mask_num_blocks
        self.mask_min_h = mask_min_h
        self.mask_max_h = mask_max_h
        self.mask_min_w = mask_min_w
        self.mask_max_w = mask_max_w
        # Image→token stride (height and width). image_h/total_stride_h = N_v
        # and image_w/total_stride_w = N_h.
        self.total_stride_h = 4 * patch_h
        self.total_stride_w = 4 * patch_w
        self.n_v = img_height // self.total_stride_h

        self.encoder = HybridCNNViTEncoder(
            img_height=img_height,
            stem_channels=stem_channels,
            patch_h=patch_h,
            patch_w=patch_w,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_n_h=max_n_h,
        )

        # Learnable scalar substituted in the input image at masked
        # positions, BEFORE the CNN stem. Token-level masking (replace
        # tokens after patch_embed) leaked masked-region info into
        # adjacent unmasked tokens via the stem's overlapping receptive
        # field — observed as msn dropping to ~0.02 in a few hundred
        # batches with sigreg staying low. Pixel-space masking keeps a
        # tiny RF leak at the boundary of each block but cuts the bulk
        # of the shortcut.
        self.mask_pixel = nn.Parameter(torch.zeros(()))

        # Plain linear CTC head — transformer already contextualises.
        self.ctc_head = CTCHead(embedding_dim, num_classes) if num_classes else None

        self.criterion = MSNLoss(
            lambda_msn=lambda_msn,
            lambda_sigreg=lambda_sigreg,
            lambda_ctc=lambda_ctc,
        )

    def _round_up_width(self, img):
        W = img.shape[-1]
        pad = (self.total_stride_w - W % self.total_stride_w) % self.total_stride_w
        if pad > 0:
            img = F.pad(img, (0, pad))
        return img

    def _convert_lengths_to_patches(self, input_lengths):
        """
        Collate gives input_lengths in Kraken units (W // 8). v9 encoder
        downsamples by total_stride_w = 16, so factor = 2.
        """
        factor = max(1, self.total_stride_w // 8)
        return input_lengths // factor

    def _padding_mask(self, n_h_valid, n_v, n_h, device):
        B = n_h_valid.size(0)
        ar = torch.arange(n_h, device=device)
        col_pad = ar[None, :] >= n_h_valid[:, None]  # (B, N_h)
        pad2d = col_pad.unsqueeze(1).expand(B, n_v, n_h)
        return pad2d.reshape(B, n_v * n_h)

    def _pool_vertical(self, tokens, n_v, n_h):
        B, _, D = tokens.shape
        return tokens.reshape(B, n_v, n_h, D).mean(dim=1)  # (B, N_h, D)

    def forward(self, img):
        """
        Inference forward: encoder + vertical pool + CTC head. Returned
        tuple follows the (pred, z_seq, ctc_logits) contract used by the
        evaluation pipeline.
        """
        img = self._round_up_width(img)
        tokens, (n_v, n_h) = self.encoder(img)
        z_seq = self._pool_vertical(tokens, n_v, n_h)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head is not None else None
        return None, z_seq, ctc_logits

    def compute_loss(self, img, targets=None, input_lengths=None, target_lengths=None):
        img = self._round_up_width(img)
        B, H, W = img.shape
        n_v = self.n_v
        n_h = W // self.total_stride_w

        n_h_valid = None
        pad_mask_flat = None
        if input_lengths is not None:
            n_h_valid = torch.clamp(
                self._convert_lengths_to_patches(input_lengths), max=n_h
            )
            pad_mask_flat = self._padding_mask(n_h_valid, n_v, n_h, img.device)

        # 1. Clean branch — full image through CNN stem + patch embed +
        # pos-embed + transformer. With grad: trains encoder through CTC
        # and SIGReg.
        tokens_clean, (n_v_g, n_h_g) = self.encoder.patchify(img)
        z_clean = self.encoder.transformer_pass(
            tokens_clean, n_v_g, n_h_g, src_key_padding_mask=pad_mask_flat
        )

        # 2. Vertical pool → Linear CTC.
        z_seq = self._pool_vertical(z_clean, n_v_g, n_h_g)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head is not None else None

        # 3. Masked branch — sample a 2D block mask on the token grid,
        # upsample it to pixel resolution, replace masked PIXELS with
        # a learnable scalar, then re-run the full encoder. The CNN
        # stem now sees zeroed (or learned-mask-value) pixels in masked
        # blocks, so adjacent unmasked tokens cannot leak masked-region
        # info via the receptive field overlap.
        z_masked = None
        mask_flat = None
        valid_flat = None
        if self.use_msn:
            mask2d = sample_2d_block_mask(
                B,
                n_v_g,
                n_h_g,
                num_blocks=self.mask_num_blocks,
                min_h=self.mask_min_h,
                max_h=self.mask_max_h,
                min_w=self.mask_min_w,
                max_w=self.mask_max_w,
                valid_h_lengths=n_h_valid,
                device=img.device,
            )
            mask_flat = mask2d.reshape(B, n_v_g * n_h_g)

            # Upsample (B, n_v, n_h) bool to (B, H, W) bool by repeating
            # each grid cell over its corresponding pixel block.
            pixel_mask = mask2d.repeat_interleave(
                self.total_stride_h, dim=1
            ).repeat_interleave(self.total_stride_w, dim=2)
            img_masked = torch.where(pixel_mask, self.mask_pixel.expand_as(img), img)

            tokens_masked, _ = self.encoder.patchify(img_masked)
            z_masked = self.encoder.transformer_pass(
                tokens_masked, n_v_g, n_h_g, src_key_padding_mask=pad_mask_flat
            )
            valid_flat = (
                ~pad_mask_flat
                if pad_mask_flat is not None
                else torch.ones_like(mask_flat)
            )

        return self.criterion(
            z_masked=z_masked,
            z_clean=z_clean,
            mask_flat=mask_flat,
            valid_flat=valid_flat,
            ctc_logits=ctc_logits,
            targets=targets,
            input_lengths=n_h_valid,
            target_lengths=target_lengths,
        )

    def adapt(self, img, input_lengths=None):
        return self.compute_loss(img, input_lengths=input_lengths)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HWMv10(nn.Module):
    """
    Handwriting World Model v10 — I-JEPA with hybrid CNN+ViT encoder.

    The encoder's own transformer is used for BOTH the clean (teacher) and
    the masked (context/student) branches.  Two transformer_pass calls
    share the same weights but differ in their key_padding_mask:

      1. Clean pass  — all tokens visible  → z_clean  (for CTC, SIGReg,
         and as JEPA targets via stop-gradient).
      2. Context pass — target+padding tokens hidden from self-attention
         → z_ctx.  Visible tokens cannot attend to masked ones, so zero
         information leakage.  The predictor then reads out predictions
         at masked positions via cross-attention.

    This is the I-JEPA recipe: the context encoder and the target encoder
    share weights (no EMA needed when SIGReg prevents collapse), and a
    lightweight cross-attention predictor maps context → target positions.

    Flow:

      image (B, H, W)
        → CNN stem + patch_embed → raw_tokens (B, N_v*N_h, D)
          │
          ├─ transformer_pass(all visible) → z_clean
          │    ├─ vertical pool → BiLSTM CTC → ctc_logits
          │    └─ SIGRegV2(z_clean)                         ← anti-collapse
          │
          └─ JEPA branch:
                 sample 2D block mask on (N_v, N_h) grid
                 mask_token at target positions in raw_tokens
                 transformer_pass(key_pad_mask=targets+pad) → z_ctx
                     ↑ target tokens invisible to self-attention
                 cross_attn_predictor(z_ctx, mask_token) → z_pred
                 LN(z_pred), LN(sg(z_clean[targets])) → MSE
    """

    def __init__(
        self,
        img_height=120,
        stem_channels=64,
        patch_h=3,
        patch_w=4,
        embedding_dim=384,
        num_layers=4,
        num_heads=8,
        ff_dim=1536,
        pred_num_layers=2,
        pred_ff_dim=768,
        dropout=0.1,
        num_classes=None,
        lambda_pred=1.0,
        lambda_sigreg=0.1,
        lambda_ctc=1.0,
        sigreg_var=25.0,
        sigreg_cov=1.0,
        sigreg_gamma=1.0,
        ctc_hidden=512,
        ctc_num_lstm=1,
        mask_num_blocks=5,
        mask_min_h=2,
        mask_max_h=6,
        mask_min_w=4,
        mask_max_w=24,
        max_n_h=400,
        use_jepa=True,
    ):
        super().__init__()
        self.img_height = img_height
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.embedding_dim = embedding_dim
        self.use_jepa = use_jepa
        self.mask_num_blocks = mask_num_blocks
        self.mask_min_h = mask_min_h
        self.mask_max_h = mask_max_h
        self.mask_min_w = mask_min_w
        self.mask_max_w = mask_max_w
        self.total_stride_h = 4 * patch_h
        self.total_stride_w = 4 * patch_w
        self.n_v = img_height // self.total_stride_h

        self.encoder = HybridCNNViTEncoder(
            img_height=img_height,
            stem_channels=stem_channels,
            patch_h=patch_h,
            patch_w=patch_w,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_n_h=max_n_h,
        )

        self.jepa_predictor = JEPACrossAttnPredictor(
            embedding_dim=embedding_dim,
            num_layers=pred_num_layers,
            num_heads=num_heads,
            ff_dim=pred_ff_dim,
            dropout=dropout,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pred_norm = nn.LayerNorm(embedding_dim)
        self.target_norm = nn.LayerNorm(embedding_dim)

        self.ctc_head = (
            CTCHeadBiLSTM(
                embedding_dim,
                num_classes,
                hidden_dim=ctc_hidden,
                num_lstm_layers=ctc_num_lstm,
            )
            if num_classes
            else None
        )

        self.criterion = JEPALoss(
            lambda_pred=lambda_pred,
            lambda_sigreg=lambda_sigreg,
            lambda_ctc=lambda_ctc,
            sigreg_var=sigreg_var,
            sigreg_cov=sigreg_cov,
            sigreg_gamma=sigreg_gamma,
        )

    def _round_up_width(self, img):
        W = img.shape[-1]
        pad = (self.total_stride_w - W % self.total_stride_w) % self.total_stride_w
        if pad > 0:
            img = F.pad(img, (0, pad))
        return img

    def _convert_lengths_to_patches(self, input_lengths):
        factor = max(1, self.total_stride_w // 8)
        return input_lengths // factor

    def _padding_mask(self, n_h_valid, n_v, n_h, device):
        B = n_h_valid.size(0)
        ar = torch.arange(n_h, device=device)
        col_pad = ar[None, :] >= n_h_valid[:, None]
        pad2d = col_pad.unsqueeze(1).expand(B, n_v, n_h)
        return pad2d.reshape(B, n_v * n_h)

    def _pool_vertical(self, tokens, n_v, n_h):
        B, _, D = tokens.shape
        return tokens.reshape(B, n_v, n_h, D).mean(dim=1)

    def _jepa_predict(self, raw_tokens, n_v, n_h, n_h_valid=None):
        B, N, D = raw_tokens.shape

        target_mask_2d = sample_2d_block_mask(
            B,
            n_v,
            n_h,
            num_blocks=self.mask_num_blocks,
            min_h=self.mask_min_h,
            max_h=self.mask_max_h,
            min_w=self.mask_min_w,
            max_w=self.mask_max_w,
            valid_h_lengths=n_h_valid,
            device=raw_tokens.device,
        )
        target_mask = target_mask_2d.reshape(B, N)

        ctx_kpm = target_mask.clone()
        if n_h_valid is not None:
            ar = torch.arange(n_h, device=target_mask.device)
            pad_mask = ar[None, :] >= n_h_valid[:, None]
            pad2d = pad_mask.unsqueeze(1).expand(B, n_v, n_h)
            ctx_kpm = target_mask | pad2d.reshape(B, N)

        all_masked = ctx_kpm.all(dim=1)
        if all_masked.any():
            ctx_kpm = ctx_kpm.clone()
            ctx_kpm[all_masked, 0] = False

        # Masked context: replace target tokens with mask_token, then
        # run the SAME encoder transformer with key_padding_mask hiding
        # targets + padding.  Self-attention among visible tokens only.
        mask_tok = self.mask_token.expand(B, N, D)
        ctx_tokens = torch.where(target_mask.unsqueeze(-1), mask_tok, raw_tokens)
        z_ctx = self.encoder.transformer_pass(
            ctx_tokens, n_v, n_h, src_key_padding_mask=ctx_kpm
        )

        # Cross-attention predictor: queries = mask_token + pos_enc at
        # every position, K/V = context output.  Memory mask hides the
        # same target+padding positions so queries cannot cheat.
        pred_full = self.jepa_predictor(
            context=z_ctx,
            memory_key_padding_mask=ctx_kpm,
            mask_token=self.mask_token,
            seq_len=N,
        )

        return pred_full, target_mask

    def forward(self, img):
        img = self._round_up_width(img)
        tokens, (n_v, n_h) = self.encoder(img)
        z_seq = self._pool_vertical(tokens, n_v, n_h)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head is not None else None
        return None, z_seq, ctc_logits

    def compute_loss(self, img, targets=None, input_lengths=None, target_lengths=None):
        img = self._round_up_width(img)
        B, H, W = img.shape
        n_v = self.n_v
        n_h = W // self.total_stride_w

        n_h_valid = None
        if input_lengths is not None:
            n_h_valid = torch.clamp(
                self._convert_lengths_to_patches(input_lengths.to(img.device)),
                max=n_h,
            )

        # 1. Patchify: CNN stem + patch_embed — benefits from AMP.
        raw_tokens, (n_v_g, n_h_g) = self.encoder.patchify(img)

        # 2-4. Transformer passes, JEPA branch, and loss — run in float32.
        # The double transformer pass (clean + context) through the same
        # encoder amplifies gradients and can overflow float16 after the
        # warmup LR ramp.  Disabling autocast here forces float32 for
        # the transformer, predictor, and all loss computations.
        _amp_ctx = (
            torch.amp.autocast("cuda", enabled=False)
            if img.is_cuda
            else contextlib.nullcontext()
        )
        with _amp_ctx:
            raw_tokens = raw_tokens.float()

            z_clean = self.encoder.transformer_pass(raw_tokens, n_v_g, n_h_g)

            z_seq = self._pool_vertical(z_clean, n_v_g, n_h_g)
            ctc_logits = self.ctc_head(z_seq) if self.ctc_head is not None else None

            z_pred = None
            z_target = None
            if self.use_jepa:
                pred_full, target_mask = self._jepa_predict(
                    raw_tokens, n_v_g, n_h_g, n_h_valid
                )

                if target_mask.any():
                    z_pred_t = pred_full[target_mask]
                    z_tgt_t = z_clean.detach()[target_mask]
                else:
                    z_pred_t = pred_full[:, 0, :]
                    z_tgt_t = z_clean.detach()[:, 0, :]

                z_pred = self.pred_norm(z_pred_t)
                z_target = self.target_norm(z_tgt_t)

            return self.criterion(
                z_pred=z_pred,
                z_target=z_target,
                z_seq=z_clean,
                ctc_logits=ctc_logits,
                targets=targets,
                input_lengths=n_h_valid,
                target_lengths=target_lengths,
            )

    def adapt(self, img, input_lengths=None):
        return self.compute_loss(img, input_lengths=input_lengths)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HWMv11(nn.Module):
    """
    Handwriting World Model v11 — Kraken 1D encoder + SimSiam consistency.

    Pretext: encode a clean view and a heavily perturbed view of the
    same line, pool both to a single line vector, and pull them
    together via cosine similarity (with a predictor MLP on the
    perturbed branch for SimSiam-style asymmetry). SIGRegV2 prevents
    collapse on the raw frame embeddings; CTC trains recognition.

    Why this is different from v5-v10:
      - 6 prior versions hit the same JEPA wall (shared encoder + stop-
        grad target collapses without EMA / contrastive negatives).
      - v11 sidesteps that entirely: the consistency target is the
        SAME encoder's output on a different view (still stop-grad),
        but with two key changes that historically work without EMA:
        (a) SimSiam predictor MLP on the student side ⇒ asymmetry,
        (b) consistency at the LINE pool level (not frame level) ⇒
            position-only collapse cannot satisfy it because the
            perturbations include horizontal shift.

    Flow
    ----
        line image (B, H, W)
          ├─► clean view  ─► KrakenEncoder ─► z_clean (B, T, D)
          │                                       │
          │                              ┌────────┴───────┐
          │                              │                │
          │                       pool over T     BiLSTM CTC head
          │                       (padding-aware)         │
          │                              │                ▼
          │                       v_clean (B, D)     log_softmax
          │                              │           (B, T, num_cls)
          │                              │
          │                          stop-grad
          │                              │
          │                              ▼  cos_sim
          │                              ▲
          │                              │
          └─► perturb(view) ─► KrakenEncoder ─► z_pert ─► pool ─► v_pert
                                  (same weights)                     │
                                                              predictor MLP
                                                                     │
                                                                  p_pert

        L = -λ_cons · cos_sim(p_pert, sg(v_clean))
            + λ_reg · SIGRegV2(z_clean)
            + λ_ctc · CTC(z_clean, targets)

    The KrakenEncoder runs twice per training step (once on clean,
    once on perturbed). At inference, only the clean path runs:
    encoder → BiLSTM → CTC. The predictor MLP is discarded.
    """

    def __init__(
        self,
        img_height=120,
        embedding_dim=384,
        pred_hidden=384,
        num_classes=None,
        lambda_cons=1.0,
        lambda_sigreg=0.1,
        lambda_ctc=1.0,
        sigreg_var=25.0,
        sigreg_cov=1.0,
        sigreg_gamma=1.0,
        ctc_hidden=256,
        ctc_num_lstm=1,
        # Perturbation hyperparameters (view 2)
        pert_shift_x=4,
        pert_shear_deg=5.0,
        pert_mask_blocks=4,
        pert_mask_w_min=16,
        pert_mask_w_max=32,
        pert_contrast_min=0.7,
        pert_contrast_max=1.3,
        pert_brightness=0.1,
        pert_noise_std=0.03,
        use_pretext=True,
    ):
        super().__init__()
        self.img_height = img_height
        self.embedding_dim = embedding_dim
        self.use_pretext = use_pretext

        self.encoder = KrakenEncoder(img_height=img_height, embedding_dim=embedding_dim)

        # SimSiam predictor MLP on the perturbed branch only. The
        # asymmetry is what prevents the trivial "encoder = identity"
        # collapse; without it the consistency loss has a degenerate
        # minimum at p_pert = v_clean = const.
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, pred_hidden),
            nn.GELU(),
            nn.Linear(pred_hidden, embedding_dim),
        )

        # Learnable scalar substituted in pixel space at masked blocks.
        # Single parameter (broadcast to image shape) so the mask value
        # adapts to the dataset's normalisation range.
        self.mask_pixel = nn.Parameter(torch.zeros(()))

        self.ctc_head = (
            CTCHeadBiLSTM(
                embedding_dim,
                num_classes,
                hidden_dim=ctc_hidden,
                num_lstm_layers=ctc_num_lstm,
            )
            if num_classes
            else None
        )

        self.criterion = SimSiamHybridLoss(
            lambda_cons=lambda_cons,
            lambda_sigreg=lambda_sigreg,
            lambda_ctc=lambda_ctc,
            sigreg_var=sigreg_var,
            sigreg_cov=sigreg_cov,
            sigreg_gamma=sigreg_gamma,
        )

        # Perturbation hyperparameters
        self.pert_shift_x = pert_shift_x
        self.pert_shear_deg = pert_shear_deg
        self.pert_mask_blocks = pert_mask_blocks
        self.pert_mask_w_min = pert_mask_w_min
        self.pert_mask_w_max = pert_mask_w_max
        self.pert_contrast_min = pert_contrast_min
        self.pert_contrast_max = pert_contrast_max
        self.pert_brightness = pert_brightness
        self.pert_noise_std = pert_noise_std

    def _perturb(self, img, input_lengths=None):
        """
        Apply the v11 perturbation stack to ``img`` (B, H, W).

        Returns the perturbed image with the same shape.

        Order:
          1. Photometric (per-sample contrast, brightness, gaussian noise)
          2. Affine (per-sample horizontal shear + horizontal shift)
          3. Pixel-space block masking
        """
        B, H, W = img.shape
        device = img.device

        # 1. Photometric: per-sample contrast and brightness.
        contrast = (
            torch.rand(B, 1, 1, device=device)
            * (self.pert_contrast_max - self.pert_contrast_min)
            + self.pert_contrast_min
        )
        brightness = (torch.rand(B, 1, 1, device=device) * 2 - 1) * self.pert_brightness
        img = img * contrast + brightness
        if self.pert_noise_std > 0:
            img = img + torch.randn_like(img) * self.pert_noise_std

        # 2. Affine: horizontal shear + horizontal shift, per sample.
        # theta maps OUTPUT normalised coords [-1, 1] -> INPUT coords:
        #   x_in = x_out + (-tan(α)) * y_out + (-2 * shift_px / W)
        #   y_in = y_out
        if self.pert_shear_deg > 0 or self.pert_shift_x > 0:
            shear_rad = (
                (torch.rand(B, device=device) * 2 - 1)
                * self.pert_shear_deg
                * math.pi
                / 180.0
            )
            shift_px = (torch.rand(B, device=device) * 2 - 1) * self.pert_shift_x
            theta = torch.zeros(B, 2, 3, device=device, dtype=img.dtype)
            theta[:, 0, 0] = 1.0
            theta[:, 0, 1] = -torch.tan(shear_rad).to(img.dtype)
            theta[:, 0, 2] = (-2.0 * shift_px / max(W, 1)).to(img.dtype)
            theta[:, 1, 1] = 1.0
            grid = F.affine_grid(theta, size=(B, 1, H, W), align_corners=False)
            img = F.grid_sample(
                img.unsqueeze(1),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).squeeze(1)

        # 3. Pixel-space block masking. Sample per-sample column blocks
        # within the valid (non-padding) region so we don't waste blocks
        # on padding. Replace masked pixels with the learned scalar.
        if self.pert_mask_blocks > 0:
            mask = torch.zeros(B, W, dtype=torch.bool, device=device)
            for b in range(B):
                eff_w = (
                    int(input_lengths[b].item() * 8) if input_lengths is not None else W
                )
                eff_w = max(eff_w, self.pert_mask_w_min + 1)
                upper_w = min(self.pert_mask_w_max, eff_w - 1)
                for _ in range(self.pert_mask_blocks):
                    w_blk = int(
                        torch.randint(self.pert_mask_w_min, upper_w + 1, (1,)).item()
                    )
                    start = int(torch.randint(0, eff_w - w_blk + 1, (1,)).item())
                    mask[b, start : start + w_blk] = True
            mask2d = mask.unsqueeze(1).expand(B, H, W)
            img = torch.where(mask2d, self.mask_pixel.to(img.dtype).expand_as(img), img)

        return img

    def _pool_temporal(self, z_seq, input_lengths=None):
        """
        Padding-aware mean over the time axis.

        Args:
            z_seq: (B, T, D)
            input_lengths: (B,) long, T units (image width // 8).
                If None, average over all T.
        Returns:
            (B, D) line vector.
        """
        B, T, D = z_seq.shape
        if input_lengths is None:
            return z_seq.mean(dim=1)
        ar = torch.arange(T, device=z_seq.device)
        valid = (ar[None, :] < input_lengths[:, None]).float()
        z_masked = z_seq * valid.unsqueeze(-1)
        denom = input_lengths.clamp(min=1).float().unsqueeze(-1)
        return z_masked.sum(dim=1) / denom

    def forward(self, img):
        """
        Inference forward: encoder + (BiLSTM) CTC head only.
        Returned tuple matches the (pred, z_seq, ctc_logits) contract.
        """
        z_seq = self.encoder(img)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head is not None else None
        return None, z_seq, ctc_logits

    def compute_loss(self, img, targets=None, input_lengths=None, target_lengths=None):
        # 1. Clean view: full image through Kraken. With grad — feeds
        # CTC, SIGRegV2, and serves as the (stop-grad) consistency target.
        z_clean = self.encoder(img)
        ctc_logits = self.ctc_head(z_clean) if self.ctc_head is not None else None

        p_pert = None
        v_clean = None
        if self.use_pretext:
            # 2. Perturbed view: same image, run through perturbation
            # stack, then through the SAME Kraken weights.
            img_pert = self._perturb(img, input_lengths=input_lengths)
            z_pert = self.encoder(img_pert)

            # 3. Pool both views to a single line vector.
            v_clean = self._pool_temporal(z_clean, input_lengths)
            v_pert = self._pool_temporal(z_pert, input_lengths)

            # 4. Predictor MLP on the perturbed branch (SimSiam asymmetry).
            p_pert = self.predictor(v_pert)

        return self.criterion(
            p_pert=p_pert,
            v_clean=v_clean,
            z_seq=z_clean,
            ctc_logits=ctc_logits,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )

    def adapt(self, img, input_lengths=None):
        return self.compute_loss(img, input_lengths=input_lengths)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HWMv12(nn.Module):
    """
    Handwriting World Model v12 — Kraken 1D conv + Transformer encoder,
    masked-segment prediction (MSN/data2vec) + Epps-Pulley SIGReg + CTC,
    with an optional writer/page contrastive branch.

    This realises the original project intuition: a Kraken decoding
    LSTM is "moved upstream into the encoder" and replaced by a
    Transformer; the remaining BiLSTM stays in the CTC head.

    Three structural choices vs v5-v11:

      1. Option B — NO final LayerNorm on the encoder. The paper's
         SIGReg cannot match a Gaussian if a per-sample LayerNorm pins
         frames to a sphere. The only LayerNorm in the v12 path sits
         inside the CTC head (``ctc_norm``), where it merely stabilises
         the BiLSTM. SIGReg, the pretext target and the writer branch
         all consume the raw, un-normalised ``z``.

      2. Real SIGReg — the Epps-Pulley regulariser from the paper
         (random projections + univariate normality test), not the
         VICReg-style ``SIGRegV2`` of v10/v11.

      3. Pixel-space masking — masked frame spans are blanked in the
         IMAGE, before the conv stem. The wide Kraken kernels would
         otherwise leak masked content into neighbour tokens (the v9
         token-masking failure). The transformer itself in-paints the
         blanked region; no separate predictor module.

    Flow
    ----
        line image (B, H, W)
          ├─► clean view ─► KrakenEncoderV12 ─► z  (B, T, D)  [raw]
          │       ├─ LayerNorm → BiLSTM → CTC
          │       ├─ SIGReg(z)                    ← anti-collapse
          │       └─ pool_T → style_proj → SupCon  [optional, writer_id]
          │
          └─► pixel-masked view ─► KrakenEncoderV12 ─► z_masked
                  InfoNCE( jepa_proj(z_masked@masked),
                           jepa_proj(sg z@masked) )

        L = λ_ctc·CTC + λ_jepa·InfoNCE + λ_sigreg·SIGReg ( + λ_wc·SupCon )

    At inference only the clean path runs: encoder → LayerNorm → BiLSTM
    → CTC. ``adapt()`` runs the self-supervised terms only (InfoNCE +
    SIGReg) — masked-segment prediction on an unlabelled new writer's
    lines forces the encoder to internalise that hand.
    """

    def __init__(
        self,
        img_height=120,
        embedding_dim=192,
        num_layers=3,
        num_heads=3,
        ff_dim=384,
        dropout=0.1,
        num_classes=None,
        lambda_ctc=1.0,
        lambda_jepa=0.5,
        lambda_sigreg=0.1,
        lambda_wc=0.2,
        ctc_hidden=192,
        ctc_num_lstm=1,
        proj_dim=128,
        proj_hidden=192,
        jepa_num_targets=4,
        jepa_min_size=8,
        jepa_max_size=20,
        sigreg_projections=256,
        sigreg_knots=17,
        infonce_temp=0.1,
        supcon_temp=0.1,
        use_pretext=True,
        use_writer_contrastive=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.img_height = img_height
        self.embedding_dim = embedding_dim
        self.use_pretext = use_pretext
        self.use_writer_contrastive = use_writer_contrastive
        self.jepa_num_targets = jepa_num_targets
        self.jepa_min_size = jepa_min_size
        self.jepa_max_size = jepa_max_size
        # Conv stem width stride (3 × MaxPool(2)). Frame t covers pixel
        # columns [8t, 8t+8); used to map a frame mask back to pixels.
        self.frame_stride = 8

        self.encoder = KrakenEncoderV12(
            img_height=img_height,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
        )

        # Learnable scalar substituted in pixel space at masked columns.
        self.mask_pixel = nn.Parameter(torch.zeros(()))

        # SSL projection head for the InfoNCE pretext. Routing the
        # pretext gradient through a projector keeps raw z aligned with
        # CTC (the v6 rationale). Discarded at inference.
        self.jepa_proj = nn.Sequential(
            nn.Linear(embedding_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_dim),
        )

        # Writer/page contrastive projection head. Built unconditionally
        # so enabling the branch later needs no architecture change;
        # only used when ``use_writer_contrastive`` and a writer_id is
        # passed. Discarded at inference.
        self.style_proj = nn.Sequential(
            nn.Linear(embedding_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_dim),
        )

        # The ONLY LayerNorm in the v12 path — stabilises the BiLSTM.
        self.ctc_norm = nn.LayerNorm(embedding_dim)
        self.ctc_head = (
            CTCHeadBiLSTM(
                embedding_dim,
                num_classes,
                hidden_dim=ctc_hidden,
                num_lstm_layers=ctc_num_lstm,
            )
            if num_classes
            else None
        )

        self.criterion = V12Loss(
            lambda_ctc=lambda_ctc,
            lambda_jepa=lambda_jepa,
            lambda_sigreg=lambda_sigreg,
            lambda_wc=lambda_wc,
            infonce_temp=infonce_temp,
            supcon_temp=supcon_temp,
            sigreg_projections=sigreg_projections,
            sigreg_knots=sigreg_knots,
        )

    def _pool_temporal(self, z_seq, input_lengths=None):
        """Padding-aware mean over the time axis → (B, D) line vector."""
        B, T, D = z_seq.shape
        if input_lengths is None:
            return z_seq.mean(dim=1)
        ar = torch.arange(T, device=z_seq.device)
        valid = (ar[None, :] < input_lengths.clamp(max=T)[:, None]).float()
        z_masked = z_seq * valid.unsqueeze(-1)
        denom = input_lengths.clamp(min=1, max=T).float().unsqueeze(-1)
        return z_masked.sum(dim=1) / denom

    def _make_masks(self, img, T, input_lengths):
        """
        Sample a frame-level block mask, return the pixel-masked image
        and the corresponding frame mask.

        The frame mask is sampled first (``sample_jepa_mask``, valid-
        region aware), then upsampled by ``frame_stride`` to pixel
        columns so frame mask and pixel mask are exactly consistent.
        """
        B, _, W = img.shape
        frame_mask = sample_jepa_mask(
            B,
            T,
            num_targets=self.jepa_num_targets,
            min_size=self.jepa_min_size,
            max_size=self.jepa_max_size,
            valid_lengths=input_lengths,
            device=img.device,
        )  # (B, T)

        pixel_mask = frame_mask.repeat_interleave(self.frame_stride, dim=1)
        if pixel_mask.shape[1] < W:
            pixel_mask = F.pad(pixel_mask, (0, W - pixel_mask.shape[1]))
        else:
            pixel_mask = pixel_mask[:, :W]

        img_masked = torch.where(
            pixel_mask.unsqueeze(1),
            self.mask_pixel.to(img.dtype).expand_as(img),
            img,
        )
        return img_masked, frame_mask

    def forward(self, img):
        """
        Inference forward: encoder + LayerNorm + BiLSTM CTC head.
        Returned tuple matches the (pred, z_seq, ctc_logits) contract.
        """
        z_seq = self.encoder(img)
        ctc_logits = (
            self.ctc_head(self.ctc_norm(z_seq)) if self.ctc_head is not None else None
        )
        return None, z_seq, ctc_logits

    def compute_loss(
        self,
        img,
        targets=None,
        input_lengths=None,
        target_lengths=None,
        writer_id=None,
    ):
        if input_lengths is not None:
            input_lengths = input_lengths.to(img.device)

        # 1. Encoder passes — run under the caller's autocast so the conv
        # stem (the dominant activation-memory cost) uses fp16 under AMP.
        # The encoder keeps its own transformer in float32 internally.
        # The transformer runs twice (clean + masked) sharing weights.
        z_clean = self.encoder(img, input_lengths)  # (B, T, D) raw
        B, T, D = z_clean.shape
        ctc_in = input_lengths.clamp(max=T) if input_lengths is not None else None

        # Valid (non-padding) frame mask, shared by SIGReg and the
        # pretext target selection.
        valid_mask = None
        if ctc_in is not None:
            ar = torch.arange(T, device=img.device)
            valid_mask = ar[None, :] < ctc_in[:, None]  # (B, T)

        z_masked = None
        frame_mask = None
        if self.use_pretext:
            img_masked, frame_mask = self._make_masks(img, T, ctc_in)
            if valid_mask is not None:
                frame_mask = frame_mask & valid_mask
            if frame_mask.any():
                z_masked = self.encoder(img_masked, input_lengths)
            else:
                frame_mask = None

        # 2. Heads + loss — forced to float32. SIGReg and InfoNCE run long
        # reductions that are sensitive to fp16 precision; the tensors
        # here are small (the sequence is /8 downsampled) so the float32
        # cost is negligible.
        _f32 = (
            torch.amp.autocast("cuda", enabled=False)
            if img.is_cuda
            else contextlib.nullcontext()
        )
        with _f32:
            z_clean = z_clean.float()

            ctc_logits = (
                self.ctc_head(self.ctc_norm(z_clean))
                if self.ctc_head is not None
                else None
            )

            # Masked-segment pretext (MSN-style, InfoNCE).
            z_pred = z_target = None
            if z_masked is not None and frame_mask is not None:
                z_masked = z_masked.float()
                z_pred = self.jepa_proj(z_masked[frame_mask])
                z_target = self.jepa_proj(z_clean.detach()[frame_mask])

            # Writer/page contrastive (optional, dormant w/o writer_id).
            line_vec = None
            if self.use_writer_contrastive and writer_id is not None:
                v = self._pool_temporal(z_clean, ctc_in)
                line_vec = self.style_proj(v)

            return self.criterion(
                z_pred=z_pred,
                z_target=z_target,
                z_seq=z_clean,
                valid_mask=valid_mask,
                ctc_logits=ctc_logits,
                targets=targets,
                input_lengths=ctc_in,
                target_lengths=target_lengths,
                line_vec=line_vec,
                writer_id=writer_id,
            )

    def adapt(self, img, input_lengths=None):
        """Self-supervised step: InfoNCE + SIGReg only (no CTC, no SupCon)."""
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
