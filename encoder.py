"""
HWM Encoder - CNN 1D for encoding handwriting columns
Converts image columns to latent embeddings
"""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class CNNEncoder(nn.Module):
    """
    Lightweight CNN 1D encoder for handwriting columns
    Input: (B, H) or (B, H, W) where W is window_size
    Output: (B, embedding_dim)
    """

    def __init__(self, img_height=32, window_size=10, embedding_dim=64):
        super().__init__()

        self.img_height = img_height
        self.window_size = window_size
        input_dim = img_height * window_size

        # Very lightweight: 2 conv layers + 1 linear
        self.net = nn.Sequential(
            # Input: (B, H*W)
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim),
        )

        # Alternative: Conv1D approach
        self.use_conv = False
        if self.use_conv:
            # Treat as 1D sequence: (B, 1, H*W)
            self.conv_net = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, embedding_dim),
            )

    def forward(self, x):
        """
        Args:
            x: (B, H, W) or (B, H*W) tensor
        Returns:
            z: (B, embedding_dim) embeddings
        """
        # Flatten if needed
        if x.dim() == 3:
            B, H, W = x.shape
            x = x.view(B, -1)

        if self.use_conv:
            # Add channel dimension for Conv1D
            x = x.unsqueeze(1)
            return self.conv_net(x)
        else:
            return self.net(x)


class Conv2DEncoder(nn.Module):
    """
    Conv2D encoder for handwriting image columns.
    Preserves spatial structure unlike the MLP encoder.

    Input: (B, H, W) where H=48, W=window_size
    Output: (B, embedding_dim)
    """

    def __init__(self, img_height=48, window_size=10, embedding_dim=96):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 1)),
        )
        self.fc = nn.Linear(128 * 4, embedding_dim)

    def forward(self, x):
        if x.dim() == 2:
            raise ValueError("Conv2DEncoder requires (B, H, W) input")
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Conv2DEncoderV2(nn.Module):
    """
    Deeper Conv2D encoder with skip connections.
    Input: (B, H, W) where H=48, W=window_size
    Output: (B, embedding_dim)
    """

    def __init__(self, img_height=48, window_size=32, embedding_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 1)),
        )
        self.fc = nn.Linear(256 * 4, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        if x.dim() == 2:
            raise ValueError("Conv2DEncoderV2 requires (B, H, W) input")
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.norm(self.fc(x))


class Conv2DEncoderV3(nn.Module):
    """
    Wider encoder with preserved horizontal resolution.
    Input: (B, H, W) where H=48, W=window_size
    Output: (B, embedding_dim)
    """

    def __init__(self, img_height=48, window_size=32, embedding_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2)),
        )
        self.fc = nn.Linear(512 * 4 * 2, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        if x.dim() == 2:
            raise ValueError("Conv2DEncoderV3 requires (B, H, W) input")
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.norm(self.fc(x))


class KrakenEncoder(nn.Module):
    """
    Kraken-style convolutional encoder for full line images.
    Rectangular kernels (3x13, 3x9) capture horizontal structure.

    Input: (B, H, W) where H=120, W=variable (full line)
    Output: (B, T, D) where T=W/8, D=embedding_dim
    """

    def __init__(self, img_height=120, embedding_dim=256):
        super().__init__()
        self.img_height = img_height

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 13), padding=(1, 6)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 32, kernel_size=(3, 13), padding=(1, 6)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=(3, 9), padding=(1, 4)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, kernel_size=(3, 9), padding=(1, 4)),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # H after 3 MaxPool(2,2): img_height / 8
        h_out = img_height // 8
        self.feature_dim = 64 * h_out
        self.proj = nn.Linear(self.feature_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        Args:
            x: (B, H, W) grayscale line image
        Returns:
            z_seq: (B, T, D) embedding sequence, T = W/8
        """
        x = x.unsqueeze(1)                    # (B, 1, H, W)
        x = self.conv(x)                      # (B, 64, H/8, W/8)
        B, C, H, T = x.shape
        x = x.permute(0, 3, 1, 2)            # (B, T, C, H)
        x = x.reshape(B, T, C * H)           # (B, T, 960)
        return self.norm(self.proj(x))         # (B, T, D)


class KrakenEncoderV12(nn.Module):
    """
    Option-B encoder for HWMv12: Kraken-style 1D conv stem + Transformer,
    with NO final LayerNorm.

    The conv stem (wide rectangular kernels) extracts stroke-level
    features and downsamples width by 8. A Transformer encoder then
    contextualises the frame sequence — this is the Kraken BiLSTM
    "moved into the encoder and replaced by attention".

    The output ``z`` is deliberately UN-normalised. A per-sample final
    LayerNorm constrains every frame to a hypersphere; the Gaussian
    target of the paper's SIGReg cannot match a sphere-supported
    distribution, so SIGReg never converges (LeWorldModel paper,
    Sec. 3.1). A LayerNorm is therefore applied only downstream, inside
    the CTC head, where it just stabilises the BiLSTM input.

    The internal transformer is pre-LN (``norm_first=True``) for
    training stability; no normalisation is applied to the final
    output (``nn.TransformerEncoder`` with ``norm=None``).

    Input:  (B, H, W) grayscale line image, H = img_height.
    Output: (B, T, D) raw embedding sequence, T = W // 8.
    """

    def __init__(
        self,
        img_height=120,
        embedding_dim=192,
        num_layers=3,
        num_heads=3,
        ff_dim=384,
        dropout=0.1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.img_height = img_height
        self.embedding_dim = embedding_dim
        # Gradient checkpointing on the conv stem: the stem runs on the
        # full-resolution image and is the dominant activation-memory
        # cost (×2 — v12 encodes a clean and a masked view per step).
        # Checkpointing drops its activations and recomputes them in
        # the backward pass — ~30% more compute for a large memory cut.
        self.use_checkpoint = use_checkpoint

        # Conv stem identical to KrakenEncoder: rectangular kernels
        # capture horizontal stroke structure, 3 MaxPools → W/8, H/8.
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 13), padding=(1, 6)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=(3, 13), padding=(1, 6)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=(3, 9), padding=(1, 4)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=(3, 9), padding=(1, 4)),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        h_out = img_height // 8
        self.feature_dim = 64 * h_out
        # Projection only — NO LayerNorm (Option B).
        self.proj = nn.Linear(self.feature_dim, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        # norm=None ⇒ no final LayerNorm on the transformer output.
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _padding_mask(self, T, input_lengths, device):
        """(B, T) bool, True = padding frame (t >= input_lengths[b])."""
        ar = torch.arange(T, device=device)
        return ar[None, :] >= input_lengths[:, None]

    def forward(self, img, input_lengths=None):
        """
        Args:
            img: (B, H, W) grayscale line image.
            input_lengths: optional (B,) long, valid frame counts in
                T units (image width // 8). Used to hide padding frames
                from self-attention.
        Returns:
            z: (B, T, D) raw (un-normalised) embedding sequence.
        """
        x = img.unsqueeze(1)               # (B, 1, H, W)
        if self.use_checkpoint and self.training:
            # use_reentrant=False: works although `x` itself carries no
            # grad (the conv parameters do), and restores the RNG state
            # so the Dropout masks match between forward and recompute.
            x = checkpoint(self.conv, x, use_reentrant=False)
        else:
            x = self.conv(x)               # (B, 64, H/8, W/8)
        B, C, H, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * H)  # (B, T, 64*H/8)

        kpm = None
        if input_lengths is not None:
            kpm = self._padding_mask(T, input_lengths.clamp(max=T), img.device)
            # A fully-padded row would make the attention softmax NaN.
            # Extremely rare; unmask position 0 to keep it defined.
            all_pad = kpm.all(dim=1)
            if all_pad.any():
                kpm = kpm.clone()
                kpm[all_pad, 0] = False

        # The conv stem (the dominant activation-memory cost — it runs on
        # the full-resolution image) executes in fp16 under AMP. The
        # projection + transformer are forced to float32: their memory is
        # negligible (the sequence is /8 downsampled), and float32 keeps
        # the attention stack clear of any fp16 overflow.
        _f32 = (
            torch.amp.autocast("cuda", enabled=False)
            if x.is_cuda
            else contextlib.nullcontext()
        )
        with _f32:
            tokens = self.proj(x.float())  # (B, T, D) — no LayerNorm
            return self.transformer(tokens, src_key_padding_mask=kpm)


class ViTEncoder(nn.Module):
    """
    Vision-Transformer-style encoder for HWMv8.

    Treats the full line image as a 2D grid of patches instead of a 1D
    sequence of thin vertical strips. Rationale (see v8 rationale in
    model.py): handwriting has genuine 2D structure — ascenders,
    descenders, diacritics, accent marks — that a 1D tall-thin-strip
    encoder compresses prematurely.

    Input:  (B, H, W)  line image, H divisible by ``patch_h``,
                       W divisible by ``patch_w`` (we pad/round externally).
    Output: (B, N_v * N_h, D) token sequence in row-major order,
            together with the grid shape (N_v, N_h).

    Positional encoding: learnable, separable (row embedding + column
    embedding). Learnable is fine here because the batch dimension is
    always fixed per run and the grid shape for a given config is
    determined by W (which is padded to the batch's max width). Row
    positions are fixed (only 8 rows); column positions go up to
    MAX_N_H_V8 which is sized for the dataset's max image width.
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
        max_n_h=400,
    ):
        super().__init__()
        assert img_height % patch_h == 0, (
            f"img_height {img_height} must be divisible by patch_h {patch_h}"
        )
        self.img_height = img_height
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.embedding_dim = embedding_dim
        self.n_v = img_height // patch_h

        # Patchify via a single Conv2d with stride = patch size. Each
        # patch becomes a D-dim token.
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=embedding_dim,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )

        # Separable positional encoding: row embed (N_v, D) + col embed
        # (max_n_h, D). Added before the transformer.
        self.row_embed = nn.Parameter(torch.zeros(self.n_v, embedding_dim))
        self.col_embed = nn.Parameter(torch.zeros(max_n_h, embedding_dim))
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # pre-LN: more stable for deeper ViTs.
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)

    def patchify_pixels(self, x):
        """
        Factor the raw image into flat patch vectors (for MAE targets).
        Args:
            x: (B, H, W) — H and W divisible by (patch_h, patch_w).
        Returns:
            patches: (B, N_v, N_h, patch_h * patch_w) raw pixel blocks.
        """
        B, H, W = x.shape
        ph, pw = self.patch_h, self.patch_w
        n_v, n_h = H // ph, W // pw
        # (B, H, W) -> (B, n_v, ph, n_h, pw) -> (B, n_v, n_h, ph, pw) -> flatten
        x = x.reshape(B, n_v, ph, n_h, pw).permute(0, 1, 3, 2, 4).contiguous()
        return x.reshape(B, n_v, n_h, ph * pw)

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: (B, H, W) line image.
            src_key_padding_mask: optional (B, N_v*N_h) bool, True = ignore
                (used to hide padding patches during self-attention).
        Returns:
            tokens: (B, N_v * N_h, D) post-LN transformer output.
            grid:   tuple (N_v, N_h) for the caller to reshape.
        """
        # Conv expects (B, 1, H, W).
        x = x.unsqueeze(1)
        feats = self.patch_embed(x)                    # (B, D, N_v, N_h)
        B, D, n_v, n_h = feats.shape
        assert n_v == self.n_v, (
            f"n_v mismatch: expected {self.n_v}, got {n_v}"
        )
        assert n_h <= self.col_embed.size(0), (
            f"n_h={n_h} exceeds max_n_h={self.col_embed.size(0)}. "
            "Increase MAX_N_H_V8 or tighten max_width."
        )

        # (B, D, N_v, N_h) -> (B, N_v, N_h, D) -> (B, N_v*N_h, D)
        tokens = feats.permute(0, 2, 3, 1).contiguous().reshape(B, n_v * n_h, D)

        # Add separable positional encoding.
        pos = (
            self.row_embed.unsqueeze(1)[:, :, :]          # (N_v, 1, D)
            + self.col_embed[:n_h].unsqueeze(0)           # (1, N_h, D)
        )                                                  # (N_v, N_h, D)
        pos = pos.reshape(n_v * n_h, D)
        tokens = tokens + pos.unsqueeze(0)                 # (B, N_v*N_h, D)

        tokens = self.transformer(
            tokens, src_key_padding_mask=src_key_padding_mask
        )
        return self.norm(tokens), (n_v, n_h)


class HybridCNNViTEncoder(nn.Module):
    """
    CNN stem + Vision Transformer encoder (HWMv9).

    The pure ViT in v8 projects raw pixels to tokens via a single
    conv-stride layer. That dumps all low-level feature extraction
    (edges, strokes, curvature) onto the transformer, which needs a
    lot of data to learn those biases from scratch. A short CNN stem
    in front of the patchifier bakes in the standard visual inductive
    biases (locality, translation equivariance) cheaply, leaving the
    transformer to focus on long-range reasoning.

    Structure:

        image (B, 1, H, W)
          ↓ 2 conv+pool blocks (~¼ H, ¼ W)
        features (B, C_stem, H/4, W/4)
          ↓ Conv2d kernel=(patch_h, patch_w), stride=(patch_h, patch_w)
        tokens (B, D, N_v, N_h)
          ↓ add separable row/col pos-embed
          ↓ transformer encoder blocks (pre-LN, GELU)
        output (B, N_v*N_h, D)

    Two entry points:

    - ``patchify(img)`` runs stem + patch-embed and returns raw tokens
      (no pos-embed, no transformer). Used by HWMv9's masked branch:
      we substitute [MASK] tokens at masked positions before running
      the transformer.
    - ``transformer_pass(tokens, n_v, n_h, key_padding_mask)`` applies
      pos-embed and the transformer stack. Called twice per step by
      HWMv9 (clean branch, masked branch).
    - ``forward(img, key_padding_mask)`` is the convenience full-path
      (patchify + transformer_pass) used at inference.
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
        max_n_h=400,
    ):
        super().__init__()
        assert img_height % 4 == 0, (
            f"img_height {img_height} must be divisible by 4 (stem pool)."
        )
        stem_h = img_height // 4
        assert stem_h % patch_h == 0, (
            f"stem_h {stem_h} must be divisible by patch_h {patch_h}"
        )

        self.img_height = img_height
        self.stem_channels = stem_channels
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.embedding_dim = embedding_dim
        self.n_v = stem_h // patch_h
        # Total vertical stride image→token = 4 (stem) * patch_h.
        # Total horizontal stride image→token = 4 (stem) * patch_w.
        self.total_stride_h = 4 * patch_h
        self.total_stride_w = 4 * patch_w

        # Light 2-block CNN stem. BatchNorm + GELU, 2 x MaxPool2d(2) →
        # divides both H and W by 4. Params kept small (~50k) so the
        # transformer still does most of the heavy lifting.
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, stem_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
        )

        # Patch embed on the feature map. Non-overlapping conv.
        self.patch_embed = nn.Conv2d(
            in_channels=stem_channels,
            out_channels=embedding_dim,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )

        # Separable learnable positional encoding (same scheme as ViT-v8).
        self.row_embed = nn.Parameter(torch.zeros(self.n_v, embedding_dim))
        self.col_embed = nn.Parameter(torch.zeros(max_n_h, embedding_dim))
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)

    def _pos_embed(self, n_v, n_h):
        pos = (
            self.row_embed.unsqueeze(1)             # (N_v, 1, D)
            + self.col_embed[:n_h].unsqueeze(0)     # (1, N_h, D)
        )
        return pos.reshape(n_v * n_h, -1)

    def patchify(self, img):
        """
        Args:
            img: (B, H, W) grayscale line image.
        Returns:
            tokens: (B, N_v*N_h, D) raw tokens (NO pos-embed, NO transformer).
            grid:   (N_v, N_h) grid shape.
        """
        x = img.unsqueeze(1)                         # (B, 1, H, W)
        feats = self.cnn_stem(x)                     # (B, C_stem, H/4, W/4)
        tokens = self.patch_embed(feats)             # (B, D, N_v, N_h)
        B, D, n_v, n_h = tokens.shape
        assert n_v == self.n_v, (
            f"n_v mismatch: expected {self.n_v}, got {n_v}"
        )
        assert n_h <= self.col_embed.size(0), (
            f"n_h={n_h} exceeds max_n_h={self.col_embed.size(0)}. "
            "Increase MAX_N_H_V9 or tighten max_width."
        )
        tokens = tokens.permute(0, 2, 3, 1).contiguous().reshape(B, n_v * n_h, D)
        return tokens, (n_v, n_h)

    def transformer_pass(self, tokens, n_v, n_h, src_key_padding_mask=None):
        """
        Args:
            tokens: (B, N_v*N_h, D) raw tokens from ``patchify``, possibly
                with some positions replaced by a learned [MASK] token.
            n_v, n_h: grid shape.
            src_key_padding_mask: (B, N_v*N_h) bool, True = ignore.
        Returns:
            out: (B, N_v*N_h, D) post-LN transformer output.
        """
        pos = self._pos_embed(n_v, n_h)
        tokens = tokens + pos.unsqueeze(0)
        out = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)
        return self.norm(out)

    def forward(self, img, src_key_padding_mask=None):
        tokens, (n_v, n_h) = self.patchify(img)
        return self.transformer_pass(tokens, n_v, n_h, src_key_padding_mask), (n_v, n_h)


class RotaryEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding) le long d'un axe 1D.

    Les embeddings de requete et de cle sont tournes d'un angle proportionnel
    a leur position : le produit scalaire q.k ne depend que de l'ECART de
    positions (encodage relatif), sans position absolue.

    Convention demi-split (LLaMA / GPT-NeoX) : la dimension de tete est
    coupee en deux moities (x1, x2) et la paire (x1_i, x2_i) tourne de
    angle_i = pos * inv_freq_i.

    Toutes les positions de v19 sont des coordonnees x de patches (le [cls]
    est en position 0, rotation identite), donc une seule table cos/sin
    suffit, precalculee jusqu'a ``max_positions``.
    """

    def __init__(self, head_dim: int, max_positions: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim impair ({head_dim}) pour RoPE"
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        t = torch.arange(max_positions, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)              # (max_positions, head_dim/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, S, head_dim) requetes ou cles.
            positions: (S,) long, coordonnee x de chaque token.
        Returns:
            (B, H, S, head_dim) version tournee.
        """
        cos = self.cos[positions]                     # (S, head_dim/2)
        sin = self.sin[positions]
        # Repeter sur les deux moities : (S, head_dim) -> broadcast (B, H, S, hd)
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)        # rotate_half
        return x * cos + rotated * sin


def build_block_causal_mask(num_frames: int, frame_size: int, device):
    """
    Masque d'attention BLOCK-CAUSAL, cle du contrat v19.

    La sequence comporte ``num_frames * frame_size`` tokens (une fenetre =
    un bloc de ``frame_size`` patches, chevauchement gere en amont par la
    duplication des patches dans leurs fenetres) :
      - intra-fenetre  : BIDIRECTIONNEL (tous les tokens d'une meme fenetre
        s'attendent mutuellement) ;
      - inter-fenetres : CAUSAL gauche -> droite (la fenetre f ne voit que
        les fenetres f' <= f).

    Returns:
        (S, S) bool, True = attention autorisee, S = num_frames * frame_size.
        (Le [cls] est gere a part par l'appelant : il interroge tout, et
        personne ne l'interroge.)
    """
    S = num_frames * frame_size
    query_frame = torch.arange(S, device=device) // frame_size   # (S,)
    key_frame = torch.arange(S, device=device) // frame_size     # (S,)
    return key_frame.unsqueeze(0) <= query_frame.unsqueeze(1)    # (S, S)


class BlockCausalAttentionLayer(nn.Module):
    """
    Un bloc transformer pre-LN avec attention self block-causale + RoPE.

    Identique en structure aux blocs eprouves du projet (pre-LN, GELU,
    residuelles), sauf que l'attention porte un masque bloc-causal et une
    rotation positionnelle relatives au lieu d'un TransformerEncoder
    standard : nn.TransformerEncoder ne sait pas combiner RoPE et masque
    arbitraire, d'ou l'attention ecrite a la main via
    ``F.scaled_dot_product_attention``.
    """

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x, attn_bias, rope, positions):
        """
        Args:
            x: (B, S, D) sequence ([cls] compris).
            attn_bias: (B, 1, S, S) float, masque additif (0 ou -inf) —
                block-causal + padding cles, [cls] visible uniquement
                de sa propre ligne.
            rope: RotaryEmbedding.
            positions: (S,) long, coordonnee x de chaque token.
        """
        B, S, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q = rope(q, positions)
        k = rope(k, positions)

        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=drop_p)
        out = out.transpose(1, 2).reshape(B, S, D)
        x = x + self.out_proj(out)

        return x + self.ffn(self.norm2(x))


class LeVJEPAEncoderV19(nn.Module):
    """
    Encodeur v19 : stem CNN Kraken -> patches carres -> fenetres coulissantes
    -> transformer block-causal avec [cls] readout.

    Transposition du papier LeVJEPA (arXiv:2608.27395) aux lignes manuscrites :
    l'axe temporel de la video devient l'axe x de la ligne.

    Structure
    ---------
        image (B, H=64, W)
          -> stem CNN 2 blocs conv Kraken (v15-v18), 2 x MaxPool(2) -> /4
          -> carte (B, C_stem, 16, W/4)
          -> Conv2d patch (16 x 16) / 16  ->  (B, D, 1, T)   T = W/64
          -> fenetres coulissantes : K patches consecutifs, stride K/2
             (chevauchement 50%) ; sequence = F x K tokens (un patch apparait
             dans jusqu'a 2 fenetres)
          -> [cls] apprenant prepende en tete
          -> N blocs block-causaux (RoPE le long de x, pas de position absolue)
          -> LayerNorm sur la SEQUENCE DE TOKENS (entree de la tete CTC)

    Roles du [cls] (contrat strict) :
      - JAMAIS droppé par le token-drop SSL ;
      - il ATTENDS a tous les tokens (toutes fenetres) ;
      - PERSONNE ne lui attends (masque cle) — pur readout global, l'information
        ne fuit pas du cls vers les tokens.

    Sorties : le [cls] est renvoye BRUT (sans LayerNorm) car la cible
    gaussienne du SIGReg ne peut pas matcher une hypersphere (lecon v12
    Option B / v17) ; les tokens normalises nourrissent la tete CTC.

    Input:  (B, H, W) image de ligne grayscale, H divisible par 4 et
            H // 4 == patch (une seule rangee de patches carres).
    Output: voir ``forward``.
    """

    def __init__(
        self,
        img_height=64,
        stem_channels=64,
        patch=16,
        window_patches=8,
        window_stride=4,
        embedding_dim=256,
        num_layers=8,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
        max_patches=512,
    ):
        super().__init__()
        stem_h = img_height // 4
        assert stem_h == patch, (
            f"img_height//4 ({stem_h}) doit egal patch ({patch}) : "
            "une seule rangee de patches carres le long de x"
        )
        self.img_height = img_height
        self.patch = patch
        self.window_patches = window_patches      # K
        self.window_stride = window_stride        # s = K/2
        self.embedding_dim = embedding_dim
        # Stride horizontal total image -> token : 4 (stem) x patch.
        self.token_stride = 4 * patch             # 64 px par token
        self.max_patches = max_patches

        # --- Stem CNN : blocs conv Kraken eprouves (recette v15-v18), a
        # kernels rectangulaires larges pour capter la structure horizontale
        # des traits. 2 MaxPool(2) -> H/4 = 16 rangees, W/4 colonnes.
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 13), padding=(1, 6)),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, stem_channels, kernel_size=(3, 9), padding=(1, 4)),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2, 2),
        )

        # --- Patch embed : un patch carre 16x16 de la carte -> un token D.
        # La hauteur de carte (16) est entièrement couverte : la ligne devient
        # une sequence 1D de tokens le long de x.
        self.patch_embed = nn.Conv2d(
            stem_channels, embedding_dim,
            kernel_size=(patch, patch), stride=(patch, patch),
        )

        # --- Token [cls] apprenable, prepende en tete.
        self.cls_token = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # --- RoPE le long de x + N blocs block-causaux.
        self.rope = RotaryEmbedding(embedding_dim // num_heads, max_patches)
        self.layers = nn.ModuleList([
            BlockCausalAttentionLayer(
                embedding_dim, num_heads, ff_dim, dropout
            )
            for _ in range(num_layers)
        ])

        # LayerNorm de sortie, sur la SEQUENCE DE TOKENS uniquement
        # (le [cls] reste brut pour SIGReg — cf. docstring de classe).
        self.norm = nn.LayerNorm(embedding_dim)

        # Cache du masque block-causal (S, S) : ne depend que de (F, K,
        # device), pas du batch.
        self._mask_cache = {}

    # ------------------------------------------------------------------
    # Aides geometrie
    # ------------------------------------------------------------------

    def _frames_count(self, n_patches: torch.Tensor) -> torch.Tensor:
        """(B,) nombre de fenetres F : dernier fenetrage finissant >= T.

        F = 1 si T <= K, sinon 1 + ceil((T - K) / s).
        """
        K, s = self.window_patches, self.window_stride
        F = 1 + torch.div(
            n_patches - K + s - 1, s, rounding_mode="floor"
        )
        return F.clamp(min=1)

    def _block_mask(self, num_frames: int, device):
        """Masque bool block-causal (S, S) caché par (F, K, device)."""
        key = (num_frames, self.window_patches, str(device))
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = build_block_causal_mask(
                num_frames, self.window_patches, device
            )
            self._mask_cache[key] = mask
        return mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, img, patch_counts=None, token_drop=0.0):
        """
        Args:
            img: (B, H, W) image de ligne grayscale.
            patch_counts: (B,) long optionnel, nombre de patches REELS
                (non-padding) par echantillon. None -> tout le batch est
                considere valide (T = T_feat).
            token_drop: proba de drop uniforme des tokens (SSL uniquement).
                Le drop est defini PAR PATCH (x) : un patch droppe
                disparait de toutes ses fenetres. Le patch 0 est JAMAIS
                droppe (cle d'ancrage de la fenetre 0 -> pas de ligne
                d'attention entierement masquee). Le [cls] n'est jamais
                droppe.
        Returns:
            cls_emb:   (B, D) embedding [cls] NORMALISE (sortie LayerNorm,
                       sur la sphere — c'est lui que proj_head projete).
            tokens:    (B, S-1, D) tokens normalises (sortie LayerNorm),
                       S-1 = F_max * K.
            token_valid: (B, S-1) bool, True = token reel (patch valide,
                       fenetre valide, non droppe) — longueurs CTC.
            meta: dict (num_frames (B,), patch_counts (B,)).
        """
        B, H, W = img.shape
        K, s = self.window_patches, self.window_stride

        # 1. Pad la largeur au multiple du stride token (64 px) pour le
        #    patch-embed conv.
        rem = W % self.token_stride
        if rem != 0:
            img = F.pad(img, (0, self.token_stride - rem))
        W_pad = img.shape[2]

        x = img.unsqueeze(1)                          # (B, 1, H, W)
        x = self.conv(x)                              # (B, C, 16, W/4)
        feats = self.patch_embed(x)                   # (B, D, 1, T_feat)
        T_feat = feats.shape[3]
        assert T_feat <= self.max_patches, (
            f"T_feat={T_feat} > max_patches={self.max_patches} (RoPE) ; "
            "augmenter MAX_PATCHES_V19 ou borner la largeur des lignes."
        )
        grid = (
            feats.squeeze(2).permute(0, 2, 1).float()
        )                                             # (B, T_feat, D)

        # 2. Geometrie par echantillon : T reels, F fenetres, T_pad tokens
        #    de grille (le dernier fenetrage s'etend jusqu'a (F-1)*s+K).
        if patch_counts is not None:
            T_b = patch_counts.to(grid.device).clamp(min=1, max=T_feat)
        else:
            T_b = torch.full((B,), T_feat, dtype=torch.long, device=grid.device)
        F_b = self._frames_count(T_b)                 # (B,)
        F_max = int(F_b.max().item())
        T_pad = (F_max - 1) * s + K

        # 3. Grille padree puis decoupee en fenetres glissantes.
        if grid.shape[1] < T_pad:
            grid = F.pad(grid, (0, 0, 0, T_pad - grid.shape[1]))
        windows = grid.unfold(1, K, s)                # (B, F', D, K)
        windows = windows[:, :F_max]                  # (B, F_max, D, K)
        seq = (
            windows.permute(0, 1, 3, 2).reshape(B, F_max * K, -1)
        )                                              # (B, F_max*K, D)

        # Coordonnee x de chaque token (f, k) -> f*s + k ; id fenetre.
        ar_f = torch.arange(F_max, device=grid.device)
        ar_k = torch.arange(K, device=grid.device)
        pos_x = (ar_f.unsqueeze(1) * s + ar_k.unsqueeze(0)).reshape(-1)  # (S_t,)
        frame_id = ar_f.repeat_interleave(K)                              # (S_t,)
        S_t = F_max * K

        # 4. Validite des tokens : patch reel (x < T_b), fenetre reelle
        #    (f < F_b), non droppe.
        valid = (pos_x[None, :] < T_b[:, None]) & (frame_id[None, :] < F_b[:, None])

        if token_drop > 0.0 and self.training:
            drop_grid = (
                torch.rand(B, T_pad, device=grid.device) < token_drop
            )
            drop_grid[:, 0] = False                    # patch 0 = ancre, jamais droppe
            # Un token (f, k) est droppe ssi son PATCH x est droppe.
            dropped = drop_grid[:, pos_x]              # (B, S_t)
            valid = valid & ~dropped

        # 5. Masque d'attention additif (B, 1, S, S) :
        #    - [cls] (ligne 0) attend a tout le monde (tokens + lui-meme) ;
        #    - les tokens attendent aux fenetres passees ET presentes
        #      (block-causal), mais JAMAIS au [cls] (col 0 masquee) ;
        #    - les cles invalides (padding / drop) sont masquees pour tous.
        allowed = self._block_mask(F_max, grid.device)           # (S_t, S_t)
        allowed = torch.cat([
            torch.zeros(S_t, 1, dtype=torch.bool, device=grid.device),  # col cls interdite aux tokens
            allowed,
        ], dim=1)                                                # (S_t, 1+S_t)
        cls_row = torch.ones(1, 1 + S_t, dtype=torch.bool, device=grid.device)
        allowed = torch.cat([cls_row, allowed], dim=0)            # (S, S)

        key_valid = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=grid.device), valid],
            dim=1,
        )                                                        # (B, S) — cls toujours cle
        attn_bool = allowed[None] & key_valid[:, None, :]         # (B, S, S)
        neg_inf = torch.finfo(torch.float32).min
        attn_bias = torch.where(
            attn_bool, torch.zeros((), device=grid.device),
            torch.full((), neg_inf, device=grid.device),
        ).unsqueeze(1)                                           # (B, 1, S, S)

        # 6. [cls] en tete + N blocs block-causaux. Transformer en float32
        #    (masques additifs -inf + RoPE sensibles au fp16), le stem
        #    reste sous l'autocast de l'appelant.
        _f32 = (
            torch.amp.autocast("cuda", enabled=False)
            if img.is_cuda
            else contextlib.nullcontext()
        )
        with _f32:
            tokens = torch.cat(
                [self.cls_token.expand(B, 1, -1), seq.float()], dim=1
            )                                                    # (B, 1+S_t, D)
            positions = torch.cat(
                [torch.zeros(1, dtype=torch.long, device=grid.device), pos_x]
            )                                                    # (S,) — cls en 0 (rotation identite)

            for layer in self.layers:
                tokens = layer(tokens, attn_bias, self.rope, positions)

            # LayerNorm sur la SEQUENCE COMPLETE (incl. [cls]) : le [cls]
            # aboutit sur la sphere — prémisse du papier LeVJEPA, le
            # projecteur h_phi l'en sort ensuite vers R^K pour SIGReg.
            toks_norm = self.norm(tokens)
            cls_emb = toks_norm[:, 0]                               # (B, D) [cls] sur la sphere
            tokens_norm = toks_norm[:, 1:]                          # (B, S_t, D)

        return cls_emb, tokens_norm, valid, {
            "num_frames": F_b,
            "patch_counts": T_b,
        }


def test_encoder():
    """Test encoder on Pi"""
    print("Testing CNNEncoder...")

    encoder = CNNEncoder(img_height=32, window_size=10, embedding_dim=64)

    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {num_params:,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 32, 10)

    with torch.no_grad():
        z = encoder(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {z.shape}")
    print(f"✓ Encoder working!")

    return encoder


if __name__ == "__main__":
    test_encoder()
