"""
HWM Encoder - CNN 1D for encoding handwriting columns
Converts image columns to latent embeddings
"""

import torch
import torch.nn as nn


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
