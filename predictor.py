"""
HWM Predictor - Transformer for next-step prediction
Predicts future embeddings from past embeddings
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Lightweight transformer for predicting next embedding
    Input: sequence of embeddings (B, T, D)
    Output: predicted next embedding (B, D)
    """

    def __init__(
        self,
        embedding_dim=64,
        num_layers=2,
        num_heads=2,
        ff_dim=128,
        dropout=0.1,
        max_seq_len=5000,
        causal=True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.causal = causal

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            embedding_dim, max_len=max_seq_len, dropout=dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def _generate_causal_mask(self, seq_len, device):
        """Upper triangular mask: True = masked (cannot attend)."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

    def forward(self, z_seq, src_key_padding_mask=None):
        """
        Args:
            z_seq: (B, T, D) sequence of embeddings
            src_key_padding_mask: (B, T) bool — True = position to ignore
                as a key. Used by the v7 I-JEPA context-encoder role so
                that target/padding positions don't contribute to the
                context representation.
        Returns:
            z_pred: (B, T, D) predicted embedding at each position.
            In causal mode: predicts the next step given the past.
            In non-causal (JEPA) mode: bidirectional attention over the
            whole sequence — used to fill in masked positions.
        """
        z_seq = self.pos_encoder(z_seq)

        if self.causal:
            causal_mask = self._generate_causal_mask(z_seq.size(1), z_seq.device)
            encoded = self.transformer(
                z_seq, mask=causal_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        else:
            encoded = self.transformer(
                z_seq, src_key_padding_mask=src_key_padding_mask
            )

        z_pred = self.output_proj(encoded)  # (B, T, D)

        return z_pred

    def predict_sequence(self, z_seq, steps=1):
        """
        Predict multiple future steps autoregressively
        Args:
            z_seq: (B, T, D)
            steps: number of future steps to predict
        Returns:
            predictions: (B, steps, D)
        """
        predictions = []
        current_seq = z_seq

        for _ in range(steps):
            z_all = self.forward(current_seq)  # (B, T, D)
            z_next = z_all[:, -1, :]           # (B, D)
            predictions.append(z_next)
            current_seq = torch.cat([current_seq, z_next.unsqueeze(1)], dim=1)

        return torch.stack(predictions, dim=1)  # (B, steps, D)


class JEPACrossAttnPredictor(nn.Module):
    """
    I-JEPA style predictor with cross-attention to a context encoder.

    The v5/v6 predictor does in-place mask-token filling: mask tokens
    are substituted in the input and self-attention fills them. That
    lets context positions see mask tokens, which can leak subtle
    information and blurs the "predict from context" boundary.

    This predictor separates the two cleanly:

      1. A *context encoder* (self-attention, runs outside this module)
         processes z_seq with src_key_padding_mask hiding target and
         padding positions, producing z_ctx.
      2. This predictor builds *queries* at every sequence position
         (mask_token + positional encoding) and runs a stack of
         TransformerDecoder layers that cross-attend to z_ctx while
         self-attending among queries.
      3. The caller reads the output at masked positions to get the
         predictions.

    Queries never leak into the context representation, and targets
    are genuinely predicted from context alone.
    """

    def __init__(
        self,
        embedding_dim,
        num_layers,
        num_heads,
        ff_dim,
        dropout=0.1,
        max_seq_len=5000,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pos_encoder = PositionalEncoding(
            embedding_dim, max_len=max_seq_len, dropout=dropout
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, context, memory_key_padding_mask, mask_token, seq_len):
        """
        Args:
            context: (B, T, D) — context encoder output (all positions).
            memory_key_padding_mask: (B, T) bool — True = ignore this
                context position in cross-attention (targets + padding).
            mask_token: (1, 1, D) — learnable [MASK] vector, expanded
                to build queries.
            seq_len: T — query sequence length (== context.size(1)).
        Returns:
            pred: (B, T, D) — predictions at every position; caller
                selects the rows at masked positions.
        """
        B, T, D = context.size(0), seq_len, context.size(2)
        # Queries = mask_token at every position + positional encoding.
        # We build queries at all T positions (not only masked) so the
        # caller can index with the same boolean mask used elsewhere.
        queries = mask_token.expand(B, T, D).contiguous()
        queries = self.pos_encoder(queries)
        out = self.decoder(
            tgt=queries,
            memory=context,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(out)


class MAEDecoder(nn.Module):
    """
    Small Masked-AutoEncoder decoder for HWMv8.

    Unlike the v5-v7 JEPA family, v8's target is PIXELS (the raw
    content of masked patches), not embeddings. That makes the loss
    structurally collapse-proof: pixels are external, fixed targets
    that the encoder can never game.

    Flow (official MAE, SimMIM-lite variant — no encoder-side patch
    drop, because we also need an all-patch encoder pass for CTC):

      1. Encoder produces tokens at every patch position (visible and
         masked alike); we treat that output as ``visible_tokens`` plus
         "dummy" tokens at masked positions that we'll replace.
      2. Build ``decoder_input`` by keeping encoder output at visible
         positions and substituting a learned ``mask_token`` + decoder
         positional encoding at masked positions.
      3. Run a shallow decoder transformer.
      4. Linear head reconstructs the patch_h * patch_w pixel values
         from each decoder output token.
      5. Caller computes MSE between reconstructed and original pixels
         at masked, non-padding positions.
    """

    def __init__(
        self,
        encoder_dim,
        decoder_dim=256,
        num_layers=2,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
        patch_h=15,
        patch_w=16,
        max_n_h=400,
        n_v=8,
    ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.patch_h = patch_h
        self.patch_w = patch_w

        # Bridge from encoder to decoder width. Decoder is usually
        # thinner than the encoder (MAE: 512 enc -> 256 dec).
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)

        # Learnable [MASK] token at decoder width.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Separate positional encoding inside the decoder. Mirrors the
        # encoder's separable scheme so the decoder knows where each
        # token sits in the 2D grid.
        self.row_embed = nn.Parameter(torch.zeros(n_v, decoder_dim))
        self.col_embed = nn.Parameter(torch.zeros(max_n_h, decoder_dim))
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(decoder_dim)

        # Reconstruct pixel values for each patch.
        self.pixel_head = nn.Linear(decoder_dim, patch_h * patch_w)

    def forward(self, enc_tokens, mask_flat, n_v, n_h, key_padding_mask=None):
        """
        Args:
            enc_tokens: (B, N_v*N_h, D_enc) — encoder output at every patch.
            mask_flat: (B, N_v*N_h) bool — True = masked (target) positions.
            n_v, n_h: grid shape.
            key_padding_mask: (B, N_v*N_h) bool — True = padding position
                to hide from decoder self-attention (we still want it to
                attend between visible and masked, but not into padding).
        Returns:
            pred_pixels: (B, N_v*N_h, patch_h * patch_w) predicted pixels.
        """
        B, N, _ = enc_tokens.shape
        D = self.decoder_dim

        tokens = self.enc_to_dec(enc_tokens)                  # (B, N, D_dec)

        # Substitute mask_token at masked positions.
        mask_tok = self.mask_token.expand(B, N, D)
        tokens = torch.where(mask_flat.unsqueeze(-1), mask_tok, tokens)

        # Add decoder positional encoding (separable).
        pos = (
            self.row_embed.unsqueeze(1)                       # (N_v, 1, D)
            + self.col_embed[:n_h].unsqueeze(0)               # (1, N_h, D)
        ).reshape(n_v * n_h, D)
        tokens = tokens + pos.unsqueeze(0)

        tokens = self.decoder(tokens, src_key_padding_mask=key_padding_mask)
        tokens = self.norm(tokens)
        return self.pixel_head(tokens)


def test_predictor():
    """Test predictor on Pi"""
    print("\nTesting TransformerPredictor...")

    predictor = TransformerPredictor(
        embedding_dim=64, num_layers=2, num_heads=2, ff_dim=128, dropout=0.1
    )

    # Count parameters
    num_params = sum(p.numel() for p in predictor.parameters())
    print(f"Predictor parameters: {num_params:,}")

    # Test forward pass
    batch_size = 4
    seq_len = 10
    z_seq = torch.randn(batch_size, seq_len, 64)

    with torch.no_grad():
        z_pred = predictor(z_seq)
        z_multi = predictor.predict_sequence(z_seq, steps=3)

    print(f"Input shape: {z_seq.shape}")
    print(f"Single prediction shape: {z_pred.shape}")
    print(f"Multi-step prediction shape: {z_multi.shape}")
    print(f"✓ Predictor working!")

    return predictor


if __name__ == "__main__":
    test_predictor()
