"""
I-JEPA style block-masking utilities.

Instead of causal next-frame prediction (trivially solved by stroke
continuity), we mask several contiguous blocks of the latent sequence
and ask the predictor to reconstruct their representations from the
surrounding bidirectional context. The targets come from the detached
encoder output (stop-grad), so the encoder is trained through the
context path only — just like I-JEPA / V-JEPA.
"""

import torch


def sample_jepa_mask(
    batch_size,
    seq_len,
    num_targets=4,
    min_size=4,
    max_size=10,
    valid_lengths=None,
    device=None,
    generator=None,
):
    """
    Sample per-sample target block masks for a batch of sequences.

    Blocks may overlap — the union defines the masked region, which is
    fine for JEPA (context is still the complement of the union).

    Args:
        batch_size: B
        seq_len: T (max length across the batch)
        num_targets: blocks to attempt per sample
        min_size, max_size: inclusive block-length range in frames
        valid_lengths: optional (B,) long tensor of non-padding lengths.
            When provided, blocks stay inside the valid region so we
            don't spend capacity predicting padding.
        device: target device for the mask
        generator: optional torch.Generator for reproducibility

    Returns:
        mask: (B, T) bool tensor, True at target positions.
    """
    device = device or torch.device("cpu")
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    for b in range(batch_size):
        if valid_lengths is not None:
            eff_len = int(valid_lengths[b].item())
        else:
            eff_len = seq_len
        if eff_len <= min_size:
            continue
        upper = min(max_size, eff_len - 1)
        for _ in range(num_targets):
            size = int(
                torch.randint(min_size, upper + 1, (1,), generator=generator).item()
            )
            start = int(
                torch.randint(0, eff_len - size + 1, (1,), generator=generator).item()
            )
            mask[b, start : start + size] = True

    return mask
