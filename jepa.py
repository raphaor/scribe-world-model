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


def sample_2d_block_mask(
    batch_size,
    n_v,
    n_h,
    num_blocks=4,
    min_h=2,
    max_h=4,
    min_w=4,
    max_w=16,
    valid_h_lengths=None,
    device=None,
    generator=None,
):
    """
    Sample per-sample 2D rectangular target-block masks.

    Used by HWMv8 where the image is processed as a 2D patch grid
    (``N_v`` vertical × ``N_h`` horizontal patches). Rectangles are
    sampled inside the valid region and their union defines the mask.

    Args:
        batch_size: B
        n_v: number of vertical patches in the grid
        n_h: number of horizontal patches (max across the batch)
        num_blocks: blocks to attempt per sample
        min_h, max_h: inclusive vertical block-size range (patches)
        min_w, max_w: inclusive horizontal block-size range (patches)
        valid_h_lengths: optional (B,) long tensor of non-padding horizontal
            extent in patches. Blocks stay inside ``[0, valid_h_lengths[b])``
            so we don't predict padding.
        device, generator: standard forwards.

    Returns:
        mask: (B, n_v, n_h) bool tensor, True at target (masked) patches.
    """
    device = device or torch.device("cpu")
    mask = torch.zeros(batch_size, n_v, n_h, dtype=torch.bool, device=device)

    if n_v < min_h:
        return mask

    for b in range(batch_size):
        eff_n_h = (
            int(valid_h_lengths[b].item())
            if valid_h_lengths is not None
            else n_h
        )
        if eff_n_h <= min_w:
            continue
        upper_h = min(max_h, n_v)
        upper_w = min(max_w, eff_n_h)
        for _ in range(num_blocks):
            h = int(
                torch.randint(min_h, upper_h + 1, (1,), generator=generator).item()
            )
            w = int(
                torch.randint(min_w, upper_w + 1, (1,), generator=generator).item()
            )
            start_v = int(
                torch.randint(0, n_v - h + 1, (1,), generator=generator).item()
            )
            start_h = int(
                torch.randint(0, eff_n_h - w + 1, (1,), generator=generator).item()
            )
            mask[b, start_v : start_v + h, start_h : start_h + w] = True

    return mask
