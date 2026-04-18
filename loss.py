"""
HWM Loss Functions
Prediction loss + SIGReg regularizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGRegLoss(nn.Module):
    """
    SIGReg: Simple Isometric Gaussian Regularization
    Forces embeddings to follow isotropic Gaussian distribution

    From LeWorldModel paper - prevents collapse without EMA/stop-gradient
    """

    def __init__(self, lambda_reg=0.1):
        super().__init__()
        self.lambda_reg = lambda_reg
        self._eye_cache = {}

    def _get_eye(self, size, device):
        """Cache identity matrix to avoid repeated GPU allocations."""
        key = (size, device)
        if key not in self._eye_cache:
            self._eye_cache[key] = torch.eye(size, device=device)
        return self._eye_cache[key]

    def forward(self, z):
        """
        Args:
            z: (B, D) or (B, T, D) embeddings
        Returns:
            loss: scalar regularization loss
        """
        # Flatten if needed
        if z.dim() == 3:
            B, T, D = z.shape
            z = z.reshape(B * T, D)

        N = z.size(0)

        # Standardize embeddings (zero mean, unit variance per dimension)
        z_mean = z.mean(dim=0)
        z_centered = z - z_mean
        z_std = z_centered / (z_centered.std(dim=0) + 1e-8)

        # Covariance matrix: (D, N) @ (N, D) -> (D, D) — much smaller than (N, N)
        cov = (z_std.T @ z_std) / N

        # Off-diagonal penalty (want identity matrix)
        # Use mean over all D*D entries so the loss scale is independent of D
        eye = self._get_eye(cov.size(0), z.device)
        off_diag_loss = ((cov - eye) ** 2).mean()

        # Variance penalty (want unit variance)
        var_loss = ((z_std.var(dim=0) - 1) ** 2).mean()

        loss = self.lambda_reg * (off_diag_loss + var_loss)

        return loss


class VICRegLoss(nn.Module):
    """
    VICReg: Variance-Invariance-Covariance Regularization (Bardes+ 2022).

    Drop-in replacement for SIGReg. Key difference: operates on RAW z
    (not standardised), so it penalises both decorrelation AND scale
    collapse — SIGReg's ``/ std`` normalisation made both invisible,
    letting the encoder shrink magnitude to trivialise MSE.

    The invariance / prediction term lives outside this module (it's
    the JEPA MSE in our case).

    Returns (total, var_loss, cov_loss) so the caller can log the
    breakdown — useful to see which branch is actually firing.
    """

    def __init__(self, lambda_var=25.0, lambda_cov=1.0, gamma=1.0, eps=1e-4):
        super().__init__()
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma
        self.eps = eps

    def forward(self, z):
        if z.dim() == 3:
            B, T, D = z.shape
            z = z.reshape(B * T, D)
        N, D = z.shape

        # Variance hinge on RAW std: forces std(z_d) >= gamma for every
        # dim. This is the anti-scale-collapse guard SIGReg lacks.
        std = torch.sqrt(z.var(dim=0) + self.eps)
        var_loss = torch.mean(F.relu(self.gamma - std))

        # Covariance: off-diagonal of raw (un-standardised) covariance,
        # L2-summed and normalised by D (standard VICReg scaling).
        z_c = z - z.mean(dim=0)
        cov = (z_c.T @ z_c) / max(N - 1, 1)
        cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D

        total = self.lambda_var * var_loss + self.lambda_cov * cov_loss
        return total, var_loss.detach(), cov_loss.detach()


class HWMLoss(nn.Module):
    """
    Combined loss for HWM training
    L = L_pred + λ * L_SIGReg
    """

    def __init__(self, lambda_sigreg=0.1, pred_loss_type="mse"):
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.pred_loss_type = pred_loss_type

        self.sigreg = SIGRegLoss(lambda_reg=1.0)

        if pred_loss_type == "mse":
            self.pred_loss = nn.MSELoss()
        elif pred_loss_type == "l1":
            self.pred_loss = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {pred_loss_type}")

    def forward(self, z_pred, z_target, z_all=None):
        """
        Args:
            z_pred: (B, D) predicted embeddings
            z_target: (B, D) target embeddings
            z_all: (B, T, D) all embeddings for SIGReg (optional)
        Returns:
            total_loss, dict with loss components
        """
        # Prediction loss
        pred_loss = self.pred_loss(z_pred, z_target)

        # SIGReg regularization — gradients flow back to encoder to prevent collapse
        if z_all is not None:
            sigreg_loss = self.sigreg(z_all)
        else:
            sigreg_loss = self.sigreg(torch.stack([z_pred, z_target], dim=1))

        # Total
        total_loss = pred_loss + self.lambda_sigreg * sigreg_loss

        return total_loss, {
            "total": total_loss.item(),
            "pred": pred_loss.item(),
            "sigreg": sigreg_loss.item(),
        }


class HybridLoss(nn.Module):
    """
    Combined loss: prediction + VICReg + CTC

    L = lambda_pred * L_pred + lambda_sigreg * L_vicreg + lambda_ctc * L_ctc

    (``lambda_sigreg`` is kept as the external knob name for back-compat
    with config / checkpoints — it now scales the VICReg regulariser.)

    Knobs:
      - lambda_pred: scales the JEPA prediction MSE. Set to 0 to disable
        the self-supervised branch entirely (baseline = CTC-only).
      - target_norm: LayerNorm predictions and targets before MSE. Off
        by default — empirically it co-operated with SIGReg to hide
        scale collapse. VICReg works on raw z and doesn't need it.
    """

    def __init__(
        self,
        lambda_sigreg=0.1,
        lambda_ctc=1.0,
        lambda_pred=1.0,
        target_norm=False,
    ):
        super().__init__()
        self.pred_loss = nn.MSELoss()
        # VICReg replaces SIGReg: variance hinge on RAW std fights
        # scale collapse, covariance term handles decorrelation.
        self.reg = VICRegLoss()
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.lambda_sigreg = lambda_sigreg
        self.lambda_ctc = lambda_ctc
        self.lambda_pred = lambda_pred
        self.target_norm = target_norm

    def forward(
        self,
        z_pred,
        z_target,
        z_all,
        ctc_logits=None,
        targets=None,
        input_lengths=None,
        target_lengths=None,
    ):
        if self.lambda_pred > 0 and z_pred is not None:
            if self.target_norm:
                D = z_pred.shape[-1]
                z_pred = F.layer_norm(z_pred, (D,))
                z_target = F.layer_norm(z_target, (D,))
            pred = self.pred_loss(z_pred, z_target)
        else:
            pred = torch.zeros((), device=z_all.device)
        # VICReg must backprop through the encoder — it's the anti-collapse
        # mechanism that replaces EMA in JEPA. Without these gradients,
        # adapt mode diverges (embeddings drift with no anchor).
        reg, var, cov = self.reg(z_all)

        total = self.lambda_pred * pred + self.lambda_sigreg * reg
        losses = {
            "pred": pred.detach().item(),
            "var": var.item(),
            "cov": cov.item(),
        }

        # Collapse diagnostic: SIGReg only constrains the global distribution
        # over (B*T, D). If intra_var << global_var, frames within a single
        # line are near-identical — the predictor then solves the JEPA task
        # trivially by copying a neighbour, independent of what it learned.
        with torch.no_grad():
            if z_all.dim() == 3:
                D = z_all.shape[-1]
                intra = z_all.var(dim=1).mean()
                glob = z_all.reshape(-1, D).var(dim=0).mean()
                losses["intra_var"] = intra.item()
                losses["global_var"] = glob.item()

        if ctc_logits is not None and targets is not None:
            ctc_input = ctc_logits.permute(1, 0, 2)
            ctc = self.ctc_loss(ctc_input, targets, input_lengths, target_lengths)
            total = total + self.lambda_ctc * ctc
            losses["ctc"] = ctc.detach().item()

        losses["total"] = total.detach().item()
        return total, losses


def test_loss():
    """Test loss functions"""
    print("\nTesting Loss Functions...")

    batch_size = 4
    seq_len = 10
    embed_dim = 64

    # Test SIGReg
    sigreg = SIGRegLoss(lambda_reg=0.1)
    z = torch.randn(batch_size, seq_len, embed_dim)
    loss = sigreg(z)
    print(f"SIGReg loss (random z): {loss.item():.4f}")

    # Test HWM loss
    criterion = HWMLoss(lambda_sigreg=0.1)
    z_pred = torch.randn(batch_size, embed_dim)
    z_target = torch.randn(batch_size, embed_dim)
    z_all = torch.randn(batch_size, seq_len, embed_dim)

    total_loss, losses_dict = criterion(z_pred, z_target, z_all)
    print(f"Total loss: {losses_dict['total']:.4f}")
    print(f"  Pred: {losses_dict['pred']:.4f}")
    print(f"  SIGReg: {losses_dict['sigreg']:.4f}")
    print(f"✓ Loss functions working!")

    return criterion


if __name__ == "__main__":
    test_loss()
