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
    Combined loss: prediction + SIGReg + CTC

    L = L_pred + lambda_sigreg * L_sigreg + lambda_ctc * L_ctc
    """

    def __init__(self, lambda_sigreg=0.1, lambda_ctc=1.0):
        super().__init__()
        self.pred_loss = nn.MSELoss()
        self.sigreg = SIGRegLoss(lambda_reg=1.0)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.lambda_sigreg = lambda_sigreg
        self.lambda_ctc = lambda_ctc

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
        pred = self.pred_loss(z_pred, z_target)
        # SIGReg must backprop through the encoder — it's the anti-collapse
        # mechanism that replaces EMA in JEPA. Without these gradients,
        # adapt mode diverges (embeddings drift with no anchor).
        sigreg = self.sigreg(z_all)

        total = pred + self.lambda_sigreg * sigreg
        losses = {"pred": pred.detach().item(), "sigreg": sigreg.detach().item()}

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
