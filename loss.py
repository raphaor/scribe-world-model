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
        
        # Standardize embeddings (zero mean, unit variance per dimension)
        z_centered = z - z.mean(dim=0)
        z_std = z_centered / (z_centered.std(dim=0) + 1e-8)
        
        # SIGReg loss:
        # 1. Penalize off-diagonal covariance (decorrelation)
        # 2. Penalize deviation from unit variance
        
        # Covariance matrix
        cov = (z_std.T @ z_std) / z_std.size(0)
        
        # Off-diagonal penalty (want identity matrix)
        off_diag = cov - torch.eye(cov.size(0), device=z.device)
        off_diag_loss = (off_diag ** 2).sum() / cov.size(0)
        
        # Variance penalty (want unit variance)
        var_loss = ((z_std.var(dim=0) - 1) ** 2).mean()
        
        loss = self.lambda_reg * (off_diag_loss + var_loss)
        
        return loss


class HWMLoss(nn.Module):
    """
    Combined loss for HWM training
    L = L_pred + λ * L_SIGReg
    """
    
    def __init__(self, lambda_sigreg=0.1, pred_loss_type='mse'):
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.pred_loss_type = pred_loss_type
        
        self.sigreg = SIGRegLoss(lambda_reg=1.0)
        
        if pred_loss_type == 'mse':
            self.pred_loss = nn.MSELoss()
        elif pred_loss_type == 'l1':
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
        
        # SIGReg regularization
        if z_all is not None:
            sigreg_loss = self.sigreg(z_all)
        else:
            # Use pred + target
            sigreg_loss = self.sigreg(torch.stack([z_pred, z_target], dim=1))
        
        # Total
        total_loss = pred_loss + self.lambda_sigreg * sigreg_loss
        
        return total_loss, {
            'total': total_loss.item(),
            'pred': pred_loss.item(),
            'sigreg': sigreg_loss.item()
        }


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
