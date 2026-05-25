"""
HWM Loss Functions
Prediction loss + SIGReg regularizer
"""

import numpy as np
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


class InfoNCELoss(nn.Module):
    """
    Contrastive (InfoNCE) prediction loss for JEPA.

    Replaces MSE regression, which has a trivial solution:
    ``pred* = E[target | context]`` collapses to the mean whenever
    targets are VICReg-constrained noise-like — observed as
    ``pred = Var(target) ≈ 1`` in our runs.

    For each of the N masked positions in the batch, the positive is
    its aligned (stop-grad) target and the negatives are every other
    target in the batch. Predicting the mean gives *chance*-level
    contrastive loss, so the trivial minimum disappears.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred, target):
        # pred, target: (N, D). target should already be stop-grad.
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        logits = pred @ target.T / self.temperature  # (N, N), row i vs all targets
        labels = torch.arange(pred.size(0), device=pred.device)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            top1 = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, top1


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
      - lambda_pred: scales the JEPA prediction loss. Set to 0 to disable
        the self-supervised branch entirely (baseline = CTC-only).
      - pred_loss_type: "mse" (legacy, next-frame regression used by
        v2-v4) or "infonce" (contrastive, used by v5). MSE has a
        trivial minimum ``pred* = E[target|context]`` that collapses to
        the mean when targets look noise-like; InfoNCE removes it.
      - target_norm: LayerNorm pred + target before MSE. Ignored when
        pred_loss_type=="infonce" (InfoNCE L2-normalises internally).
    """

    def __init__(
        self,
        lambda_sigreg=0.1,
        lambda_ctc=1.0,
        lambda_pred=1.0,
        target_norm=False,
        pred_loss_type="mse",
        infonce_temp=0.1,
    ):
        super().__init__()
        self.pred_loss_type = pred_loss_type
        if pred_loss_type == "mse":
            self.pred_loss = nn.MSELoss()
        elif pred_loss_type == "infonce":
            self.pred_loss = InfoNCELoss(temperature=infonce_temp)
        else:
            raise ValueError(f"Unknown pred_loss_type: {pred_loss_type}")
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
        pred_top1 = None
        if self.lambda_pred > 0 and z_pred is not None:
            if self.pred_loss_type == "infonce":
                pred, pred_top1 = self.pred_loss(z_pred, z_target)
            else:
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
        if pred_top1 is not None:
            losses["acc"] = pred_top1.item()

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


class MAEHybridLoss(nn.Module):
    """
    HWMv8 loss: pixel-reconstruction MAE + optional CTC.

    Fundamentally different from HybridLoss:
      - No VICReg / InfoNCE — the MAE target is raw pixels, an
        external signal the encoder cannot collapse onto. The
        scale-collapse and trivial-mean failure modes that motivated
        VICReg and InfoNCE in v5-v7 do not apply.
      - Reconstruction loss is patch-normalised MSE (each patch
        target is standardised, as in the MAE paper) so bright/dark
        regions contribute equally.
      - Loss averages only over ``valid_mask`` positions (masked AND
        non-padding), not over the whole tensor.
    """

    def __init__(self, lambda_mae=1.0, lambda_ctc=1.0, norm_pixel_loss=True):
        super().__init__()
        self.lambda_mae = lambda_mae
        self.lambda_ctc = lambda_ctc
        self.norm_pixel_loss = norm_pixel_loss
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def _patch_norm(self, target_pixels):
        mean = target_pixels.mean(dim=-1, keepdim=True)
        var = target_pixels.var(dim=-1, keepdim=True, unbiased=False)
        return (target_pixels - mean) / torch.sqrt(var + 1e-6)

    def forward(
        self,
        pred_pixels,  # (B, N, P) full grid predictions, or None
        target_pixels,  # (B, N, P) ground-truth pixel patches, or None
        valid_mask,  # (B, N) bool — positions to score (masked & valid)
        ctc_logits=None,
        targets=None,
        input_lengths=None,
        target_lengths=None,
    ):
        losses = {}
        if pred_pixels is not None and target_pixels is not None:
            if self.norm_pixel_loss:
                target_pixels = self._patch_norm(target_pixels)
            per_patch = ((pred_pixels - target_pixels) ** 2).mean(dim=-1)
            denom = valid_mask.sum().clamp(min=1)
            mae = (per_patch * valid_mask).sum() / denom
            total = self.lambda_mae * mae
            losses["mae"] = mae.detach().item()
        else:
            device = (
                ctc_logits.device if ctc_logits is not None else torch.device("cpu")
            )
            total = torch.zeros((), device=device)

        if ctc_logits is not None and targets is not None:
            ctc_input = ctc_logits.permute(1, 0, 2)
            ctc = self.ctc_loss(ctc_input, targets, input_lengths, target_lengths)
            total = total + self.lambda_ctc * ctc
            losses["ctc"] = ctc.detach().item()

        losses["total"] = (
            total.detach().item() if isinstance(total, torch.Tensor) else float(total)
        )
        return total, losses


class SIGRegLossV2(nn.Module):
    """
    SIGReg corrected: operates on RAW z (no standardisation).

    The original SIGRegLoss divides by per-dim std before computing the
    covariance.  That makes the loss scale-invariant — tiny-magnitude
    quasi-constant embeddings pass the regulariser because after
    normalisation they look like unit-variance noise.  Result: the
    encoder collapses and MSN / MSE loss drops to ~0 trivially.

    V2 keeps the off-diagonal covariance penalty from SIGReg but adds
    a variance hinge (like VICReg) on the RAW std, so scale collapse
    is penalised directly.

    Components:
      - var_loss: mean(relu(gamma - std(z_d)))  over all D dims.
        Ensures every dimension has std >= gamma.  Anti-scale-collapse.
      - cov_loss: mean over all D*D entries of (cov_raw - I)^2.
        Decorrelates dimensions, same as original SIGReg but on raw z.
    """

    def __init__(self, lambda_var=25.0, lambda_cov=1.0, gamma=1.0, eps=1e-4):
        super().__init__()
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma
        self.eps = eps
        self._eye_cache = {}

    def _get_eye(self, size, device):
        key = (size, device)
        if key not in self._eye_cache:
            self._eye_cache[key] = torch.eye(size, device=device)
        return self._eye_cache[key]

    def forward(self, z):
        z = z.float()
        if z.dim() == 3:
            B, T, D = z.shape
            z = z.reshape(B * T, D)
        N, D = z.shape

        std = torch.sqrt(z.var(dim=0) + self.eps)
        var_loss = torch.mean(F.relu(self.gamma - std))

        z_c = z - z.mean(dim=0)
        cov = (z_c.T @ z_c) / N
        eye = self._get_eye(D, z.device)
        cov_loss = ((cov - eye) ** 2).mean()

        total = self.lambda_var * var_loss + self.lambda_cov * cov_loss
        return total, var_loss.detach(), cov_loss.detach()


class JEPALoss(nn.Module):
    """
    Loss for HWMv10: JEPA prediction + SIGRegV2 + optional CTC.

    L = lambda_pred * MSE(proj(z_pred), proj(sg(z_target)))
      + lambda_sigreg * SIGRegV2(z_seq)
      + lambda_ctc * CTC(z_pooled, targets)
    """

    def __init__(
        self,
        lambda_pred=1.0,
        lambda_sigreg=0.1,
        lambda_ctc=1.0,
        sigreg_var=25.0,
        sigreg_cov=1.0,
        sigreg_gamma=1.0,
    ):
        super().__init__()
        self.lambda_pred = lambda_pred
        self.lambda_sigreg = lambda_sigreg
        self.lambda_ctc = lambda_ctc
        self.reg = SIGRegLossV2(
            lambda_var=sigreg_var,
            lambda_cov=sigreg_cov,
            gamma=sigreg_gamma,
        )
        self.mse = nn.MSELoss()
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(
        self,
        z_pred=None,
        z_target=None,
        z_seq=None,
        ctc_logits=None,
        targets=None,
        input_lengths=None,
        target_lengths=None,
    ):
        losses = {}
        device = (
            z_seq.device
            if z_seq is not None
            else ctc_logits.device
            if ctc_logits is not None
            else torch.device("cpu")
        )

        if z_pred is not None and z_target is not None:
            pred = self.mse(z_pred.float(), z_target.float())
            losses["pred"] = pred.detach().item()
        else:
            pred = torch.zeros((), device=device)

        reg_total = torch.zeros((), device=device)
        if z_seq is not None:
            reg_total, var_l, cov_l = self.reg(z_seq)
            losses["sigreg"] = reg_total.detach().item()
            losses["var"] = var_l.item()
            losses["cov"] = cov_l.item()

        total = self.lambda_pred * pred + self.lambda_sigreg * reg_total

        if ctc_logits is not None and targets is not None:
            ctc_input = ctc_logits.permute(1, 0, 2)
            ctc = self.ctc_loss(ctc_input, targets, input_lengths, target_lengths)
            total = total + self.lambda_ctc * ctc
            losses["ctc"] = ctc.detach().item()

        losses["total"] = total.detach().item()
        return total, losses


class MSNLoss(nn.Module):
    """
    HWMv9 loss: MSN-style masked-image consistency + SIGReg + optional CTC.

    The pretext is "same abstraction with and without the mask":

        image ─┬─► encoder ─► z_clean (gradient flows → CTC + SIGReg)
               │
               └─► [mask 2D patches as [MASK] tokens] ─► encoder
                           ─► z_masked (gradient flows → MSN)

        L_msn = MSE(z_masked, z_clean.detach()) at masked, non-padding positions

    Key design points:

    - The MSN target is ``z_clean.detach()`` — only the masked branch
      gets gradients from the MSN term. The clean branch still gets
      gradients from CTC and SIGReg, so the encoder is trained end-to-end.
    - No EMA teacher, no projection head, no prototype assignments.
      Just "predict your own clean representation from partial input".
    - SIGReg (LeWorldModel paper, `SIGRegLoss`) on the raw clean output
      is the anti-collapse guard. Without it, the trivial solution
      ``z_masked = z_clean = const`` satisfies MSN perfectly.
    """

    def __init__(self, lambda_msn=1.0, lambda_sigreg=0.1, lambda_ctc=1.0):
        super().__init__()
        self.lambda_msn = lambda_msn
        self.lambda_sigreg = lambda_sigreg
        self.lambda_ctc = lambda_ctc
        self.sigreg = SIGRegLoss(lambda_reg=1.0)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(
        self,
        z_masked,  # (B, N, D) student output, or None (CTC-only)
        z_clean,  # (B, N, D) teacher output (not yet detached)
        mask_flat,  # (B, N) bool — True = masked (target) position
        valid_flat,  # (B, N) bool — True = non-padding position
        ctc_logits=None,
        targets=None,
        input_lengths=None,
        target_lengths=None,
    ):
        losses = {}
        total = None

        if z_masked is not None and z_clean is not None and mask_flat is not None:
            # MSE at masked, non-padding positions only.
            scored = mask_flat & valid_flat if valid_flat is not None else mask_flat
            per_pos = ((z_masked - z_clean.detach()) ** 2).mean(dim=-1)
            denom = scored.sum().clamp(min=1)
            msn = (per_pos * scored).sum() / denom
            losses["msn"] = msn.detach().item()

            # SIGReg on the clean encoder output (trains encoder).
            reg = self.sigreg(z_clean)
            losses["sigreg"] = reg.detach().item()

            total = self.lambda_msn * msn + self.lambda_sigreg * reg
        else:
            device = (
                ctc_logits.device if ctc_logits is not None else torch.device("cpu")
            )
            total = torch.zeros((), device=device)

        if ctc_logits is not None and targets is not None:
            ctc_input = ctc_logits.permute(1, 0, 2)
            ctc = self.ctc_loss(ctc_input, targets, input_lengths, target_lengths)
            total = total + self.lambda_ctc * ctc
            losses["ctc"] = ctc.detach().item()

        losses["total"] = (
            total.detach().item() if isinstance(total, torch.Tensor) else float(total)
        )
        return total, losses


class SimSiamHybridLoss(nn.Module):
    """
    HWMv11 loss: SimSiam-style cosine consistency + SIGRegV2 + optional CTC.

    Pretext: encode a clean view and a perturbed view of the same line,
    pool each to a single line vector, run the perturbed pool through a
    predictor MLP, then maximise the cosine similarity between the
    student prediction and the (stop-grad) clean target.

        L = -lambda_cons * cos_sim(p_pert, sg(v_clean))
            + lambda_sigreg * SIGRegV2(z_clean)
            + lambda_ctc    * CTC(z_clean, targets)

    Notes
    -----
    - Cosine (not MSE) makes the target scale-invariant: shrinking the
      embedding magnitudes does NOT minimise the consistency loss. This
      removes one collapse mode by construction.
    - SIGRegV2 (variance hinge on RAW std + covariance decorrelation)
      keeps embeddings spread out per-dim and decorrelated. Gradient
      flows through z_clean only; the perturbed branch trains via the
      consistency term.
    - The predictor MLP lives on the perturbed branch only (set
      asymmetry, SimSiam recipe). It is what prevents the "encoder ==
      identity" collapse that would otherwise satisfy cos_sim = 1.
    - Loss is computed at the LINE level (after temporal pooling), so
      perturbations that shift positions horizontally (the v11 default)
      do not break alignment.
    """

    def __init__(
        self,
        lambda_cons=1.0,
        lambda_sigreg=0.1,
        lambda_ctc=1.0,
        sigreg_var=25.0,
        sigreg_cov=1.0,
        sigreg_gamma=1.0,
    ):
        super().__init__()
        self.lambda_cons = lambda_cons
        self.lambda_sigreg = lambda_sigreg
        self.lambda_ctc = lambda_ctc
        self.reg = SIGRegLossV2(
            lambda_var=sigreg_var,
            lambda_cov=sigreg_cov,
            gamma=sigreg_gamma,
        )
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(
        self,
        p_pert=None,         # (B, D) predictor output on perturbed view, or None
        v_clean=None,        # (B, D) pooled clean view, or None
        z_seq=None,          # (B, T, D) raw clean encoder output, or None
        ctc_logits=None,
        targets=None,
        input_lengths=None,
        target_lengths=None,
    ):
        losses = {}
        device = (
            z_seq.device
            if z_seq is not None
            else ctc_logits.device
            if ctc_logits is not None
            else torch.device("cpu")
        )
        total = torch.zeros((), device=device)

        if p_pert is not None and v_clean is not None:
            # Cosine similarity on line vectors. Stop-grad on the clean
            # target — gradients flow only through the perturbed branch
            # (predictor + encoder of the perturbed view).
            cos = F.cosine_similarity(
                p_pert.float(), v_clean.detach().float(), dim=-1
            )
            cons = -cos.mean()
            losses["cons"] = cons.detach().item()
            total = total + self.lambda_cons * cons

        if z_seq is not None:
            reg_total, var_l, cov_l = self.reg(z_seq)
            losses["sigreg"] = reg_total.detach().item()
            losses["var"] = var_l.item()
            losses["cov"] = cov_l.item()
            total = total + self.lambda_sigreg * reg_total

        if ctc_logits is not None and targets is not None:
            ctc_input = ctc_logits.permute(1, 0, 2)
            ctc = self.ctc_loss(ctc_input, targets, input_lengths, target_lengths)
            total = total + self.lambda_ctc * ctc
            losses["ctc"] = ctc.detach().item()

        losses["total"] = total.detach().item()
        return total, losses


class SIGRegEppsPulleyLoss(nn.Module):
    """
    SIGReg as defined in the LeWorldModel paper (Maes et al.) — the
    *real* one, not the VICReg-style ``SIGRegLossV2``.

    Embeddings are projected onto M random unit directions; on each 1D
    projection the Epps-Pulley normality statistic is optimised against
    a standard normal N(0, 1). By the Cramér-Wold theorem, matching
    every 1D marginal to N(0, 1) matches the full joint distribution to
    an isotropic Gaussian N(0, I). This single term therefore pins the
    mean to 0, the variance to 1, AND enforces Gaussianity — which is
    why the paper needs only one regularisation weight.

    Epps-Pulley statistic. ``T(h)`` is the squared L2 distance between
    the empirical characteristic function (ECF) of a 1D sample and that
    of N(0, 1), ``phi_0(t) = exp(-t^2 / 2)``, integrated against the
    Gaussian weight ``exp(-t^2 / 2)``::

        T(h) = integral |phi_n(t) - phi_0(t)|^2 exp(-t^2/2) dt
             ~ sum_k w_k [ (Re phi_n(t_k) - phi_0(t_k))^2
                           + (Im phi_n(t_k))^2 ]

    with ``phi_n(t) = mean_j exp(i t h_j)``. The integral is evaluated
    by Gauss-Hermite quadrature: the substitution ``t = sqrt(2) x``
    turns the ``exp(-t^2/2)`` weight into the ``exp(-x^2)`` weight that
    Gauss-Hermite integrates exactly, so the knots are ``sqrt(2) x_k``
    and the quadrature weights ``sqrt(2) a_k``. Everything is
    differentiable in ``h`` and hence in the encoder parameters.

    IMPORTANT: the encoder output must be UN-normalised (no final
    LayerNorm). A per-sample LayerNorm constrains samples to a sphere,
    a support an isotropic Gaussian cannot have — SIGReg then never
    settles (paper Sec. 3.1). HWMv12's encoder follows this (Option B).
    """

    def __init__(
        self,
        num_projections=256,
        num_knots=17,
        resample=True,
        eps=1e-6,
    ):
        super().__init__()
        self.num_projections = num_projections
        self.resample = resample
        self.eps = eps

        # Gauss-Hermite nodes/weights: integral f(x) exp(-x^2) dx
        #   ~ sum_k a_k f(x_k).  Rescale to the exp(-t^2/2) weight.
        nodes, gh_w = np.polynomial.hermite.hermgauss(num_knots)
        sqrt2 = float(np.sqrt(2.0))
        knots = torch.tensor(nodes * sqrt2, dtype=torch.float32)
        quad_w = torch.tensor(gh_w * sqrt2, dtype=torch.float32)
        self.register_buffer("knots", knots)                    # (K,)
        self.register_buffer("quad_w", quad_w)                   # (K,)
        # Target ECF of N(0,1) at the knots (purely real).
        self.register_buffer("phi0", torch.exp(-0.5 * knots * knots))  # (K,)
        # Optional fixed projection set (used when resample=False).
        self._fixed_U = None

    def _directions(self, dim, device):
        u = torch.randn(self.num_projections, dim, device=device)
        return F.normalize(u, dim=1)

    def forward(self, z, valid_mask=None):
        """
        Args:
            z: (B, T, D) or (N, D) embeddings — raw, un-normalised.
            valid_mask: optional (B, T) or (N,) bool, True = real
                (non-padding) position. Padding frames are excluded so
                they don't bias the distribution towards a spike.
        Returns:
            scalar SIGReg loss = Epps-Pulley statistic on raw z.
        """
        z = z.float()
        if z.dim() == 3:
            z = z.reshape(-1, z.shape[-1])
        if valid_mask is not None:
            z = z[valid_mask.reshape(-1)]

        N, D = z.shape
        if N < 2:
            return z.new_zeros(())

        if self.resample or self._fixed_U is None or self._fixed_U.shape[1] != D:
            U = self._directions(D, z.device)
            if not self.resample:
                self._fixed_U = U
        else:
            U = self._fixed_U

        h = z @ U.t()                              # (N, M) projections
        # ECF at every knot:  phi_n(t_k) = mean_j exp(i t_k h_j).
        th = h.unsqueeze(-1) * self.knots.view(1, 1, -1)   # (N, M, K)
        cos = th.cos().mean(dim=0)                     # (M, K) = Re phi_n
        sin = th.sin().mean(dim=0)                     # (M, K) = Im phi_n
        diff2 = (cos - self.phi0.view(1, -1)) ** 2 + sin ** 2  # (M, K)
        shape_loss = (diff2 * self.quad_w.view(1, -1)).sum(dim=-1).mean()  # ()

        return shape_loss


class SupConLoss(nn.Module):
    """
    Supervised contrastive loss (Khosla et al. 2020).

    For each anchor, every other sample sharing its label is a positive
    and all remaining samples are negatives. Used by HWMv12 to make line
    embeddings of the same writer/page cluster together — an explicit
    "writer style" signal on top of the masked-segment pretext.

    Note on false negatives: with a page id used as a writer proxy, two
    lines from different pages of the *same* clerk are treated as a
    negative pair. This is label noise, tolerated as standard in
    contrastive learning; the branch is optional and weighted low.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, feats, labels):
        """
        Args:
            feats: (B, D) line embeddings (already projected).
            labels: (B,) long writer/page ids.
        Returns:
            (loss, coverage) — coverage is the fraction of anchors that
            had at least one positive in the batch (0 ⇒ loss is 0).
        """
        feats = F.normalize(feats.float(), dim=-1)
        B = feats.shape[0]
        device = feats.device
        if B < 2:
            return feats.new_zeros(()), feats.new_zeros(())

        sim = feats @ feats.t() / self.temperature        # (B, B)
        eye = torch.eye(B, dtype=torch.bool, device=device)
        labels = labels.view(-1)
        pos = (labels[:, None] == labels[None, :]) & ~eye  # (B, B)

        # log-softmax over all non-self entries. Mask the diagonal with
        # finfo.min (a large *finite* negative) rather than -inf: -inf
        # would survive into log_prob and give -inf * 0 = NaN when
        # multiplied by the (False) diagonal of ``pos``.
        sim = sim.masked_fill(eye, torch.finfo(sim.dtype).min)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        pos_count = pos.sum(dim=1)
        loss_i = -(log_prob * pos).sum(dim=1) / pos_count.clamp(min=1)
        valid = pos_count > 0
        if not valid.any():
            return feats.new_zeros(()), feats.new_zeros(())
        return loss_i[valid].mean(), valid.float().mean()


class V12Loss(nn.Module):
    """
    HWMv12 objective — four terms, three active by default:

        L = lambda_ctc    * CTC
          + lambda_jepa   * InfoNCE(masked-segment prediction)
          + lambda_sigreg * SIGReg(Epps-Pulley, paper)
          + lambda_wc     * SupCon(writer/page id)        [optional]

    - InfoNCE (not MSE) on the masked-segment pretext: MSE has the
      trivial minimum ``pred = E[target|context]`` that collapses to
      the mean (the v5-v10 wall). In-batch negatives remove it.
    - SIGReg is the real Epps-Pulley regulariser; it needs the
      un-normalised encoder output (Option B).
    - SupCon fires only when ``writer_id`` is supplied; otherwise the
      branch is inert at zero cost.
    """

    def __init__(
        self,
        lambda_ctc=1.0,
        lambda_jepa=0.5,
        lambda_sigreg=0.1,
        lambda_wc=0.2,
        infonce_temp=0.1,
        supcon_temp=0.1,
        sigreg_projections=256,
        sigreg_knots=17,
    ):
        super().__init__()
        self.lambda_ctc = lambda_ctc
        self.lambda_jepa = lambda_jepa
        self.lambda_sigreg = lambda_sigreg
        self.lambda_wc = lambda_wc
        self.infonce = InfoNCELoss(temperature=infonce_temp)
        self.sigreg = SIGRegEppsPulleyLoss(
            num_projections=sigreg_projections, num_knots=sigreg_knots
        )
        self.supcon = SupConLoss(temperature=supcon_temp)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(
        self,
        z_pred=None,        # (N, P) projected masked-frame predictions
        z_target=None,      # (N, P) projected stop-grad targets
        z_seq=None,         # (B, T, D) raw clean encoder output (SIGReg)
        valid_mask=None,    # (B, T) bool, True = non-padding frame
        ctc_logits=None,
        targets=None,
        input_lengths=None,
        target_lengths=None,
        line_vec=None,      # (B, P) projected line vectors (SupCon)
        writer_id=None,     # (B,) long writer/page ids, or None
    ):
        losses = {}
        device = (
            z_seq.device
            if z_seq is not None
            else ctc_logits.device
            if ctc_logits is not None
            else torch.device("cpu")
        )
        total = torch.zeros((), device=device)

        # 1. Masked-segment InfoNCE. Needs >=2 masked frames for negatives.
        if z_pred is not None and z_target is not None and z_pred.shape[0] >= 2:
            jepa, acc = self.infonce(z_pred, z_target)
            total = total + self.lambda_jepa * jepa
            losses["jepa"] = jepa.detach().item()
            losses["jepa_acc"] = acc.item()

        # 2. SIGReg anti-collapse on the raw clean embeddings.
        if z_seq is not None and self.lambda_sigreg > 0:
            reg = self.sigreg(z_seq, valid_mask=valid_mask)
            total = total + self.lambda_sigreg * reg
            losses["sigreg"] = reg.detach().item()

        # 3. Writer/page SupCon (optional).
        if line_vec is not None and writer_id is not None and self.lambda_wc > 0:
            wc, cov = self.supcon(line_vec, writer_id)
            total = total + self.lambda_wc * wc
            losses["wc"] = wc.detach().item()
            losses["wc_cov"] = cov.detach().item()

        # 4. CTC recognition.
        if ctc_logits is not None and targets is not None:
            ctc = self.ctc_loss(
                ctc_logits.permute(1, 0, 2), targets, input_lengths, target_lengths
            )
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
