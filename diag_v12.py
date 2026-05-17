"""
Diagnostic for hwm_v12.pt — distinguishes a benign SIGReg plateau from a
dimensional collapse.

Loads the trained KrakenEncoderV12, runs real ALTO lines through it, and
reports on the encoder-output marginal:
  - per-dimension variance (collapse = many near-zero dims)
  - effective rank / participation ratio of the covariance
  - the SIGReg (Epps-Pulley) statistic, calibrated against a true
    Gaussian and against a full point-mass collapse for the same N, D.
"""
import torch
import numpy as np

import config
from encoder import KrakenEncoderV12
from loss import SIGRegEppsPulleyLoss
from data_alto import AltoLineDataset, collate_alto_v5_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_LINES = 128
CHUNK = 16

# --- 1. Load the trained encoder ----------------------------------------
ck = torch.load("hwm_v12.pt", map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}

encoder = KrakenEncoderV12(
    img_height=config.IMG_HEIGHT_V12,
    embedding_dim=config.EMBEDDING_DIM_V12,
    num_layers=config.NUM_LAYERS_V12,
    num_heads=config.NUM_HEADS_V12,
    ff_dim=config.FF_DIM_V12,
    dropout=0.0,
)
missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
print(f"Checkpoint: epoch {ck.get('epoch')} | loss {ck.get('loss'):.4f} | mode {ck.get('mode')}")
print(f"Encoder load: {len(missing)} missing, {len(unexpected)} unexpected")
encoder = encoder.to(device).eval()

# --- 2. Real ALTO lines --------------------------------------------------
dataset = AltoLineDataset(config.ALTO_DIRS, img_height=config.IMG_HEIGHT_V12, augment=False)
char_to_idx, _ = dataset.get_alphabet()
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(dataset), generator=g)[:N_LINES].tolist()
batch = [dataset[i] for i in idx]
padded, _t, input_lengths, _tl, _rt = collate_alto_v5_fn(batch, char_to_idx=char_to_idx)
print(f"Lines: {len(batch)} | padded img {tuple(padded.shape)}")

# --- 3. Encode, collect valid (non-padding) tokens -----------------------
valid_tokens = []
with torch.no_grad():
    for s in range(0, padded.shape[0], CHUNK):
        img = padded[s:s + CHUNK].to(device)
        il = input_lengths[s:s + CHUNK].to(device)
        z = encoder(img, il)                       # (b, T, D)
        T = z.shape[1]
        ar = torch.arange(T, device=device)
        vmask = ar[None, :] < il.clamp(max=T)[:, None]
        valid_tokens.append(z[vmask].float().cpu())
z = torch.cat(valid_tokens, dim=0)                 # (N, D)
N, D = z.shape
print(f"Valid tokens: N={N}, D={D}")

# --- 4. Marginal statistics ---------------------------------------------
mean = z.mean(0)
var = z.var(0, unbiased=False)
norms = z.norm(dim=1)
print("\n=== Encoder-output marginal ===")
print(f"  token L2 norm   : mean {norms.mean():.3f}  std {norms.std():.3f}  "
      f"(cv {norms.std() / norms.mean():.3f})")
print(f"  per-dim mean    : |mean| avg {mean.abs().mean():.4f}  max {mean.abs().max():.4f}")
print(f"  per-dim variance: min {var.min():.4e}  median {var.median():.4e}  "
      f"max {var.max():.4e}")
near_zero = (var < 0.01 * var.max()).sum().item()
print(f"  dims with var < 1% of max : {near_zero} / {D}")

# Effective rank via covariance eigenspectrum.
zc = z - mean
cov = (zc.t() @ zc) / N
eig = torch.linalg.eigvalsh(cov).clamp(min=0)
eig_sorted = torch.sort(eig, descending=True).values
part_ratio = (eig.sum() ** 2) / (eig.pow(2).sum())          # participation ratio
p = eig / eig.sum()
p = p[p > 0]
entropy_rank = torch.exp(-(p * p.log()).sum())              # exp(spectral entropy)
cum = torch.cumsum(eig_sorted, 0) / eig_sorted.sum()
dims_90 = int((cum < 0.90).sum().item()) + 1
dims_99 = int((cum < 0.99).sum().item()) + 1
print(f"\n=== Effective rank (D={D}) ===")
print(f"  participation ratio   : {part_ratio:.1f}")
print(f"  spectral-entropy rank : {entropy_rank:.1f}")
print(f"  dims for 90% variance : {dims_90}")
print(f"  dims for 99% variance : {dims_99}")
print(f"  top-5 eigenvalues     : {[round(float(e), 4) for e in eig_sorted[:5]]}")

# --- 5. SIGReg value, calibrated ----------------------------------------
sig = SIGRegEppsPulleyLoss(
    num_projections=config.SIGREG_PROJECTIONS_V12,
    num_knots=config.SIGREG_KNOTS_V12,
)
torch.manual_seed(0)
with torch.no_grad():
    s_real = float(sig(z))
    s_gauss = float(sig(torch.randn(N, D)))                 # ideal: ~0
    s_collapse = float(sig(torch.zeros(N, D)))              # full point-mass
    # standardised: subtract mean, scale each dim to unit variance.
    z_std = (z - mean) / var.clamp(min=1e-8).sqrt()
    s_std = float(sig(z_std))
print("\n=== SIGReg (Epps-Pulley) ===")
print(f"  trained encoder z       : {s_real:.4f}   <-- training plateaued here (~0.697)")
print(f"  same z, per-dim standardised : {s_std:.4f}")
print(f"  reference N(0,1)        : {s_gauss:.4f}   (ideal floor)")
print(f"  reference point-mass    : {s_collapse:.4f}   (full collapse)")

# --- 6. Verdict ----------------------------------------------------------
print("\n=== Verdict ===")
frac_rank = float(part_ratio) / D
if frac_rank < 0.15:
    print(f"  DIMENSIONAL COLLAPSE: rank uses only {frac_rank * 100:.0f}% of {D} dims.")
elif frac_rank < 0.4:
    print(f"  PARTIAL anisotropy: rank uses {frac_rank * 100:.0f}% of {D} dims.")
else:
    print(f"  Healthy rank: {frac_rank * 100:.0f}% of {D} dims used — plateau is benign.")
if s_std < 0.5 * s_real:
    print(f"  SIGReg drops {s_real:.3f} -> {s_std:.3f} after standardising:"
          f" plateau is mostly wrong scale/mean, not non-Gaussian shape.")
else:
    print(f"  SIGReg stays high ({s_std:.3f}) after standardising:"
          f" the marginal shape itself is non-Gaussian (multimodal/heavy-tailed).")
