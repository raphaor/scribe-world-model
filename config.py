"""
HWM-v1 Configuration
Pi-friendly hyperparameters for lightweight world model
"""

# Architecture
EMBEDDING_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 2
FF_DIM = 128
DROPOUT = 0.1

# Input processing
WINDOW_SIZE = 10
STRIDE = 5
IMG_HEIGHT = 32

# Training
BATCH_SIZE = 4
SEQ_LEN = 50
LEARNING_RATE = 1e-3
SIGREG_LAMBDA = 0.1
MAX_PARAMS = 1_000_000

# Data
NUM_SYNTHETIC_LINES = 100
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Logging
LOG_INTERVAL = 5

# --- HWM-v2 additions ---

CTC_BLANK = 0
LAMBDA_CTC = 1.0

ENCODER_TYPE = "conv2d"
IMG_HEIGHT_V2 = 48
EMBEDDING_DIM_V2 = 96
FF_DIM_V2 = 192

ALTO_DIRS = [
    #"/media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie/Alto/bars_dordogne_alto",
    #"/media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie/Alto/saint_chamassy_dordogne_alto_set_1",
    ## "/media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie/Alto/saint_chamassy_dordogne_alto_set_train",
    "/media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie/Alto/lectaurep_bronod_notaire_paris_18e",
    "/media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie/Alto/lectaurep_mariages_divorces_paris_19e",
    "/media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie/Alto/lectaurep_repertoires_notaires_paris_1830-1939",
    "/media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie/Alto/timeuscorpus_prudhommes_paris_1858-1878",
]

# --- HWM-v3 ---

WINDOW_SIZE_V3 = 32
STRIDE_V3 = 4
IMG_HEIGHT_V3 = 48
EMBEDDING_DIM_V3 = 128
NUM_LAYERS_V3 = 4
NUM_HEADS_V3 = 4
FF_DIM_V3 = 384
LAMBDA_CTC_V3 = 2.0

# --- HWM-v4 ---

WINDOW_SIZE_V4 = 32
STRIDE_V4 = 4
IMG_HEIGHT_V4 = 48
EMBEDDING_DIM_V4 = 256
NUM_LAYERS_V4 = 4
NUM_HEADS_V4 = 8
FF_DIM_V4 = 512
LAMBDA_CTC_V4 = 0.5
CTC_HIDDEN_V4 = 256


# --- HWM-v5 ---

IMG_HEIGHT_V5 = 120
# Dropped from 960 to 384 after VICReg+MSE runs showed pred stuck at
# the MSE trivial minimum (predictor outputs the mean). With 960 dim,
# information is spread too thin — each dim looks like noise to the
# predictor. 384 concentrates content per dim and aligns with the
# wav2vec2 / I-JEPA literature scale. FF_DIM follows 2x.
EMBEDDING_DIM_V5 = 384
NUM_LAYERS_V5 = 4
NUM_HEADS_V5 = 8
FF_DIM_V5 = 768
LAMBDA_CTC_V5 = 1.0
CTC_HIDDEN_V5 = 512
CTC_NUM_LSTM_V5 = 1

# "mse" (legacy next-frame regression) vs "infonce" (contrastive). MSE
# has a trivial minimum pred*=E[target|context] that collapses to the
# mean whenever targets look noise-like — observed as pred=Var(target).
# InfoNCE: positive = aligned target, negatives = all other targets in
# the batch. Predicting the mean gives chance-level contrastive loss,
# so the trivial minimum disappears.
PRED_LOSS_V5 = "infonce"
INFONCE_TEMP_V5 = 0.1

# I-JEPA block masking for v5.
# With T = W/8 frames (~100-200 for a typical line), 6 blocks of 12-30
# frames mask roughly 25-40% of the sequence — large enough that
# receptive-field overlap cannot leak target content into the context.
JEPA_NUM_TARGETS_V5 = 6
JEPA_MIN_SIZE_V5 = 12
JEPA_MAX_SIZE_V5 = 30

# Apply LayerNorm to predictor output AND stop-grad target before MSE.
# Disabled by default: empirically, combining target_norm with SIGReg's
# scale-invariant regularisation let the encoder emit tiny-magnitude
# embeddings (scale collapse) while the loss still looked healthy —
# pred ≈ Var(z), i.e. the predictor just returned the mean. Removing
# LayerNorm surfaces the collapse in the raw loss.
TARGET_NORM_V5 = False

# Weight on the JEPA prediction loss. Set to 0 to disable the world-model
# branch entirely — useful for the CTC-only baseline ablation.
LAMBDA_PRED_V5 = 1.0


# --- HWM-v6 ---
# Same architecture as v5 + a projection head on the JEPA branch
# (SimCLR/VICReg-style). The single v5 encoder has to serve two goals
# at once — CTC wants character-discriminative frames, JEPA wants
# frames that are mutually predictable. A small MLP on the JEPA side
# absorbs the "SSL-specific compromise" so raw z_seq stays focused on
# the CTC objective. The CTC path is unchanged at inference; the
# projection head is only used during training.

IMG_HEIGHT_V6 = IMG_HEIGHT_V5
EMBEDDING_DIM_V6 = EMBEDDING_DIM_V5
NUM_LAYERS_V6 = NUM_LAYERS_V5
NUM_HEADS_V6 = NUM_HEADS_V5
FF_DIM_V6 = FF_DIM_V5
LAMBDA_CTC_V6 = LAMBDA_CTC_V5
LAMBDA_PRED_V6 = LAMBDA_PRED_V5
CTC_HIDDEN_V6 = CTC_HIDDEN_V5
CTC_NUM_LSTM_V6 = CTC_NUM_LSTM_V5
PRED_LOSS_V6 = PRED_LOSS_V5
INFONCE_TEMP_V6 = INFONCE_TEMP_V5
JEPA_NUM_TARGETS_V6 = JEPA_NUM_TARGETS_V5
JEPA_MIN_SIZE_V6 = JEPA_MIN_SIZE_V5
JEPA_MAX_SIZE_V6 = JEPA_MAX_SIZE_V5
TARGET_NORM_V6 = TARGET_NORM_V5

# Projection head: 2-layer MLP (Linear -> GELU -> Linear).
# Output dim defaults to the encoder dim — we do not shrink the SSL
# representation, just route it through a learned transformation.
PROJ_DIM_V6 = EMBEDDING_DIM_V5
PROJ_HIDDEN_V6 = EMBEDDING_DIM_V5


# --- HWM-v7 ---
# v6 + true I-JEPA cross-attention predictor (queries = mask_token +
# pos_enc at every position, K/V = context encoder output; targets
# never see mask tokens). The existing transformer is reused as the
# context encoder via src_key_padding_mask hiding target + padding.

IMG_HEIGHT_V7 = IMG_HEIGHT_V5
EMBEDDING_DIM_V7 = EMBEDDING_DIM_V5
NUM_LAYERS_V7 = NUM_LAYERS_V5
NUM_HEADS_V7 = NUM_HEADS_V5
FF_DIM_V7 = FF_DIM_V5
LAMBDA_CTC_V7 = LAMBDA_CTC_V5
LAMBDA_PRED_V7 = LAMBDA_PRED_V5
CTC_HIDDEN_V7 = CTC_HIDDEN_V5
CTC_NUM_LSTM_V7 = CTC_NUM_LSTM_V5
PRED_LOSS_V7 = PRED_LOSS_V5
INFONCE_TEMP_V7 = INFONCE_TEMP_V5
JEPA_NUM_TARGETS_V7 = JEPA_NUM_TARGETS_V5
JEPA_MIN_SIZE_V7 = JEPA_MIN_SIZE_V5
JEPA_MAX_SIZE_V7 = JEPA_MAX_SIZE_V5
TARGET_NORM_V7 = TARGET_NORM_V5
PROJ_DIM_V7 = PROJ_DIM_V6
PROJ_HIDDEN_V7 = PROJ_HIDDEN_V6

# Cross-attention predictor depth. Kept smaller than the context
# encoder (NUM_LAYERS_V5 = 4): the bulk of representational work is
# in the context encoder; the cross-attn block is a lightweight read-
# out that maps "context + query position" -> "predicted embedding".
JEPA_PRED_LAYERS_V7 = 2


# --- HWM-v8 ---
# ViT encoder + MAE pretext. Structural change vs v5-v7: 2D patches
# over the full line image (instead of 1D vertical strips), and raw
# pixel reconstruction as the self-supervised target (instead of
# InfoNCE on stop-grad embeddings). Pixels are external targets, so
# there is no trivial-mean / scale-collapse failure mode.

IMG_HEIGHT_V8 = 120
# 15 divides 120 cleanly -> 8 vertical patches.
PATCH_H_V8 = 15
# 16 halves CTC horizontal resolution vs Kraken (W/16 vs W/8). Trade-off:
# smaller patches multiply token count quadratically in attention. 16
# keeps compute manageable for W up to ~2000.
PATCH_W_V8 = 16

EMBEDDING_DIM_V8 = 384
NUM_LAYERS_V8 = 4
NUM_HEADS_V8 = 8
FF_DIM_V8 = 1536  # 4x expansion (standard ViT)

# Decoder is smaller than the encoder (official MAE recipe: the decoder
# exists only during pretraining and can afford to be cheap).
DEC_DIM_V8 = 256
DEC_LAYERS_V8 = 2
DEC_HEADS_V8 = 8
DEC_FF_V8 = 1024

# 2D block masking. With N_v=8 and N_h typically ~50-100, 4 blocks of
# (2-4 rows) x (4-16 cols) mask roughly 30-50 % of the grid.
MASK_NUM_BLOCKS_V8 = 4
MASK_MIN_H_V8 = 2
MASK_MAX_H_V8 = 4
MASK_MIN_W_V8 = 4
MASK_MAX_W_V8 = 16

# Upper bound on N_h. For patch_w=16 and max line width 2000, N_h<=125.
# Positional embedding is sized up to MAX_N_H_V8 to allow larger lines.
MAX_N_H_V8 = 400

LAMBDA_MAE_V8 = 1.0
LAMBDA_CTC_V8 = 0.5

CTC_HIDDEN_V8 = 256
CTC_NUM_LSTM_V8 = 1


# --- HWM-v9 ---
# Return to the LeWorldModel recipe (SIGReg) with image-level masking
# (MSN / data2vec style) and a hybrid CNN+ViT encoder. The CNN stem
# halves the representation-learning load that a pure ViT would carry,
# which matters on small HTR datasets.

IMG_HEIGHT_V9 = 120
# CNN stem reduces 120 → 30 vertically and W → W/4 horizontally via
# 2 × MaxPool(2,2). Patch embed on the feature map uses a small
# non-overlapping patch. Net image→token stride: 12 rows, 16 cols.
STEM_CHANNELS_V9 = 64
PATCH_H_V9 = 3
PATCH_W_V9 = 4

EMBEDDING_DIM_V9 = 384
NUM_LAYERS_V9 = 4
NUM_HEADS_V9 = 8
FF_DIM_V9 = 1536

# Loss weights. lambda_msn is the SSL consistency term; lambda_sigreg
# is the anti-collapse regulariser (paper default style).
LAMBDA_MSN_V9 = 1.0
LAMBDA_SIGREG_V9 = 0.1
LAMBDA_CTC_V9 = 0.5

# 2D block mask. Grid is N_v=10 rows × (W/16) cols. Blocks of 2-4 rows
# by 4-16 cols mask ~30-50 % of the valid region.
MASK_NUM_BLOCKS_V9 = 4
MASK_MIN_H_V9 = 2
MASK_MAX_H_V9 = 4
MASK_MIN_W_V9 = 4
MASK_MAX_W_V9 = 16

# Upper bound on N_h for positional embedding. For patch stride 16 and
# max line width ~2000, N_h <= 125; 400 leaves plenty of slack.
MAX_N_H_V9 = 400


# --- HWM-v10 ---
# v7 JEPA architecture (cross-attention predictor + projection head) with
# the hybrid CNN+ViT encoder from v9.  SIGReg corrected (no normalisation,
# variance hinge on raw z) replaces VICReg.  MSE prediction loss (LeWM-
# style) replaces InfoNCE.  CTC with BiLSTM restored.

IMG_HEIGHT_V10 = 120
STEM_CHANNELS_V10 = 64
PATCH_H_V10 = 3
PATCH_W_V10 = 4

EMBEDDING_DIM_V10 = 384
NUM_LAYERS_V10 = 4
NUM_HEADS_V10 = 8
FF_DIM_V10 = 1536

PRED_NUM_LAYERS_V10 = 2
PRED_FF_DIM_V10 = 768

LAMBDA_PRED_V10 = 1.0
LAMBDA_SIGREG_V10 = 0.1
LAMBDA_CTC_V10 = 1.0

SIGREG_VAR_V10 = 25.0
SIGREG_COV_V10 = 1.0
SIGREG_GAMMA_V10 = 1.0

CTC_HIDDEN_V10 = 512
CTC_NUM_LSTM_V10 = 1

MASK_NUM_BLOCKS_V10 = 5
MASK_MIN_H_V10 = 2
MASK_MAX_H_V10 = 6
MASK_MIN_W_V10 = 4
MASK_MAX_W_V10 = 24

MAX_N_H_V10 = 400


# --- HWM-v11 ---
# Return to a pure 1D Kraken-style encoder, with a SimSiam-inspired
# consistency pretext: encode a clean view and a heavily perturbed view
# of the same line, pool both to a single line vector, and pull them
# together via cosine similarity. SIGRegV2 prevents collapse on the
# raw frame embeddings; CTC trains the recognition path.
#
# Why this design:
#   - 6 prior versions (v5-v10) hit the JEPA "moving target" wall.
#     Symmetric encoder with stop-grad target collapses without EMA.
#   - SimSiam-style asymmetry (predictor MLP on one branch only) breaks
#     the trivial-identity solution without needing an EMA teacher.
#   - Consistency at the pooled (line-level) granularity targets style
#     INVARIANCE, which matches the project goal of transfer to new
#     scribes — not predictive reconstruction at the frame level.

IMG_HEIGHT_V11 = 120
EMBEDDING_DIM_V11 = 384

# Predictor MLP: 2 layers, hidden = embed_dim. Standard SimSiam scale.
PRED_HIDDEN_V11 = 384

# CTC head — BiLSTM provides temporal context for character recognition.
CTC_HIDDEN_V11 = 256
CTC_NUM_LSTM_V11 = 1

# Loss weights.
LAMBDA_CONS_V11 = 1.0      # consistency (cosine, perturbed -> clean)
LAMBDA_SIGREG_V11 = 0.1    # anti-collapse on raw z
LAMBDA_CTC_V11 = 1.0       # supervised recognition

# SIGRegV2 (variance hinge on raw std + cov decorrelation).
SIGREG_VAR_V11 = 25.0
SIGREG_COV_V11 = 1.0
SIGREG_GAMMA_V11 = 1.0

# Perturbations applied to view 2 (the "hard" view).
#
# Original v11 release used softer values (shift=4, shear=5, mask_blocks=4,
# mask_w=[16,32], noise=0.03). Observed result: ``cons`` saturated at -0.92
# from epoch 1 of the adapt run — the encoder was trivially invariant to
# those weak perturbations and the pretext stopped producing gradient
# signal. Bumped to harder defaults so the encoder must actually learn an
# invariant representation instead of getting it for free.
PERT_V11_SHIFT_X = 8               # was 4 — ±8 px horizontal shift
PERT_V11_SHEAR_DEG = 10.0          # was 5°  — ±10° shear (more style variation)
PERT_V11_MASK_BLOCKS = 6           # was 4  — more occlusion blocks
PERT_V11_MASK_W_MIN = 16           # unchanged
PERT_V11_MASK_W_MAX = 40           # was 32 — wider blocks possible
PERT_V11_CONTRAST_MIN = 0.65       # was 0.7
PERT_V11_CONTRAST_MAX = 1.35       # was 1.3
PERT_V11_BRIGHTNESS = 0.15         # was 0.1
PERT_V11_NOISE_STD = 0.05          # was 0.03


# --- HWM-v12 ---
# Kraken 1D conv stem + Transformer encoder (the Kraken BiLSTM "moved
# into the encoder and replaced by attention"), trained with three
# objectives:
#   - CTC                      : supervised recognition
#   - InfoNCE masked-segment   : MSN/data2vec pretext — predict the
#       embedding of pixel-masked frame spans from context. The
#       transformer IS the predictor (no separate module).
#   - SIGReg (Epps-Pulley)     : the paper's real anti-collapse term
#       (random projections + normality test), NOT the VICReg-style
#       SIGRegV2 of v10/v11.
# Optional 4th term: SupCon over a writer/page id (dormant until the
# collate provides writer_id; see use_writer_contrastive).
#
# Design decisions (see design discussion):
#   - Option B: NO final LayerNorm on the encoder output. A per-sample
#     LayerNorm pins frames to a sphere, which the Gaussian SIGReg
#     target cannot match. LN is applied only inside the CTC head.
#   - Masking is in PIXEL space, before the conv stem: the wide Kraken
#     kernels would otherwise leak masked content into neighbour tokens.
# Dimensions kept deliberately light — scale up if it trains well.

IMG_HEIGHT_V12 = 120
EMBEDDING_DIM_V12 = 192
NUM_LAYERS_V12 = 3
NUM_HEADS_V12 = 3
FF_DIM_V12 = 384

# CTC head — BiLSTM, fed by the only LayerNorm in the v12 path.
CTC_HIDDEN_V12 = 192
CTC_NUM_LSTM_V12 = 1

# SSL projection heads (InfoNCE + SupCon). Discarded at inference.
PROJ_DIM_V12 = 128
PROJ_HIDDEN_V12 = 192

# Loss weights.
LAMBDA_CTC_V12 = 1.0
LAMBDA_JEPA_V12 = 0.5      # InfoNCE masked-segment prediction
LAMBDA_SIGREG_V12 = 0.1    # Epps-Pulley SIGReg (paper default lambda)
LAMBDA_WC_V12 = 0.2        # SupCon writer/page contrastive (if enabled)

INFONCE_TEMP_V12 = 0.1
SUPCON_TEMP_V12 = 0.1

# Masked-segment pretext: T = W/8 frames (~100-250 for a typical line).
# 4 blocks of 8-20 frames mask roughly 10-30 % of the sequence.
JEPA_NUM_TARGETS_V12 = 4
JEPA_MIN_SIZE_V12 = 8
JEPA_MAX_SIZE_V12 = 20

# Epps-Pulley SIGReg. The paper shows performance is insensitive to
# both quantities; 256 projections keeps the (N x M x K) tensor small.
SIGREG_PROJECTIONS_V12 = 256
SIGREG_KNOTS_V12 = 17

# Writer/page contrastive branch. Off by default: the collate does not
# yet emit a writer_id. To enable later, plumb a per-line page id
# through the collate and set this True.
USE_WRITER_CONTRASTIVE_V12 = False

# --- HWM-v13 ---
# Capacity bump: same v12 architecture (KrakenEncoderV12 + Transformer + BiLSTM CTC)
# but wider embedding (384 vs 192) and 2 BiLSTM layers in the CTC head (vs 1).
# Target: match lectaurep_base (4.0M, 3×BiLSTM200, 960-dim input) capacity.
#
# Dimension comparison with lectaurep_base:
#   lectaurep_base: conv→960 features → 3×BiLSTM(200) → 400 bidir
#   v13:           conv→960 features → proj→384 → Transformer(384) → 2×BiLSTM(384)
#
# The conv stem is identical (4 conv layers, same kernels, 64 filters out).
# KrakenEncoderV12.feature_dim = 64 * (120/8) = 960, then projected to 384.

EMBEDDING_DIM_V13 = 384
NUM_LAYERS_V13 = 3          # same transformer depth as v12
NUM_HEADS_V13 = 6           # 384/6 = 64 per head (was 192/3=64, same ratio)
FF_DIM_V13 = 768            # 2× embedding_dim (same ratio as v12)

# CTC head — 2 BiLSTM layers (vs 1 in v12), hidden stays at 192.
CTC_HIDDEN_V13 = 192
CTC_NUM_LSTM_V13 = 2

# SSL projection heads — scaled up with embedding_dim.
PROJ_DIM_V13 = 128
PROJ_HIDDEN_V13 = 384

# Loss weights — keep v12 defaults, they worked.
LAMBDA_CTC_V13 = 1.0
LAMBDA_JEPA_V13 = 0.5
LAMBDA_SIGREG_V13 = 0.1
LAMBDA_WC_V13 = 0.2

INFONCE_TEMP_V13 = 0.1
SUPCON_TEMP_V13 = 0.1

# Masked-segment pretext — same as v12.
JEPA_NUM_TARGETS_V13 = 4
JEPA_MIN_SIZE_V13 = 8
JEPA_MAX_SIZE_V13 = 20

# SIGReg — same as v12.
SIGREG_PROJECTIONS_V13 = 256
SIGREG_KNOTS_V13 = 17

USE_WRITER_CONTRASTIVE_V13 = False


# --- HWM-v14 ---
# Compromise between v12 (192-dim, 2M params) and v13 (384-dim, 6.3M params).
# embed_dim=256 keeps the model lighter while 3 BiLSTM layers in the CTC head
# (vs 2 in v13, 1 in v12) compensate with more temporal modeling capacity.
# SIGReg is reverted to the unified form (no shape/scale split) — the split
# was introduced for an adapt-training experiment that is now abandoned.
# Full training only, letting CTC guide the encoder from epoch 1.

EMBEDDING_DIM_V14 = 256
NUM_LAYERS_V14 = 3          # same transformer depth as v12/v13
NUM_HEADS_V14 = 4           # 256/4 = 64 per head (same ratio)
FF_DIM_V14 = 512            # 2× embedding_dim (same ratio)

# CTC head — 3 BiLSTM layers (vs 2 in v13, 1 in v12).
CTC_HIDDEN_V14 = 192
CTC_NUM_LSTM_V14 = 3

# SSL projection heads.
PROJ_DIM_V14 = 128
PROJ_HIDDEN_V14 = 256

# Loss weights — same as v12/v13.
LAMBDA_CTC_V14 = 1.0
LAMBDA_JEPA_V14 = 0.5
LAMBDA_SIGREG_V14 = 0.1
LAMBDA_WC_V14 = 0.2

INFONCE_TEMP_V14 = 0.1
SUPCON_TEMP_V14 = 0.1

# Masked-segment pretext — same as v12/v13.
JEPA_NUM_TARGETS_V14 = 4
JEPA_MIN_SIZE_V14 = 8
JEPA_MAX_SIZE_V14 = 20

# SIGReg — same as v12/v13 (unified, no shape/scale split).
SIGREG_PROJECTIONS_V14 = 256
SIGREG_KNOTS_V14 = 17

USE_WRITER_CONTRASTIVE_V14 = False


# --- Lectaurep Clone (v15) ---
# Faithful reproduction of lectaurep_base:
#   CNN (4 conv, Kraken stem) → 960-dim → 3×BiLSTM(200) → Linear → CTC
# No Transformer, no JEPA, no SIGReg.  Pure CTC baseline to see if we
# reproduce Lectaurep's 9.8% CER on their data.
# Total params: ~4.0M (same as the official model).
#
# Training variations planned:
#   - with / without elastic deformations (augmentation)
#   - with / without width bucketing (grouping lines by size)
LECTAUREP_IMG_HEIGHT = 120
LECTAUREP_HIDDEN = 200
LECTAUREP_NUM_LSTM = 3
LECTAUREP_DROPOUT = 0.1
LECTAUREP_LR = 1e-4        # from the official ketos command: -r 0.0001


# --- v16: v15 training recipe + v14 encoder (Transformer + JEPA + SIGReg) ---
# CNN v15 → Projection(960→256) + LayerNorm → Transformer 3L pre-LN 256-dim 4 heads
# → 2× BiLSTM(128) → CTC Head
# ~3.6M params total. Adam, cosine+warmup, no AMP, single param group.
EMBEDDING_DIM_V16 = 256
NUM_LAYERS_V16 = 3              # Transformer depth
NUM_HEADS_V16 = 4               # 256/4 = 64 per head
FF_DIM_V16 = 1024               # 4× embedding_dim
DROPOUT_V16 = 0.1               # Transformer / conv stem dropout

LSTM_HIDDEN_V16 = 128           # BiLSTM hidden (output=256 bidir)
NUM_LSTM_V16 = 2                # 2 layers (Transformer already does sequential)
LSTM_DROPOUT_MID_V16 = 0.1      # Dropout between LSTM layers
LSTM_DROPOUT_LAST_V16 = 0.3     # Dropout on last LSTM (less aggressive than v15's 0.5)

LAMBDA_CTC_V16 = 1.0
LAMBDA_JEPA_V16 = 0.5
LAMBDA_SIGREG_V16 = 0.1
LAMBDA_WC_V16 = 0.2

INFONCE_TEMP_V16 = 0.1
SUPCON_TEMP_V16 = 0.1

PROJ_DIM_V16 = 128
PROJ_HIDDEN_V16 = 256

JEPA_NUM_TARGETS_V16 = 4
JEPA_MIN_SIZE_V16 = 8
JEPA_MAX_SIZE_V16 = 20

SIGREG_PROJECTIONS_V16 = 256
SIGREG_KNOTS_V16 = 17

USE_WRITER_CONTRASTIVE_V16 = False


# --- v17: retour BiLSTM (v15 recipe) + JEPA + SIGReg, sans Transformer ---
# CNN → 3×BiLSTM(128) → CTC. Premier BiLSTM mange 960 directement.
# Pas de Linear(960→256), pas de Transformer.
# SIGReg debrayable (--lambda-sigreg 0). JEPA debrayable (--no-jepa).
# Pas de LayerNorm avant SIGReg (LeWorldModel paper: la normalisation
# avant SIGReg empeche la cible gaussienne de matcher la distribution).
# ~3.7M params. Adam, cosine, no AMP, single param group.

# LSTM
LSTM_HIDDEN_V17 = 128             # BiLSTM hidden (output=256 bidir)
NUM_LSTM_V17 = 3                  # 3 couches comme v15
LSTM_DROPOUT_MID_V17 = 0.1        # Dropout entre couches LSTM
LSTM_DROPOUT_LAST_V17 = 0.3       # Dropout derniere couche (modere vs v15's 0.5)

# Loss weights
LAMBDA_CTC_V17 = 1.0
LAMBDA_JEPA_V17 = 0.2            # 0.5 donnait 26% du gradient a JEPA, trop. 0.2 → ~7%.
LAMBDA_SIGREG_V17 = 0.1           # Debrayable: --lambda-sigreg 0
LAMBDA_WC_V17 = 0.2

# JEPA
JEPA_NUM_TARGETS_V17 = 4
JEPA_MIN_SIZE_V17 = 8
JEPA_MAX_SIZE_V17 = 20

# Projection heads
PROJ_DIM_V17 = 128
PROJ_HIDDEN_V17 = 256

# SIGReg
SIGREG_PROJECTIONS_V17 = 256
SIGREG_KNOTS_V17 = 17

# InfoNCE / SupCon temperatures
INFONCE_TEMP_V17 = 0.1
SUPCON_TEMP_V17 = 0.1

# Writer contrastive (off by default)
USE_WRITER_CONTRASTIVE_V17 = False


# --- HWM-v18 ---
# Decoupled JEPA / CTC branches: CNN partagee, JEPA via Linear(960->384)
# sans LayerNorm, BiLSTM uniquement sur le chemin CTC. SIGReg sur la
# sortie JEPA (z_jepa_clean), pas sur z_seq (BiLSTM).
#
# Motivation: v17 voyait JEPA polluer le gradient des BiLSTM et etouffer
# CTC (CTC bloque a ~3.5 nats/char = quasi random). v18 isole les deux
# branches au-dela du CNN partage.

JEPA_DIM_V18 = 384

# LSTM (identique a v17)
LSTM_HIDDEN_V18 = 320
NUM_LSTM_V18 = 3
LSTM_DROPOUT_MID_V18 = 0.2
LSTM_DROPOUT_LAST_V18 = 0.5

# Loss weights — lambda_jepa garde 0.2 (le decouplage devrait suffire).
LAMBDA_CTC_V18 = 1.0
LAMBDA_JEPA_V18 = 0.2
LAMBDA_SIGREG_V18 = 0.1
LAMBDA_WC_V18 = 0.2

# JEPA masking (identique a v17)
JEPA_NUM_TARGETS_V18 = 4
JEPA_MIN_SIZE_V18 = 8
JEPA_MAX_SIZE_V18 = 20

# Projection heads (input dim = JEPA_DIM_V18, sortie 128)
PROJ_DIM_V18 = 128
PROJ_HIDDEN_V18 = 256

# Transformer predictor asymetrique (remplace le MLP symetrique jepa_proj).
# Le predicteur recoit la sequence complete de features masquees, remplace
# les positions cibles par un mask token, et attend le contexte temporel
# pour predire. La cible utilise target_proj (Linear separe).
JEPA_PRED_NUM_LAYERS_V18 = 2
JEPA_PRED_NUM_HEADS_V18 = 4
JEPA_PRED_DIM_FF_V18 = 1536
JEPA_PRED_DROPOUT_V18 = 0.1

# SIGReg Epps-Pulley
SIGREG_PROJECTIONS_V18 = 256
SIGREG_KNOTS_V18 = 17

# InfoNCE / SupCon temperatures
INFONCE_TEMP_V18 = 0.1
SUPCON_TEMP_V18 = 0.1

USE_WRITER_CONTRASTIVE_V18 = False


# --- HWM-v19 : LeVJEPA (arXiv:2608.27395) transpose aux lignes manuscrites ---
# Re-test de la regularisation JEPA+SIGReg abandonnee en v10-v18, avec la
# recette LeVJEPA : 1 vue globale + V vues locales, invariance MSE symetrique
# (pas de stop-gradient, pas de predicteur, pas d'EMA) + SIGReg Epps-Pulley
# sur les embeddings [cls] du batch. lambda_sigreg = seul hyperparametre SSL.
#
# Chaine :
#   image ligne (h=64, W variable)
#     -> stem CNN Kraken 2 blocs (stride 4) -> carte (64, 16, W/4)
#     -> patches CARRES p=16 -> T = W/64 tokens le long de x
#     -> fenetres coulissantes de K=8 patches, stride 4 (chevauchement 50%)
#     -> N=8 blocs transformer BLOCK-CAUSAUX (intra-fenetre bidirectionnel,
#        inter-fenetres causal gauche->droite), RoPE le long de x
#     -> [cls] apprenable (readout global) + LayerNorm sur les tokens
#     -> BiLSTM(320)x2 -> CTC
#
# NB resolution : 1 token = 64 px (stem /4 x patch 16). La collate v5 filtre
# les lignes non-alignables avec T = W//8 ; a cette resolution CTC voit
# ~2xW/64 tokens valides par ligne (fenetres chevauchees) — utiliser
# --min-frames-per-char avec le stride 64 du registry pour ecarter les
# lignes trop courtes si besoin.

IMG_HEIGHT_V19 = 64

# Stem CNN : 2 blocs conv Kraken eprouves (v15-v18) -> stride 4, sortie 16
# rangees de hauteur = PATCH_V19, d'ou des patches 16x16 carres.
STEM_CHANNELS_V19 = 64
PATCH_V19 = 16

# Fenetres coulissantes (« frames ») : K patches consecutifs le long de x,
# stride K/2 -> chevauchement 50% entre fenetres consecutives.
WINDOW_PATCHES_V19 = 8      # K
WINDOW_STRIDE_V19 = 4       # s = K/2

# Encodeur transformer block-causal.
EMBEDDING_DIM_V19 = 256     # d (192/256 selon le plan ; 256 par defaut)
NUM_LAYERS_V19 = 8          # N (6-8 ; 8 par défaut)
NUM_HEADS_V19 = 8           # 256/8 = 32 par tete
FF_DIM_V19 = 1024           # 4x embedding_dim
DROPOUT_V19 = 0.1
MAX_PATCHES_V19 = 512       # borne RoPE (512 tokens = ~32768 px de ligne)

# Projecteur h_phi (SSL uniquement, jete apres pre-entraînement) :
# Linear(d->2048) -> BatchNorm -> GELU -> Linear(2048->K), K ~ 96-128.
PROJ_HIDDEN_V19 = 2048
PROJ_DIM_V19 = 128

# Tete de decodage CTC : features par-token (sortie LayerNorm, PAS le [cls])
# -> BiLSTM(320) x 2-3 -> CTC.
LSTM_HIDDEN_V19 = 320
NUM_LSTM_V19 = 2
LSTM_DROPOUT_MID_V19 = 0.1
LSTM_DROPOUT_LAST_V19 = 0.3

# Poids de perte.
LAMBDA_CTC_V19 = 1.0
LAMBDA_INV_V19 = 1.0        # invariance MSE vue globale <- vues locales
LAMBDA_SIGREG_V19 = 0.1     # SIGReg Epps-Pulley sur les [cls] (seul hp SSL)

# Objectif SSL (phase adapt) : V vues locales = recadrage horizontal
# + augmentation photometrique ; drop uniforme de 50% des tokens
# (jamais le [cls], jamais le patch 0 pour garder une cle d'ancrage).
NUM_LOCAL_VIEWS_V19 = 4
TOKEN_DROP_V19 = 0.5
LOCAL_CROP_MIN_V19 = 0.3    # fraction min de la largeur valide
LOCAL_CROP_MAX_V19 = 0.6    # fraction max
PHOTO_CONTRAST_V19 = 0.3    # +/- 30%
PHOTO_BRIGHTNESS_V19 = 0.15 # +/- 0.15
PHOTO_NOISE_STD_V19 = 0.05

# SIGReg Epps-Pulley (identique a v12-v18).
SIGREG_PROJECTIONS_V19 = 256
SIGREG_KNOTS_V19 = 17

# =============================================================================
# v20 : la construction v19 (stem CNN + [cls] + SIGReg LeVJEPA) portee en
# production — CNNs renforces, comparateur CLS-vs-all, LSTM leger en tete CTC.
# =============================================================================
IMG_HEIGHT_V20 = 64

# Stem CNN RENFORCE : meme recette Kraken (kernels larges captant la structure
# horizontale des traits), mais le bloc de sortie passe de 64 a 128 canaux —
# c'est le « on reprend les cnn » : plus de capacite d'extraction locale, sans
# casser la tokenisation (img_height//4 == patch).
STEM_CHANNELS_V20 = 128
PATCH_V20 = 16

# Fenetres coulissantes (meme fenetration que v19) : K patches, stride K/2.
WINDOW_PATCHES_V20 = 8
WINDOW_STRIDE_V20 = 4

# Encodeur transformer block-causal (construction v19).
EMBEDDING_DIM_V20 = 256
NUM_LAYERS_V20 = 8
NUM_HEADS_V20 = 8
FF_DIM_V20 = 1024
DROPOUT_V20 = 0.1
MAX_PATCHES_V20 = 512

# Projecteur h_phi (SSL uniquement, jete apres pre-entrainement) : meme
# forme que v19 (LayerNorm de sortie pour garder l'Epps-Pulley actif).
PROJ_HIDDEN_V20 = 2048
PROJ_DIM_V20 = 128

# REPRESENTATION COMPAREE en SSL (le test central de v20) :
#   'cls' : on compare les embeddings [cls] (comportement v19) ;
#   'all' : on compare les features TOUT-TOKENS moyennees (sortie LayerNorm,
#           la representation que la CTC consomme ensuite). SIGReg reste pose
#           sur le batch des [cls] dans les deux cas (fidelite paper).
COMPARE_MODE_V20 = "cls"

# Tete de decodage CTC : LSTM LEGER (espoir : le transformer fait le boulot,
# donc 1 couche suffit ; basculer a 2 si le CER le justifie).
LSTM_HIDDEN_V20 = 320
NUM_LSTM_V20 = 1
LSTM_DROPOUT_MID_V20 = 0.1
LSTM_DROPOUT_LAST_V20 = 0.3

# Poids de perte.
LAMBDA_CTC_V20 = 1.0
LAMBDA_INV_V20 = 1.0        # invariance MSE vue globale <- vues locales
LAMBDA_SIGREG_V20 = 0.1     # SIGReg Epps-Pulley sur les [cls] (seul hp SSL)

# Objectif SSL (phase adapt) : V vues locales = recadrage horizontal +
# augmentation photometrique ; drop uniforme des tokens, INDEPENDANT par vue
# (chaque vue tire ses propres tokens a jeter — jamais le [cls], jamais le
# patch 0, cle d'ancrage). Le taux (drop = difficulte de l'invariance) se
# balaie via --token-drop (defaut 0.5 ; 95% = regime video, probablement trop
# pour des lignes HTR, a tester).
NUM_LOCAL_VIEWS_V20 = 4
TOKEN_DROP_V20 = 0.5
LOCAL_CROP_MIN_V20 = 0.3    # fraction min de la largeur valide
LOCAL_CROP_MAX_V20 = 0.6    # fraction max
PHOTO_CONTRAST_V20 = 0.3    # +/- 30%
PHOTO_BRIGHTNESS_V20 = 0.15 # +/- 0.15
PHOTO_NOISE_STD_V20 = 0.05

# SIGReg Epps-Pulley (identique a v12-v19).
SIGREG_PROJECTIONS_V20 = 256
SIGREG_KNOTS_V20 = 17


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
