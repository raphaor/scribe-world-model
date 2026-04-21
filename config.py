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
    "D:/OCR_genealogie/Alto/bars_dordogne_alto",
    "D:/OCR_genealogie/Alto/saint_chamassy_dordogne_alto_set_1",
    ## "D:/OCR_genealogie/Alto/saint_chamassy_dordogne_alto_set_train",
    "D:/OCR_genealogie/Alto/lectaurep_bronod_notaire_paris_18e",
    "D:/OCR_genealogie/Alto/lectaurep_mariages_divorces_paris_19e",
    "D:/OCR_genealogie/Alto/lectaurep_repertoires_notaires_paris_1830-1939",
    "D:/OCR_genealogie/Alto/timeuscorpus_prudhommes_paris_1858-1878",
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
