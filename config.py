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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
