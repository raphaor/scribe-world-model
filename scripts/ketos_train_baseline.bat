@echo off
REM ============================================================================
REM ketos_train_baseline.bat
REM ============================================================================
REM
REM Windows batch script that trains a lectaurep_base kraken model using the
REM EXACT same architecture, hyperparameters, and data as the original
REM Lectaurep training run.  Uses ketos (kraken 7.x) to:
REM
REM   1. Compile ALTO XML data into a .arrow binary dataset
REM   2. Train the model (VGSL spec matching lectaurep_base)
REM   3. Test the model and report CER
REM   4. Dump one complete training batch + alphabet/codec to disk for
REM      comparison with the scribe-world-model LectaurepClone
REM
REM PREREQUISITES
REM -------------
REM   - Python 3.10+ with pip
REM   - CUDA toolkit (optional but recommended)
REM   - kraken 7.0.2:  pip install kraken==7.0.2
REM   - The ALTO XML directories exist and contain paired .xml + .jpg files
REM
REM USAGE
REM -----
REM   Edit the configuration block below (ALTO_DIRS, OUTPUT_DIR, etc.), then:
REM
REM     ketos_train_baseline.bat
REM
REM   Or override from the command line:
REM
REM     set ALTO_DIRS=/media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie\Alto\lectaurep_foo /media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie\Alto\lectaurep_bar
REM     set OUTPUT_DIR=D:\ketos_baseline
REM     ketos_train_baseline.bat
REM
REM ============================================================================

REM --- Configuration (edit these or set them as env vars before running) -------

if not defined ALTO_DIRS (
    set ALTO_DIRS=/media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie\Alto\lectaurep_bronod_notaire_paris_18e /media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie\Alto\lectaurep_mariages_divorces_paris_19e /media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie\Alto\lectaurep_repertoires_notaires_paris_1830-1939 /media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie\Alto\timeuscorpus_prudhommes_paris_1858-1878
)

if not defined OUTPUT_DIR (
    set OUTPUT_DIR=D:\ketos_baseline_output
)

REM Number of epochs (-1 = early stopping, N = fixed epochs).
REM The original Lectaurep run used early stopping with --lag 10.
if not defined NUM_EPOCHS (
    set NUM_EPOCHS=50
)

REM Batch size. The original Lectaurep command used -B 1 (gradient accumulated
REM by Lightning's automatic tuning).  We keep 1 here for faithfulness.
if not defined BATCH_SIZE (
    set BATCH_SIZE=1
)

REM Learning rate. The original was -r 0.0001.
if not defined LEARNING_RATE (
    set LEARNING_RATE=0.0001
)

REM Train/val partition ratio (0.9 = 90%% train, 10%% val).
if not defined PARTITION (
    set PARTITION=0.9
)

REM Number of worker processes for data loading.
if not defined NUM_WORKERS (
    set NUM_WORKERS=4
)

REM ----------------------------------------------------------------------------

echo ========================================================================
echo  Ketos Lectaurep Baseline Training
echo ========================================================================
echo  ALTO dirs   : %ALTO_DIRS%
echo  Output dir  : %OUTPUT_DIR%
echo  Epochs      : %NUM_EPOCHS%
echo  Batch size  : %BATCH_SIZE%
echo  Learning rate: %LEARNING_RATE%
echo  Partition   : %PARTITION%
echo ========================================================================

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM --- Locate Python ----------------------------------------------------------

REM Prefer the venv if it exists (same dir as this script's parent)
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..

if exist "%PROJECT_DIR%\venv\Scripts\python.exe" (
    set PYTHON=%PROJECT_DIR%\venv\Scripts\python.exe
    echo Using venv Python: %PYTHON%
) else (
    set PYTHON=python
    echo Using system Python: %PYTHON%
)

REM Check that kraken is installed
%PYTHON% -c "import kraken; print(f'kraken {kraken.__version__}')" 2>nul
if errorlevel 1 (
    echo ERROR: kraken is not installed. Run:  pip install kraken==7.0.2
    exit /b 1
)

REM ============================================================================
REM  STEP 1: Compile ALTO data into .arrow
REM ============================================================================

set ARROW_FILE=%OUTPUT_DIR%\lectaurep_dataset.arrow

if exist "%ARROW_FILE%" (
    echo.
    echo [STEP 1] Arrow dataset already exists at %ARROW_FILE%
    echo          Skipping compilation. Delete it to recompile.
    echo.
) else (
    echo.
    echo [STEP 1] Compiling ALTO data into %ARROW_FILE% ...
    echo.

    REM Build the argument list of ALTO XML files (find all .xml in each dir)
    REM We pass the directories directly to ketos compile with -f alto.
    REM ketos compile accepts glob patterns or individual files.

    %PYTHON% %SCRIPT_DIR%ketos_train_baseline.py ^
        --step compile ^
        --alto-dirs %ALTO_DIRS% ^
        --output-dir "%OUTPUT_DIR%" ^
        --num-workers %NUM_WORKERS%

    if errorlevel 1 (
        echo ERROR: Compilation failed.
        exit /b 1
    )
    echo.
    echo [STEP 1] Done. Arrow dataset: %ARROW_FILE%
)

REM ============================================================================
REM  STEP 2: Train the model
REM ============================================================================

echo.
echo [STEP 2] Training lectaurep_base model ...
echo.

REM The VGSL spec for lectaurep_base (from the official training command):
REM   [1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2
REM    Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2
REM    S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]
REM
REM Training hyperparameters (from the official command):
REM   - Optimizer: AdamW
REM   - LR: 0.0001
REM   - Schedule: constant (no decay)
REM   - Batch size: 1
REM   - Quit: early stopping, lag 10
REM   - Partition: 0.9
REM   - Normalization: NFD
REM   - No augmentation
REM
REM We use the Python companion script for full control over the pipeline.

%PYTHON% %SCRIPT_DIR%ketos_train_baseline.py ^
    --step train ^
    --arrow-file "%ARROW_FILE%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --batch-size %BATCH_SIZE% ^
    --lr %LEARNING_RATE% ^
    --epochs %NUM_EPOCHS% ^
    --partition %PARTITION% ^
    --num-workers %NUM_WORKERS%

if errorlevel 1 (
    echo ERROR: Training failed.
    exit /b 1
)

REM ============================================================================
REM  STEP 3: Test the model and report CER
REM ============================================================================

echo.
echo [STEP 3] Testing model and computing CER ...
echo.

%PYTHON% %SCRIPT_DIR%ketos_train_baseline.py ^
    --step test ^
    --arrow-file "%ARROW_FILE%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --batch-size %BATCH_SIZE%

if errorlevel 1 (
    echo ERROR: Testing failed.
    exit /b 1
)

REM ============================================================================
REM  STEP 4: Dump one batch + alphabet for comparison with scribe
REM ============================================================================

echo.
echo [STEP 4] Dumping one training batch + alphabet to disk ...
echo.

%PYTHON% %SCRIPT_DIR%ketos_train_baseline.py ^
    --step dump ^
    --arrow-file "%ARROW_FILE%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --batch-size %BATCH_SIZE%

if errorlevel 1 (
    echo ERROR: Batch dump failed.
    exit /b 1
)

echo.
echo ========================================================================
echo  All steps complete.
echo  Output directory: %OUTPUT_DIR%
echo.
echo  Files produced:
echo    lectaurep_dataset.arrow  - Compiled dataset
echo    model\                    - Trained model checkpoints
echo    test_results.txt         - CER report
echo    batch_dump\               - One batch for comparison
echo      images.pt              - (B, H, W) float tensor
echo      targets.pt             - Flattened target indices
echo      input_lengths.pt       - (B,) sequence lengths
echo      target_lengths.pt      - (B,) target lengths
echo      raw_texts.json         - List of ground-truth strings
echo    alphabet.json             - char_to_idx + idx_to_char mapping
echo ========================================================================

pause
