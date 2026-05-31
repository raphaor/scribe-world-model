#!/usr/bin/env python3
"""
ketos_train_baseline.py — Python companion to ketos_train_baseline.bat

Uses kraken 7.x / ketos Python APIs to:
  1. compile   — Build a .arrow dataset from ALTO XML files
  2. train     — Train a lectaurep_base model (exact VGSL spec + hyperparams)
  3. test      — Evaluate on the validation split, report CER
  4. dump      — Dump one complete batch + alphabet/codec to disk for
                 comparison with scribe-world-model's LectaurepClone

USAGE
-----
  This script is designed to be called from ketos_train_baseline.bat but can
  also be run standalone:

    python ketos_train_baseline.py --step compile  --alto-dirs <...> --output-dir <...>
    python ketos_train_baseline.py --step train    --arrow-file <...> --output-dir <...>
    python ketos_train_baseline.py --step test     --arrow-file <...> --output-dir <...>
    python ketos_train_baseline.py --step dump     --arrow-file <...> --output-dir <...>

PREREQUISITES
-------------
  pip install kraken==7.0.2

NOTES
-----
  - The VGSL spec and hyperparameters below exactly match the official
    Lectaurep training command (lectaurep_base):
        ketos train -s '[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2
          Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3
          Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]'
        -r 0.0001 -B 1 --schedule constant --optimizer AdamW
        -p 0.9 -q early --lag 10 -u NFD

  - The dump step extracts one batch in the SAME format as scribe-world-model's
    collate_alto_v5_fn: (images, targets, input_lengths, target_lengths, raw_texts)
    plus the full codec/alphabet mapping. This enables a direct numerical
    comparison between the kraken pipeline and the scribe LectaurepClone.
"""

import argparse
import json
import os
import sys
import glob
import logging
import unicodedata

# ---------------------------------------------------------------------------
# Fix up path so this script can be run from anywhere
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The official lectaurep_base VGSL spec
# ---------------------------------------------------------------------------
LECTAUREP_SPEC = (
    "[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 "
    "Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 "
    "S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]"
)

# Training hyperparameters (matching the official ketos command)
LECTAUREP_LR = 1e-4          # -r 0.0001
LECTAUREP_OPTIMIZER = "AdamW"
LECTAUREP_SCHEDULE = "constant"
LECTAUREP_PARTITION = 0.9    # -p 0.9
LECTAUREP_NORMALIZATION = "NFD"
LECTAUREP_QUIT = "early"
LECTAUREP_LAG = 10           # --lag 10
LECTAUREP_BATCH_SIZE = 1     # -B 1


# ============================================================================
#  STEP 1: Compile ALTO XML → .arrow
# ============================================================================

def step_compile(alto_dirs, output_dir, num_workers=4):
    """
    Parse ALTO XML files and build a .arrow dataset using kraken's
    build_binary_dataset API.
    """
    from kraken.lib.arrow_dataset import build_binary_dataset

    arrow_path = os.path.join(output_dir, "lectaurep_dataset.arrow")
    os.makedirs(output_dir, exist_ok=True)

    # Collect all XML files from the given directories
    xml_files = []
    for d in alto_dirs:
        found = sorted(glob.glob(os.path.join(d, "*.xml")))
        # Filter out METS.xml (not line-level data)
        found = [f for f in found if os.path.basename(f).upper() != "METS.XML"]
        xml_files.extend(found)
        logger.info(f"  {d}: {len(found)} XML files")

    if not xml_files:
        raise FileNotFoundError(
            f"No XML files found in {alto_dirs}. "
            f"Check that the directories exist and contain .xml files."
        )

    logger.info(f"Compiling {len(xml_files)} XML files into {arrow_path} ...")

    def _progress(chunk, total):
        if total > 0:
            pct = chunk / total * 100
            sys.stdout.write(f"\r  Compiling: {pct:.1f}% ({chunk}/{total} lines flushed)")
            sys.stdout.flush()

    build_binary_dataset(
        files=xml_files,
        output_file=arrow_path,
        format_type="alto",
        num_workers=num_workers,
        skip_empty_lines=True,
        callback=_progress,
    )
    sys.stdout.write("\n")
    logger.info(f"Dataset compiled: {arrow_path}")
    return arrow_path


# ============================================================================
#  STEP 2: Train
# ============================================================================

def step_train(arrow_file, output_dir, batch_size, lr, epochs, partition,
               num_workers=4):
    """
    Train a lectaurep_base model using ketos train.

    We invoke the ketos CLI via subprocess because the internal Python API
    (Lightning-based) is complex to drive programmatically and the CLI is the
    recommended interface. The CLI arguments exactly mirror the original
    Lectaurep training command.
    """
    import subprocess

    model_dir = os.path.join(output_dir, "model")

    cmd = [
        sys.executable, "-m", "kraken.ketos", "train",
        "-f", "binary",
        "-s", LECTAUREP_SPEC,
        "-o", model_dir,
        "-B", str(batch_size),
        "-r", str(lr),
        "--optimizer", LECTAUREP_OPTIMIZER,
        "--schedule", LECTAUREP_SCHEDULE,
        "-p", str(partition),
        "-u", LECTAUREP_NORMALIZATION,
        "-q", LECTAUREP_QUIT,
        "--lag", str(LECTAUREP_LAG),
        "-N", str(epochs),
        "--num-workers", str(num_workers),
        "--weights-format", "safetensors",
        arrow_file,
    ]

    logger.info("Running ketos train ...")
    logger.info("  Command: " + " ".join(cmd))

    result = subprocess.run(cmd, env=os.environ.copy())
    if result.returncode != 0:
        raise RuntimeError(f"ketos train failed with exit code {result.returncode}")

    logger.info(f"Training complete. Model saved in {model_dir}")


# ============================================================================
#  STEP 3: Test + report CER
# ============================================================================

def step_test(arrow_file, output_dir, batch_size):
    """
    Evaluate the trained model on the validation split of the arrow dataset
    and report CER.

    Uses ketos test CLI.
    """
    import subprocess

    model_dir = os.path.join(output_dir, "model")

    # Find the best model weights file
    model_file = _find_best_model(model_dir)
    if model_file is None:
        raise FileNotFoundError(
            f"No trained model found in {model_dir}. Run --step train first."
        )

    test_output = os.path.join(output_dir, "test_results.txt")

    cmd = [
        sys.executable, "-m", "kraken.ketos", "test",
        "-f", "binary",
        "-m", model_file,
        "-B", str(batch_size),
        arrow_file,
    ]

    logger.info("Running ketos test ...")
    logger.info("  Command: " + " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    # Write full output to file
    with open(test_output, "w", encoding="utf-8") as f:
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)

    # Print to console
    print(result.stdout)
    if result.stderr:
        # Filter for important lines only (not debug spam)
        for line in result.stderr.splitlines():
            if any(kw in line.lower() for kw in ["error", "cer", "accuracy", "warn"]):
                print(f"  [stderr] {line}")

    if result.returncode != 0:
        raise RuntimeError(f"ketos test failed with exit code {result.returncode}")

    logger.info(f"Test results written to {test_output}")


# ============================================================================
#  STEP 4: Dump batch + alphabet
# ============================================================================

def step_dump(arrow_file, output_dir, batch_size):
    """
    Load the compiled .arrow dataset, extract one batch in the same format
    as scribe-world-model's collate_alto_v5_fn, and dump everything to disk.

    Also dumps the alphabet/codec mapping.

    This enables a direct numerical comparison between:
      - kraken's data pipeline (this script)
      - scribe-world-model's LectaurepClone (data_alto.py + collate_alto_v5_fn)

    Output files in <output_dir>/batch_dump/:
      images.pt          — (B, H, W) float32 tensor (inverted: white-on-black)
      targets.pt         — 1D long tensor of flattened character indices
      input_lengths.pt   — (B,) long tensor of CNN output sequence lengths
      target_lengths.pt  — (B,) long tensor of target lengths
      raw_texts.json     — list of ground-truth text strings (NFD-normalised)

    Output files in <output_dir>/:
      alphabet.json      — {char_to_idx: {...}, idx_to_char: {...}}
    """
    import torch
    import numpy as np
    from PIL import Image
    import io
    import pyarrow as pa
    import pyarrow.ipc as ipc

    dump_dir = os.path.join(output_dir, "batch_dump")
    os.makedirs(dump_dir, exist_ok=True)

    # --- Load the arrow dataset ---
    logger.info(f"Loading arrow dataset: {arrow_file}")
    with pa.memory_map(arrow_file, "rb") as source:
        reader = ipc.open_file(source)
        table = reader.read_all()

    # Read metadata to get the alphabet
    metadata = table.schema.metadata
    lines_meta = json.loads(metadata[b"lines"])

    # The alphabet from the compiled dataset is a dict {char: count}
    raw_alphabet = lines_meta["alphabet"]
    # Build char_to_idx with blank=0 (same as kraken's codec)
    # kraken uses NFD normalisation and sorts characters
    chars = sorted(raw_alphabet.keys())
    char_to_idx = {}
    for i, c in enumerate(chars):
        char_to_idx[c] = i + 1  # blank=0
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    idx_to_char[0] = ""  # blank

    logger.info(f"Alphabet: {len(char_to_idx)} characters")

    # Save alphabet
    alphabet_path = os.path.join(output_dir, "alphabet.json")
    with open(alphabet_path, "w", encoding="utf-8") as f:
        json.dump(
            {"char_to_idx": char_to_idx, "idx_to_char": {str(k): v for k, v in idx_to_char.items()}},
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Alphabet saved to {alphabet_path}")

    # --- Extract lines from the arrow dataset ---
    # The arrow table has columns: lines (struct with text + im), train, validation, test
    lines_col = table.column("lines")
    train_col = table.column("train")

    # Collect training lines
    train_lines = []
    for i in range(len(lines_col)):
        if train_col[i].as_py():
            line_struct = lines_col[i]
            text = line_struct["text"].as_py()
            im_bytes = line_struct["im"].as_py()
            if text and im_bytes:
                train_lines.append((text, im_bytes))

    logger.info(f"Training lines in dataset: {len(train_lines)}")

    if not train_lines:
        raise ValueError("No training lines found in the arrow dataset.")

    # --- Take one batch ---
    batch_lines = train_lines[:batch_size]

    images_list = []
    all_targets = []
    input_lengths = []
    target_lengths = []
    raw_texts = []

    for text, im_bytes in batch_lines:
        # Load line image from bytes
        pil_img = Image.open(io.BytesIO(im_bytes))
        # Convert to grayscale if needed
        if pil_img.mode != "L":
            pil_img = pil_img.convert("L")

        # Resize to height=120 (matching lectaurep_base spec: [1,120,0,1])
        img_height = 120
        w, h = pil_img.size
        new_w = int(w * img_height / h)
        pil_img = pil_img.resize((new_w, img_height), Image.LANCZOS)

        # Convert to numpy float [0, 1], then invert (white text on black bg)
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        arr = 1.0 - arr  # invert: matches both ketos and scribe-world-model

        # Convert to torch tensor: (H, W)
        img_tensor = torch.from_numpy(arr)

        # NFD-normalise the text (matches ketos pipeline)
        text_nfd = unicodedata.normalize("NFD", text)

        # Encode targets
        encoded = []
        for c in text_nfd:
            if c in char_to_idx:
                encoded.append(char_to_idx[c])
            # else: skip OOV (shouldn't happen since alphabet is from this data)

        # The CNN subsamples width by 8 (3 MaxPool2d(2,2) + final conv gives W/8)
        # Actually with the VGSL spec: 3x Mp2,2 → W/8
        input_len = new_w // 8

        images_list.append(img_tensor)
        input_lengths.append(input_len)
        all_targets.extend(encoded)
        target_lengths.append(len(encoded))
        raw_texts.append(text_nfd)

    # Pad images to same width
    B = len(images_list)
    H = images_list[0].shape[0]  # 120
    W_max = max(img.shape[1] for img in images_list)

    padded_images = torch.zeros(B, H, W_max, dtype=torch.float32)
    for i, img in enumerate(images_list):
        padded_images[i, :, : img.shape[1]] = img

    targets = torch.tensor(all_targets, dtype=torch.long)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    # --- Save tensors ---
    torch.save(padded_images, os.path.join(dump_dir, "images.pt"))
    torch.save(targets, os.path.join(dump_dir, "targets.pt"))
    torch.save(input_lengths, os.path.join(dump_dir, "input_lengths.pt"))
    torch.save(target_lengths, os.path.join(dump_dir, "target_lengths.pt"))

    with open(os.path.join(dump_dir, "raw_texts.json"), "w", encoding="utf-8") as f:
        json.dump(raw_texts, f, ensure_ascii=False, indent=2)

    # Print batch summary
    logger.info(f"Batch dump ({B} samples):")
    logger.info(f"  images:         {padded_images.shape}")
    logger.info(f"  targets:        {targets.shape}  (total {len(targets)} chars)")
    logger.info(f"  input_lengths:  {input_lengths.tolist()}")
    logger.info(f"  target_lengths: {target_lengths.tolist()}")
    for i, t in enumerate(raw_texts[:5]):
        logger.info(f"  text[{i}]: {t[:80]}{'...' if len(t) > 80 else ''}")
    logger.info(f"All files saved to {dump_dir}")

    # --- Also dump the raw line images (un-resized) for visual inspection ---
    raw_dir = os.path.join(dump_dir, "raw_line_images")
    os.makedirs(raw_dir, exist_ok=True)
    for i, (text, im_bytes) in enumerate(batch_lines[:10]):
        img_path = os.path.join(raw_dir, f"line_{i:03d}.png")
        with open(img_path, "wb") as f:
            f.write(im_bytes)
    logger.info(f"Saved {min(10, len(batch_lines))} raw line images to {raw_dir}")


# ============================================================================
#  Utilities
# ============================================================================

def _find_best_model(model_dir):
    """Find the best model file in the ketos output directory."""
    if not os.path.isdir(model_dir):
        return None

    # Look for .safetensors files (newer kraken format)
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, "best_*.safetensors")))
    if safetensor_files:
        return safetensor_files[-1]

    # Look for any .safetensors files
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if safetensor_files:
        return safetensor_files[-1]

    # Look for checkpoint files
    ckpt_files = sorted(glob.glob(os.path.join(model_dir, "*.ckpt")))
    if ckpt_files:
        return ckpt_files[-1]

    # Look in subdirectories (ketos sometimes creates versioned dirs)
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".safetensors") or f.endswith(".ckpt"):
                return os.path.join(root, f)

    return None


# ============================================================================
#  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ketos lectaurep_base training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=["compile", "train", "test", "dump"],
        help="Which step to run.",
    )

    # --- compile args ---
    parser.add_argument(
        "--alto-dirs",
        nargs="+",
        default=None,
        help="Directories containing ALTO XML + JPG files (for compile step).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for all artifacts.",
    )

    # --- train / test / dump args ---
    parser.add_argument(
        "--arrow-file",
        default=None,
        help="Path to the compiled .arrow dataset (for train/test/dump).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=LECTAUREP_BATCH_SIZE,
        help=f"Batch size (default: {LECTAUREP_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LECTAUREP_LR,
        help=f"Learning rate (default: {LECTAUREP_LR}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs (default: 50).",
    )
    parser.add_argument(
        "--partition",
        type=float,
        default=LECTAUREP_PARTITION,
        help=f"Train/val partition (default: {LECTAUREP_PARTITION}).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4).",
    )

    args = parser.parse_args()

    # Defaults for arrow file
    if args.arrow_file is None:
        args.arrow_file = os.path.join(args.output_dir, "lectaurep_dataset.arrow")

    if args.step == "compile":
        if not args.alto_dirs:
            parser.error("--alto-dirs is required for the compile step.")
        step_compile(
            alto_dirs=args.alto_dirs,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
        )

    elif args.step == "train":
        if not os.path.exists(args.arrow_file):
            parser.error(
                f"Arrow dataset not found: {args.arrow_file}\n"
                f"Run --step compile first."
            )
        step_train(
            arrow_file=args.arrow_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            partition=args.partition,
            num_workers=args.num_workers,
        )

    elif args.step == "test":
        if not os.path.exists(args.arrow_file):
            parser.error(
                f"Arrow dataset not found: {args.arrow_file}\n"
                f"Run --step compile first."
            )
        step_test(
            arrow_file=args.arrow_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )

    elif args.step == "dump":
        if not os.path.exists(args.arrow_file):
            parser.error(
                f"Arrow dataset not found: {args.arrow_file}\n"
                f"Run --step compile first."
            )
        step_dump(
            arrow_file=args.arrow_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
