#!/usr/bin/env python3
"""
Compare two data-loading pipelines on the lectaurep-bronod PAGE XML dataset:

Pipeline A (kraken-native / ketos-style):
  - XMLPage(filetype='page') + extract_polygons
  - resize to height 120, invert (text=white), normalize to float

Pipeline B (scribe-style, adapted for PAGE XML):
  - Same kraken parsing but using scribe's data_alto.py preprocessing logic
  - Identical resize + normalize + invert
  - NFD-normalized alphabet build (matching scribe's get_alphabet)

Compares: line counts, alphabets, image tensors, encoded targets, input_lengths.
"""

import os
import sys
import glob
import unicodedata
import warnings

warnings.filterwarnings("ignore", message="divide by zero", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value", category=RuntimeWarning)

import numpy as np
import torch
from PIL import Image
from kraken.lib.xml import XMLPage
from kraken.lib.segmentation import extract_polygons

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = "/home/raph/projects/lectaurep-bronod/data/bronod1"
PAGE_DIR = os.path.join(DATA_DIR, "page")
IMG_HEIGHT = 120
MAX_WIDTH = 2000
N_COMPARE = 5  # number of sample lines to compare in detail


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline A: kraken-native / ketos-style
# ═══════════════════════════════════════════════════════════════════════════════

def load_pipeline_a(page_dir, img_height, max_width):
    """
    Load using kraken's native PAGE XML support.
    Preprocessing matches ketos convention:
      - resize to img_height
      - convert to grayscale uint8
      - __getitem__ does /255 and 1-x (invert to text=white)
    """
    xml_files = sorted(glob.glob(os.path.join(page_dir, "*.xml")))
    samples = []  # (arr_uint8, text, page_xml_basename)
    chars = set()

    for xml_path in xml_files:
        basename = os.path.splitext(os.path.basename(xml_path))[0]
        jpg_path = os.path.join(DATA_DIR, basename + ".jpg")
        if not os.path.exists(jpg_path):
            continue

        try:
            page = XMLPage(xml_path, filetype="page")
            seg = page.to_container()
            pil_img = Image.open(jpg_path)
        except Exception as e:
            print(f"  [A] Skipping {xml_path}: {e}")
            continue

        gen = extract_polygons(pil_img, seg)
        while True:
            try:
                line_img, line_obj = next(gen)
            except StopIteration:
                break
            except (ValueError, RuntimeError):
                continue

            text = getattr(line_obj, "text", None)
            if not text or not text.strip():
                continue

            w, h = line_img.size
            if w == 0 or h == 0:
                continue
            new_w = int(w * img_height / h)
            if new_w > max_width or new_w < 10:
                continue

            line_img = line_img.resize((new_w, img_height), Image.LANCZOS)
            arr = np.array(line_img.convert("L"), dtype=np.uint8)
            samples.append((arr, text, basename))
            chars.update(text)

    return samples, chars


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline B: scribe-style (adapted from data_alto.py for PAGE XML)
# ═══════════════════════════════════════════════════════════════════════════════

def load_pipeline_b(page_dir, img_height, max_width):
    """
    Replicate scribe's data_alto.py logic but with filetype='page' and
    corrected image path resolution (images in parent dir, not page/).
    """
    xml_files = sorted(glob.glob(os.path.join(page_dir, "*.xml")))
    samples = []  # (arr_uint8, text, page_xml_basename)
    chars = set()

    for xml_path in xml_files:
        # scribe's _parse_page does xml_path.replace(".xml", ".jpg")
        # which would be wrong for our layout (page/*.xml vs ../*.jpg).
        # Adapt to find the image correctly.
        basename = os.path.splitext(os.path.basename(xml_path))[0]
        jpg_path = os.path.join(DATA_DIR, basename + ".jpg")
        if not os.path.exists(jpg_path):
            continue

        try:
            # This is the key adaptation: scribe uses filetype='alto',
            # but our data is PAGE XML
            page = XMLPage(xml_path, filetype="page")
            seg = page.to_container()
            pil_img = Image.open(jpg_path)
        except Exception as e:
            print(f"  [B] Skipping {xml_path}: {e}")
            continue

        gen = extract_polygons(pil_img, seg)
        while True:
            try:
                line_img, line_obj = next(gen)
            except StopIteration:
                break
            except (ValueError, RuntimeError):
                continue

            text = getattr(line_obj, "text", None)
            if not text or not text.strip():
                continue

            w, h = line_img.size
            if w == 0 or h == 0:
                continue
            new_w = int(w * img_height / h)
            if new_w > max_width or new_w < 10:
                continue

            # scribe's preprocessing: resize, convert L, store uint8
            line_img = line_img.resize((new_w, img_height), Image.LANCZOS)
            arr = np.array(line_img.convert("L"), dtype=np.uint8)
            samples.append((arr, text, basename))
            chars.update(text)

    return samples, chars


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing functions (matching scribe's __getitem__)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_scribe(arr):
    """Mimic scribe AltoLineDataset.__getitem__: /255, invert."""
    img = torch.from_numpy(arr.copy()) / 255.0
    img = 1.0 - img  # invert: text=white on bg=black
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# Alphabet builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_alphabet_scribe(chars_set):
    """Mimic scribe's AltoLineDataset.get_alphabet: NFD normalize, sort, dedup."""
    chars = sorted(unicodedata.normalize("NFD", "".join(chars_set)))
    seen = set()
    unique_chars = []
    for c in chars:
        if c not in seen:
            seen.add(c)
            unique_chars.append(c)
    chars = unique_chars
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    idx_to_char[0] = ""
    return char_to_idx, idx_to_char


def encode_text_scribe(text, char_to_idx):
    """Mimic collate_alto_v5_fn encoding: NFD normalize first."""
    text_nfd = unicodedata.normalize("NFD", text)
    encoded = [char_to_idx[c] for c in text_nfd if c in char_to_idx]
    return encoded


# ═══════════════════════════════════════════════════════════════════════════════
# Main comparison
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("COMPARISON: kraken-native vs scribe-style pipelines")
    print("Dataset: lectaurep-bronod PAGE XML")
    print(f"img_height={IMG_HEIGHT}, max_width={MAX_WIDTH}")
    print("=" * 70)

    # ── Load both pipelines ──────────────────────────────────────────────────
    print("\n[1] Loading Pipeline A (kraken-native, filetype='page')...")
    samples_a, chars_a = load_pipeline_a(PAGE_DIR, IMG_HEIGHT, MAX_WIDTH)
    print(f"    → {len(samples_a)} lines loaded")

    print("\n[2] Loading Pipeline B (scribe-style, adapted for PAGE XML)...")
    samples_b, chars_b = load_pipeline_b(PAGE_DIR, IMG_HEIGHT, MAX_WIDTH)
    print(f"    → {len(samples_b)} lines loaded")

    # ── Compare line counts ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("COMPARISON (a): Number of lines loaded")
    print(f"  Pipeline A: {len(samples_a)}")
    print(f"  Pipeline B: {len(samples_b)}")
    print(f"  Match: {'YES' if len(samples_a) == len(samples_b) else 'NO'}")

    # ── Compare alphabets ────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("COMPARISON (b): Alphabet / character sets")
    c2i_a, i2c_a = build_alphabet_scribe(chars_a)
    c2i_b, i2c_b = build_alphabet_scribe(chars_b)
    print(f"  Pipeline A: {len(c2i_a)} unique NFD characters")
    print(f"  Pipeline B: {len(c2i_b)} unique NFD characters")
    only_a = set(c2i_a.keys()) - set(c2i_b.keys())
    only_b = set(c2i_b.keys()) - set(c2i_a.keys())
    if only_a:
        print(f"  Only in A: {sorted(only_a)}")
    if only_b:
        print(f"  Only in B: {sorted(only_b)}")
    print(f"  Alphabets identical: {'YES' if not only_a and not only_b else 'NO'}")
    # Show first 20 chars
    print(f"  A first 20: {''.join(list(c2i_a.keys())[:20])}")
    print(f"  B first 20: {''.join(list(c2i_b.keys())[:20])}")

    # ── Compare image tensors for sample lines ──────────────────────────────
    print("\n" + "─" * 70)
    print(f"COMPARISON (c): Image tensor comparison ({N_COMPARE} sample lines)")
    n = min(N_COMPARE, len(samples_a), len(samples_b))
    # Pick evenly-spaced samples
    if n > 0:
        indices = [int(i * (min(len(samples_a), len(samples_b)) - 1) / max(n - 1, 1))
                    for i in range(n)]
    else:
        indices = []

    for idx in indices:
        arr_a, text_a, page_a = samples_a[idx]
        arr_b, text_b, page_b = samples_b[idx]

        tensor_a = preprocess_scribe(arr_a)
        tensor_b = preprocess_scribe(arr_b)

        # Shapes
        print(f"\n  Line {idx} (page {page_a}):")
        print(f"    Text A: {repr(text_a[:70])}")
        print(f"    Text B: {repr(text_b[:70])}")
        print(f"    Text match: {'YES' if text_a == text_b else 'NO'}")
        print(f"    Array shape A: {arr_a.shape}, B: {arr_b.shape}")
        print(f"    Tensor shape A: {tensor_a.shape}, B: {tensor_b.shape}")

        if tensor_a.shape == tensor_b.shape:
            diff = (tensor_a - tensor_b).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"    Max abs diff:  {max_diff:.6f}")
            print(f"    Mean abs diff: {mean_diff:.6f}")
            print(f"    Tensors identical: {'YES' if max_diff == 0 else 'NO'}")
        else:
            print(f"    Shape MISMATCH — cannot compare element-wise")

    # ── Compare encoded target sequences ─────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"COMPARISON (d): Encoded target sequences ({N_COMPARE} sample lines)")

    for idx in indices:
        _, text_a, _ = samples_a[idx]
        _, text_b, _ = samples_b[idx]

        enc_a = encode_text_scribe(text_a, c2i_a)
        enc_b = encode_text_scribe(text_b, c2i_b)

        print(f"\n  Line {idx}:")
        print(f"    Encoded A ({len(enc_a)}): {enc_a[:30]}{'...' if len(enc_a)>30 else ''}")
        print(f"    Encoded B ({len(enc_b)}): {enc_b[:30]}{'...' if len(enc_b)>30 else ''}")
        print(f"    Encoded match: {'YES' if enc_a == enc_b else 'NO'}")

    # ── Compare input_lengths computation ────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"COMPARISON (e): input_lengths computation")
    print(f"  scribe's collate_alto_v5_fn: input_len = img.shape[1] // 8")
    print(f"  (encoder subsamples width by factor 8)")
    print()

    for idx in indices:
        arr_a, text_a, _ = samples_a[idx]
        arr_b, text_b, _ = samples_b[idx]
        tensor_a = preprocess_scribe(arr_a)
        tensor_b = preprocess_scribe(arr_b)

        # scribe's input_lengths: width // 8
        input_len_a = tensor_a.shape[1] // 8
        input_len_b = tensor_b.shape[1] // 8

        enc_a = encode_text_scribe(text_a, c2i_a)
        enc_b = encode_text_scribe(text_b, c2i_b)
        target_len_a = len(enc_a)
        target_len_b = len(enc_b)

        # CTC check
        ctc_ok_a = input_len_a >= target_len_a
        ctc_ok_b = input_len_b >= target_len_b

        print(f"  Line {idx}:")
        print(f"    A: width={tensor_a.shape[1]}, input_len={input_len_a}, "
              f"target_len={target_len_a}, CTC_ok={ctc_ok_a}")
        print(f"    B: width={tensor_b.shape[1]}, input_len={input_len_b}, "
              f"target_len={target_len_b}, CTC_ok={ctc_ok_b}")
        print(f"    Match: {'YES' if input_len_a == input_len_b else 'NO'}")

    # ── Overall summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    lines_match = len(samples_a) == len(samples_b)
    alpha_match = not only_a and not only_b
    texts_match = all(samples_a[i][1] == samples_b[i][1]
                      for i in range(min(len(samples_a), len(samples_b))))

    # Image comparison for all matching-shape samples
    img_diffs = []
    for i in range(min(len(samples_a), len(samples_b))):
        ta = preprocess_scribe(samples_a[i][0])
        tb = preprocess_scribe(samples_b[i][0])
        if ta.shape == tb.shape:
            img_diffs.append((ta - tb).abs().max().item())
        else:
            img_diffs.append(None)

    all_identical = all(d == 0 for d in img_diffs if d is not None)
    max_diff_overall = max((d for d in img_diffs if d is not None), default=0)
    n_shape_mismatch = sum(1 for d in img_diffs if d is None)

    print(f"  Line counts:         {'MATCH' if lines_match else 'MISMATCH'}")
    print(f"  Alphabets:           {'MATCH' if alpha_match else 'DIFFER'}")
    print(f"  Text strings:        {'MATCH' if texts_match else 'DIFFER'}")
    print(f"  Image tensors:       {'ALL IDENTICAL' if all_identical else f'max_diff={max_diff_overall:.6f}'}")
    print(f"  Shape mismatches:    {n_shape_mismatch}")
    print(f"  Total lines:         {len(samples_a)}")

    if lines_match and alpha_match and texts_match and all_identical:
        print("\n  ✓ Both pipelines produce IDENTICAL output.")
    else:
        print("\n  ✗ Pipelines differ — see details above.")


if __name__ == "__main__":
    main()
