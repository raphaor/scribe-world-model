"""
Train a character-level n-gram language model for CTC beam search.

Extracts the TRAINING split (seed 42, 80/20 — same split as train.py)
transcriptions from the ALTO dataset and builds a character n-gram model
saved as a .pkl file, loadable by beam_decode.CharNgramLM.

The n-gram counts do NOT cross line boundaries: each transcription is
treated as an independent character sequence.

Usage:
    python train_lm.py --model-version v18 --output char_8gram.pkl --order 8
    python train_lm.py --model-version v18 --output char_8gram.pkl --order 8 \\
        --min-frames-per-char 1.0
"""

import sys
import os
import argparse
import pickle
from collections import defaultdict, Counter

import torch
from torch.utils.data import random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from model_registry import get_spec, default_train_args, known_versions
from data_alto import AltoLineDataset, build_alphabet


def extract_train_texts(alto_dirs, img_height, model_version, min_frames_per_char):
    """
    Extract training texts using the same split as train.py (seed 42, 80/20).

    Returns: (list[str], set[str]) — texts and vocabulary.
    """
    spec = get_spec(model_version)
    char_to_idx, _ = build_alphabet(alto_dirs)

    dataset = AltoLineDataset(alto_dirs, img_height=img_height)

    if min_frames_per_char > 0.0:
        removed, kept = dataset.filter_unlearnable(
            char_to_idx,
            width_stride=spec.cnn_width_stride,
            min_frames_per_char=min_frames_per_char,
        )
        print(f"Filtered {removed} unlearnable lines; {kept} remain")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, _val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Train split: {len(train_ds)} lines")

    # Extract texts directly from the underlying dataset (skip image loading)
    base = train_ds.dataset if hasattr(train_ds, "dataset") else dataset
    indices = train_ds.indices if hasattr(train_ds, "indices") else range(len(base))

    texts = []
    vocab = set()
    for idx in indices:
        # samples is a list of (numpy_array, text) tuples
        _, text = base.samples[idx]
        texts.append(text)
        vocab.update(text)

    print(f"Vocabulary: {len(vocab)} unique characters")
    return texts, vocab


def build_ngram_counts(texts, order):
    """
    Count character n-grams up to *order*.

    Each text is treated as an independent character sequence (n-grams
    do not cross text boundaries).

    Returns:
        counts: dict {1: {(c,): n}, 2: {(c1,c2): n}, ...}
        total_chars: total number of character tokens
    """
    counts = {k: defaultdict(int) for k in range(1, order + 1)}
    total_chars = 0

    for text in texts:
        chars = list(text)
        total_chars += len(chars)

        for k in range(1, min(order + 1, len(chars) + 1)):
            for i in range(len(chars) - k + 1):
                ngram = tuple(chars[i : i + k])
                counts[k][ngram] += 1

    return dict(counts), total_chars


def main():
    parser = argparse.ArgumentParser(
        description="Train a character n-gram LM for CTC beam search"
    )
    parser.add_argument(
        "--model-version",
        choices=known_versions(),
        default="v18",
        help="Model version (for img_height and cnn_width_stride)",
    )
    parser.add_argument(
        "--alto-dirs",
        nargs="+",
        default=config.ALTO_DIRS,
        help="ALTO directories (default: config.ALTO_DIRS)",
    )
    parser.add_argument(
        "--output",
        default="char_8gram.pkl",
        help="Output .pkl path (default: char_8gram.pkl)",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=8,
        help="N-gram order (default: 8)",
    )
    parser.add_argument(
        "--min-frames-per-char",
        type=float,
        default=0.0,
        help="Must match training time value (default: 0.0). "
        "v18 R8 used 1.0",
    )
    args = parser.parse_args()

    spec = get_spec(args.model_version)
    img_h = spec.img_height

    print(f"Model version: {args.model_version} (img_height={img_h})")
    print(f"ALTO dirs: {args.alto_dirs}")
    print(f"N-gram order: {args.order}")
    print()

    # 1. Extract training texts
    print("Loading dataset and extracting train split texts...")
    texts, vocab = extract_train_texts(
        args.alto_dirs, img_h, args.model_version, args.min_frames_per_char
    )
    print(f"Extracted {len(texts)} training lines")

    # Stats
    lengths = [len(t) for t in texts]
    print(
        f"  Line lengths: min={min(lengths)}, "
        f"max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}"
    )

    # 2. Count n-grams
    print(f"\nCounting n-grams (order 1..{args.order})...")
    counts, total_chars = build_ngram_counts(texts, args.order)

    for k in range(1, args.order + 1):
        print(f"  {k}-grams: {len(counts[k]):,} unique")

    print(f"  Total character tokens: {total_chars:,}")
    print(f"  Vocabulary: {len(vocab)} characters")

    # 3. Save
    lm_data = {
        "counts": counts,
        "total_chars": total_chars,
        "vocab_size": len(vocab),
        "order": args.order,
    }
    with open(args.output, "wb") as f:
        pickle.dump(lm_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nSaved LM to {args.output} ({file_size:.1f} MB)")
    print(f"\nUsage in recognize.py:")
    print(
        f"  python recognize.py --model hwm_v18_322_no_jepa.pt "
        f"--model-version {args.model_version} "
        f"--beam-search --lm-path {args.output} "
        f"--min-frames-per-char {args.min_frames_per_char}"
    )


if __name__ == "__main__":
    main()
