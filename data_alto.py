"""
Data loader for ALTO XML + JPG page images.
Uses kraken's parsers for XML and line extraction.
With pickle cache, parallel loading, and merged alphabet build.
"""

import warnings

warnings.filterwarnings("ignore", message="divide by zero", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value", category=RuntimeWarning)

from PIL import Image, ImageEnhance
from kraken.lib.xml import XMLPage
from kraken.lib.segmentation import extract_polygons
import torch
from torch.utils.data import Dataset
import os
import sys
import glob
import pickle
import hashlib
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache_alto")


def _parse_page(args):
    xml_path, img_height, max_width = args
    jpg_path = xml_path.replace(".xml", ".jpg")
    if not os.path.exists(jpg_path):
        return [], set()

    try:
        page = XMLPage(xml_path, filetype="alto")
        seg = page.to_container()
        pil_img = Image.open(jpg_path)
    except Exception:
        return [], set()

    gen = extract_polygons(pil_img, seg)
    all_lines = []
    while True:
        try:
            all_lines.append(next(gen))
        except StopIteration:
            break
        except (ValueError, RuntimeError):
            continue

    samples = []
    chars = set()
    for line_img, line_obj in all_lines:
        try:
            text = line_obj.text
        except (ValueError, RuntimeError):
            continue
        if not text or not text.strip():
            continue

        w, h = line_img.size
        if w == 0 or h == 0:
            continue
        new_w = int(w * img_height / h)
        if new_w > max_width or new_w < 10:
            continue

        line_img = line_img.resize((new_w, img_height), Image.LANCZOS)
        arr = np.array(line_img.convert("L"), dtype=np.float32)
        samples.append((arr, text))
        chars.update(text)

    return samples, chars


def _cache_key(alto_dirs, img_height, max_width):
    h = hashlib.md5()
    for d in sorted(alto_dirs):
        h.update(d.encode())
        for f in sorted(glob.glob(os.path.join(d, "*.xml"))):
            h.update(f.encode())
            h.update(str(os.path.getmtime(f)).encode())
            jpg = f.replace(".xml", ".jpg")
            if os.path.exists(jpg):
                h.update(str(os.path.getmtime(jpg)).encode())
    h.update(str(img_height).encode())
    h.update(str(max_width).encode())
    return h.hexdigest()


class AltoLineDataset(Dataset):
    def __init__(
        self, alto_dirs, img_height=48, max_width=2000, augment=False, max_workers=4
    ):
        self.samples = []
        self.img_height = img_height
        self.augment = augment
        self.chars = set()

        os.makedirs(CACHE_DIR, exist_ok=True)
        key = _cache_key(alto_dirs, img_height, max_width)
        cache_path = os.path.join(CACHE_DIR, f"dataset_{key}.pkl")

        if os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path} ...")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.samples = cached["samples"]
            self.chars = cached["chars"]
            print(f"Loaded {len(self.samples)} lines (from cache)")
            return

        xml_files = []
        for alto_dir in alto_dirs:
            for xml_path in sorted(glob.glob(os.path.join(alto_dir, "*.xml"))):
                if os.path.basename(xml_path) == "METS.xml":
                    continue
                xml_files.append(xml_path)

        tasks = [(xml_path, img_height, max_width) for xml_path in xml_files]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_parse_page, t): t for t in tasks}
            done = 0
            for future in as_completed(futures):
                done += 1
                page_samples, page_chars = future.result()
                self.samples.extend(page_samples)
                self.chars.update(page_chars)
                sys.stdout.write(f"\r  Parsing pages: {done}/{len(tasks)}")
                sys.stdout.flush()

        sys.stdout.write("\n")
        print(f"Loaded {len(self.samples)} lines from {len(alto_dirs)} dirs")

        with open(cache_path, "wb") as f:
            pickle.dump({"samples": self.samples, "chars": self.chars}, f)
        print(f"Cache saved to {cache_path}")

    def get_alphabet(self):
        chars = sorted(self.chars)
        char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        idx_to_char = {i + 1: c for i, c in enumerate(chars)}
        idx_to_char[0] = ""
        return char_to_idx, idx_to_char

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, text = self.samples[idx]
        if self.augment:
            arr = self._augment(arr)
        img = torch.from_numpy(arr.copy()) / 255.0
        return img, text

    @staticmethod
    def _augment(arr):
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
        if random.random() < 0.5:
            angle = random.uniform(-3, 3)
            img = img.rotate(angle, fillcolor=255, expand=False)
        if random.random() < 0.5:
            factor = random.uniform(0.85, 1.15)
            img = ImageEnhance.Contrast(img).enhance(factor)
        arr = np.array(img, dtype=np.float32)
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.02 * 255, arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0, 255)
        return arr


def collate_alto_fn(batch, window_size=10, stride=5, char_to_idx=None):
    from generate_data import extract_columns

    img_seqs = []
    all_targets = []
    input_lengths = []
    target_lengths = []

    for img, text in batch:
        cols = extract_columns(img, window_size=window_size, stride=stride)
        img_seqs.append(cols)
        input_lengths.append(cols.shape[0])

        encoded = [char_to_idx[c] for c in text if c in char_to_idx]
        all_targets.extend(encoded)
        target_lengths.append(len(encoded))

    max_len = max(seq.shape[0] for seq in img_seqs)
    B = len(img_seqs)
    H = img_seqs[0].shape[1]

    padded = torch.zeros(B, max_len, H, window_size)
    for i, seq in enumerate(img_seqs):
        padded[i, : seq.shape[0]] = seq

    targets = torch.tensor(all_targets, dtype=torch.long)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return padded, targets, input_lengths, target_lengths


def build_alphabet(alto_dirs):
    chars = set()
    for alto_dir in alto_dirs:
        for xml_path in glob.glob(os.path.join(alto_dir, "*.xml")):
            if "METS" in xml_path:
                continue
            page = XMLPage(xml_path, filetype="alto")
            for line in page.get_sorted_lines():
                if hasattr(line, "text") and line.text:
                    chars.update(line.text)

    chars = sorted(chars)
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    idx_to_char[0] = ""

    print(f"Alphabet: {len(chars)} characters")
    return char_to_idx, idx_to_char
