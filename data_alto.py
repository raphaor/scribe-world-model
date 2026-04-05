"""
Data loader for ALTO XML + JPG page images.
Uses kraken's parsers for XML and line extraction.
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
import glob
import numpy as np
import random


class AltoLineDataset(Dataset):
    def __init__(self, alto_dirs, img_height=48, max_width=2000, augment=False):
        self.samples = []
        self.img_height = img_height
        self.augment = augment

        for alto_dir in alto_dirs:
            xml_files = sorted(glob.glob(os.path.join(alto_dir, "*.xml")))
            for xml_path in xml_files:
                if os.path.basename(xml_path) == "METS.xml":
                    continue

                jpg_path = xml_path.replace(".xml", ".jpg")
                if not os.path.exists(jpg_path):
                    continue

                page = XMLPage(xml_path, filetype="alto")
                seg = page.to_container()
                pil_img = Image.open(jpg_path)

                gen = extract_polygons(pil_img, seg)
                all_lines = []
                while True:
                    try:
                        all_lines.append(next(gen))
                    except StopIteration:
                        break
                    except (ValueError, RuntimeError):
                        continue

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
                    self.samples.append((arr, text))

        print(f"Loaded {len(self.samples)} lines from {len(alto_dirs)} dirs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, text = self.samples[idx]
        if self.augment:
            arr = self._augment(arr)
        img = torch.from_numpy(arr) / 255.0
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
