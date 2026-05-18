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
        # Store as uint8 (1 byte/pixel): the in-RAM dataset is otherwise
        # 4x larger for nothing. __getitem__ does `/ 255.0` which promotes
        # to float anyway, and _augment casts via uint8 already.
        arr = np.array(line_img.convert("L"), dtype=np.uint8)
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

        # Parse in parallel but reassemble in deterministic task order.
        # as_completed yields by completion time, so extending samples
        # directly would make the list order — and hence the seeded
        # random_split train/val partition — depend on thread timing.
        # Index the results to keep the split a pure function of the data.
        results = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_parse_page, t): i for i, t in enumerate(tasks)}
            done = 0
            for future in as_completed(futures):
                done += 1
                results[futures[future]] = future.result()
                sys.stdout.write(f"\r  Parsing pages: {done}/{len(tasks)}")
                sys.stdout.flush()
        for page_samples, page_chars in results:
            self.samples.extend(page_samples)
            self.chars.update(page_chars)

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


class UnannotatedLineDataset(Dataset):
    """
    Dataset for unannotated line images (adapt / self-supervised mode).
    Loads lines from ALTO pages but discards the text.
    """

    def __init__(
        self, dirs, img_height=48, max_width=2000, augment=False, max_workers=4
    ):
        self.samples = []
        self.img_height = img_height
        self.augment = augment

        os.makedirs(CACHE_DIR, exist_ok=True)
        key = _cache_key(dirs, img_height, max_width) + "_unannotated"
        cache_path = os.path.join(CACHE_DIR, f"dataset_{key}.pkl")

        if os.path.exists(cache_path):
            print(f"Loading cached unannotated dataset from {cache_path} ...")
            with open(cache_path, "rb") as f:
                self.samples = pickle.load(f)
            print(f"Loaded {len(self.samples)} unannotated lines (from cache)")
            return

        xml_files = []
        for d in dirs:
            xml_files.extend(
                p for p in sorted(glob.glob(os.path.join(d, "*.xml")))
                if os.path.basename(p) != "METS.xml"
            )

        tasks = [(xml_path, img_height, max_width) for xml_path in xml_files]

        # Deterministic task-order reassembly (see AltoLineDataset).
        results = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_parse_page, t): i for i, t in enumerate(tasks)}
            done = 0
            for future in as_completed(futures):
                done += 1
                results[futures[future]] = future.result()
                sys.stdout.write(f"\r  Parsing unannotated pages: {done}/{len(tasks)}")
                sys.stdout.flush()
        for page_samples, _chars in results:
            self.samples.extend([arr for arr, _text in page_samples])

        sys.stdout.write("\n")
        print(f"Loaded {len(self.samples)} unannotated lines from {len(dirs)} dirs")

        with open(cache_path, "wb") as f:
            pickle.dump(self.samples, f)
        print(f"Cache saved to {cache_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr = self.samples[idx]
        if self.augment:
            arr = AltoLineDataset._augment(arr)
        img = torch.from_numpy(arr.copy()) / 255.0
        return (img,)


def collate_unannotated_fn(batch, window_size=10, stride=5, max_seq_len=512):
    """Collate for unannotated batches — returns only (padded_img_seqs,)."""
    from generate_data import extract_columns

    img_seqs = []
    for (img,) in batch:
        cols = extract_columns(img, window_size=window_size, stride=stride)
        if cols.shape[0] > max_seq_len:
            cols = cols[:max_seq_len]
        img_seqs.append(cols)

    max_len = max(seq.shape[0] for seq in img_seqs)
    B = len(img_seqs)
    H = img_seqs[0].shape[1]

    padded = torch.zeros(B, max_len, H, window_size)
    for i, seq in enumerate(img_seqs):
        padded[i, : seq.shape[0]] = seq

    return (padded,)


def collate_alto_fn(batch, window_size=10, stride=5, char_to_idx=None, max_seq_len=512):
    from generate_data import extract_columns

    img_seqs = []
    all_targets = []
    input_lengths = []
    target_lengths = []
    raw_texts = []

    for img, text in batch:
        cols = extract_columns(img, window_size=window_size, stride=stride)
        # Truncate overly long sequences to cap memory usage
        if cols.shape[0] > max_seq_len:
            cols = cols[:max_seq_len]
        img_seqs.append(cols)
        input_lengths.append(cols.shape[0])

        encoded = [char_to_idx[c] for c in text if c in char_to_idx]
        all_targets.extend(encoded)
        target_lengths.append(len(encoded))
        raw_texts.append(text)

    max_len = max(seq.shape[0] for seq in img_seqs)
    B = len(img_seqs)
    H = img_seqs[0].shape[1]

    padded = torch.zeros(B, max_len, H, window_size)
    for i, seq in enumerate(img_seqs):
        padded[i, : seq.shape[0]] = seq

    targets = torch.tensor(all_targets, dtype=torch.long)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return padded, targets, input_lengths, target_lengths, raw_texts


def collate_alto_v5_fn(batch, char_to_idx=None):
    """Collate for v5: full line images padded in width, no frame extraction."""
    imgs = []
    all_targets = []
    input_lengths = []
    target_lengths = []
    raw_texts = []

    for img, text in batch:
        imgs.append(img)
        # Conv reduces width by factor 8 (3 MaxPool(2,2))
        input_lengths.append(img.shape[1] // 8)

        encoded = [char_to_idx[c] for c in text if c in char_to_idx]
        all_targets.extend(encoded)
        target_lengths.append(len(encoded))
        raw_texts.append(text)

    B = len(imgs)
    H = imgs[0].shape[0]
    W_max = max(img.shape[1] for img in imgs)

    padded = torch.zeros(B, H, W_max)
    for i, img in enumerate(imgs):
        padded[i, :, :img.shape[1]] = img

    targets = torch.tensor(all_targets, dtype=torch.long)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return padded, targets, input_lengths, target_lengths, raw_texts


def collate_unannotated_v5_fn(batch):
    """Collate for v5 unannotated: full line images padded in width."""
    imgs = []
    input_lengths = []
    for (img,) in batch:
        imgs.append(img)
        # Match collate_alto_v5_fn: encoder subsamples width by 8.
        input_lengths.append(img.shape[1] // 8)

    B = len(imgs)
    H = imgs[0].shape[0]
    W_max = max(img.shape[1] for img in imgs)

    padded = torch.zeros(B, H, W_max)
    for i, img in enumerate(imgs):
        padded[i, :, :img.shape[1]] = img

    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    return padded, input_lengths


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


def line_widths(ds):
    """
    Return the pixel width of every sample of ``ds``, in dataset order.

    Handles a bare AltoLineDataset / UnannotatedLineDataset as well as a
    ``torch.utils.data.Subset`` (as produced by ``random_split``). The
    returned widths are positional — width[p] is the width of the sample
    the loader sees at position ``p`` — so a LengthBucketBatchSampler
    built on them yields valid indices for that (sub)dataset.
    """
    if isinstance(ds, torch.utils.data.Subset):
        base, idxs = ds.dataset, ds.indices
    else:
        base, idxs = ds, range(len(ds))
    samples = base.samples
    widths = []
    for i in idxs:
        s = samples[i]
        arr = s[0] if isinstance(s, tuple) else s   # (arr, text) or bare arr
        widths.append(int(arr.shape[1]))
    return widths


class LengthBucketBatchSampler:
    """
    Width-homogeneous, fixed-count batch sampler.

    Plain shuffled batching of variable-width lines pads every batch to
    its widest line — one ~2000px line in an otherwise-narrow batch
    wastes most of the tensor and spikes VRAM unpredictably.

    This sampler sorts lines by width inside shuffled pools and cuts
    fixed ``batch_size``-line batches from each pool, so every batch is
    near-uniform in width: padding waste is near zero, and peak VRAM is
    the *widest* batch (``batch_size`` lines at the maximum line width)
    — choose ``batch_size`` so that batch fits.

    Every batch holds exactly ``batch_size`` lines (bar the last of each
    pool). Uniform count matters for training: with CTC's mean reduction
    and JEPA's in-batch InfoNCE negatives, a variable count would
    silently re-weight long lines and swing the contrastive loss scale.
    Batch *order* is shuffled, so an epoch never drifts short->long.
    Yields positional indices, so it works directly on a Subset.
    """

    def __init__(self, widths, batch_size, shuffle=True, pool_factor=50):
        self.widths = [int(w) for w in widths]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pool_size = max(batch_size, batch_size * pool_factor)
        self._len = len(self._build(shuffle=False))

    def _build(self, shuffle):
        n = len(self.widths)
        order = list(range(n))
        if shuffle:
            random.shuffle(order)
        batches = []
        for ps in range(0, n, self.pool_size):
            # Sort the pool by width, then cut fixed-size chunks: each
            # batch is width-homogeneous AND holds batch_size lines.
            pool = sorted(order[ps:ps + self.pool_size], key=lambda i: self.widths[i])
            for bs in range(0, len(pool), self.batch_size):
                batches.append(pool[bs:bs + self.batch_size])
        if shuffle:
            random.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._build(self.shuffle))

    def __len__(self):
        return self._len
