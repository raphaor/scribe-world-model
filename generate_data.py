"""
Synthetic Data Generator for HWM
Generates simple handwriting-like images from fonts
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset, DataLoader

import config


class SyntheticHandwritingDataset(Dataset):
    """
    Generates synthetic handwriting lines from fonts
    For PoC: uses system fonts or simple rendering
    """

    def __init__(
        self,
        num_lines=100,
        img_height=32,
        max_width=200,
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        min_len=3,
        max_len=8,
        seed=42,
    ):
        super().__init__()

        self.num_lines = num_lines
        self.img_height = img_height
        self.max_width = max_width
        self.alphabet = alphabet
        self.min_len = min_len
        self.max_len = max_len

        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Generate data
        self.samples = self._generate_samples()

        print(f"Generated {len(self.samples)} synthetic handwriting lines")

    def _generate_samples(self):
        """Generate all samples"""
        samples = []

        # Try to find a font, fall back to default
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
            None,  # Default font
        ]

        font = None
        font_size = 24
        for font_path in font_paths:
            try:
                if font_path and os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    print(f"Using font: {font_path}")
                    break
            except:
                continue

        if font is None:
            font = ImageFont.load_default()
            print("Using default font")

        for i in range(self.num_lines):
            # Generate random text
            text_len = random.randint(self.min_len, self.max_len)
            text = "".join(random.choices(self.alphabet, k=text_len))

            # Create image
            img = Image.new("L", (self.max_width, self.img_height), color=255)
            draw = ImageDraw.Draw(img)

            # Draw text
            draw.text((5, 2), text, font=font, fill=0)

            # Add some noise for realism
            img_array = np.array(img)
            noise = np.random.randint(0, 20, img_array.shape, dtype=np.uint8)
            img_array = np.clip(img_array.astype(np.int32) - noise, 0, 255).astype(
                np.uint8
            )

            samples.append(
                {"image": img_array, "text": text, "width": img_array.shape[1]}
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            img: (H, W) tensor normalized to [0, 1]
            text: string
        """
        sample = self.samples[idx]
        img = torch.from_numpy(sample["image"]).float() / 255.0
        text = sample["text"]

        return img, text


def extract_columns(img, window_size=10, stride=5):
    """
    Extract sliding window columns from image
    Args:
        img: (H, W) tensor
        window_size: width of each window
        stride: stride between windows
    Returns:
        columns: (T, H, window_size) tensor
    """
    H, W = img.shape
    columns = []

    for start in range(0, W - window_size + 1, stride):
        col = img[:, start : start + window_size]
        columns.append(col)

    if len(columns) == 0:
        # Fallback: pad and take one window
        padded = torch.nn.functional.pad(img, (0, window_size - W))
        columns = [padded[:, :window_size]]

    return torch.stack(columns)


def collate_fn(batch, window_size=10, stride=5):
    """
    Collate function for DataLoader
    Converts images to column sequences
    """
    img_seqs = []
    texts = []

    for img, text in batch:
        # Extract columns
        cols = extract_columns(img, window_size=window_size, stride=stride)
        img_seqs.append(cols)
        texts.append(text)

    # Pad sequences to same length
    max_len = max(seq.shape[0] for seq in img_seqs)
    batch_size = len(img_seqs)
    H = img_seqs[0].shape[1]

    padded_seqs = torch.zeros(batch_size, max_len, H, window_size)
    lengths = []

    for i, seq in enumerate(img_seqs):
        T = seq.shape[0]
        padded_seqs[i, :T] = seq
        lengths.append(T)

    return padded_seqs, texts, torch.tensor(lengths)


def create_dataloader(
    num_lines=100, batch_size=4, window_size=10, stride=5, shuffle=True
):
    """Create DataLoader for synthetic data"""
    dataset = SyntheticHandwritingDataset(num_lines=num_lines)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, window_size, stride),
    )

    return loader


def encode_text(text, char_to_idx):
    """Encode text string to list of CTC label indices."""
    return [char_to_idx[c] for c in text if c in char_to_idx]


def decode_indices(indices, idx_to_char):
    """Decode CTC output indices to string (greedy, with collapse)."""
    result = []
    prev = None
    for idx in indices:
        if idx != 0 and idx != prev:
            result.append(idx_to_char.get(idx, "?"))
        prev = idx
    return "".join(result)


def test_data_generation():
    """Test data generation"""
    print("\n" + "=" * 60)
    print("Testing Synthetic Data Generation")
    print("=" * 60)

    # Create small dataset
    dataset = SyntheticHandwritingDataset(num_lines=5)

    print(f"\nSample shapes:")
    for i in range(min(3, len(dataset))):
        img, text = dataset[i]
        print(f"  [{i}] Image: {img.shape}, Text: '{text}'")

    # Test DataLoader
    print(f"\nTesting DataLoader...")
    loader = create_dataloader(num_lines=10, batch_size=2)

    batch = next(iter(loader))
    img_seqs, texts, lengths = batch

    print(f"  Batch shapes:")
    print(f"    Images: {img_seqs.shape}")
    print(f"    Texts: {texts}")
    print(f"    Lengths: {lengths}")

    print(f"\n✓ Data generation working!")
    print("=" * 60)


if __name__ == "__main__":
    test_data_generation()
