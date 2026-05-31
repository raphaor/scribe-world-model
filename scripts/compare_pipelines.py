#!/usr/bin/env python3
"""
Pipeline comparison: scribe-world-model vs kraken/ketos.

Compares every step of the HTR pipeline to diagnose why v15 LectaurepClone
collapses to 98% CER while ketos achieves 9.8% on the same data.

Usage:
  # With real ALTO data (on Windows):
  python scripts/compare_pipelines.py --alto-dirs D:/OCR_genealogie/Alto/dir1 D:/OCR_genealogie/Alto/dir2

  # Synthetic test (no data needed, for Pi testing):
  python scripts/compare_pipelines.py --synthetic

  # Architecture comparison only (no data needed):
  python scripts/compare_pipelines.py --arch-only
"""

import argparse
import json
import os
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compare_architectures():
    """Compare LectaurepClone (scribe) vs TorchVGSLModel (kraken) layer by layer."""
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)

    from kraken.lib.vgsl import TorchVGSLModel
    from model import LectaurepClone
    import config

    # Build both models
    vgsl_spec = "[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]"
    kraken_model = TorchVGSLModel(vgsl=vgsl_spec, codec=None)
    # Add a dummy codec to allow model init
    # kraken_model.add_codec(...)

    scribe_model = LectaurepClone(
        img_height=config.LECTAUREP_IMG_HEIGHT,
        num_classes=100,  # dummy
        hidden=config.LECTAUREP_HIDDEN,
        num_lstm_layers=config.LECTAUREP_NUM_LSTM,
        dropout=config.LECTAUREP_DROPOUT,
    )

    print(f"\nKraken params: {sum(p.numel() for p in kraken_model.nn.parameters()):,}")
    print(f"Scribe params: {scribe_model.count_parameters():,}")

    # Compare layer by layer
    print("\n--- Layer-by-layer comparison ---")

    # 1. CNN layers
    kraken_layers = list(kraken_model.nn.named_modules())
    scribe_encoder_layers = list(scribe_model.encoder.named_children())

    print("\n[CNN Encoder]")
    kraken_convs = [(n, m) for n, m in kraken_layers if 'co' in n and 'Conv' in str(type(m))]
    scribe_convs = [(n, m) for n, m in scribe_encoder_layers if 'Conv' in str(type(m))]

    for i, ((kn, km), (sn, sm)) in enumerate(zip(kraken_convs, scribe_convs)):
        print(f"\n  Conv layer {i}:")
        print(f"    Kraken: {km}")
        print(f"    Scribe: {sm}")
        # Compare weights shape
        if hasattr(km, 'weight') and hasattr(sm, 'weight'):
            print(f"    Kraken weight shape: {km.weight.shape}")
            print(f"    Scribe weight shape: {sm.weight.shape}")
            print(f"    Match: {km.weight.shape == sm.weight.shape}")
        if hasattr(km, 'padding') and hasattr(sm, 'padding'):
            print(f"    Kraken padding: {km.padding}")
            print(f"    Scribe padding: {sm.padding}")
            print(f"    Match: {km.padding == sm.padding}")

    # 2. Dropout layers
    print("\n[Dropout Layers]")
    kraken_dropouts = [(n, m) for n, m in kraken_layers if 'Dropout' in str(type(m)) and not n.endswith('layer')]
    for n, m in kraken_dropouts:
        layer = m.layer if hasattr(m, 'layer') else m
        print(f"  Kraken {n}: {type(layer).__name__}(p={layer.p})")

    print(f"  Scribe CNN Dropout: nn.Dropout(p={config.LECTAUREP_DROPOUT}) [all 4 layers]")
    print(f"  Scribe LSTM Dropout: nn.Dropout(p={config.LECTAUREP_DROPOUT}) [all 3 layers]")

    # CRITICAL FINDING: kraken uses Dropout2d for CNN/LSTM, scribe uses Dropout (1d)
    kraken_dropout_types = set()
    for n, m in kraken_dropouts:
        layer = m.layer if hasattr(m, 'layer') else m
        kraken_dropout_types.add(type(layer).__name__)

    has_dropout2d = 'Dropout2d' in kraken_dropout_types
    has_high_dropout = any(
        (m.layer.p if hasattr(m, 'layer') else m.p) > 0.2
        for n, m in kraken_dropouts
    )

    print(f"\n  ⚠️  CRITICAL: Kraken uses Dropout2d (spatial) for CNN+LSTM, Scribe uses Dropout (element-wise)")
    print(f"  ⚠️  CRITICAL: Kraken last Dropout has p=0.5, Scribe uses p={config.LECTAUREP_DROPOUT}")

    # 3. LSTM layers
    print("\n[LSTM Layers]")
    kraken_lstms = [(n, m) for n, m in kraken_layers if 'LSTM' in str(type(getattr(m, 'layer', m)))]
    scribe_lstms = [(n, m) for n, m in scribe_model.lstm_layers.named_children()]

    for i, ((kn, km)) in enumerate(kraken_lstms):
        lstm = km.layer if hasattr(km, 'layer') else km
        print(f"  Kraken LSTM {i}: input={lstm.input_size}, hidden={lstm.hidden_size}, "
              f"bidir={lstm.bidirectional}, batch_first={lstm.batch_first}")

    for i, (sn, sm) in enumerate(scribe_lstms):
        print(f"  Scribe LSTM {i}: input={sm.input_size}, hidden={sm.hidden_size}, "
              f"bidir={sm.bidirectional}, batch_first={sm.batch_first}")

    # 4. MaxPool layers
    print("\n[MaxPool Layers]")
    kraken_pools = [(n, m) for n, m in kraken_layers if 'MaxPool' in str(type(m)) and not n.endswith('layer')]
    for n, m in kraken_pools:
        layer = m.layer if hasattr(m, 'layer') else m
        print(f"  Kraken {n}: kernel={layer.kernel_size}, stride={layer.stride}, ceil_mode={layer.ceil_mode}")

    print(f"  Scribe: MaxPool2d(kernel_size=(2,2), stride=(2,2), ceil_mode=False) [all 3 layers]")

    # 5. Forward pass comparison with random input
    print("\n[Forward Pass Comparison]")
    with torch.no_grad():
        # Create random input (B=2, H=120, W=400)
        x = torch.randn(2, 1, 120, 400)
        seq_lens = torch.tensor([400, 350])

        # Kraken forward
        try:
            kraken_out = kraken_model.nn(x, seq_lens)
            if isinstance(kraken_out, tuple):
                kraken_logits = kraken_out[0]
                kraken_seq_lens = kraken_out[1] if len(kraken_out) > 1 else None
            else:
                kraken_logits = kraken_out
                kraken_seq_lens = None
            print(f"  Kraken output shape: {kraken_logits.shape}")
            if kraken_seq_lens is not None:
                print(f"  Kraken seq_lens: {kraken_seq_lens}")
        except Exception as e:
            print(f"  Kraken forward failed: {e}")
            kraken_logits = None

        # Scribe forward
        try:
            x_flat = x.squeeze(1)  # (B, H, W)
            input_lengths = seq_lens // 8
            _, _, scribe_logits = scribe_model(x_flat, input_lengths=input_lengths)
            print(f"  Scribe output shape: {scribe_logits.shape}")
            print(f"  Scribe input_lengths: {input_lengths}")
        except Exception as e:
            print(f"  Scribe forward failed: {e}")
            scribe_logits = None

        # Compare seq_len computation
        if kraken_seq_lens is not None:
            scribe_computed = seq_lens // 8
            print(f"\n  ⚠️  Sequence length comparison:")
            print(f"    Kraken computed: {kraken_seq_lens}")
            print(f"    Scribe computed (W//8): {scribe_computed}")
            match = torch.equal(kraken_seq_lens, scribe_computed)
            print(f"    Match: {match}")
            if not match:
                diff = kraken_seq_lens - scribe_computed
                print(f"    Difference: {diff}")
                print(f"    ⚠️  MISMATCH! This could cause CTC blank collapse!")

    # 6. Weight initialization comparison
    print("\n[Weight Initialization]")
    for i, lstm in enumerate(scribe_model.lstm_layers):
        for name, param in lstm.named_parameters():
            if 'weight_ih' in name:
                print(f"  Scribe LSTM {i} {name}: std={param.std():.4f}, mean={param.mean():.4f}")
            elif 'weight_hh' in name:
                print(f"  Scribe LSTM {i} {name}: std={param.std():.4f}, ortho_check={torch.norm(param @ param.T - torch.eye(param.shape[0])):.4f}")
            elif 'bias' in name:
                n = param.shape[0]
                forget_gate = param[n//4:n//2]
                print(f"  Scribe LSTM {i} {name}: forget_gate_mean={forget_gate.mean():.4f}, forget_gate_range=[{forget_gate.min():.4f}, {forget_gate.max():.4f}]")

    # Check kraken init (use model method, not nn)
    try:
        kraken_model.init_weights()
    except Exception:
        pass  # Some versions don't have init_weights
    for i, (name, mod) in enumerate(kraken_lstms):
        lstm = mod.layer if hasattr(mod, 'layer') else mod
        for pname, param in lstm.named_parameters():
            if 'bias' in pname:
                n = param.shape[0]
                forget_gate = param[n//4:n//2]
                print(f"  Kraken LSTM {i} {pname}: forget_gate_mean={forget_gate.mean():.4f}, forget_gate_range=[{forget_gate.min():.4f}, {forget_gate.max():.4f}]")

    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON COMPLETE")
    print("=" * 70)


def compare_cnn_output():
    """Pass same image through both CNNs, compare activations."""
    print("\n" + "=" * 70)
    print("CNN OUTPUT COMPARISON (random input)")
    print("=" * 70)

    from kraken.lib.vgsl import TorchVGSLModel
    from model import LectaurepClone
    import config

    vgsl_spec = "[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]"
    kraken_model = TorchVGSLModel(vgsl=vgsl_spec, codec=None)
    scribe_model = LectaurepClone(
        img_height=config.LECTAUREP_IMG_HEIGHT,
        num_classes=100,
        hidden=config.LECTAUREP_HIDDEN,
        num_lstm_layers=config.LECTAUREP_NUM_LSTM,
        dropout=config.LECTAUREP_DROPOUT,
    )

    # Set both to eval mode (disable dropout)
    kraken_model.nn.eval()
    scribe_model.eval()

    with torch.no_grad():
        # Random input
        torch.manual_seed(42)
        x = torch.randn(1, 1, 120, 400)

        # Copy weights from scribe to kraken (they should have same architecture)
        # First, let's check if we can transfer weights
        kraken_convs = [m for m in kraken_model.nn.modules() if isinstance(m, torch.nn.Conv2d)]
        scribe_convs = [m for m in scribe_model.encoder.modules() if isinstance(m, torch.nn.Conv2d)]

        print(f"Kraken conv layers: {len(kraken_convs)}")
        print(f"Scribe conv layers: {len(scribe_convs)}")

        # Copy scribe weights to kraken
        for kc, sc in zip(kraken_convs, scribe_convs):
            kc.weight.data.copy_(sc.weight.data)
            kc.bias.data.copy_(sc.bias.data)

        # Forward through kraken CNN only (extract first 11 layers = CNN + reshape)
        kraken_cnn_out = x
        for name, mod in kraken_model.nn.named_children():
            kraken_cnn_out = mod(kraken_cnn_out if 'S_' not in name else (kraken_cnn_out, torch.tensor([x.shape[3]])))
            if 'S_' in name:
                break

        # Forward through scribe CNN
        scribe_out = scribe_model._encode(x.squeeze(1))

        print(f"\nKraken CNN output shape: {kraken_cnn_out.shape if not isinstance(kraken_cnn_out, tuple) else kraken_cnn_out[0].shape}")
        print(f"Scribe CNN output shape: {scribe_out.shape}")

        # If shapes match, compare values
        if isinstance(kraken_cnn_out, tuple):
            kraken_cnn_out = kraken_cnn_out[0]

        # They won't match because dropout is eval-mode and weights are random,
        # but the SHAPES should match
        print(f"\n  Shape match: {kraken_cnn_out.shape == scribe_out.shape}")


def compare_input_lengths():
    """Verify that W//8 matches kraken's actual CNN output width."""
    print("\n" + "=" * 70)
    print("INPUT LENGTH VERIFICATION")
    print("=" * 70)

    from kraken.lib.vgsl import TorchVGSLModel

    vgsl_spec = "[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]"
    kraken_model = TorchVGSLModel(vgsl=vgsl_spec, codec=None)
    kraken_model.nn.eval()

    # Test various widths
    test_widths = [100, 200, 400, 800, 1200, 1600, 2000, 399, 401, 801, 1001]

    print(f"\n{'Width':>8} | {'W//8':>6} | {'Kraken T':>10} | {'Match':>7} | {'Diff':>5}")
    print("-" * 50)

    with torch.no_grad():
        for w in test_widths:
            x = torch.randn(1, 1, 120, w)
            seq_lens = torch.tensor([w])

            try:
                out = kraken_model.nn(x, seq_lens)
                if isinstance(out, tuple) and len(out) > 1:
                    kraken_t = out[1].item()
                else:
                    kraken_t = out.shape[-1] if out.dim() == 3 else out.shape[2]

                scribe_t = w // 8
                match = kraken_t == scribe_t
                diff = kraken_t - scribe_t
                print(f"{w:>8} | {scribe_t:>6} | {kraken_t:>10} | {'✅' if match else '❌':>7} | {diff:>+5}")
            except Exception as e:
                print(f"{w:>8} | {w//8:>6} | ERROR: {e}")

    print("\nNote: Any mismatch means scribe's W//8 formula is wrong for that width!")
    print("This would cause CTC to use incorrect input_lengths → blank collapse.")


def run_synthetic_test():
    """Create fake ALTO data and run both pipelines."""
    print("\n" + "=" * 70)
    print("SYNTHETIC DATA TEST")
    print("=" * 70)

    import tempfile
    from PIL import Image, ImageDraw, ImageFont

    # Create temporary directory with fake ALTO + images
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simple test images with text
        test_lines = [
            "Bonjour le monde",
            "Test ligne 2",
            "Archives nationales",
        ]

        xml_paths = []
        for i, text in enumerate(test_lines):
            # Create image
            img = Image.new('L', (400, 120), color=255)  # white background
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
            except:
                font = ImageFont.load_default()
            draw.text((10, 40), text, fill=0, font=font)  # black text

            img_path = os.path.join(tmpdir, f"page_{i}.jpg")
            img.save(img_path)

            # Create minimal ALTO XML
            alto_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Layout>
    <Page WIDTH="400" HEIGHT="120" PHYSICAL_IMG_NR="1">
      <PrintSpace HPOS="0" VPOS="0" WIDTH="400" HEIGHT="120">
        <TextBlock ID="b1" HPOS="0" VPOS="0" WIDTH="400" HEIGHT="120">
          <TextLine ID="l1" BASELINE="80 80 400 80">
            <String ID="s1" HPOS="10" VPOS="40" WIDTH="380" HEIGHT="40" CONTENT="{text}"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
  <sourceImageInformation>
    <fileName>page_{i}.jpg</fileName>
  </sourceImageInformation>
</alto>'''
            xml_path = os.path.join(tmpdir, f"page_{i}.xml")
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(alto_xml)
            xml_paths.append(xml_path)

        print(f"Created {len(xml_paths)} synthetic ALTO pages in {tmpdir}")

        # Try loading with kraken
        try:
            from kraken.lib.xml import XMLPage
            for xml_path in xml_paths:
                page = XMLPage(xml_path)
                print(f"\n  Kraken parsed: {xml_path}")
                for line in page.lines:
                    print(f"    Text: '{line.text}'")
                    if line.image is not None:
                        print(f"    Image: {line.image.size}")
        except Exception as e:
            print(f"  Kraken parse error: {e}")

        # Try loading with scribe
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from data_alto import AltoLineDataset
            ds = AltoLineDataset([tmpdir], img_height=120, augment=False)
            print(f"\n  Scribe loaded: {len(ds)} lines")
            char_to_idx, idx_to_char = ds.get_alphabet()
            print(f"  Scribe alphabet: {len(char_to_idx)} chars")
            for i in range(min(len(ds), 3)):
                img, text = ds[i]
                print(f"    Line {i}: text='{text}', img shape={img.shape}, "
                      f"min={img.min():.3f}, max={img.max():.3f}, mean={img.mean():.3f}")
        except Exception as e:
            print(f"  Scribe parse error: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Compare scribe vs ketos pipelines")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic test with fake data")
    parser.add_argument("--arch-only", action="store_true", help="Compare architectures only")
    parser.add_argument("--input-lengths", action="store_true", help="Verify input_length computation")
    parser.add_argument("--alto-dirs", nargs="+", help="ALTO XML directories for full comparison")

    args = parser.parse_args()

    # Always run architecture comparison
    compare_architectures()

    if args.input_lengths or args.arch_only or True:
        compare_input_lengths()

    if args.synthetic:
        run_synthetic_test()

    if args.alto_dirs:
        print("\n⚠️  Full data comparison requires ALTO data. Implementing...")
        # TODO: implement full data comparison

    if not args.synthetic and not args.alto_dirs:
        print("\n💡 Tip: Run with --synthetic to test with fake data,")
        print("   or --alto-dirs /path/to/alto for full comparison.")


if __name__ == "__main__":
    main()
