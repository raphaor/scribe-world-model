"""Throwaway: peak-VRAM of a worst-case v12 full-mode training step."""
import sys
import torch
import config
from model import HWMv12

ckpt = len(sys.argv) > 1 and sys.argv[1] == "ckpt"
B = int(sys.argv[2]) if len(sys.argv) > 2 else 32
device = "cuda"
H, W = config.IMG_HEIGHT_V12, 2000                 # B lines at max width
T = W // 8

model = HWMv12(
    img_height=config.IMG_HEIGHT_V12,
    embedding_dim=config.EMBEDDING_DIM_V12,
    num_layers=config.NUM_LAYERS_V12,
    num_heads=config.NUM_HEADS_V12,
    ff_dim=config.FF_DIM_V12,
    num_classes=130,
    use_checkpoint=ckpt,
).to(device)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
scaler = torch.amp.GradScaler("cuda")

img = torch.rand(B, H, W, device=device)
input_lengths = torch.full((B,), T, dtype=torch.long)
tgt_len = torch.full((B,), 30, dtype=torch.long)
targets = torch.randint(1, 130, (B * 30,), dtype=torch.long)

torch.cuda.reset_peak_memory_stats()
try:
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss, d = model.compute_loss(img, targets, input_lengths, tgt_len)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    alloc = torch.cuda.max_memory_allocated() / 1e9
    resv = torch.cuda.max_memory_reserved() / 1e9
    print(f"batch={B:3d}  checkpoint={str(ckpt):5s}  peak_allocated={alloc:.2f} GB  "
          f"peak_reserved={resv:.2f} GB")
except RuntimeError as e:
    msg = "OUT OF MEMORY" if "out of memory" in str(e).lower() else str(e)[:120]
    print(f"batch={B:3d}  checkpoint={str(ckpt):5s}  -> {msg}")
