"""Trains the U-Net on the training regions, keeps the weights that do best on the validation
regions, and exports them to ONNX for the tile server.

    python -m canopy.train [--epochs 60]

Writes models/unet.pt (weights + normalisation), models/unet.onnx, models/norm.json and
results/train_log.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import CLASSES, NODATA, ROOT
from .dataset import Patches, load, normalise, region_names, stats
from .metrics import confusion, summary
from .model import UNet, multiple

MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, dev: torch.device, tile: int = 512, margin: int = 64) -> np.ndarray:
    """Class map of a whole region (normalised C x H x W input), in overlapping tiles so memory
    stays bounded; each tile keeps only its centre, where the network saw enough context."""
    model.eval()
    _, h, w = x.shape
    out = np.zeros((h, w), np.uint8)
    step = tile - 2 * margin
    for r0 in range(0, h, step):
        for c0 in range(0, w, step):
            r1, c1 = max(0, r0 - margin), max(0, c0 - margin)
            r2, c2 = min(h, r0 + step + margin), min(w, c0 + step + margin)
            patch = x[:, r1:r2, c1:c2]
            ph, pw = patch.shape[1:]
            m = multiple()
            H, W = -(-ph // m) * m, -(-pw // m) * m
            buf = np.zeros((x.shape[0], H, W), np.float32)
            buf[:, :ph, :pw] = patch
            t = torch.from_numpy(buf)[None].to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                logits = model(t)
            cls = logits[0, :, :ph, :pw].argmax(0).cpu().numpy().astype(np.uint8)
            rr, cc = min(r0 + step, h), min(c0 + step, w)
            out[r0:rr, c0:cc] = cls[r0 - r1 : rr - r1, c0 - c1 : cc - c1]
    return out


def evaluate(model, regions, dev) -> dict:
    cm = np.zeros((10, 10), np.int64)
    for x, y in regions:
        cm += confusion(y, predict(model, x, dev))
    return summary(cm, min_pixels=1000) | {"confusion": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patches", type=int, default=800, help="patches per epoch")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = device()
    train_names, val_names = region_names("train"), region_names("val")
    mean, std = stats(train_names)

    def prep(names):
        out = []
        for n in names:
            x, y, _ = load(n)
            out.append((normalise(x, mean, std), y))
        return out

    train, val = prep(train_names), prep(val_names)
    # Class weights: inverse square root of frequency, so rare classes count without dominating.
    counts = sum(np.bincount(y[y != NODATA].ravel(), minlength=10) for _, y in train).astype(np.float64)
    freq = counts / counts.sum()
    weights = np.where(counts > 0, 1 / np.sqrt(np.maximum(freq, 1e-6)), 0)
    weights = weights / weights[counts > 0].mean()
    print("training pixels per class:", {CLASSES[k]: int(c) for k, c in enumerate(counts)})

    model = UNet(width=a.width).to(dev)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M on {dev}")
    loader = torch.utils.data.DataLoader(Patches(train, a.size, a.patches, a.seed), batch_size=a.batch, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=dev), ignore_index=NODATA)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    log, best = [], -1.0
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    for epoch in range(a.epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        ev = evaluate(model, val, dev)
        entry = {"epoch": epoch + 1, "loss": total / len(loader), "val_miou": ev["miou"], "val_accuracy": ev["accuracy"], "seconds": round(time.time() - t0)}
        log.append(entry)
        print(entry, flush=True)
        if ev["miou"] > best:
            best = ev["miou"]
            torch.save({"state": model.state_dict(), "mean": mean, "std": std, "width": a.width, "epoch": epoch + 1}, MODELS / "unet.pt")
    (RESULTS / "train_log.json").write_text(json.dumps({"args": vars(a), "class_weights": weights.tolist(), "log": log}, indent=1))
    export()


def load_model(dev=None) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    dev = dev or device()
    ck = torch.load(MODELS / "unet.pt", map_location=dev, weights_only=False)
    model = UNet(width=ck["width"]).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck["mean"], ck["std"]


def export():
    model, mean, std = load_model(torch.device("cpu"))
    x = torch.zeros(1, 20, 256, 256)
    torch.onnx.export(model, x, MODELS / "unet.onnx", input_names=["x"], output_names=["logits"], opset_version=17, dynamo=False)
    (MODELS / "norm.json").write_text(json.dumps({"mean": [float(v) for v in mean], "std": [float(v) for v in std],
                                                    "classes": CLASSES, "tile": 256, "multiple": multiple()}, indent=1))
    print("exported", MODELS / "unet.onnx")


if __name__ == "__main__":
    main()
