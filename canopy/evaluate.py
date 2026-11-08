"""Per-class IoU on the held-out test regions: the U-Net against two Random Forest baselines.

    python -m canopy.evaluate

Writes results/evaluation.json and docs/iou.png, docs/maps.png.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import baseline
from .data import CLASSES, COLORS, NODATA, ROOT
from .dataset import load, normalise, region_names
from .metrics import confusion, summary
from .train import load_model, predict

RESULTS, DOCS = ROOT / "results", ROOT / "docs"
MIN_PIXELS = 10_000  # a class needs this many true pixels in the test regions to enter the mean IoU


def rgb(x: np.ndarray) -> np.ndarray:
    """Summer true colour, stretched."""
    v = x[[12, 11, 10]].astype(np.float32) / 10000  # summer B04, B03, B02
    return np.clip((v.transpose(1, 2, 0) / 0.18) ** (1 / 1.6), 0, 1)


def colorize(c: np.ndarray) -> np.ndarray:
    lut = np.array([[int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)] for h in COLORS] + [[0, 0, 0]] * 246)
    return lut[c]


def main():
    test, train_names = region_names("test"), region_names("train")
    model, mean, std = load_model()
    import torch

    dev = next(model.parameters()).device
    t = time.time()
    rf_train = [load(n)[:2] for n in train_names]
    rfs = {k: baseline.train(rf_train, k) for k in ["indices", "bands+indices"]}
    print(f"random forests trained in {time.time() - t:.0f} s", flush=True)
    del rf_train
    methods = ["unet", "rf-indices", "rf-bands+indices"]
    cms = {m: np.zeros((10, 10), np.int64) for m in methods}
    per_region, examples = {}, {}
    for name in test:
        x, y, _ = load(name)
        preds = {"unet": predict(model, normalise(x, mean, std), dev)}
        for k, rf in rfs.items():
            preds[f"rf-{k}"] = baseline.predict(rf, x, k)
        per_region[name] = {}
        for m in methods:
            cm = confusion(y, preds[m])
            cms[m] += cm
            per_region[name][m] = summary(cm, MIN_PIXELS)["miou"]
        examples[name] = (rgb(x), y, preds["unet"], preds["rf-bands+indices"])
        print(name, {m: round(v, 3) for m, v in per_region[name].items()}, flush=True)
    out = {"test_regions": test, "min_pixels_for_mean": MIN_PIXELS, "classes": CLASSES, "per_region_miou": per_region}
    for m in methods:
        out[m] = summary(cms[m], MIN_PIXELS) | {"confusion": cms[m].tolist()}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "evaluation.json").write_text(json.dumps(out, indent=1))
    for m in methods:
        print(m, "mIoU", round(out[m]["miou"], 3), "accuracy", round(out[m]["accuracy"], 3))
    figures(out, examples)


def figures(out: dict, examples: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DOCS.mkdir(exist_ok=True)
    support = np.array(out["unet"]["support"])
    ks = [k for k in range(10) if support[k] >= MIN_PIXELS]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    labels = {"unet": "U-Net", "rf-indices": "Random Forest, indices", "rf-bands+indices": "Random Forest, bands + indices"}
    for j, m in enumerate(labels):
        v = [out[m]["iou"][k] or 0 for k in ks]
        ax.bar(np.arange(len(ks)) + (j - 1) * 0.27, v, 0.27, label=f"{labels[m]} (mIoU {out[m]['miou']:.2f})")
    ax.set_xticks(np.arange(len(ks)), [f"{CLASSES[k]}\n{support[k] / 1e6:.2f}M px" for k in ks], fontsize=8)
    ax.set(ylabel="IoU", ylim=(0, 1), title="Held-out test regions")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "iou.png", dpi=130)
    plt.close(fig)

    names = list(examples)
    fig, axes = plt.subplots(len(names), 4, figsize=(12, 3 * len(names)))
    for i, n in enumerate(names):
        im, y, pu, pr = examples[n]
        yy = y.copy()
        for j, (a, title) in enumerate(zip(axes[i], ["Sentinel-2, summer 2021", "WorldCover 2021", "U-Net", "Random Forest"])):
            a.imshow([im, colorize(yy), colorize(pu), colorize(pr)][j], interpolation="nearest")
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title(title, fontsize=10)
        axes[i, 0].set_ylabel(n, fontsize=10)
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=COLORS[k], label=CLASSES[k]) for k in range(10)], loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(DOCS / "maps.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
