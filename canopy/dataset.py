"""Regions on disk as arrays, normalisation, and random training patches."""
from __future__ import annotations

import json

import numpy as np
import torch

from .data import DATA, NODATA, load_regions

LABEL_YEAR = 2021  # WorldCover 2021 labels go with the 2021 composites


def region_names(split: str) -> list[str]:
    return [r["name"] for r in load_regions()["regions"] if r["split"] == split]


def load(name: str, year: int = LABEL_YEAR, with_labels: bool = True):
    x = np.load(DATA / name / f"{year}.npy")
    meta = json.loads((DATA / name / "meta.json").read_text())
    y = None
    if with_labels:
        y = np.load(DATA / name / "labels.npy").copy()
        y[(x == 0).any(axis=0)] = NODATA  # no imagery, no label
    return x, y, meta


def stats(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation of log reflectance over the given regions."""
    acc = []
    for n in names:
        x, _, _ = load(n, with_labels=False)
        valid = (x > 0).all(axis=0)
        acc.append(np.log1p(x[:, valid][:, ::50].astype(np.float32)))
    a = np.concatenate(acc, axis=1)
    return a.mean(axis=1), a.std(axis=1)


def normalise(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """uint16 reflectance (C, H, W) -> float32 standardised log reflectance; no data -> 0."""
    v = (np.log1p(x.astype(np.float32)) - mean[:, None, None]) / std[:, None, None]
    v[:, (x == 0).any(axis=0)] = 0
    return v


class Patches(torch.utils.data.Dataset):
    """Random square crops with random flips and quarter turns."""

    def __init__(self, regions: list[tuple[np.ndarray, np.ndarray]], size: int, count: int, seed: int = 0):
        self.regions, self.size, self.count = regions, size, count
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, i: int):
        rng = self.rng
        while True:
            x, y = self.regions[rng.integers(len(self.regions))]
            _, h, w = x.shape
            r, c = rng.integers(h - self.size + 1), rng.integers(w - self.size + 1)
            yp = y[r : r + self.size, c : c + self.size]
            if (yp != NODATA).mean() > 0.5:
                break
        xp = x[:, r : r + self.size, c : c + self.size]
        k = int(rng.integers(4))
        xp, yp = np.rot90(xp, k, axes=(1, 2)), np.rot90(yp, k)
        if rng.random() < 0.5:
            xp, yp = xp[:, :, ::-1], yp[:, ::-1]
        return torch.from_numpy(xp.copy()), torch.from_numpy(yp.astype(np.int64))
