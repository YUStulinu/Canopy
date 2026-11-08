"""Random Forest baselines, pixel by pixel.

Two feature sets:
* "indices": spectral indices only, per season: NDVI (vegetation), NDWI (open water, McFeeters),
  MNDWI (water, with short-wave infrared) and NDBI (built-up): 8 features;
* "bands+indices": the 20 reflectances as well: 28 features.

Trained on a class-balanced sample of pixels from the training regions.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .data import BAND_IDS, NODATA

B = {b: k for k, b in enumerate(BAND_IDS)}


def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (a - b) / (a + b)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def indices(x: np.ndarray) -> np.ndarray:
    """(20, ...) reflectance -> (8, ...) indices, two seasons."""
    x = x.astype(np.float32)
    out = []
    for s in range(2):
        o = 10 * s
        g, r, nir, swir = x[o + B["B03"]], x[o + B["B04"]], x[o + B["B08"]], x[o + B["B11"]]
        out += [nd(nir, r), nd(g, nir), nd(g, swir), nd(swir, nir)]
    return np.stack(out)


def features(x: np.ndarray, kind: str) -> np.ndarray:
    """(C, H, W) uint16 -> (H*W, F) float32."""
    ind = indices(x)
    f = ind if kind == "indices" else np.concatenate([np.log1p(x.astype(np.float32)), ind])
    return f.reshape(f.shape[0], -1).T


def sample(regions: list[tuple[np.ndarray, np.ndarray]], kind: str, per_class: int, rng) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for x, y in regions:
        xs.append(features(x, kind))
        ys.append(y.ravel())
    X, Y = np.concatenate(xs), np.concatenate(ys)
    keep = []
    for k in range(10):
        idx = np.flatnonzero(Y == k)
        if len(idx):
            keep.append(rng.choice(idx, min(per_class, len(idx)), replace=False))
    keep = np.concatenate(keep)
    return X[keep], Y[keep]


def train(regions, kind: str, per_class: int = 20000, seed: int = 0) -> RandomForestClassifier:
    rng = np.random.default_rng(seed)
    X, Y = sample(regions, kind, per_class, rng)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    rf.fit(X, Y)
    return rf


def predict(rf: RandomForestClassifier, x: np.ndarray, kind: str) -> np.ndarray:
    f = features(x, kind)
    out = np.zeros(f.shape[0], np.uint8)
    for s in range(0, len(f), 500_000):
        out[s : s + 500_000] = rf.predict(f[s : s + 500_000])
    return out.reshape(x.shape[1:])


def ignore_nodata(y: np.ndarray) -> np.ndarray:
    return y != NODATA
