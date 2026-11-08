"""Confusion matrices and intersection over union."""
from __future__ import annotations

import numpy as np

from .data import NODATA


def confusion(truth: np.ndarray, pred: np.ndarray, classes: int = 10) -> np.ndarray:
    """Rows: truth, columns: prediction. Pixels without a label are left out."""
    keep = truth != NODATA
    t, p = truth[keep].astype(np.int64), pred[keep].astype(np.int64)
    return np.bincount(t * classes + p, minlength=classes * classes).reshape(classes, classes)


def iou(cm: np.ndarray) -> np.ndarray:
    """Per class: TP / (TP + FP + FN); NaN for a class absent from both truth and prediction."""
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(axis=0) + cm.sum(axis=1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, tp / denom, np.nan)


def summary(cm: np.ndarray, min_pixels: int = 0) -> dict:
    """IoU per class, mean IoU over classes with at least `min_pixels` true pixels, accuracy."""
    v = iou(cm)
    support = cm.sum(axis=1)
    counted = (support >= max(min_pixels, 1)) & ~np.isnan(v)
    return {
        "iou": [None if np.isnan(x) else float(x) for x in v],
        "support": [int(s) for s in support],
        "miou": float(np.nanmean(v[counted])) if counted.any() else float("nan"),
        "miou_classes": [int(k) for k in np.flatnonzero(counted)],
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
    }
