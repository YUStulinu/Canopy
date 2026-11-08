"""Tests that need no downloaded data: metrics, indices, grids, the network's shapes."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from canopy import baseline, checks, data, metrics
from canopy.data import NODATA
from canopy.dataset import Patches, normalise
from canopy.model import UNet


def test_confusion_and_iou():
    truth = np.array([0, 0, 1, 1, 2, NODATA], np.uint8)
    pred = np.array([0, 1, 1, 1, 2, 0], np.uint8)
    cm = metrics.confusion(truth, pred, classes=3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [0, 0, 1]]
    iou = metrics.iou(cm)
    assert iou == pytest.approx([1 / 2, 2 / 3, 1.0])
    s = metrics.summary(np.pad(cm, (0, 7)), min_pixels=1)
    assert s["miou_classes"] == [0, 1, 2]
    assert s["miou"] == pytest.approx((1 / 2 + 2 / 3 + 1) / 3)
    assert s["accuracy"] == pytest.approx(4 / 5)


def test_absent_classes_do_not_count():
    cm = np.zeros((10, 10), np.int64)
    cm[0, 0] = 100
    cm[1, 1] = 5
    s = metrics.summary(cm, min_pixels=10)
    assert s["miou_classes"] == [0]
    assert s["iou"][2] is None


def test_spectral_indices():
    x = np.zeros((20, 1, 2), np.uint16)
    # pixel 0: vegetation (high NIR), pixel 1: water (high green, low NIR and SWIR)
    for s in (0, 10):
        x[s + 1] = [[500, 900]]   # green
        x[s + 2] = [[400, 300]]   # red
        x[s + 6] = [[4000, 100]]  # NIR (B08)
        x[s + 8] = [[2000, 50]]   # SWIR (B11)
    ind = baseline.indices(x)
    ndvi, ndwi = ind[0, 0], ind[1, 0]
    assert ndvi[0] > 0.8 and ndvi[1] < 0
    assert ndwi[0] < 0 and ndwi[1] > 0.7
    assert baseline.features(x, "indices").shape == (2, 8)
    assert baseline.features(x, "bands+indices").shape == (2, 28)


def test_grid_is_aligned_to_20_m(monkeypatch):
    monkeypatch.setattr(data, "scene_epsg", lambda r: 32634)
    epsg, tr, n = data.grid({"name": "x", "lon": 23.64, "lat": 46.77}, 16)
    assert epsg == 32634 and n == 1600
    assert tr.a == 10 and tr.e == -10 and tr.c % 20 == 0 and tr.f % 20 == 0


def test_worldcover_tile_names():
    assert data.worldcover_tiles((23.5, 46.7, 23.7, 46.8)) == ["N45E021"]
    assert data.worldcover_tiles((23.9, 47.9, 24.1, 48.1)) == ["N45E021", "N45E024", "N48E021", "N48E024"]


def test_unet_shapes():
    m = UNet(width=8).eval()
    with torch.no_grad():
        y = m(torch.zeros(1, 20, 64, 48))
    assert y.shape == (1, 10, 64, 48)


def test_normalise_and_patches():
    x = np.full((20, 40, 40), 1000, np.uint16)
    x[:, 0, 0] = 0
    v = normalise(x, np.full(20, np.log1p(1000.0), np.float32), np.ones(20, np.float32))
    assert v[:, 0, 0].tolist() == [0.0] * 20 and abs(v[:, 5, 5]).max() < 1e-6
    y = np.zeros((40, 40), np.uint8)
    xb, yb = Patches([(v, y)], size=16, count=3)[0]
    assert xb.shape == (20, 16, 16) and yb.shape == (16, 16)


def test_wilson_interval():
    lo, hi = checks.wilson(45, 50)
    assert lo < 0.9 < hi and 0.75 < lo < 0.85 and 0.95 < hi < 0.98
