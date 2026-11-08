"""Downloads Sentinel-2 composites and ESA WorldCover labels for the regions in regions.json.

    python -m canopy.data [--regions name,...] [--years 2021] [--labels]

Imagery: Sentinel-2 L2A from the Earth Search STAC catalogue (Copernicus data, Cloud-Optimised
GeoTIFFs on AWS, no account needed). Only the window of each region is read. For every region and
year, two seasonal composites ("spring" April-June, "summer" July-September): per season, up to
three to eight of the clearest scenes (more while clouds leave gaps), clouds and shadows masked with the scene
classification layer, then the per-pixel median. Ten bands each, 20 channels in all, on a 10 m
grid in the tile's UTM zone.

Labels: ESA WorldCover 2021 v200 (CC BY 4.0), resampled (nearest) onto the same grid.

Writes data/<region>/<year>.npy (uint16, 20 x H x W, reflectance x 10000, 0 = no data),
data/<region>/labels.npy (uint8 class index, 255 = no data) and data/<region>/meta.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform as warp_transform, transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STAC = "https://earth-search.aws.element84.com/v1/search"
WORLDCOVER = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("VSI_CACHE", "TRUE")

# Earth Search asset names of the ten bands used, in channel order.
BANDS = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]
BAND_IDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SEASONS = {"spring": ("04-01", "06-30"), "summer": ("07-01", "09-30")}
CHANNELS = [f"{s}_{b}" for s in SEASONS for b in BAND_IDS]

# WorldCover classes. Mangroves (95) do not occur in Romania; the other ten are kept.
WORLDCOVER_CODES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
CLASSES = ["tree cover", "shrubland", "grassland", "cropland", "built-up", "bare / sparse vegetation",
           "snow and ice", "permanent water", "herbaceous wetland", "moss and lichen"]
COLORS = ["#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000", "#b4b4b4", "#f0f0f0", "#0064c8", "#0096a0", "#fae6a0"]
NODATA = 255

# Scene classification values treated as unusable: saturated, cloud shadow, cloud (medium, high),
# thin cirrus, and no data.
SCL_BAD = {0, 1, 3, 8, 9, 10}


def load_regions() -> dict:
    return json.loads((ROOT / "regions.json").read_text())


def all_regions() -> list[dict]:
    r = load_regions()
    return r["regions"] + [dict(x, split="change") for x in r["change"]]


def utm_epsg(lon: float, lat: float) -> int:
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def scene_epsg(region: dict) -> int:
    """The UTM zone of the Sentinel-2 tiles that cover a region. Near a zone boundary that can be
    the neighbouring zone (tiles overlap the boundary), so it is looked up, not computed; the
    answer is kept in meta.json so every year of a region shares one grid."""
    meta = DATA / region["name"] / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        if "epsg" in m:
            return int(m["epsg"])
    items = search(region, "2021-06-01", "2021-08-31")
    codes = [i["properties"].get("proj:epsg") or i["properties"].get("proj:code") for i in items]
    codes = [int(str(c).replace("EPSG:", "")) for c in codes if c]
    epsg = max(set(codes), key=codes.count) if codes else utm_epsg(region["lon"], region["lat"])
    meta.parent.mkdir(parents=True, exist_ok=True)
    m = json.loads(meta.read_text()) if meta.exists() else {}
    m["epsg"] = epsg
    meta.write_text(json.dumps(m, indent=1))
    return epsg


def grid(region: dict, size_km: float) -> tuple[int, rasterio.Affine, int]:
    """The region's 10 m grid: (EPSG, transform, size in pixels); edges on multiples of 20 m so the
    20 m bands line up."""
    epsg = scene_epsg(region)
    (x,), (y,) = warp_transform("EPSG:4326", f"EPSG:{epsg}", [region["lon"]], [region["lat"]])
    half = size_km * 500
    left = math.floor((x - half) / 20) * 20
    top = math.ceil((y + half) / 20) * 20
    n = int(size_km * 100)
    return epsg, from_origin(left, top, 10, 10), n


def search(region: dict, start: str, end: str, max_cloud: float = 70) -> list[dict]:
    body = {"collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [region["lon"], region["lat"]]},
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "query": {"eo:cloud_cover": {"lt": max_cloud}},
            "limit": 100}
    for attempt in range(5):
        try:
            r = requests.post(STAC, json=body, timeout=90)
            r.raise_for_status()
            return r.json()["features"]
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
    return []


def read_band(href: str, bounds, epsg: int, n: int, resampling=Resampling.bilinear) -> np.ndarray:
    for attempt in range(4):
        try:
            with rasterio.open(href) as ds:
                if ds.crs.to_epsg() != epsg:
                    raise ValueError(f"scene in {ds.crs}, region grid in EPSG:{epsg}")
                w = from_bounds(*bounds, ds.transform)
                return ds.read(1, window=w, out_shape=(n, n), resampling=resampling, boundless=True, fill_value=0)
        except rasterio.errors.RasterioIOError:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def scene_stack(item: dict, bounds, epsg: int, n: int, pool: ThreadPoolExecutor) -> np.ndarray | None:
    """Ten bands of one scene with bad pixels set to 0, or None if mostly unusable."""
    scl = read_band(item["assets"]["scl"]["href"], bounds, epsg, n, Resampling.nearest)  # a class layer
    good = ~np.isin(scl, list(SCL_BAD))
    if good.mean() < 0.3:
        return None
    bands = list(pool.map(lambda b: read_band(item["assets"][b]["href"], bounds, epsg, n), BANDS))
    stack = np.stack(bands)
    if not item["properties"].get("earthsearch:boa_offset_applied", False) and float(item["properties"].get("s2:processing_baseline", "0")) >= 4:
        # Baseline 04.00+ stores reflectance + 1000; Earth Search usually removes it, but not always.
        stack = np.where(stack > 1000, stack - 1000, 0).astype(np.uint16)
    stack[:, ~good] = 0
    stack[:, (stack == 0).any(axis=0)] = 0
    return stack


def composite(region: dict, year: int, size_km: float, log=print) -> tuple[np.ndarray, dict]:
    epsg, tr, n = grid(region, size_km)
    bounds = (tr.c, tr.f - 10 * n, tr.c + 10 * n, tr.f)
    out, used = [], {}
    with ThreadPoolExecutor(10) as pool:
        for season, (a, b) in SEASONS.items():
            items = search(region, f"{year}-{a}", f"{year}-{b}")
            # Scenes of the region's own UTM zone, clearest first; prefer the MGRS tile seen most.
            items = [i for i in items if i["properties"].get("proj:epsg", i["properties"].get("proj:code")) in (epsg, f"EPSG:{epsg}")]
            items.sort(key=lambda i: i["properties"]["eo:cloud_cover"])
            # The catalogue holds some scenes twice (reprocessed: _0 and _1); keep one per tile and day.
            seen, unique = set(), []
            for it in items:
                key = it["id"].rsplit("_", 2)[0]
                if key not in seen:
                    seen.add(key)
                    unique.append(it)
            items = unique
            stacks, ids = [], []
            covered = None
            for it in items:
                # At least three scenes (the median needs them), more while clouds leave holes.
                if len(stacks) >= 8 or (len(stacks) >= 3 and covered.mean() > 0.995):
                    break
                s = scene_stack(it, bounds, epsg, n, pool)
                if s is None:
                    continue
                if (s[0] > 0).mean() < 0.3:
                    continue  # mostly cloud, or the region lies mostly outside this scene's footprint
                stacks.append(s)
                ids.append(it["id"])
                covered = (s[0] > 0) if covered is None else covered | (s[0] > 0)
                log(f"    {region['name']} {year} {season}: {it['id']} cloud {it['properties']['eo:cloud_cover']:.1f}%, valid {(s[0] > 0).mean():.0%}")
            if not stacks:
                raise RuntimeError(f"no usable scene for {region['name']} {year} {season}")
            a3 = np.stack(stacks).astype(np.float32)
            a3[a3 == 0] = np.nan
            with np.errstate(all="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    med = np.nanmedian(a3, axis=0)
            out.append(np.nan_to_num(med, nan=0).round().astype(np.uint16))
            used[season] = ids
    meta = {"epsg": epsg, "transform": list(tr)[:6], "size": n, "scenes": used}
    return np.concatenate(out), meta


def worldcover_tiles(bounds4326) -> list[str]:
    w, s, e, n = bounds4326
    tiles = []
    for lat in range(int(math.floor(s / 3) * 3), int(math.floor(n / 3) * 3) + 1, 3):
        for lon in range(int(math.floor(w / 3) * 3), int(math.floor(e / 3) * 3) + 1, 3):
            tiles.append(f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}")
    return tiles


def labels(region: dict, size_km: float) -> np.ndarray:
    epsg, tr, n = grid(region, size_km)
    bounds = (tr.c, tr.f - 10 * n, tr.c + 10 * n, tr.f)
    b4326 = transform_bounds(f"EPSG:{epsg}", "EPSG:4326", *bounds, densify_pts=21)
    dst = np.zeros((n, n), np.uint8)
    for tile in worldcover_tiles(b4326):
        with rasterio.open(WORLDCOVER.format(tile=tile)) as ds:
            m = 0.01
            w = from_bounds(b4326[0] - m, b4326[1] - m, b4326[2] + m, b4326[3] + m, ds.transform).round_offsets().round_lengths()
            src = ds.read(1, window=w, boundless=True, fill_value=0)
            part = np.zeros_like(dst)
            reproject(src, part, src_transform=ds.window_transform(w), src_crs=ds.crs, dst_transform=tr,
                      dst_crs=f"EPSG:{epsg}", resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
            dst = np.where(dst == 0, part, dst)
    lut = np.full(256, NODATA, np.uint8)
    for k, code in enumerate(WORLDCOVER_CODES):
        lut[code] = k
    return lut[dst]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", help="comma-separated names (default: all)")
    ap.add_argument("--years", default="2021")
    ap.add_argument("--labels", action="store_true", help="also fetch WorldCover labels")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_regions()
    regs = all_regions()
    if a.regions:
        names = set(a.regions.split(","))
        regs = [r for r in regs if r["name"] in names]
    for r in regs:
        d = DATA / r["name"]
        d.mkdir(parents=True, exist_ok=True)
        meta_path = d / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        for year in map(int, a.years.split(",")):
            p = d / f"{year}.npy"
            if p.exists() and not a.force:
                continue
            t = time.time()
            print(f"{r['name']} {year}", flush=True)
            try:
                arr, m = composite(r, year, cfg["size_km"], log=lambda s: print(s, flush=True))
            except Exception as e:  # noqa: BLE001 - one region failing should not stop the others
                print(f"  FAILED: {e}", flush=True)
                continue
            np.save(p, arr)
            meta.update({k: v for k, v in m.items() if k != "scenes"})
            meta.setdefault("scenes", {})[str(year)] = m["scenes"]
            meta.update({"name": r["name"], "lon": r["lon"], "lat": r["lat"], "split": r["split"], "channels": CHANNELS})
            meta_path.write_text(json.dumps(meta, indent=1))
            print(f"  done in {time.time() - t:.0f} s; valid {(arr[0] > 0).mean():.1%}", flush=True)
        if a.labels and r["split"] != "change" and (not (d / "labels.npy").exists() or a.force):
            lab = labels(r, cfg["size_km"])
            np.save(d / "labels.npy", lab)
            counts = np.bincount(lab.ravel(), minlength=256)
            print(f"  labels: " + ", ".join(f"{CLASSES[k]} {counts[k] / lab.size:.1%}" for k in range(10) if counts[k]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
