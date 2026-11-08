"""Manual checks of the change detector.

    python -m canopy.checks sample [--api http://127.0.0.1:8720] [--per-kind 50]
    python -m canopy.checks sheets          # image chips to look at
    python -m canopy.checks score

`sample` asks the change API (Go) what changed between 2019 and 2025 in the four change regions
and draws a random sample of the patches it reports, per kind. It also draws "missed?" points:
places where vegetation dropped sharply (NDVI down by 0.3 or more, summer to summer) that the
detector did not report, to see what it leaves out.

Every sampled item is then looked at, before and after side by side, and labelled by hand in
results/change_checks.json ("label": "yes" / "no" / "unclear", with a note). `score` turns the
labels into precision per kind, with Wilson 95% intervals.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import requests
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform

from .baseline import nd
from .data import DATA, ROOT, load_regions

FROM, TO = 2019, 2025
CHIP = 64  # pixels: 640 m around the point


def grid(name: str):
    m = json.loads((DATA / name / "meta.json").read_text())
    return m["epsg"], Affine(*m["transform"]), m["size"]


def to_pixel(name: str, lon: float, lat: float) -> tuple[float, float]:
    epsg, tr, _ = grid(name)
    (x,), (y,) = warp_transform("EPSG:4326", f"EPSG:{epsg}", [lon], [lat])
    c, r = ~tr * (x, y)
    return r, c


def sample(api: str, per_kind: int, missed: int, seed: int, kinds: list[str], path, dnbr: float):
    rng = np.random.default_rng(seed)
    items = []
    names = [r["name"] for r in load_regions()["change"]]
    found = {}
    for name in names:
        res = requests.get(f"{api}/api/change", params={"region": name, "from": FROM, "to": TO, "dnbr": dnbr}, timeout=1800).json()
        found[name] = res
        print(name, {k: round(v, 1) for k, v in res["totals_ha"].items()}, len(res["patches"]["features"]), "patches", flush=True)
    for kind in kinds:
        pool = [(n, f) for n in names for f in found[n]["patches"]["features"] if f["properties"]["kind"] == kind]
        pick = rng.choice(len(pool), min(per_kind, len(pool)), replace=False) if pool else []
        for k in sorted(pick):
            n, f = pool[k]
            p = f["properties"]
            items.append({"id": f"{n}/{p['id']}", "region": n, "kind": kind, "area_ha": p["area_ha"], "lon": p["centroid"][0],
                          "lat": p["centroid"][1], "ring": f["geometry"]["coordinates"][0], "label": None, "note": ""})
        print(kind, len(pool), "patches,", len(pick), "sampled", flush=True)
    # Sharp vegetation loss the detector did not report.
    per_region = missed // len(names)
    for n in names if per_region else []:
        a, b = np.load(DATA / n / f"{FROM}.npy"), np.load(DATA / n / f"{TO}.npy")
        ndvi = lambda x: nd(x[16].astype(np.float32), x[12].astype(np.float32))  # summer B08, B04
        drop = (ndvi(a) - ndvi(b)) >= 0.3
        drop &= (a > 0).all(0) & (b > 0).all(0)
        epsg, tr, size = grid(n)
        flagged = np.zeros((size, size), bool)
        for f in found[n]["patches"]["features"]:
            ring = np.array(f["geometry"]["coordinates"][0])
            xs, ys = warp_transform("EPSG:4326", f"EPSG:{epsg}", ring[:, 0].tolist(), ring[:, 1].tolist())
            cs, rs = zip(*[~tr * (x, y) for x, y in zip(xs, ys)])
            flagged[max(0, int(min(rs))) : int(math.ceil(max(rs))), max(0, int(min(cs))) : int(math.ceil(max(cs)))] = True
        cand = np.flatnonzero(drop & ~flagged)
        for k in rng.choice(len(cand), min(per_region, len(cand)), replace=False):
            r, c = divmod(int(cand[k]), size)
            x, y = tr * (c + 0.5, r + 0.5)
            (lon,), (lat,) = warp_transform(f"EPSG:{epsg}", "EPSG:4326", [x], [y])
            items.append({"id": f"{n}/missed-{r}-{c}", "region": n, "kind": "missed?", "lon": lon, "lat": lat, "label": None, "note": ""})
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"from": FROM, "to": TO, "seed": seed, "dnbr_min": dnbr, "totals_ha": {n: found[n]["totals_ha"] for n in names},
                                  "patch_counts": {n: len(found[n]["patches"]["features"]) for n in names}, "items": items}, indent=1))
    print(f"{len(items)} items in {path}")


def rgb(x: np.ndarray) -> np.ndarray:
    v = x[[12, 11, 10]].astype(np.float32) / 10000
    return np.clip((v.transpose(1, 2, 0) / 0.18) ** (1 / 1.6), 0, 1)


def sheets(path, outdir, per_sheet: int = 12):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(path.read_text())
    outdir.mkdir(parents=True, exist_ok=True)
    cache = {}
    items = d["items"]
    for s in range(0, len(items), per_sheet):
        chunk = items[s : s + per_sheet]
        rows = (len(chunk) + 1) // 2
        fig, grid_axes = plt.subplots(rows, 4, figsize=(11, 2.9 * rows), squeeze=False)
        for ax in grid_axes.ravel():
            ax.axis("off")
        for i, it in enumerate(chunk):
            axes_pair = grid_axes[i // 2, 2 * (i % 2) : 2 * (i % 2) + 2]
            n = it["region"]
            if n not in cache:
                cache[n] = (rgb(np.load(DATA / n / f"{FROM}.npy")), rgb(np.load(DATA / n / f"{TO}.npy")))
            r, c = to_pixel(n, it["lon"], it["lat"])
            r0, c0 = int(r) - CHIP // 2, int(c) - CHIP // 2
            for j in range(2):
                img = cache[n][j]
                pad = np.zeros((CHIP, CHIP, 3))
                rr0, cc0 = max(0, r0), max(0, c0)
                rr1, cc1 = min(img.shape[0], r0 + CHIP), min(img.shape[1], c0 + CHIP)
                pad[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0] = img[rr0:rr1, cc0:cc1]
                ax = axes_pair[j]
                ax.axis("on")
                ax.imshow(pad, interpolation="nearest", extent=(c0, c0 + CHIP, r0 + CHIP, r0))
                if "ring" in it:
                    pts = [to_pixel(n, lon, lat) for lon, lat in it["ring"]]
                    ax.plot([p[1] for p in pts], [p[0] for p in pts], color="cyan" if j else "yellow", lw=1)
                else:
                    ax.plot([c], [r], marker="+", color="cyan", ms=14, mew=1.5)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"#{s + i} {it['kind']} {FROM if j == 0 else TO}" + (f" {it['area_ha']:.1f} ha" if j and "area_ha" in it else ""), fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"sheet{s // per_sheet:02d}.png", dpi=80)
        plt.close(fig)
    print("sheets in", outdir)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def score(path) -> dict:
    d = json.loads(path.read_text())
    out = {}
    for kind in ["deforestation", "construction", "missed?"]:
        its = [i for i in d["items"] if i["kind"] == kind]
        yes = sum(i["label"] == "yes" for i in its)
        no = sum(i["label"] == "no" for i in its)
        unclear = sum(i["label"] == "unclear" for i in its)
        lo, hi = wilson(yes, yes + no)
        out[kind] = {"checked": len(its), "yes": yes, "no": no, "unclear": unclear,
                     "share_yes": yes / (yes + no) if yes + no else None, "ci95": [lo, hi]}
    d["score"] = out
    path.write_text(json.dumps(d, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sample", "sheets", "score"])
    ap.add_argument("--api", default="http://127.0.0.1:8720")
    ap.add_argument("--per-kind", type=int, default=50)
    ap.add_argument("--missed", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--kinds", default="deforestation,construction")
    ap.add_argument("--dnbr", type=float, default=0.1, help="the change API's NBR check (0 = off)")
    ap.add_argument("--name", default="change_checks", help="results/<name>.json, docs/<name>/")
    a = ap.parse_args()
    path, outdir = ROOT / "results" / f"{a.name}.json", ROOT / "docs" / a.name
    if a.what == "sample":
        sample(a.api, a.per_kind, a.missed, a.seed, a.kinds.split(","), path, a.dnbr)
    elif a.what == "sheets":
        sheets(path, outdir)
    else:
        score(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
