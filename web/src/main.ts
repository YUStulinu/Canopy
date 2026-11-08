import * as maplibregl from "maplibre-gl";
import type { Map as MlMap } from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import "maplibre-gl/dist/maplibre-gl.css";
import "./style.css";
import { CLASSES, bounds, clipRight, defaultYears, hectares, tileUrl, type Layer, type Region } from "./util";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const STYLE = "https://tiles.openfreemap.org/styles/positron";
maplibregl.setWorkerUrl(workerUrl);

function makeMap(container: string): MlMap {
  return new maplibregl.Map({ container, style: STYLE, center: [25, 45.9], zoom: 6, attributionControl: false });
}

const before = makeMap("before");
const after = makeMap("after");
after.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
after.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-left");

// Keep the two cameras together: whichever map the user moves leads.
let syncing = false;
function follow(from: MlMap, to: MlMap) {
  from.on("move", () => {
    if (syncing) return;
    syncing = true;
    to.jumpTo({ center: from.getCenter(), zoom: from.getZoom(), bearing: from.getBearing(), pitch: from.getPitch() });
    syncing = false;
  });
}
follow(before, after);
follow(after, before);

// The slider: the right-hand map is clipped to the right of the handle.
let fraction = 0.5;
function place() {
  const w = $("maps").clientWidth;
  $("after").style.clipPath = clipRight(fraction, w);
  $("handle").style.left = `${fraction * w}px`;
}
let dragging = false;
$("handle").addEventListener("pointerdown", (e) => {
  dragging = true;
  $("handle").setPointerCapture(e.pointerId);
});
$("handle").addEventListener("pointermove", (e) => {
  if (!dragging) return;
  const r = $("maps").getBoundingClientRect();
  fraction = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
  place();
});
$("handle").addEventListener("pointerup", () => (dragging = false));
window.addEventListener("resize", place);
place();

function setLayer(map: MlMap, layer: Layer, year: number) {
  const apply = () => {
    if (map.getLayer("canopy")) map.removeLayer("canopy");
    if (map.getSource("canopy")) map.removeSource("canopy");
    map.addSource("canopy", { type: "raster", tiles: [location.origin + tileUrl(layer, year)], tileSize: 256, minzoom: 8, maxzoom: 18 });
    const firstSymbol = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
    map.addLayer({ id: "canopy", type: "raster", source: "canopy", paint: { "raster-opacity": layer === "classes" ? 0.85 : 1, "raster-resampling": "nearest" } }, firstSymbol);
  };
  if (map.isStyleLoaded()) apply();
  else map.once("load", apply);
}

let regions: Region[] = [];
const sel = <T extends HTMLSelectElement>(id: string) => $<T>(id);

function current(): Region {
  return regions.find((r) => r.name === sel("region").value)!;
}

function refresh() {
  const yb = Number(sel("year-before").value), ya = Number(sel("year-after").value);
  const lb = sel("layer-before").value as Layer, la = sel("layer-after").value as Layer;
  setLayer(before, lb, yb);
  setLayer(after, la, ya);
  const name = (l: Layer) => (l === "rgb" ? "image" : "land cover");
  $("tag-before").textContent = `${yb} · ${name(lb)}`;
  $("tag-after").textContent = `${ya} · ${name(la)}`;
  clearChange();
}

function chooseRegion() {
  const r = current();
  for (const id of ["year-before", "year-after"]) {
    sel(id).innerHTML = r.years.map((y) => `<option>${y}</option>`).join("");
  }
  const [a, b] = defaultYears(r);
  sel("year-before").value = String(a);
  sel("year-after").value = String(b);
  // A single year: compare the image with the predicted land cover instead.
  sel("layer-before").value = "rgb";
  sel("layer-after").value = a === b ? "classes" : "rgb";
  ($("detect") as HTMLButtonElement).disabled = a === b;
  before.fitBounds(bounds(r), { padding: 20, duration: 0 });
  refresh();
}

function clearChange() {
  $("change").innerHTML = "";
  for (const m of [after]) {
    for (const id of ["change-outline", "change-image"]) if (m.getLayer(id)) m.removeLayer(id);
    for (const id of ["change-patches", "change-image"]) if (m.getSource(id)) m.removeSource(id);
  }
}

interface Patch {
  type: "Feature";
  geometry: { type: "Polygon"; coordinates: number[][][] };
  properties: { id: string; kind: string; pixels: number; area_ha: number; centroid: [number, number] };
}

interface ChangeResult {
  totals_ha: Record<string, number>;
  patches: { type: "FeatureCollection"; features: Patch[] };
  overlay: { url: string; corners: [number, number][] };
  min_pixels: number;
}

async function detect() {
  const r = current();
  const from = sel("year-before").value, to = sel("year-after").value;
  $("change").innerHTML = `<p class="muted">Classifying ${r.name} for ${from} and ${to}…</p>`;
  const resp = await fetch(`/api/change?region=${r.name}&from=${from}&to=${to}`);
  if (!resp.ok) {
    $("change").innerHTML = `<p class="error">${await resp.text()}</p>`;
    return;
  }
  const res = (await resp.json()) as ChangeResult;
  clearChange();
  after.addSource("change-image", { type: "image", url: res.overlay.url, coordinates: res.overlay.corners as [[number, number], [number, number], [number, number], [number, number]] });
  after.addLayer({ id: "change-image", type: "raster", source: "change-image", paint: { "raster-resampling": "nearest" } });
  after.addSource("change-patches", { type: "geojson", data: res.patches as unknown as maplibregl.GeoJSONSourceSpecification["data"] });
  after.addLayer({ id: "change-outline", type: "line", source: "change-patches", paint: { "line-color": ["match", ["get", "kind"], "deforestation", "#ff8c00", "#e6005a"], "line-width": 1.2 } });
  const feats = res.patches.features.slice().sort((a, b) => b.properties.area_ha - a.properties.area_ha);
  const rows = feats.slice(0, 12).map((f) => {
    const p = f.properties;
    return `<li><button data-c="${p.centroid.join(",")}" class="${p.kind}">${p.kind === "deforestation" ? "Forest lost" : "New built-up"} · ${hectares(p.area_ha)}</button></li>`;
  });
  $("change").innerHTML = `
    <dl>
      <dt><i class="sw deforestation"></i>Forest lost</dt><dd>${hectares(res.totals_ha.deforestation)}</dd>
      <dt><i class="sw construction"></i>New built-up</dt><dd>${hectares(res.totals_ha.construction)}</dd>
    </dl>
    <p class="muted">${feats.length} patches of at least ${res.min_pixels / 100} ha. Largest:</p>
    <ul class="patches">${rows.join("")}</ul>`;
  $("change").querySelectorAll<HTMLButtonElement>("button[data-c]").forEach((b) =>
    b.addEventListener("click", () => {
      const [lon, lat] = b.dataset.c!.split(",").map(Number);
      after.flyTo({ center: [lon, lat], zoom: 15 });
    }),
  );
}

$("legend").innerHTML = CLASSES.map(([n, c]) => `<li><i style="background:${c}"></i>${n}</li>`).join("");
for (const id of ["year-before", "year-after", "layer-before", "layer-after"]) sel(id).addEventListener("change", refresh);
sel("region").addEventListener("change", chooseRegion);
$("detect").addEventListener("click", () => void detect());

fetch("/tiles/regions")
  .then((r) => r.json())
  .then((rs: Region[]) => {
    regions = rs.sort((a, b) => (a.years.length > 1 ? 0 : 1) - (b.years.length > 1 ? 0 : 1) || a.name.localeCompare(b.name));
    sel("region").innerHTML = regions.map((r) => `<option value="${r.name}">${r.name}${r.years.length > 1 ? " (change)" : ` (${r.split})`}</option>`).join("");
    chooseRegion();
  })
  .catch(() => {
    $("change").innerHTML = `<p class="error">The tile server is not running (cargo run --release in tiles/).</p>`;
  });
