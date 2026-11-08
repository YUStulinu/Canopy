// Small pure helpers, kept apart from the map so they can be tested.

export const CLASSES = [
  ["tree cover", "#006400"],
  ["shrubland", "#ffbb22"],
  ["grassland", "#ffff4c"],
  ["cropland", "#f096ff"],
  ["built-up", "#fa0000"],
  ["bare / sparse vegetation", "#b4b4b4"],
  ["snow and ice", "#f0f0f0"],
  ["permanent water", "#0064c8"],
  ["herbaceous wetland", "#0096a0"],
  ["moss and lichen", "#fae6a0"],
] as const;

export type Layer = "rgb" | "classes";

export interface Region {
  name: string;
  split: string;
  epsg: number;
  size: number;
  years: number[];
  corners: [number, number][];
  bbox: [number, number, number, number];
}

export function tileUrl(layer: Layer, year: number): string {
  return `/tiles/${layer}/${year}/{z}/{x}/{y}.png`;
}

/** [[west, south], [east, north]] of a region, for fitBounds. */
export function bounds(r: Region): [[number, number], [number, number]] {
  const [w, s, e, n] = r.bbox;
  return [
    [w, s],
    [e, n],
  ];
}

/** Hectares, readably: "0.6 ha", "38 ha", "1,250 ha". */
export function hectares(v: number): string {
  if (v < 10) return `${v.toFixed(1)} ha`;
  return `${Math.round(v).toLocaleString("en-US")} ha`;
}

/** The clip rectangle that shows the "after" map right of the slider. */
export function clipRight(fraction: number, width: number): string {
  const x = Math.round(Math.min(1, Math.max(0, fraction)) * width);
  return `inset(0 0 0 ${x}px)`;
}

/** Regions with two or more years can be compared; the default pair is the first and last. */
export function defaultYears(r: Region): [number, number] {
  const ys = [...r.years].sort((a, b) => a - b);
  return [ys[0], ys[ys.length - 1]];
}
