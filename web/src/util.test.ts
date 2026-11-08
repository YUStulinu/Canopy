import { describe, expect, it } from "vitest";
import { CLASSES, bounds, clipRight, defaultYears, hectares, tileUrl, type Region } from "./util";

const region: Region = {
  name: "pipera",
  split: "change",
  epsg: 32635,
  size: 1600,
  years: [2025, 2019, 2021],
  corners: [
    [26.04, 44.57],
    [26.24, 44.57],
    [26.24, 44.43],
    [26.04, 44.43],
  ],
  bbox: [26.04, 44.43, 26.24, 44.57],
};

describe("helpers", () => {
  it("lists the ten WorldCover classes with their official colours", () => {
    expect(CLASSES).toHaveLength(10);
    expect(CLASSES[0]).toEqual(["tree cover", "#006400"]);
    expect(CLASSES[4][1]).toBe("#fa0000");
  });
  it("builds tile URLs and bounds", () => {
    expect(tileUrl("classes", 2021)).toBe("/tiles/classes/2021/{z}/{x}/{y}.png");
    expect(bounds(region)).toEqual([
      [26.04, 44.43],
      [26.24, 44.57],
    ]);
  });
  it("formats areas", () => {
    expect(hectares(0.55)).toBe("0.6 ha");
    expect(hectares(38.2)).toBe("38 ha");
    expect(hectares(1250.4)).toBe("1,250 ha");
  });
  it("clips at the slider, within the map", () => {
    expect(clipRight(0.5, 800)).toBe("inset(0 0 0 400px)");
    expect(clipRight(-1, 800)).toBe("inset(0 0 0 0px)");
    expect(clipRight(2, 800)).toBe("inset(0 0 0 800px)");
  });
  it("compares the earliest and latest year by default", () => {
    expect(defaultYears(region)).toEqual([2019, 2025]);
  });
});
