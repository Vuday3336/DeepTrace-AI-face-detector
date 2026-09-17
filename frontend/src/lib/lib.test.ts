import { describe, expect, it } from "vitest";
import { formatBytes, validateUpload } from "./validation";
import { formatPercent, meterGeometry } from "./verdict";

const file = (type: string, size: number) => new File([new Uint8Array(size)], "x", { type });

describe("validateUpload", () => {
  it("accepts supported images within the limit", () => {
    expect(validateUpload(file("image/jpeg", 1000))).toBeNull();
    expect(validateUpload(file("image/webp", 1000))).toBeNull();
  });
  it("rejects other types, empty and oversized files", () => {
    expect(validateUpload(file("image/gif", 10))).toMatch(/JPG, PNG or WebP/);
    expect(validateUpload(file("image/png", 0))).toMatch(/empty/);
    expect(validateUpload(file("image/png", 10 * 1024 * 1024 + 1))).toMatch(/limit is 10 MB/);
  });
  it("formats bytes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
  });
});

describe("meterGeometry", () => {
  it("places marker, threshold and band as percentages", () => {
    const g = meterGeometry(0.87, 0.62, 0.05);
    expect(g.marker).toBeCloseTo(87);
    expect(g.threshold).toBeCloseTo(62);
    expect(g.bandStart).toBeCloseTo(57);
    expect(g.bandEnd).toBeCloseTo(67);
  });
  it("clamps the band to the bar", () => {
    const g = meterGeometry(0.01, 0.02, 0.1);
    expect(g.bandStart).toBe(0);
    expect(g.marker).toBe(1);
  });
});

describe("formatPercent", () => {
  it("never shows a misleading 0% or 100%", () => {
    expect(formatPercent(0.004)).toBe("<1%");
    expect(formatPercent(0.996)).toBe(">99%");
    expect(formatPercent(0.5)).toBe("50%");
    expect(formatPercent(0)).toBe("0%");
  });
});
