import { describe, it, expect } from "vitest";
import type { Advice } from "../src/types.js";

describe("Advice type", () => {
  it("has required fields", () => {
    const a: Advice = {
      domain: "tax",
      steering_text: "cite sources",
      confidence: 0.9,
      version: "1.0",
    };
    expect(a.domain).toBe("tax");
    expect(a.confidence).toBeGreaterThanOrEqual(0);
    expect(a.confidence).toBeLessThanOrEqual(1);
  });
});
