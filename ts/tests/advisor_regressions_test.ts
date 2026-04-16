import { describe, it, expect, vi, beforeEach } from "vitest";
import { RemoteAdvisor } from "../src/advisor.js";

describe("RemoteAdvisor regressions", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("includes version field in returned advice (defaults to 1.0)", async () => {
    const advisor = new RemoteAdvisor("http://localhost:4321");
    const mockData = {
      domain: "tax",
      steering_text: "Be accurate",
      confidence: 0.9,
    };

    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockData),
    } as Response);

    const result = await advisor.generateAdvice({ prompt: "test" });

    // Before fix: version was undefined
    expect(result.version).toBe("1.0");
  });

  it("preserves version from server response", async () => {
    const advisor = new RemoteAdvisor("http://localhost:4321");
    const mockData = {
      domain: "tax",
      steering_text: "Be accurate",
      confidence: 0.9,
      version: "2.1",
    };

    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockData),
    } as Response);

    const result = await advisor.generateAdvice({ prompt: "test" });

    expect(result.version).toBe("2.1");
  });
});
