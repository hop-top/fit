import { describe, it, expect, vi, beforeEach } from "vitest";
import { RemoteAdvisor } from "../src/advisor.js";
import type { Advice } from "../src/types.js";

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

  it("Advice.version is required by interface (no undefined)", async () => {
    // This line compiles only because version is present.
    // If version were optional, omitting it would also compile;
    // the fact that this fails to compile without version proves
    // the field is required.
    const _advice: Advice = {
      domain: "conformance",
      steering_text: "check",
      confidence: 1,
      version: "1.0",
    };

    // Runtime: version must always be a string, never undefined
    expect(typeof _advice.version).toBe("string");
  });
});
