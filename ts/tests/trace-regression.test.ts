import { describe, it, expect } from "vitest";
import { TraceWriter, TraceReader } from "../src/trace.js";
import { mkdtempSync, rmSync, existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import type { Trace, Advice, Reward } from "../src/types.js";

function makeTrace(id: string, sessionId: string): Trace {
  const advice: Advice = {
    domain: "test",
    steering_text: "steer",
    confidence: 0.5,
  };
  const reward: Reward = {
    score: 0.9,
    breakdown: { accuracy: 0.9 },
  };
  return {
    id,
    session_id: sessionId,
    timestamp: new Date().toISOString(),
    input: { prompt: "test" },
    advice,
    frontier: { model: "stub", provider: "test" },
    reward,
  };
}

describe("TraceWriter and TraceReader imports work", () => {
  it("TraceWriter.write creates a file and TraceReader.listSessions lists it", async () => {
    const dir = mkdtempSync(join(tmpdir(), "fit-trace-regression-"));
    try {
      const writer = new TraceWriter(dir);
      const trace = makeTrace("id-1", "sess-001");
      const path = await writer.write(trace, 1);

      expect(existsSync(path)).toBe(true);

      const reader = new TraceReader(dir);
      const sessions = await reader.listSessions();
      expect(sessions).toContain("sess-001");

      const loaded = await reader.read("sess-001", 1);
      expect(loaded.id).toBe("id-1");
      expect(loaded.session_id).toBe("sess-001");
      expect(loaded.advice.domain).toBe("test");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("readdir is importable alongside other fs/promises functions", async () => {
    // This test verifies that the merged import in trace.ts
    // correctly exports TraceWriter and TraceReader which both
    // use fs/promises functions (writeFile, mkdir, readdir, readFile).
    const dir = mkdtempSync(join(tmpdir(), "fit-trace-regression-"));
    try {
      const writer = new TraceWriter(dir);
      await writer.write(makeTrace("id-2", "sess-002"), 1);
      await writer.write(makeTrace("id-3", "sess-003"), 1);

      const reader = new TraceReader(dir);
      const sessions = await reader.listSessions();
      sessions.sort();
      expect(sessions).toEqual(["sess-002", "sess-003"]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
