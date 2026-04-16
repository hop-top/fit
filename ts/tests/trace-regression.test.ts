import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, existsSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import type { Trace, Advice, Reward } from "../src/types.js";

function makeTrace(id: string, sessionId: string): Trace {
  const advice: Advice = {
    domain: "test",
    steering_text: "steer",
    confidence: 0.5,
    version: "1.0",
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
      const { TraceWriter, TraceReader } = await import("../src/trace.js");
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
    const { TraceWriter, TraceReader } = await import("../src/trace.js");
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

// PR#11 regression: listSessions() must only catch ENOENT, not all errors.
//
// Before fix: catch {} swallowed every error (EACCES, EMFILE, etc.)
// returning []. After fix: only ENOENT returns []; other errors rethrow.

// We must hoist vi.mock so it runs before the module is imported.
// Use a factory that delegates to a mutable variable so each test
// can control the behavior.
let _readdirMock:
  | ((...args: unknown[]) => Promise<unknown>)
  | null = null;

vi.mock("node:fs/promises", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs/promises")>();
  return {
    ...actual,
    readdir: (...args: unknown[]) => {
      if (_readdirMock) return _readdirMock(...args);
      return actual.readdir(...args);
    },
  };
});

describe("listSessions error handling", () => {
  afterEach(() => {
    _readdirMock = null;
  });

  it("rethrows non-ENOENT errors instead of returning []", async () => {
    _readdirMock = async () => {
      const err = new Error("permission denied");
      (err as any).code = "EACCES";
      throw err;
    };
    const { TraceReader } = await import("../src/trace.js");
    const reader = new TraceReader("/tmp");
    await expect(reader.listSessions()).rejects.toThrow("permission denied");
  });

  it("returns [] for ENOENT (dir not found)", async () => {
    _readdirMock = async () => {
      const err = new Error("ENOENT: no such file");
      (err as any).code = "ENOENT";
      throw err;
    };
    const { TraceReader } = await import("../src/trace.js");
    const reader = new TraceReader("/tmp");
    const sessions = await reader.listSessions();
    expect(sessions).toEqual([]);
  });
});
