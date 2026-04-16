import { describe, it, expect } from "vitest";
import { mkdtempSync, mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

// PR#17 regression: listSessions() must return sorted directory names,
// not filesystem-order names.

describe("listSessions returns sorted results", () => {
  it("returns session directories in alphabetical order regardless of creation order", async () => {
    const dir = mkdtempSync(join(tmpdir(), "fit-trace-sort-"));
    try {
      // Create subdirectories in reverse alphabetical order.
      // readdir may return them in creation/insertion order on some
      // filesystems (e.g., ext4), so this is a meaningful probe.
      const names = ["sess-charlie", "sess-alpha", "sess-bravo"];
      for (const name of names) {
        mkdirSync(join(dir, name));
      }

      const { TraceReader } = await import("../src/trace.js");
      const reader = new TraceReader(dir);
      const sessions = await reader.listSessions();

      expect(sessions).toEqual(["sess-alpha", "sess-bravo", "sess-charlie"]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
