import { readFile, writeFile, mkdir, readdir } from "node:fs/promises";
import { join } from "node:path";
import type { Trace } from "./types.js";
import * as yaml from "./yaml-stub.js";

export class TraceWriter {
  constructor(private outputDir: string) {}

  async write(trace: Trace, step = 1): Promise<string> {
    const sessionDir = join(this.outputDir, trace.session_id);
    await mkdir(sessionDir, { recursive: true });
    const path = join(sessionDir, `step-${String(step).padStart(3, "0")}.yaml`);
    await writeFile(path, yaml.dump(trace), "utf-8");
    return path;
  }
}

export class TraceReader {
  constructor(private outputDir: string) {}

  async listSessions(): Promise<string[]> {
    try {
      const entries = await readdir(this.outputDir, { withFileTypes: true });
      return entries.filter((e) => e.isDirectory()).map((e) => e.name).sort();
    } catch (err: any) {
      if (err?.code === "ENOENT") return [];
      throw err;
    }
  }

  async read(sessionId: string, step = 1): Promise<Trace> {
    const path = join(
      this.outputDir,
      sessionId,
      `step-${String(step).padStart(3, "0")}.yaml`,
    );
    const content = await readFile(path, "utf-8");
    return yaml.load(content) as Trace;
  }
}
