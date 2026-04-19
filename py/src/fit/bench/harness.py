"""External benchmark evaluation harness wrappers."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run_swebench(
    endpoint: str,
    dataset: str = "princeton-nlp/SWE-bench_Lite",
    output_path: str = "./predictions.jsonl",
    instance_ids: list[str] | None = None,
    max_workers: int = 4,
) -> dict:
    """Run SWE-bench evaluation. Returns results summary dict."""
    try:
        swebench_harness = importlib.import_module("swebench.harness")
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "swebench is not installed. "
            "Install with: pip install swebench"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    kwargs: dict = {
        "endpoint": endpoint,
        "dataset": dataset,
        "output_path": output_path,
        "max_workers": max_workers,
    }
    if instance_ids is not None:
        kwargs["instance_ids"] = instance_ids

    raw = swebench_harness.run_evaluation(**kwargs)

    resolved = raw.get("resolved", 0)
    total = raw.get("total", 0)
    pct = (resolved / total * 100.0) if total > 0 else 0.0

    return {
        "resolved": resolved,
        "total": total,
        "resolution_pct": pct,
    }


def run_tau(
    endpoint: str,
    domain: str = "retail",
    user_sim_model: str = "gpt-4o-mini",
    max_concurrency: int = 4,
    task_ids: list[int] | None = None,
) -> dict:
    """Run TAU-bench evaluation. Returns results summary dict."""
    try:
        tau2_runner = importlib.import_module("tau2.runner")
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "tau2 is not installed. "
            "Install with: pip install tau-bench"
        )

    kwargs: dict = {
        "domain": domain,
        "endpoint": endpoint,
        "user_sim_model": user_sim_model,
        "max_concurrency": max_concurrency,
    }
    if task_ids is not None:
        kwargs["task_ids"] = task_ids

    result = tau2_runner.run_domain(**kwargs)

    tasks = result.task_results
    total = len(tasks)
    passed = sum(1 for t in tasks if t.reward >= 1.0)
    pass_rate = (passed / total) if total > 0 else 0.0

    return {
        "passed": passed,
        "total": total,
        "pass_rate": pass_rate,
    }


def _collect_results(results_dir: Path) -> list[dict]:
    """Parse .aider.results.json files under *results_dir*."""
    out: list[dict] = []
    for f in sorted(results_dir.glob("**/.aider.results.json")):
        data = json.loads(f.read_text())
        if isinstance(data, list):
            out.extend(data)
        else:
            out.append(data)
    return out


def run_aider(
    endpoint: str,
    exercises_dir: str = "vendor/polyglot-benchmark",
    edit_format: str = "diff",
    languages: list[str] | None = None,
    tries: int = 2,
    threads: int = 10,
) -> dict:
    """Run Aider polyglot benchmark. Returns results summary dict."""
    if shutil.which("aider") is None:
        raise FileNotFoundError(
            "aider not found on PATH. "
            "Install with: pip install aider-chat"
        )

    cmd = [
        sys.executable, "-m", "aider.benchmark",
        "--endpoint", endpoint,
        "--exercises-dir", exercises_dir,
        "--edit-format", edit_format,
        "--tries", str(tries),
        "--threads", str(threads),
    ]
    if languages is not None:
        cmd.extend(["--languages", ",".join(languages)])

    subprocess.run(cmd, check=True)

    entries = _collect_results(Path(exercises_dir))
    total = len(entries)
    if total == 0:
        return {
            "pass_rate_1": 0.0,
            "pass_rate_2": 0.0,
            "total": 0,
            "correct_edit_format_pct": 0.0,
        }

    pass_1 = sum(1 for e in entries if e.get("pass_1"))
    pass_2 = sum(1 for e in entries if e.get("pass_2"))
    fmt_ok = sum(1 for e in entries if e.get("edit_format_correct"))

    return {
        "pass_rate_1": pass_1 / total * 100.0,
        "pass_rate_2": pass_2 / total * 100.0,
        "total": total,
        "correct_edit_format_pct": fmt_ok / total * 100.0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark harness")
    sub = parser.add_subparsers(dest="command")

    swe = sub.add_parser("swe")
    swe.add_argument("--endpoint", required=True)
    swe.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    swe.add_argument("--output", default="./predictions.jsonl")
    swe.add_argument("--instance-ids", nargs="*", default=None)
    swe.add_argument("--max-workers", type=int, default=4)

    tau = sub.add_parser("tau")
    tau.add_argument("--endpoint", required=True)
    tau.add_argument("--domain", default="retail")
    tau.add_argument("--user-sim-model", default="gpt-4o-mini")
    tau.add_argument("--max-concurrency", type=int, default=4)
    tau.add_argument("--task-ids", nargs="*", type=int, default=None)

    aid = sub.add_parser("aider")
    aid.add_argument("--endpoint", required=True)
    aid.add_argument("--exercises-dir", default="vendor/polyglot-benchmark")
    aid.add_argument("--edit-format", default="diff")
    aid.add_argument("--languages", nargs="*", default=None)
    aid.add_argument("--tries", type=int, default=2)
    aid.add_argument("--threads", type=int, default=10)

    args = parser.parse_args()
    if args.command == "swe":
        result = run_swebench(
            endpoint=args.endpoint,
            dataset=args.dataset,
            output_path=args.output,
            instance_ids=args.instance_ids,
            max_workers=args.max_workers,
        )
        print(result)
    elif args.command == "tau":
        result = run_tau(
            endpoint=args.endpoint,
            domain=args.domain,
            user_sim_model=args.user_sim_model,
            max_concurrency=args.max_concurrency,
            task_ids=args.task_ids,
        )
        print(result)
    elif args.command == "aider":
        result = run_aider(
            endpoint=args.endpoint,
            exercises_dir=args.exercises_dir,
            edit_format=args.edit_format,
            languages=args.languages,
            tries=args.tries,
            threads=args.threads,
        )
        print(result)
    else:
        parser.print_help()
