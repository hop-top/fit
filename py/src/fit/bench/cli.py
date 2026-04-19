"""CLI entry points for fit bench commands."""

from __future__ import annotations

import argparse
import json
import sys

from . import harness, setup


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fit-bench",
        description="External benchmark evaluation harness",
    )
    sub = p.add_subparsers(dest="command")

    # ── serve ────────────────────────────────────────────────
    srv = sub.add_parser("serve", help="Start advisor proxy server")
    srv.add_argument(
        "--mode", default="plan",
        choices=("plan", "session", "oneshot"),
    )
    srv.add_argument("--advisor", default=None, help="Path to advisor config")
    srv.add_argument("--upstream", required=True, help="Upstream provider")
    srv.add_argument("--port", type=int, default=8781)

    # ── run-swe ──────────────────────────────────────────────
    swe = sub.add_parser("run-swe", help="Run SWE-bench evaluation")
    swe.add_argument("--endpoint", required=True)
    swe.add_argument(
        "--dataset", default="princeton-nlp/SWE-bench_Lite",
    )
    swe.add_argument("--output", default="./predictions.jsonl")
    swe.add_argument("--instance-ids", nargs="*", default=None)
    swe.add_argument("--max-workers", type=int, default=4)

    # ── run-tau ──────────────────────────────────────────────
    tau = sub.add_parser("run-tau", help="Run TAU-bench evaluation")
    tau.add_argument("--endpoint", required=True)
    tau.add_argument("--domain", default="retail")
    tau.add_argument("--user-sim-model", default="gpt-4o-mini")
    tau.add_argument("--max-concurrency", type=int, default=4)
    tau.add_argument("--task-ids", nargs="*", type=int, default=None)

    # ── run-aider ────────────────────────────────────────────
    aid = sub.add_parser("run-aider", help="Run Aider polyglot benchmark")
    aid.add_argument("--endpoint", required=True)
    aid.add_argument(
        "--exercises-dir", default="vendor/polyglot-benchmark",
    )
    aid.add_argument("--edit-format", default="diff")
    aid.add_argument("--languages", nargs="*", default=None)
    aid.add_argument("--tries", type=int, default=2)
    aid.add_argument("--threads", type=int, default=10)

    # ── setup ────────────────────────────────────────────────
    sub.add_parser("setup", help="Check environment readiness")

    return p


def _emit(data: dict) -> None:
    json.dump(data, sys.stdout)
    sys.stdout.write("\n")


def _run_serve(args: argparse.Namespace) -> None:
    print(
        f"mode={args.mode} upstream={args.upstream} "
        f"port={args.port} advisor={args.advisor}\n"
        "HTTP server not yet implemented, use uvicorn directly",
        file=sys.stderr,
    )


def _run_swe(args: argparse.Namespace) -> None:
    print("Running SWE-bench...", file=sys.stderr)
    result = harness.run_swebench(
        endpoint=args.endpoint,
        dataset=args.dataset,
        output_path=args.output,
        instance_ids=args.instance_ids,
        max_workers=args.max_workers,
    )
    _emit(result)


def _run_tau(args: argparse.Namespace) -> None:
    print("Running TAU-bench...", file=sys.stderr)
    result = harness.run_tau(
        endpoint=args.endpoint,
        domain=args.domain,
        user_sim_model=args.user_sim_model,
        max_concurrency=args.max_concurrency,
        task_ids=args.task_ids,
    )
    _emit(result)


def _run_aider(args: argparse.Namespace) -> None:
    print("Running Aider benchmark...", file=sys.stderr)
    result = harness.run_aider(
        endpoint=args.endpoint,
        exercises_dir=args.exercises_dir,
        edit_format=args.edit_format,
        languages=args.languages,
        tries=args.tries,
        threads=args.threads,
    )
    _emit(result)


def _run_setup(args: argparse.Namespace) -> None:
    env = setup.check_environment()
    # table to stderr
    for name, status in env.items():
        print(f"  {name:<25} {status}", file=sys.stderr)
    _emit(env)


_DISPATCH = {
    "serve": _run_serve,
    "run-swe": _run_swe,
    "run-tau": _run_tau,
    "run-aider": _run_aider,
    "setup": _run_setup,
}


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv[1:] if argv else None)

    if not args.command:
        p.print_help(sys.stderr)
        raise SystemExit(2)

    handler = _DISPATCH[args.command]
    try:
        handler(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv)
