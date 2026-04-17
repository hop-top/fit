"""serve_advisor.py -- HTTP advisor service.

Serves a trained advisor over HTTP so Go/TS/other binaries can call
the ``POST /advise`` endpoint via ``RemoteAdvisor``.

Usage:
    python -m examples.serve_advisor --model-path ./advisor-output --port 8080

    # With environment variables:
    ADVISOR_MODEL_PATH=./advisor-output python -m examples.serve_advisor

Endpoints:
    POST /advise  -- accept JSON context, return advice-format-v1 JSON
    GET  /health  -- liveness check
    GET  /model   -- model id and metadata
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fit.advisor import Advisor
from fit.types import Advice


# ---------------------------------------------------------------------------
# File-based advisor (loads config from disk)
# ---------------------------------------------------------------------------

class FileAdvisor(Advisor):
    """Advisor backed by a JSON or YAML config on disk."""

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"model path not found: {path}")

        cfg = self._load_config(path)
        self._domain: str = cfg.get("domain", "general")
        self._steering: str = cfg.get("steering_text", "")
        self._confidence: float = float(cfg.get("confidence", 0.5))
        self._constraints: list[str] = cfg.get("constraints", [])
        self._meta: dict[str, Any] = cfg.get("metadata", {})
        self._model_id: str = self._meta.get("model", f"file:{path}")

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        for name in ("advisor.json", "config.json"):
            candidate = path / name
            if candidate.is_file():
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {candidate}: {exc}"
                    ) from exc
                return data if isinstance(data, dict) else {}

        for name in ("config.yaml", "advisor.yaml"):
            candidate = path / name
            if candidate.is_file():
                try:
                    import yaml

                    try:
                        data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
                    except yaml.YAMLError as exc:
                        raise ValueError(
                            f"Invalid YAML in {candidate}: {exc}"
                        ) from exc
                    return data if isinstance(data, dict) else {}
                except ImportError:
                    print(
                        f"warning: {candidate} found but pyyaml not installed; "
                        "skipping",
                        file=sys.stderr,
                    )

        print(
            f"warning: no config found in {path}; using rule-based fallback",
            file=sys.stderr,
        )
        return {}

    def generate_advice(self, context: dict[str, Any]) -> Advice:
        text = self._steering
        topic = context.get("topic", context.get("prompt", ""))
        if topic:
            text = f"[{topic}] {text}"
        return Advice(
            domain=self._domain,
            steering_text=text,
            confidence=self._confidence,
            constraints=list(self._constraints),
            metadata={**self._meta, "context_keys": list(context.keys())},
        )

    def model_id(self) -> str:
        return self._model_id


# ---------------------------------------------------------------------------
# HTTP handlers (stdlib -- no external deps required)
# ---------------------------------------------------------------------------

def _json_response(code: int, body: Any) -> tuple[dict[str, str], int, bytes]:
    headers = {"Content-Type": "application/json"}
    payload = json.dumps(body, indent=2).encode("utf-8")
    return headers, code, payload


def _build_app(advisor: Advisor) -> Any:
    """Return a WSGI-like callable using stdlib http.server."""

    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        _advisor: Advisor = advisor  # type: ignore[assignment]

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[serve] {fmt % args}", file=sys.stderr)

        def _read_body(self) -> Any:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                hdr, code, body = _json_response(200, {"status": "ok"})
            elif self.path == "/model":
                hdr, code, body = _json_response(
                    200,
                    {"model_id": self._advisor.model_id()},
                )
            else:
                hdr, code, body = _json_response(404, {"error": "not found"})

            self.send_response(code)
            for k, v in hdr.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/advise":
                hdr, code, body = _json_response(404, {"error": "not found"})
            else:
                try:
                    ctx = self._read_body()
                except (json.JSONDecodeError, ValueError) as exc:
                    hdr, code, body = _json_response(
                        400, {"error": f"invalid json: {exc}"}
                    )
                else:
                    if not isinstance(ctx, dict):
                        hdr, code, body = _json_response(
                            400,
                            {"error": "request body must be a JSON object"},
                        )
                    else:
                        advice = self._advisor.generate_advice(ctx)
                        hdr, code, body = _json_response(
                            200, asdict(advice)
                        )

            self.send_response(code)
            for k, v in hdr.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(advisor: Advisor, host: str, port: int) -> None:
    from http.server import HTTPServer

    handler = _build_app(advisor)
    server = HTTPServer((host, port), handler)
    print(f"advisor serving on http://{host}:{port}", file=sys.stderr)
    print(f"  model: {advisor.model_id()}", file=sys.stderr)
    print("  endpoints: POST /advise | GET /health | GET /model", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve a trained advisor over HTTP",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("ADVISOR_MODEL_PATH", "."),
        help="directory with advisor config (default: ADVISOR_MODEL_PATH or .)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="listen host")
    parser.add_argument("--port", type=int, default=8080, help="listen port")
    args = parser.parse_args()

    try:
        advisor = FileAdvisor(args.model_path)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    serve(advisor, args.host, args.port)


if __name__ == "__main__":
    main()
