"""Trace ingestion from any fit port (Go/TS/Py). Normalizes into TraceRecord."""
from __future__ import annotations

import json
import re
import sqlite3
import types
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_VALID_TABLE_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


@dataclass(frozen=True)
class TraceRecord:
    """Normalized trace record for training pipeline consumption."""

    id: str
    session_id: str
    timestamp: str
    prompt: str
    context: dict[str, Any]
    advice_text: str
    advice_domain: str
    advice_confidence: float
    frontier_output: str
    frontier_model: str
    reward_score: float | None
    reward_breakdown: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_raw(raw: dict[str, Any]) -> TraceRecord:
    """Parse a raw trace dict into a normalized TraceRecord.

    Handles traces from any fit port — all share the same schema.
    Defensively normalizes non-dict field values to empty dicts.
    """
    inp_raw = raw.get("input", {})
    advice_raw = raw.get("advice", {})
    frontier_raw = raw.get("frontier", {})
    reward_raw = raw.get("reward", {})
    metadata_raw = raw.get("metadata", {})

    inp = inp_raw if isinstance(inp_raw, dict) else {}
    advice = advice_raw if isinstance(advice_raw, dict) else {}
    frontier = frontier_raw if isinstance(frontier_raw, dict) else {}
    reward = reward_raw if isinstance(reward_raw, dict) else {}
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

    context = inp.get("context", {})
    if not isinstance(context, dict):
        context = {}

    reward_breakdown = reward.get("breakdown", {})
    if not isinstance(reward_breakdown, dict):
        reward_breakdown = {}

    return TraceRecord(
        id=raw.get("id", ""),
        session_id=raw.get("session_id", ""),
        timestamp=raw.get("timestamp", ""),
        prompt=inp.get("prompt", ""),
        context=context,
        advice_text=advice.get("steering_text", ""),
        advice_domain=advice.get("domain", "unknown"),
        advice_confidence=_safe_float(advice.get("confidence", 0.0)),
        frontier_output=frontier.get("output", ""),
        frontier_model=frontier.get("model", ""),
        reward_score=reward.get("score"),
        reward_breakdown=reward_breakdown,
        metadata=metadata,
    )


@dataclass(frozen=True)
class TraceIngestConfig:
    """Configuration for TraceIngester ingestion behavior.

    Defaults match current hardcoded behavior exactly.

    ``required_keys`` uses **any-of** semantics: a trace is eligible
    if it contains *at least one* of the listed keys, not all of them.
    """

    yaml_glob: str = "*.y*ml"
    required_keys: tuple[str, ...] = ("input", "frontier")
    sqlite_data_column: str = "data"
    metadata_filters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata_filters",
            types.MappingProxyType(self.metadata_filters),
        )


class TraceIngester:
    """Load and filter trace data from JSONL, YAML cassettes, or SQLite."""

    def __init__(self, config: TraceIngestConfig | None = None) -> None:
        self._config = config or TraceIngestConfig()
        self._records: list[TraceRecord] = []

    # -- loaders --

    def load_jsonl(self, path: str | Path) -> TraceIngester:
        """Load traces from a JSONL file (one JSON object per line)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc

                if not isinstance(raw, dict):
                    raise ValueError(
                        f"Invalid trace record at {path}:{line_no}: "
                        f"expected JSON object, got {type(raw).__name__}"
                    )
                self._records.append(_parse_raw(raw))
        return self

    def load_yaml_dir(self, path: str | Path) -> TraceIngester:
        """Load trace YAML files from a directory tree.

        Scans recursively using the glob pattern from
        ``self._config.yaml_glob`` (default ``*.y*ml``) and ingests
        any dict containing at least one of
        ``self._config.required_keys`` (default ``("input", "frontier")``).
        Non-trace YAML (configs, schemas) is skipped automatically.
        """
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f"YAML dir not found: {path}")

        for yaml_file in sorted(path.rglob(self._config.yaml_glob)):
            with yaml_file.open("r", encoding="utf-8") as f:
                try:
                    raw = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    raise ValueError(
                        f"Invalid YAML in {yaml_file}: {exc}"
                    ) from exc
            if isinstance(raw, dict):
                candidates = [raw]
            elif isinstance(raw, list):
                candidates = raw
            else:
                candidates = []
            for item in candidates:
                if isinstance(item, dict) and _has_required_keys(
                    item, self._config.required_keys
                ):
                    self._records.append(_parse_raw(item))
        return self

    def load_sqlite(self, path: str | Path, table: str = "traces") -> TraceIngester:
        """Load traces from a SQLite database table.

        Tries the JSON blob column named by
        ``self._config.sqlite_data_column`` (default ``"data"``) first.
        Falls back to individual columns matching the trace schema
        (``input``, ``advice``, ``frontier``, ``reward``, ``metadata``
        as JSON text).
        """
        if not table or not _VALID_TABLE_RE.match(table):
            raise ValueError(f"Invalid table name: {table!r}")
        col = self._config.sqlite_data_column
        if not col or not _VALID_TABLE_RE.match(col):
            raise ValueError(f"Invalid column name: {col!r}")
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SQLite file not found: {path}")

        db_uri = f"{path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True)
        try:
            conn.row_factory = sqlite3.Row
            # Try JSON blob column first
            try:
                cursor = conn.execute(f"SELECT {col} FROM {table}")
                for row_index, row in enumerate(cursor, start=1):
                    try:
                        raw = json.loads(row[col])
                    except (json.JSONDecodeError, TypeError) as exc:
                        raise ValueError(
                            f"Invalid JSON in SQLite table {table!r} at "
                            f"row {row_index} from {path}: {exc}"
                        ) from exc
                    if not isinstance(raw, dict):
                        raise ValueError(
                            f"Invalid JSON object in SQLite table {table!r} at row "
                            f"{row_index} from {path}: expected decoded 'data' to be "
                            f"a dict, got {type(raw).__name__}"
                        )
                    self._records.append(_parse_raw(raw))
                return self
            except (sqlite3.OperationalError, KeyError):
                pass

            # Fallback: individual columns matching trace schema
            probe = conn.execute(f"SELECT * FROM {table} LIMIT 1")
            cols = [desc[0] for desc in probe.description] if probe.description else []
            probe.close()

            for row in conn.execute(f"SELECT * FROM {table}"):
                raw = dict(zip(cols, row))
                # Decode JSON text columns into dicts
                for key in ("input", "advice", "frontier", "reward", "metadata"):
                    val = raw.get(key)
                    if isinstance(val, str):
                        try:
                            raw[key] = json.loads(val)
                        except (json.JSONDecodeError, ValueError) as exc:
                            raise ValueError(
                                f"Invalid JSON in column {key!r} of "
                                f"SQLite table {table!r} from {path}: {exc}"
                            ) from exc
                self._records.append(_parse_raw(raw))
        finally:
            conn.close()
        return self

    def load_batch(
        self,
        paths: list[str | Path],
        fmt: str = "auto",
    ) -> TraceIngester:
        """Auto-detect format and batch-load from multiple paths.

        For directories in auto mode, all supported formats are loaded
        (YAML, JSONL/NDJSON, JSON) so mixed-format directories work.
        """
        for p in paths:
            p = Path(p)
            if not p.exists():
                continue

            # Directories in auto mode: load all supported formats
            if p.is_dir() and fmt == "auto":
                self.load_yaml_dir(p)
                for jsonl_path in sorted(p.rglob("*.jsonl")):
                    self.load_jsonl(jsonl_path)
                for jsonl_path in sorted(p.rglob("*.ndjson")):
                    self.load_jsonl(jsonl_path)
                for json_path in sorted(p.rglob("*.json")):
                    self.load_batch([json_path], fmt="json")
                continue

            detected = fmt
            if detected == "auto":
                detected = _detect_format(p)

            if detected == "jsonl":
                if p.is_dir():
                    for jsonl_path in sorted(p.rglob("*.jsonl")):
                        self.load_jsonl(jsonl_path)
                    for jsonl_path in sorted(p.rglob("*.ndjson")):
                        self.load_jsonl(jsonl_path)
                else:
                    self.load_jsonl(p)
            elif detected == "yaml":
                if p.is_dir():
                    self.load_yaml_dir(p)
                else:
                    try:
                        with p.open("r", encoding="utf-8") as f:
                            raw = yaml.safe_load(f)
                    except yaml.YAMLError as exc:
                        raise ValueError(
                            f"Invalid YAML in {p}: {exc}"
                        ) from exc
                    if isinstance(raw, dict):
                        if _has_required_keys(raw, self._config.required_keys):
                            self._records.append(_parse_raw(raw))
                    elif isinstance(raw, list):
                        for item in raw:
                            if isinstance(item, dict) and (
                                _has_required_keys(item, self._config.required_keys)
                            ):
                                self._records.append(_parse_raw(item))
            elif detected == "sqlite":
                self.load_sqlite(p)
            elif detected == "json":
                try:
                    with p.open("r", encoding="utf-8") as f:
                        raw = json.load(f)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {p}: {exc}"
                    ) from exc
                if isinstance(raw, list):
                    for idx, item in enumerate(raw):
                        if not isinstance(item, dict):
                            raise ValueError(
                                f"Expected each item in JSON array {p} to be "
                                f"a dict (JSON object), but item at index {idx} is "
                                f"{type(item).__name__}"
                            )
                        if _has_required_keys(item, self._config.required_keys):
                            self._records.append(_parse_raw(item))
                elif isinstance(raw, dict):
                    if _has_required_keys(raw, self._config.required_keys):
                        self._records.append(_parse_raw(raw))
                else:
                    raise ValueError(
                        f"Unexpected top-level JSON type in {p}: "
                        f"expected object or array, got "
                        f"{type(raw).__name__}"
                    )
        return self

    # -- filtering --

    def filter(
        self,
        domain: str | None = None,
        tenant: str | None = None,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
    ) -> TraceIngester:
        """Filter loaded traces by domain, tenant, or time range.

        Returns a new TraceIngester with filtered records (non-mutating).
        Tenant is extracted from metadata.tenant if present.
        Call-time kwargs take precedence over config filters.
        """
        # Merge config metadata filters (config is base, kwargs override).
        effective_domain = domain
        effective_tenant = tenant
        if effective_domain is None:
            effective_domain = self._config.metadata_filters.get("domain")
        if effective_tenant is None:
            effective_tenant = self._config.metadata_filters.get("tenant")

        result = TraceIngester(config=self._config)
        for rec in self._records:
            if effective_domain and rec.advice_domain != effective_domain:
                continue
            if effective_tenant and rec.metadata.get("tenant") != effective_tenant:
                continue
            if since and not _ts_gte(rec.timestamp, since):
                continue
            if until and not _ts_lte(rec.timestamp, until):
                continue
            result._records.append(rec)
        return result

    # -- output --

    def to_trace_records(self) -> list[TraceRecord]:
        """Return all loaded (and optionally filtered) records."""
        return list(self._records)

    def count(self) -> int:
        return len(self._records)


def _has_required_keys(d: dict, keys: tuple[str, ...]) -> bool:
    """Return True if d contains at least one of the required keys."""
    return any(k in d for k in keys)


def _detect_format(path: Path) -> str:
    """Detect trace format from file extension or directory contents."""
    if path.is_dir():
        # Check for YAML cassettes
        if any(path.rglob("*.y*ml")):
            return "yaml"
        # Check for JSONL files recursively
        if any(path.rglob("*.jsonl")) or any(path.rglob("*.ndjson")):
            return "jsonl"
        return "yaml"  # default for dirs

    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    if suffix in (".yaml", ".yml"):
        return "yaml"
    if suffix == ".db":
        return "sqlite"
    if suffix == ".json":
        return "json"
    return "jsonl"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_ts(value: str | datetime) -> datetime:
    """Parse a timestamp string or datetime into datetime."""
    if isinstance(value, datetime):
        return value
    # Handle ISO 8601 with optional Z suffix
    s = value.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def _ts_gte(ts: str, since: str | datetime) -> bool:
    try:
        return _parse_ts(ts) >= _parse_ts(since)
    except (ValueError, TypeError):
        return True


def _ts_lte(ts: str, until: str | datetime) -> bool:
    try:
        return _parse_ts(ts) <= _parse_ts(until)
    except (ValueError, TypeError):
        return True
