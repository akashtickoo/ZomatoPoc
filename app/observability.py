"""Structured logging, one log line per request, and LLM cost (architecture §10).

Every log record is written as one JSON object per line, with Groq API keys scrubbed
(EC-CFG-11). Each request also adds exactly one line from the "app.request" logger: the
request middleware opens a `request_log()` scope, the code handling the request adds
fields with `annotate()`, and the line is written when the scope closes.

Summarize a log file (requests, tokens, cost and latency):
    python -m app.observability logs/api.jsonl
"""

import argparse
import contextvars
import json
import logging
import re
import statistics
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings, settings
from app.llm.client import USAGE_KEYS

REQUEST_LOGGER = "app.request"
REDACTED = "[REDACTED]"
GROQ_KEY = re.compile(r"gsk_[A-Za-z0-9]{8,}")  # real keys are gsk_ plus 52 characters

_current: contextvars.ContextVar[dict | None] = contextvars.ContextVar("request_log", default=None)


class JsonFormatter(logging.Formatter):
    """One JSON object per record. Known secrets and anything shaped like a Groq key are
    replaced in the finished line, wherever they came from (EC-CFG-11)."""

    def __init__(self, secrets: Iterable[str] = ()):
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return self.redact(json.dumps(entry, ensure_ascii=False, default=str))

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        return GROQ_KEY.sub(REDACTED, text)


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever sys.stderr is now, so a replaced stream (pytest swaps it per
    test) never leaves the handler writing to a closed one."""

    def __init__(self):
        logging.Handler.__init__(self)

    @property
    def stream(self):
        return sys.stderr


def configure_logging(cfg: Settings = settings) -> None:
    """JSON lines to stderr, and to LOG_FILE if set. Safe to call more than once.

    The handlers go on the root logger, so library warnings are formatted and scrubbed
    too; only the app's own loggers are opened up to LOG_LEVEL.
    """
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, "app_json", False)]:
        root.removeHandler(handler)
        handler.close()
    secret = cfg.groq_api_key.get_secret_value() if cfg.groq_api_key else ""
    formatter = JsonFormatter([secret])
    handlers: list[logging.Handler] = [_StderrHandler()]
    if cfg.log_file:
        cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(cfg.log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.app_json = True
        root.addHandler(handler)
    logging.getLogger("app").setLevel(cfg.log_level)


@contextmanager
def request_log(**fields) -> Iterator[dict]:
    """Collect fields for one request; write them as one line when the scope closes.

    Set "status" for the HTTP status, and "exc_info" to attach a traceback.
    """
    entry = dict(fields)
    token = _current.set(entry)
    start = time.perf_counter()
    try:
        yield entry
    finally:
        _current.reset(token)
        entry["duration_ms"] = round((time.perf_counter() - start) * 1000)
        exc_info = entry.pop("exc_info", None)
        logging.getLogger(REQUEST_LOGGER).log(
            _level(entry), _message(entry), exc_info=exc_info, extra={"fields": entry}
        )


def annotate(**fields) -> None:
    """Add fields to the current request's log line; does nothing outside a request.

    The middleware's context is copied into the worker thread that runs an endpoint, and
    the copy holds the same dict, so fields added there reach the line.
    """
    entry = _current.get()
    if entry is not None:
        entry.update(fields)


def _level(entry: dict) -> int:
    llm = entry.get("llm") or {}
    if entry.get("status", 200) >= 500 or llm.get("config_error"):
        return logging.ERROR  # needs someone to fix something (EC-CFG-05, EC-LLM-05)
    if llm.get("error"):
        return logging.WARNING  # a passing outage; the fallback ranking was served
    return logging.INFO


def _message(entry: dict) -> str:
    parts = [str(entry[key]) for key in ("method", "path", "status") if key in entry]
    return " ".join(parts) or "request"


def llm_cost_usd(usage: dict[str, int], cfg: Settings = settings) -> float:
    """What one LLM call cost, from its token usage and the configured prices.

    Cached prompt tokens are billed at the cached rate, the rest of the prompt at the
    input rate; reasoning tokens are part of completion_tokens.
    """
    cached = usage.get("cached_tokens", 0)
    uncached = usage.get("prompt_tokens", 0) - cached
    completion = usage.get("completion_tokens", 0)
    dollars = (
        uncached * cfg.llm_price_input
        + cached * cfg.llm_price_cached_input
        + completion * cfg.llm_price_output
    ) / 1_000_000
    return round(dollars, 8)


def read_requests(lines: Iterable[str]) -> list[dict]:
    """The recommendation request lines in a log; other lines and non-JSON are skipped."""
    found = []
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("logger") == REQUEST_LOGGER and "llm_status" in entry:
            found.append(entry)
    return found


def summarize(requests: list[dict]) -> dict:
    """Counts, average tokens, cost and latency over request log lines.

    A "billed call" is one that reached Groq and spent tokens: cache hits and failures
    before a response arrived are not.
    """
    llm = [r["llm"] for r in requests if r.get("llm")]
    billed = [x for x in llm if not x.get("cache_hit") and any(x["usage"].values())]
    ranked = [r for r in requests if r["llm_status"] == "ok" and not r["llm"].get("cache_hit")]
    total_cost = sum(x["cost_usd"] for x in llm)

    def mean(values: list[float]) -> float | None:
        return round(statistics.fmean(values), 1) if values else None

    return {
        "requests": len(requests),
        "ranked_by_llm": sum(r["llm_status"] == "ok" for r in requests),
        "cache_hits": sum(bool(x.get("cache_hit")) for x in llm),
        "fallback": sum(r["llm_status"] == "fallback" for r in requests),
        "not_called": sum(r["llm_status"] == "not_called" for r in requests),
        "billed_calls": len(billed),
        **{f"avg_{key}": mean([x["usage"][key] for x in billed]) for key in USAGE_KEYS},
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_billed_call_usd": round(total_cost / len(billed), 6) if billed else None,
        "avg_cost_per_request_usd": round(total_cost / len(requests), 6) if requests else None,
        "avg_latency_ms_llm_ranked": mean([r["duration_ms"] for r in ranked]),
        "max_latency_ms_llm_ranked": max((r["duration_ms"] for r in ranked), default=None),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize request cost and latency from a log.")
    parser.add_argument("log_file", type=Path, help="a file written with LOG_FILE set")
    args = parser.parse_args(argv)
    with args.log_file.open(encoding="utf-8") as f:
        report = summarize(read_requests(f))
    for key, value in report.items():
        print(f"{key:30} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
