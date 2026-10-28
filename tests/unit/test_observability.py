"""Tests for structured logging, the per-request log line and cost (architecture §10)."""

import json
import logging
import sys

import pytest

from app.observability import (
    REDACTED,
    REQUEST_LOGGER,
    JsonFormatter,
    annotate,
    configure_logging,
    llm_cost_usd,
    main,
    read_requests,
    request_log,
    summarize,
)

KEY = "gsk_" + "A1b2C3d4" * 6 + "zz"  # shaped like a real Groq key


def record(message="hello", level=logging.INFO, exc_info=None, **fields) -> logging.LogRecord:
    rec = logging.LogRecord("app.test", level, __file__, 1, message, None, exc_info)
    if fields:
        rec.fields = fields
    return rec


def app_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if getattr(h, "app_json", False)]


# --- Formatting and redaction -------------------------------------------------------------


def test_each_record_is_one_json_object_with_its_fields():
    line = JsonFormatter().format(record("served", request_id="abc", status=200))
    assert "\n" not in line
    entry = json.loads(line)
    assert entry["level"] == "INFO"
    assert entry["logger"] == "app.test"
    assert entry["message"] == "served"
    assert (entry["request_id"], entry["status"]) == ("abc", 200)
    assert entry["ts"].endswith("+00:00")


def test_exceptions_are_included():
    try:
        raise ValueError("boom")
    except ValueError:
        line = JsonFormatter().format(record(level=logging.ERROR, exc_info=sys.exc_info()))
    assert "ValueError: boom" in json.loads(line)["exception"]


@pytest.mark.parametrize(
    "rec",
    [
        record(f"calling Groq with {KEY}"),  # in the message
        record("ok", headers={"Authorization": f"Bearer {KEY}"}),  # in the fields
    ],
)
def test_groq_keys_are_redacted(rec):  # EC-CFG-11
    line = JsonFormatter().format(rec)
    assert KEY not in line
    assert REDACTED in line


def test_redacts_the_configured_secret_even_if_unusual():
    line = JsonFormatter(["not-a-gsk-shaped-secret"]).format(
        record("leaked not-a-gsk-shaped-secret here")
    )
    assert "not-a-gsk-shaped-secret" not in line


def test_redacts_keys_in_tracebacks():
    try:
        raise RuntimeError(f"auth failed for {KEY}")
    except RuntimeError:
        line = JsonFormatter().format(record(level=logging.ERROR, exc_info=sys.exc_info()))
    assert KEY not in line


# --- configure_logging --------------------------------------------------------------------


def test_configure_logging_is_idempotent(make_settings):
    configure_logging(make_settings())
    configure_logging(make_settings(log_level="WARNING"))
    assert len(app_handlers()) == 1
    assert logging.getLogger("app").level == logging.WARNING


def test_logs_go_to_stderr_and_the_log_file(make_settings, tmp_path, capsys):
    log_file = tmp_path / "logs" / "api.jsonl"
    configure_logging(make_settings(log_file=log_file, groq_api_key=KEY))
    logging.getLogger("app.test").info("hello %s", KEY)
    logging.getLogger("app.test").debug("not at INFO")
    stderr_lines = capsys.readouterr().err.splitlines()
    file_lines = log_file.read_text(encoding="utf-8").splitlines()
    for lines in (stderr_lines, file_lines):
        (line,) = [x for x in lines if x.startswith("{")]
        assert json.loads(line)["message"] == f"hello {REDACTED}"


def test_library_warnings_are_formatted_and_redacted(make_settings, capsys):
    configure_logging(make_settings())
    logging.getLogger("groq").info("library chatter stays off")
    logging.getLogger("groq").warning("retrying with %s", KEY)
    err = capsys.readouterr().err
    assert "library chatter" not in err
    assert json.loads(err.strip())["logger"] == "groq"
    assert KEY not in err


# --- request_log --------------------------------------------------------------------------


def request_lines(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == REQUEST_LOGGER]


def test_a_request_scope_writes_exactly_one_line(caplog):
    caplog.set_level(logging.INFO, logger=REQUEST_LOGGER)
    with request_log(request_id="r1", method="POST", path="/recommendations") as entry:
        annotate(candidates=15)
        annotate(llm_status="ok")
        entry["status"] = 200
    (line,) = request_lines(caplog)
    assert line.getMessage() == "POST /recommendations 200"
    assert line.levelno == logging.INFO
    assert line.fields["candidates"] == 15
    assert line.fields["llm_status"] == "ok"
    assert line.fields["duration_ms"] >= 0


def test_annotate_outside_a_request_does_nothing(caplog):
    annotate(candidates=3)  # no error, no line
    assert request_lines(caplog) == []


@pytest.mark.parametrize(
    ("fields", "level"),
    [
        ({"status": 200}, logging.INFO),
        ({"status": 422}, logging.INFO),
        ({"status": 200, "llm": {"error": "API error 429"}}, logging.WARNING),
        ({"status": 200, "llm": {"error": "API error 401", "config_error": True}}, logging.ERROR),
        ({"status": 500}, logging.ERROR),
    ],
)
def test_line_level_reflects_what_happened(caplog, fields, level):  # EC-CFG-05, EC-LLM-05
    caplog.set_level(logging.INFO, logger=REQUEST_LOGGER)
    with request_log(**fields):
        pass
    (line,) = request_lines(caplog)
    assert line.levelno == level


def test_line_is_written_even_if_the_request_fails(caplog):
    caplog.set_level(logging.INFO, logger=REQUEST_LOGGER)
    with pytest.raises(RuntimeError):
        with request_log(request_id="r2"):
            annotate(candidates=1)
            raise RuntimeError("boom")
    (line,) = request_lines(caplog)
    assert line.fields["candidates"] == 1


# --- Cost ---------------------------------------------------------------------------------


def test_cost_uses_input_cached_and_output_prices(make_settings):
    cfg = make_settings()  # $0.15 / $0.075 / $0.60 per million tokens
    usage = {"prompt_tokens": 4000, "cached_tokens": 1000, "completion_tokens": 2000}
    expected = (3000 * 0.15 + 1000 * 0.075 + 2000 * 0.60) / 1_000_000
    assert llm_cost_usd(usage, cfg) == pytest.approx(expected)
    assert llm_cost_usd({}, cfg) == 0


def test_prices_come_from_config(make_settings):
    cfg = make_settings(llm_price_input=1.0, llm_price_cached_input=0.0, llm_price_output=2.0)
    usage = {"prompt_tokens": 1_000_000, "cached_tokens": 500_000, "completion_tokens": 1_000_000}
    assert llm_cost_usd(usage, cfg) == pytest.approx(0.5 + 2.0)


# --- Cost report --------------------------------------------------------------------------


def line(
    status="ok",
    cache_hit=False,
    error=None,
    usage=(4000, 0, 1500, 900),
    cost=0.0015,
    duration_ms=5000,
) -> str:
    """One request log line as the API writes it."""
    keys = ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens")
    llm = {"cache_hit": cache_hit, "usage": dict(zip(keys, usage, strict=True)), "cost_usd": cost}
    if error:
        llm["error"] = error
    entry = {"logger": REQUEST_LOGGER, "llm_status": status, "duration_ms": duration_ms}
    if status != "not_called":
        entry["llm"] = llm
    return json.dumps(entry)


def test_report_averages_over_billed_calls_and_requests():
    lines = [
        "INFO:     Uvicorn running on http://127.0.0.1:8000",  # not JSON: skipped
        '{"logger": "app.main", "message": "Loaded 12415 restaurants"}',  # not a request
        line(cost=0.002, duration_ms=6000),
        line(cost=0.001, usage=(2000, 1000, 1000, 500), duration_ms=4000),
        line(cache_hit=True, usage=(0, 0, 0, 0), cost=0.0, duration_ms=10),
        line(status="fallback", error="API error 429", usage=(0, 0, 0, 0), cost=0.0),
        line(status="not_called"),
    ]
    report = summarize(read_requests(lines))
    assert report["requests"] == 5
    assert (report["ranked_by_llm"], report["cache_hits"]) == (3, 1)
    assert (report["fallback"], report["not_called"]) == (1, 1)
    assert report["billed_calls"] == 2
    assert report["avg_prompt_tokens"] == 3000
    assert report["avg_cached_tokens"] == 500
    assert report["total_cost_usd"] == pytest.approx(0.003)
    assert report["avg_cost_per_billed_call_usd"] == pytest.approx(0.0015)
    assert report["avg_cost_per_request_usd"] == pytest.approx(0.0006)
    assert report["avg_latency_ms_llm_ranked"] == 5000  # cache hits left out
    assert report["max_latency_ms_llm_ranked"] == 6000


def test_report_counts_tokens_spent_by_a_cut_off_answer():
    # finish_reason "length": no usable answer, but the tokens were billed.
    lines = [line(status="fallback", error="finish_reason length", usage=(4000, 0, 8192, 8000))]
    report = summarize(read_requests(lines))
    assert report["billed_calls"] == 1
    assert report["avg_completion_tokens"] == 8192


def test_report_cli(tmp_path, capsys):
    log_file = tmp_path / "api.jsonl"
    log_file.write_text(line() + "\n" + line(cost=0.0025) + "\n", encoding="utf-8")
    assert main([str(log_file)]) == 0
    out = capsys.readouterr().out
    assert "requests" in out and "avg_cost_per_billed_call_usd" in out and "0.002" in out
