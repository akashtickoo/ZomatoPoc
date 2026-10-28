"""The evaluation runner (tests/eval/run_eval.py): what it records, and how it keeps
infrastructure trouble out of the quality scores (docs/eval.md §6)."""

from dataclasses import replace

import pytest

from app.llm.client import LLMUnavailableError
from tests.eval.common import RunStopped, Scenario, TokenPacer, read_jsonl, text_sha256
from tests.eval.run_eval import InfraError, RecordingClient, RunContext, run, run_one
from tests.fakes import FakeLLM


class Flaky:
    """Raises the given errors in turn, then answers like FakeLLM."""

    def __init__(self, *errors: Exception, model: str = "fake-model"):
        self.errors = list(errors)
        self.model = model
        self.calls: list[str] = []  # the system prompts it received

    def rank(self, system_prompt: str, user_message: str):
        self.calls.append(system_prompt)
        if self.errors:
            raise self.errors.pop(0)
        return replace(FakeLLM().rank(system_prompt, user_message), model=self.model)


def api_error(status: int, retry_after: float | None = None) -> LLMUnavailableError:
    return LLMUnavailableError(f"API error {status}", status_code=status, retry_after=retry_after)


def recorder(inner, **kwargs) -> tuple[RecordingClient, list[float]]:
    slept: list[float] = []
    client = RecordingClient(inner, expected_model="fake-model", sleep=slept.append, **kwargs)
    return client, slept


def test_rate_limits_and_outages_are_waited_out():
    client, slept = recorder(Flaky(api_error(429, retry_after=7), api_error(503)))
    result = client.rank("sys", '"restaurant_id":"aaaaaaaaaaaa"')
    assert client.result is result and client.error is None
    assert client.attempts == 3
    assert slept == [7.5, 20.5]  # Groq's retry-after, then backoff
    assert client.waited_s == 27


@pytest.mark.parametrize(
    ("errors", "raised", "attempts"),
    [
        ([api_error(429, retry_after=900)], RunStopped, 1),  # the daily limit is spent
        ([api_error(401)], RunStopped, 1),  # a bad key fails every call the same way
        ([LLMUnavailableError("no Groq API key configured")], RunStopped, 1),
        ([api_error(503)] * 6, InfraError, 6),
        ([api_error(413)] * 2, InfraError, 2),  # too large for the per-minute limit
    ],
)
def test_infrastructure_failures_never_become_scores(errors, raised, attempts):
    client, _ = recorder(Flaky(*errors))
    with pytest.raises(raised):
        client.rank("sys", "user")
    assert client.attempts == attempts
    assert client.error is None  # not recorded as a product outcome


def test_bad_answers_are_product_outcomes():
    cut_off = LLMUnavailableError("finish_reason length", finish_reason="length")
    client, slept = recorder(Flaky(cut_off))
    with pytest.raises(LLMUnavailableError):  # the service falls back, as for a user
        client.rank("sys", "user")
    assert client.error is cut_off and client.attempts == 1 and slept == []


def test_wrong_model_is_an_infrastructure_error():
    client, _ = recorder(Flaky(model="some-other-model"))
    with pytest.raises(InfraError, match="served model some-other-model"):
        client.rank("sys", "user")


def test_prompt_override_and_pacing():
    now = [0.0]
    pacer = TokenPacer(8000, clock=lambda: now[0], sleep=lambda s: None)
    inner = Flaky()
    client, _ = recorder(inner, system_prompt_override="prompt v2", pacer=pacer)
    client.rank("prompt v1", "user")
    assert inner.calls == ["prompt v2"] and client.system_prompt == "prompt v2"
    assert pacer._log[0][1] == 5500  # FakeLLM's 4,000 prompt + 1,500 completion tokens


# --- One scenario run --------------------------------------------------------------------


@pytest.fixture
def records(make_record):
    return [make_record(f"R{i}", rating=round(4.6 - i * 0.1, 1), votes=500) for i in range(1, 5)]


@pytest.fixture
def make_ctx(make_repo, make_settings, records):
    repo = make_repo(records)

    def _make(mode="llm", llm=None) -> RunContext:
        return RunContext(
            repo=repo,
            cfg=make_settings(llm_model="fake-model"),
            mode=mode,
            system_prompt="SYS",
            llm=llm if llm is not None else FakeLLM(),
            sleep=lambda s: None,
        )

    return _make


def scenario(inject: str | None = None, **prefs) -> Scenario:
    data = {
        "id": "T1",
        "tags": ["baseline"],
        "preferences": {"location": "Indiranagar", "top_n": 2} | prefs,
        "expectations": {"expect_relaxation": False},
    }
    if inject:
        data["inject_review"] = {"restaurant_id": inject, "fixture": "injection_review.json"}
    return Scenario.model_validate(data)


def test_a_run_records_the_raw_answer_and_the_final_response(make_ctx, records):
    kind, row = run_one(scenario(), 1, "main", make_ctx())
    assert kind == "result" and row["status"] == "ok"
    ids = [r["restaurant_id"] for r in records]
    assert row["candidate_ids"] == ids  # pre-rank order, as sent to the model
    # FakeLLM ranks all four in reverse; the validator keeps top_n of them.
    assert [r["restaurant_id"] for r in row["raw_llm"]["recommendations"]] == ids[::-1]
    assert [i["restaurant_id"] for i in row["response"]["recommendations"]] == ids[:1:-1]
    assert row["prompt"]["system_sha256"] == text_sha256("SYS")[:12]
    assert "<candidates>" in row["prompt"]["user_message"]
    assert row["llm"]["usage"]["prompt_tokens"] == 4000 and row["llm"]["cost_usd"] > 0
    assert row["llm"]["model"] == "fake-model" and row["llm"]["attempts"] == 1
    baseline = row["baseline_response"]["recommendations"]
    assert [i["source"] for i in baseline] == ["fallback", "fallback"]  # for the J4 judge


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (LLMUnavailableError("finish_reason length", finish_reason="length"), "truncated"),
        (LLMUnavailableError("response did not match the expected schema"), "fallback"),
    ],
)
def test_failed_answers_are_recorded_with_their_status(make_ctx, error, status):
    llm = FakeLLM()
    llm.error = error
    kind, row = run_one(scenario(), 1, "main", make_ctx(llm=llm))
    assert (kind, row["status"]) == ("result", status)
    assert row["llm"]["error"] == error.reason and row["raw_llm"] is None
    assert all(i["source"] == "fallback" for i in row["response"]["recommendations"])


def test_no_candidates_means_no_call_in_any_mode(make_ctx, make_repo, make_record):
    for mode in ("llm", "no_llm"):
        ctx = make_ctx(mode=mode)
        ctx.repo = make_repo([make_record("Only", rating=3.0)])  # below 5.0 even relaxed
        kind, row = run_one(scenario(min_rating=5.0), 1, "main", ctx)
        assert row["status"] == "not_called" and row["llm"] is None
        assert row["response"]["recommendations"] == []


def test_deterministic_mode(make_ctx):
    llm = FakeLLM()
    _, row = run_one(scenario(), 1, "main", make_ctx(mode="no_llm", llm=llm))
    assert row["status"] == "no_llm" and llm.calls == 0
    assert row["prompt"] is None and row["baseline_response"] is None


def test_outages_go_to_errors_not_results(make_ctx):
    llm = FakeLLM()
    llm.error = api_error(503)
    kind, row = run_one(scenario(), 1, "main", make_ctx(llm=llm))
    assert kind == "error" and "after 6 attempts" in row["error"]


def test_injected_review_is_in_memory_and_main_only(make_ctx, records):
    target = records[2]["restaurant_id"]
    ctx = make_ctx()
    s = scenario(inject=target)
    _, main = run_one(s, 1, "main", ctx)
    _, control = run_one(s, 1, "control", ctx)
    assert "ignore your ranking rules" in main["prompt"]["user_message"]
    assert "ignore your ranking rules" not in control["prompt"]["user_message"]
    (original,) = ctx.repo.get_by_ids([target])
    assert original.review_snippets == []  # the shared data is untouched


def test_run_appends_rows_and_stops_cleanly(make_ctx, tmp_path):
    llm = Flaky(api_error(429, retry_after=900))  # the first call hits the daily limit
    ctx = make_ctx(llm=llm)
    tasks = [(scenario(), 1, "main"), (scenario(), 2, "main"), (scenario(), 3, "main")]
    with pytest.raises(RunStopped):
        run(tasks, ctx, tmp_path)
    assert read_jsonl(tmp_path / "results.jsonl") == []

    run(tasks[:2], ctx, tmp_path)  # resumed later
    rows = read_jsonl(tmp_path / "results.jsonl")
    assert [(r["scenario_id"], r["rep"]) for r in rows] == [("T1", 1), ("T1", 2)]
