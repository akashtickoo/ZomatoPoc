"""The LLM judge (tests/eval/judge.py): what each metric is shown, how replies are parsed,
and that unusable replies are never counted as a pass or a fail (docs/eval.md §5)."""

import json
from types import SimpleNamespace

import groq
import httpx
import pytest

from app.llm.prompt_builder import user_message
from app.models.preferences import UserPreferences
from tests.eval.common import RunStopped, Scenario
from tests.eval.judge import (
    JudgeClient,
    JudgeSettings,
    Preference,
    Verdict,
    build_jobs,
    records_for,
    run_job,
)

REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


class FakeGroq:
    """Plays back scripted replies: a string is the reply's content, an exception is raised."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=900, completion_tokens=40),
        )


def status_error(status: int, headers=None, code=None) -> groq.APIStatusError:
    body = {"error": {"code": code, "message": "x"}} if code else None
    response = httpx.Response(status, request=REQUEST, headers=headers or {})
    return groq.APIStatusError("x", response=response, body=body)


@pytest.fixture
def judge(monkeypatch):
    for name in ("EVAL_JUDGE_MODEL", "EVAL_JUDGE_REASONING_EFFORT"):
        monkeypatch.delenv(name, raising=False)

    def _make(*replies) -> tuple[JudgeClient, FakeGroq, list[float]]:
        fake, slept = FakeGroq(*replies), []
        client = JudgeClient(JudgeSettings(_env_file=None), client=fake, sleep=slept.append)
        return client, fake, slept

    return _make


def test_a_verdict_is_parsed_and_the_request_is_json_mode(judge):
    client, fake, _ = judge('{"verdict": " Pass ", "reason": "All supported."}')
    call = client.ask("system", "user", Verdict)
    assert call.parsed == Verdict(verdict="pass", reason="All supported.")
    assert call.usage == {"prompt_tokens": 900, "completion_tokens": 40}
    (request,) = fake.requests
    assert request["model"] == "qwen/qwen3.8-27b"  # not the gpt-oss family under test
    assert request["response_format"] == {"type": "json_object"}
    assert request["temperature"] == 0 and request["reasoning_effort"] == "none"


@pytest.mark.parametrize("content", ["not json", '{"verdict": "maybe", "reason": "?"}', ""])
def test_unusable_replies_are_judge_errors(judge, content):
    client, _, _ = judge(content)
    call = client.ask("s", "u", Verdict)
    assert call.parsed is None and "schema" in call.error


def test_invalid_json_rejected_by_groq_is_a_judge_error(judge):
    client, _, _ = judge(status_error(400, code="json_validate_failed"))
    assert client.ask("s", "u", Verdict).error == "invalid JSON from the judge"


def test_rate_limits_are_waited_out_and_long_waits_stop(judge):
    client, _, slept = judge(
        status_error(429, {"retry-after": "3"}), '{"winner": "b", "reason": "r"}'
    )
    assert client.ask("s", "u", Preference).parsed.winner == "B"
    assert slept == [3.5]
    client, _, _ = judge(status_error(429, {"retry-after": "1800"}))
    with pytest.raises(RunStopped):
        client.ask("s", "u", Verdict)
    client, _, _ = judge(status_error(401))
    with pytest.raises(RunStopped):
        client.ask("s", "u", Verdict)


# --- What gets judged --------------------------------------------------------------------


@pytest.fixture
def setup(make_repo, make_record):
    records = [make_record(f"R{i}", review_snippets=[f"Review of R{i}."]) for i in range(1, 4)]
    repo = make_repo(records)
    prefs = UserPreferences(location="Indiranagar", top_n=3, additional_preferences="rooftop")
    scenario = Scenario.model_validate(
        {
            "id": "T1",
            "tags": ["free_text_unverifiable"],
            "preferences": prefs.model_dump(),
            "expectations": {"expect_relaxation": False, "free_text_unverifiable": ["rooftop"]},
        }
    )
    candidates = repo.get_by_ids([r["restaurant_id"] for r in records])

    def item(r, rank, source="llm"):
        return {
            "rank": rank,
            "restaurant_id": r.restaurant_id,
            "name": r.name,
            "location": r.location,
            "cuisines": r.cuisines,
            "rating": r.rating,
            "votes": r.votes,
            "cost_for_two": r.cost_for_two,
            "rest_type": r.rest_type,
            "book_table": r.book_table,
            "online_order": r.online_order,
            "popular_dishes": [],
            "explanation": f"About {r.name}.",
            "source": source,
        }

    def row(status="ok", variant="main"):
        items = [item(candidates[0], 1), item(candidates[1], 2), item(candidates[2], 3, "fallback")]
        if status != "ok":
            items = [item(c, i, "fallback") for i, c in enumerate(candidates, 1)]
        return {
            "scenario_id": "T1",
            "rep": 1,
            "variant": variant,
            "status": status,
            "preferences": prefs.model_dump(),
            "candidate_ids": [c.restaurant_id for c in candidates],
            "prompt": {"user_message": user_message(prefs, candidates, [])},
            "response": {"recommendations": items, "relaxations": []},
            "baseline_response": {
                "recommendations": [item(c, i, "fallback") for i, c in enumerate(candidates, 1)]
            },
        }

    return SimpleNamespace(repo=repo, scenarios={"T1": scenario}, row=row, candidates=candidates)


def test_llm_text_in_successful_rows_is_judged(setup):
    jobs = build_jobs([setup.row()], setup.scenarios, setup.repo, "run-1")
    assert sorted(j.metric for j in jobs) == [
        "J1",
        "J1",
        "J2",
        "J3",
        "J3",
        "J4",
    ]  # not the backfill
    first_j1 = next(j for j in jobs if j.metric == "J1")
    # J1 sees only the record of the restaurant it explains, as the model saw it.
    assert "Review of R1." in first_j1.user and "Review of R2." not in first_j1.user
    assert "low is up to ₹300" in first_j1.system


def test_fallback_and_control_rows_are_skipped_unless_asked(setup):
    rows = [setup.row("fallback"), setup.row(variant="control")]
    assert build_jobs(rows, setup.scenarios, setup.repo, "run-1") == []
    jobs = build_jobs(rows, setup.scenarios, setup.repo, "run-1", include_fallback=True)
    assert sorted(j.metric for j in jobs) == ["J1"] * 3 + ["J2"] + ["J3"] * 3  # null baseline


def test_records_come_from_the_saved_prompt(setup):
    row = setup.row()
    records = records_for(row, setup.repo)
    assert list(records) == row["candidate_ids"]
    no_prompt = row | {"prompt": None}
    assert records_for(no_prompt, setup.repo) == records


def test_pairwise_order_is_random_but_reproducible_and_mapped_back(setup, judge):
    def j4(run_id):
        jobs = build_jobs([setup.row()], setup.scenarios, setup.repo, run_id, ("J4",))
        return jobs[0]

    assert j4("run-1").llm_side == j4("run-1").llm_side
    sides = {j4(f"run-{n}").llm_side for n in range(20)}
    assert sides == {"A", "B"}

    job = j4("run-1")
    other = "B" if job.llm_side == "A" else "A"
    for winner, verdict in [(job.llm_side, "llm"), (other, "baseline"), ("tie", "tie")]:
        client, fake, _ = judge(json.dumps({"winner": winner, "reason": "r"}))
        result = run_job(job, client)
        assert (result["verdict"], result["llm_side"]) == (verdict, job.llm_side)
        assert "LLM" not in fake.requests[0]["messages"][1]["content"]  # sides are unlabelled
