"""Tests for RecommendationService with a fake LLM. Case IDs refer to docs/edge-case.md §6–7."""

import logging

import pytest

from app.llm.client import LLMRanking, LLMRecommendation, LLMResult, LLMUnavailableError
from app.models.preferences import UserPreferences
from app.observability import REQUEST_LOGGER, llm_cost_usd, request_log
from app.services.recommender import (
    NOTICE_FALLBACK,
    NOTICE_NO_RESULTS,
    RecommendationService,
    shorten,
)

USAGE = {
    "prompt_tokens": 4200,
    "cached_tokens": 0,
    "completion_tokens": 1800,
    "reasoning_tokens": 1100,
}
NO_USAGE = dict.fromkeys(USAGE, 0)


class FakeLLM:
    def __init__(
        self,
        ranking: LLMRanking | None = None,
        error: Exception | None = None,
        cached: bool = False,
    ):
        self.ranking = ranking
        self.error = error
        self.cached = cached
        self.messages: list[str] = []

    def rank(self, system_prompt: str, user_message: str) -> LLMResult:
        self.messages.append(user_message)
        if self.error is not None:
            raise self.error
        return LLMResult(
            ranking=self.ranking,
            model="openai/gpt-oss-120b",
            request_id="req_1",
            stop_reason="stop",
            latency_ms=5,
            usage=dict(USAGE),
            cached=self.cached,
        )


def ranking(*picks: tuple[str, str]) -> LLMRanking:
    return LLMRanking(
        recommendations=[
            LLMRecommendation(restaurant_id=rid, rank=i, explanation=text)
            for i, (rid, text) in enumerate(picks, start=1)
        ]
    )


@pytest.fixture
def records(make_record):
    """Six restaurants in Indiranagar; pre-rank order is R1, R2, ... R6."""
    return [make_record(f"R{i}", rating=round(4.6 - i * 0.1, 1), votes=500) for i in range(1, 7)]


@pytest.fixture
def ids(records):
    return {r["name"]: r["restaurant_id"] for r in records}


@pytest.fixture
def service(make_repo, make_settings, records):
    repo = make_repo(records)

    def _make(llm):
        return RecommendationService(repo, llm, make_settings())

    return _make


def recommend(service, llm, **prefs):
    return service(llm).recommend(UserPreferences(**({"location": "Indiranagar"} | prefs)))


def picked(response) -> list[tuple[str, str]]:
    return [(i.name, i.source) for i in response.recommendations]


def test_valid_ranking_is_used_in_order(service, ids, records):
    llm = FakeLLM(ranking((ids["R3"], "Quiet and cheap."), (ids["R1"], "Best rated.")))
    response = recommend(service, llm, top_n=2)
    assert picked(response) == [("R3", "llm"), ("R1", "llm")]
    assert [i.rank for i in response.recommendations] == [1, 2]
    assert response.recommendations[0].explanation == "Quiet and cheap."
    # Facts come from the data, not from the LLM.
    assert response.recommendations[0].rating == records[2]["rating"]
    assert response.notice is None
    assert response.meta.llm_status == "ok"
    assert response.meta.model == "openai/gpt-oss-120b"


def test_hallucinated_ids_never_reach_the_output(service, ids):  # EC-LLM-24
    llm = FakeLLM(
        ranking(
            ("made-up-id", "Invented."), (ids["R2"], "Real."), ("ffffffffffff", "Also invented.")
        )
    )
    response = recommend(service, llm, top_n=3)
    assert set(i.restaurant_id for i in response.recommendations) <= set(ids.values())
    assert picked(response) == [("R2", "llm"), ("R1", "fallback"), ("R3", "fallback")]


def test_duplicate_ids_are_dropped(service, ids):  # EC-LLM-26
    llm = FakeLLM(ranking((ids["R2"], "a"), (ids["R2"], "b"), (ids["R4"], "c")))
    response = recommend(service, llm, top_n=2)
    assert picked(response) == [("R2", "llm"), ("R4", "llm")]
    assert response.recommendations[0].explanation == "a"


def test_ids_are_normalized(service, ids):  # EC-LLM-25
    llm = FakeLLM(ranking((f"  {ids['R5'].upper()} ", "Fine.")))
    assert picked(recommend(service, llm, top_n=1)) == [("R5", "llm")]


def test_too_few_picks_are_backfilled_after_the_llm_picks(service, ids):  # EC-LLM-22, EC-VAL-03
    llm = FakeLLM(ranking((ids["R4"], "Good.")))
    response = recommend(service, llm, top_n=3)
    assert picked(response) == [("R4", "llm"), ("R1", "fallback"), ("R2", "fallback")]
    assert response.notice is None  # some explanations are still from the LLM


def test_too_many_picks_are_cut(service, ids):  # EC-LLM-21
    llm = FakeLLM(ranking(*[(ids[f"R{i}"], "ok") for i in range(1, 6)]))
    assert len(recommend(service, llm, top_n=2).recommendations) == 2


def test_empty_ranking_falls_back_entirely(service):  # EC-LLM-23, EC-VAL-04
    response = recommend(service, FakeLLM(ranking()), top_n=2)
    assert picked(response) == [("R1", "fallback"), ("R2", "fallback")]
    assert response.notice == NOTICE_FALLBACK


def test_blank_explanation_gets_a_template(service, ids):  # EC-LLM-28
    response = recommend(service, FakeLLM(ranking((ids["R2"], "   "))), top_n=1)
    item = response.recommendations[0]
    assert (item.name, item.source) == ("R2", "fallback")
    assert "★" in item.explanation


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Short.", "Short."),
        ("First sentence is here. " * 30, None),  # cut at a sentence end
        ("word " * 200, None),  # no sentence end: cut at a word, with an ellipsis
    ],
)
def test_long_explanations_are_shortened(text, expected):  # EC-LLM-29
    result = shorten(text.strip(), max_chars=100)
    if expected:
        assert result == expected
    else:
        assert len(result) <= 100
        assert result.endswith((".", "…"))


@pytest.mark.parametrize(
    "error",
    [
        LLMUnavailableError("finish_reason length"),  # EC-LLM-11
        LLMUnavailableError("connection failed: APITimeoutError"),  # EC-LLM-01
    ],
)
def test_llm_failure_uses_the_fallback_ranking(service, error):
    response = recommend(service, FakeLLM(error=error), top_n=3)
    assert picked(response) == [("R1", "fallback"), ("R2", "fallback"), ("R3", "fallback")]
    assert response.notice == NOTICE_FALLBACK
    assert response.meta.llm_status == "fallback"
    assert response.meta.model is None
    assert all("★" in i.explanation for i in response.recommendations)


def test_disabled_llm_uses_the_fallback_ranking(service):
    response = recommend(service, None, top_n=2)
    assert response.meta.llm_status == "fallback"
    assert response.notice == NOTICE_FALLBACK


def test_no_candidates_means_no_llm_call(make_repo, make_settings, make_record):  # EC-FIL-17
    # Thai exists only in Whitefield, and cuisine is never relaxed: nothing in Indiranagar.
    repo = make_repo(
        [make_record("Local"), make_record("Far", location="Whitefield", cuisines=["Thai"])]
    )
    llm = FakeLLM(ranking())
    response = RecommendationService(repo, llm, make_settings()).recommend(
        UserPreferences(location="Indiranagar", cuisines=["Thai"])
    )
    assert llm.messages == []
    assert response.recommendations == []
    assert response.notice == NOTICE_NO_RESULTS
    assert response.meta.llm_status == "not_called"


def test_fewer_candidates_than_top_n(make_repo, make_settings, make_record):  # EC-IN-51
    record = make_record("Only one")
    llm = FakeLLM(ranking((record["restaurant_id"], "The only option.")))
    service = RecommendationService(make_repo([record]), llm, make_settings())
    response = service.recommend(UserPreferences(location="Indiranagar", top_n=5))
    assert [i.name for i in response.recommendations] == ["Only one"]
    assert "Recommend 1 of these 1 candidates." in llm.messages[0]


@pytest.mark.parametrize(("top_n", "expected"), [(3, 10), (5, 10), (8, 16), (10, 20)])
def test_candidate_pool_is_at_least_twice_top_n(
    make_repo, make_settings, make_record, top_n, expected
):
    records = [make_record(f"Place {i}", votes=100 + i) for i in range(30)]
    llm = FakeLLM(ranking())
    service = RecommendationService(make_repo(records), llm, make_settings(candidate_top_k=10))
    response = service.recommend(UserPreferences(location="Indiranagar", top_n=top_n))
    assert response.meta.candidates_considered == expected
    assert f"of these {expected} candidates" in llm.messages[0]


def test_relaxations_reach_the_prompt_and_response(make_repo, make_settings, make_record):
    record = make_record("Medium place", cost_for_two=400)
    llm = FakeLLM(ranking((record["restaurant_id"], "Slightly above budget.")))
    service = RecommendationService(make_repo([record]), llm, make_settings())
    response = service.recommend(UserPreferences(location="Indiranagar", budget="low"))
    assert response.relaxations == ["Budget widened from low to low or medium."]
    assert "<relaxations>" in llm.messages[0]
    assert response.meta.total_matched == 1
    assert response.meta.candidates_considered == 1


# --- The request log line (architecture §10) ------------------------------------------------


def logged(service, llm, caplog, **prefs) -> logging.LogRecord:
    """Run one recommendation inside a request scope and return its log line."""
    caplog.set_level(logging.INFO, logger=REQUEST_LOGGER)
    with request_log(request_id="r1"):
        recommend(service, llm, **prefs)
    (line,) = [r for r in caplog.records if r.name == REQUEST_LOGGER]
    return line


def test_log_line_for_an_llm_ranked_request(service, ids, caplog, make_settings):
    llm = FakeLLM(ranking((ids["R2"], "Good.")))
    line = logged(service, llm, caplog, top_n=2, budget="medium", min_rating=3.5)
    fields = line.fields
    assert line.levelno == logging.INFO
    assert fields["preferences"] == {
        "location": "Indiranagar",
        "budget": "medium",
        "cuisines": [],
        "min_rating": 3.5,
        "top_n": 2,
        "additional_preferences_chars": 0,
    }
    assert (fields["candidates"], fields["total_matched"], fields["relaxations"]) == (6, 6, [])
    assert (fields["llm_status"], fields["fallback"]) == ("ok", False)
    assert (fields["results"], fields["fallback_items"]) == (2, 1)  # one backfilled
    assert "summary_source" not in fields
    llm_fields = fields["llm"]
    assert llm_fields["model"] == "openai/gpt-oss-120b"
    assert llm_fields["finish_reason"] == "stop"
    assert llm_fields["groq_request_id"] == "req_1"
    assert llm_fields["cache_hit"] is False
    assert llm_fields["usage"] == USAGE
    assert llm_fields["cost_usd"] == llm_cost_usd(USAGE, make_settings()) > 0
    assert llm_fields["latency_ms"] >= 0


def test_free_text_is_logged_as_its_length_only(service, caplog):  # architecture §10
    text = "my number is 98450 12345, call me"
    fields = logged(service, FakeLLM(ranking()), caplog, additional_preferences=text).fields
    assert fields["preferences"]["additional_preferences_chars"] == len(text)
    assert "98450" not in str(fields)


@pytest.mark.parametrize(
    ("error", "level", "config_error"),
    [
        (
            LLMUnavailableError("API error 429 (rate_limit_exceeded)", status_code=429),
            "WARNING",
            False,
        ),
        (LLMUnavailableError("connection failed: APITimeoutError"), "WARNING", False),
        (LLMUnavailableError("API error 401 (invalid_api_key)", status_code=401), "ERROR", True),
        (LLMUnavailableError("API error 404 (model_not_found)", status_code=404), "ERROR", True),
    ],
)
def test_log_line_for_a_fallback(service, caplog, error, level, config_error):
    # EC-CFG-05, EC-CFG-06, EC-LLM-05: configuration problems are logged at ERROR.
    line = logged(service, FakeLLM(error=error), caplog, top_n=2)
    fields = line.fields
    assert line.levelname == level
    assert (fields["llm_status"], fields["fallback"]) == ("fallback", True)
    assert fields["fallback_reason"] == error.reason
    assert fields["fallback_items"] == 2
    assert fields["llm"]["error"] == error.reason
    assert fields["llm"]["status_code"] == error.status_code
    assert fields["llm"]["config_error"] is config_error
    assert (fields["llm"]["usage"], fields["llm"]["cost_usd"]) == (NO_USAGE, 0)


def test_tokens_spent_on_a_cut_off_answer_are_logged(service, caplog):  # EC-LLM-11
    usage = {
        "prompt_tokens": 4200,
        "cached_tokens": 0,
        "completion_tokens": 8192,
        "reasoning_tokens": 8100,
    }
    error = LLMUnavailableError(
        "finish_reason length", request_id="req_9", finish_reason="length", usage=usage
    )
    fields = logged(service, FakeLLM(error=error), caplog).fields
    assert fields["llm"]["finish_reason"] == "length"
    assert fields["llm"]["groq_request_id"] == "req_9"
    assert fields["llm"]["usage"] == usage
    assert fields["llm"]["cost_usd"] > 0


def test_a_cache_hit_spends_nothing(service, ids, caplog):  # D-10
    llm = FakeLLM(ranking((ids["R1"], "x")), cached=True)
    fields = logged(service, llm, caplog, top_n=1).fields
    assert fields["llm"]["cache_hit"] is True
    assert (fields["llm"]["usage"], fields["llm"]["cost_usd"]) == (NO_USAGE, 0)


def test_log_line_with_the_llm_disabled(service, caplog):
    fields = logged(service, None, caplog).fields
    assert (fields["llm_status"], fields["fallback_reason"]) == ("fallback", "LLM disabled")
    assert "llm" not in fields


def test_log_line_when_nothing_matches(make_repo, make_settings, make_record, caplog):
    repo = make_repo([make_record("Local")])
    service = RecommendationService(repo, FakeLLM(ranking()), make_settings())
    caplog.set_level(logging.INFO, logger=REQUEST_LOGGER)
    with request_log(request_id="r1"):
        service.recommend(
            UserPreferences(location="Indiranagar", cuisines=["North Indian"], min_rating=5.0)
        )
    (line,) = [r for r in caplog.records if r.name == REQUEST_LOGGER]
    assert line.fields["llm_status"] == "not_called"
    assert line.fields["candidates"] == 0
    assert "llm" not in line.fields
