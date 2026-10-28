"""Tests for GroqRankingClient against a fake SDK. Case IDs refer to docs/edge-case.md §6."""

import threading
from types import SimpleNamespace

import groq
import httpx
import pytest

from app.llm.client import (
    MAX_COMPLETION_TOKENS,
    CachingRankingClient,
    GroqRankingClient,
    LLMRanking,
    LLMRecommendation,
    LLMResult,
    LLMUnavailableError,
    make_ranking_client,
    strict_json_schema,
)

REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
RANKING = LLMRanking(
    recommendations=[LLMRecommendation(restaurant_id="abc", rank=1, explanation="Great.")],
)


def response(content: str | None = RANKING.model_dump_json(), finish_reason="stop", **overrides):
    base = {
        "choices": [
            SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content))
        ],
        "model": "openai/gpt-oss-120b",
        "usage": SimpleNamespace(
            prompt_tokens=1200,
            completion_tokens=300,
            prompt_tokens_details=SimpleNamespace(cached_tokens=1000),
            completion_tokens_details=None,
        ),
        "x_groq": SimpleNamespace(id="req_123"),
    }
    return SimpleNamespace(**(base | overrides))


class FakeSDK:
    def __init__(self, result=None, error=None):
        self.calls: list[dict] = []

        def create(**kwargs):
            self.calls.append(kwargs)
            if error is not None:
                raise error
            return result

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def client(make_settings, sdk, **settings_overrides):
    return GroqRankingClient(make_settings(**settings_overrides), client=sdk)


def test_strict_schema_requires_every_field_and_forbids_extras():
    schema = strict_json_schema(LLMRanking)
    objects = [schema, *schema["$defs"].values()]
    assert len(objects) == 2
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert sorted(obj["required"]) == sorted(obj["properties"])


def test_success_returns_ranking_and_metadata(make_settings):
    result = client(make_settings, FakeSDK(response())).rank("system", "user")
    assert result.ranking == RANKING
    assert result.model == "openai/gpt-oss-120b"
    assert result.request_id == "req_123"
    assert result.stop_reason == "stop"
    assert result.usage == {
        "prompt_tokens": 1200,
        "completion_tokens": 300,
        "cached_tokens": 1000,
        "reasoning_tokens": 0,
    }


def test_request_shape(make_settings):
    sdk = FakeSDK(response())
    client(make_settings, sdk, llm_effort="low").rank("system text", "user text")
    (call,) = sdk.calls
    assert call["model"] == "openai/gpt-oss-120b"
    assert call["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert call["reasoning_effort"] == "low"
    assert call["include_reasoning"] is False
    assert call["max_completion_tokens"] == MAX_COMPLETION_TOKENS
    response_format = call["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == strict_json_schema(LLMRanking)


def test_missing_key_fails_without_a_request(make_settings):  # EC-CFG-04
    llm = GroqRankingClient(make_settings())  # conftest clears GROQ_API_KEY
    assert not llm.is_configured
    with pytest.raises(LLMUnavailableError, match="no Groq API key"):
        llm.rank("s", "u")


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_key_means_no_key(make_settings, monkeypatch, blank):  # EC-CFG-04
    monkeypatch.setenv("GROQ_API_KEY", blank)  # e.g. "GROQ_API_KEY=" in .env
    cfg = make_settings()
    assert cfg.groq_api_key is None
    assert not GroqRankingClient(cfg).is_configured  # fallback, not a 401 on every call


def test_key_from_settings_is_used(make_settings):
    assert GroqRankingClient(make_settings(groq_api_key="gsk_test")).is_configured


def test_cut_off_response(make_settings):  # EC-LLM-11
    with pytest.raises(LLMUnavailableError, match="finish_reason length") as exc:
        client(make_settings, FakeSDK(response(finish_reason="length"))).rank("s", "u")
    assert exc.value.request_id == "req_123"
    assert exc.value.finish_reason == "length"
    # The tokens were billed even though the answer is unusable; the log records them.
    assert exc.value.usage["completion_tokens"] == 300


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (response(content=None), "no content"),
        (response(content=""), "no content"),
        (response(content='{"recommendations": ['), "schema"),
        (response(content='{"picks": []}'), "schema"),
        (response(choices=[]), "no choices"),
    ],
)
def test_unusable_output(make_settings, result, message):  # EC-LLM-13
    with pytest.raises(LLMUnavailableError, match=message):
        client(make_settings, FakeSDK(result)).rank("s", "u")


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (groq.APITimeoutError(request=REQUEST), "APITimeoutError"),  # EC-LLM-01
        (groq.APIConnectionError(request=REQUEST), "APIConnectionError"),  # EC-LLM-04
        (
            groq.RateLimitError(
                "rate limited", response=httpx.Response(429, request=REQUEST), body=None
            ),
            "API error 429",
        ),  # EC-LLM-02
        (
            groq.InternalServerError(
                "unavailable", response=httpx.Response(503, request=REQUEST), body=None
            ),
            "API error 503",
        ),  # EC-LLM-03
        (
            groq.AuthenticationError(
                "bad key", response=httpx.Response(401, request=REQUEST), body=None
            ),
            "API error 401",
        ),  # EC-CFG-05
        (
            groq.BadRequestError(
                "model does not support json_schema",
                response=httpx.Response(400, request=REQUEST),
                body=None,
            ),
            "API error 400",
        ),  # EC-LLM-05, e.g. LLM_MODEL set to an unsupported model
    ],
)
def test_transport_and_api_errors(make_settings, error, message):
    with pytest.raises(LLMUnavailableError, match=message) as exc:
        client(make_settings, FakeSDK(error=error)).rank("s", "u")
    # Only API errors carry a status; 400 and 401 need a configuration fix (EC-CFG-05).
    status = getattr(error, "status_code", None)
    assert exc.value.status_code == status
    assert exc.value.is_config_error is (status in (400, 401))
    assert exc.value.usage == {}


def api_error(status: int) -> groq.APIStatusError:
    return groq.APIStatusError("x", response=httpx.Response(status, request=REQUEST), body=None)


def test_a_rejected_key_is_remembered_until_a_call_succeeds(make_settings):  # EC-CFG-05
    sdk = FakeSDK(error=api_error(401))
    llm = client(make_settings, sdk)
    assert llm.key_rejected is False
    with pytest.raises(LLMUnavailableError):
        llm.rank("s", "u")
    assert llm.key_rejected is True
    assert CachingRankingClient(llm, ttl_seconds=60).key_rejected is True  # /health sees it
    sdk.chat.completions.create = lambda **kwargs: response()  # the key was fixed
    llm.rank("s", "u")
    assert llm.key_rejected is False


@pytest.mark.parametrize("status", [429, 503, 413])
def test_passing_outages_do_not_mark_the_key_rejected(make_settings, status):
    llm = client(make_settings, FakeSDK(error=api_error(status)))
    with pytest.raises(LLMUnavailableError) as exc:
        llm.rank("s", "u")
    assert llm.key_rejected is False
    assert exc.value.is_config_error is False


def test_api_error_reason_includes_groq_error_code(make_settings):  # EC-LLM-02
    # What the free tier returns when a request would exceed its tokens-per-minute limit.
    body = {
        "error": {
            "message": "Request too large for model on tokens per minute (TPM): Limit 8000",
            "type": "tokens",
            "code": "rate_limit_exceeded",
        }
    }
    error = groq.APIStatusError(
        "too large", response=httpx.Response(413, request=REQUEST), body=body
    )
    with pytest.raises(LLMUnavailableError) as exc:
        client(make_settings, FakeSDK(error=error)).rank("s", "u")
    assert exc.value.reason == "API error 413 (rate_limit_exceeded)"
    assert "TPM" not in exc.value.reason  # the provider's message stays out of logs


@pytest.mark.parametrize(("headers", "expected"), [({"retry-after": "7"}, 7.0), ({}, None)])
def test_api_errors_carry_retry_after(make_settings, headers, expected):
    response = httpx.Response(429, request=REQUEST, headers=headers)
    error = groq.RateLimitError("limited", response=response, body=None)
    with pytest.raises(LLMUnavailableError) as exc:
        client(make_settings, FakeSDK(error=error)).rank("s", "u")
    assert exc.value.retry_after == expected


@pytest.mark.parametrize(
    ("status", "headers"),
    [
        (429, {"retry-after": "0"}),  # the free tier's per-minute limit
        (503, {}),
        (429, {"retry-after": "0", "x-should-retry": "true"}),  # Groq says "retry"
    ],
)
def test_one_search_sends_exactly_one_request(make_settings, monkeypatch, status, headers):
    """No SDK retries: a failed call falls back at once instead of sending more (EC-LLM-02)."""
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        body = {"error": {"message": "limited", "type": "tokens", "code": "rate_limit_exceeded"}}
        return httpx.Response(status, headers=headers, json=body)

    real_groq = groq.Groq

    def groq_over_fake_transport(**kwargs):  # our own constructor arguments, fake network
        return real_groq(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(groq, "Groq", groq_over_fake_transport)
    llm = GroqRankingClient(make_settings(groq_api_key="gsk_test"))
    with pytest.raises(LLMUnavailableError, match=f"API error {status}"):
        llm.rank("system", "user")
    assert len(sent) == 1


class CountingLLM:
    """Inner client for the cache tests: counts calls and can fail on demand."""

    def __init__(self):
        self.calls = 0
        self.fail = False

    def rank(self, system_prompt: str, user_message: str) -> LLMResult:
        self.calls += 1
        if self.fail:
            raise LLMUnavailableError("API error 413 (rate_limit_exceeded)")
        return LLMResult(
            ranking=RANKING,
            model="m",
            request_id=f"req_{self.calls}",
            stop_reason="stop",
            latency_ms=900,
        )


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_cache_reuses_an_identical_request():  # D-10
    inner = CountingLLM()
    cache = CachingRankingClient(inner, ttl_seconds=60)
    first = cache.rank("s", "u")
    second = cache.rank("s", "u")
    assert inner.calls == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.ranking == first.ranking
    assert second.request_id == "req_1"


def test_cache_misses_when_any_part_of_the_prompt_differs():
    inner = CountingLLM()
    cache = CachingRankingClient(inner, ttl_seconds=60)
    cache.rank("s", "u")
    cache.rank("s", "u2")  # different preferences or candidates
    cache.rank("s2", "u")  # new prompt version
    assert inner.calls == 3


def test_cache_never_stores_failures():
    inner = CountingLLM()
    cache = CachingRankingClient(inner, ttl_seconds=60)
    inner.fail = True
    with pytest.raises(LLMUnavailableError):
        cache.rank("s", "u")
    inner.fail = False
    assert cache.rank("s", "u").cached is False  # the retry reached the LLM
    assert inner.calls == 2


def test_cache_entries_expire():
    inner, clock = CountingLLM(), FakeClock()
    cache = CachingRankingClient(inner, ttl_seconds=60, clock=clock)
    cache.rank("s", "u")
    clock.now = 59.9
    assert cache.rank("s", "u").cached is True
    clock.now = 60.0
    assert cache.rank("s", "u").cached is False
    assert inner.calls == 2


def test_cache_evicts_the_least_recently_used_entry():
    inner = CountingLLM()
    cache = CachingRankingClient(inner, ttl_seconds=60, max_entries=2)
    cache.rank("s", "a")
    cache.rank("s", "b")
    cache.rank("s", "a")  # "a" is now the most recently used
    cache.rank("s", "c")  # evicts "b"
    assert cache.rank("s", "a").cached is True
    assert cache.rank("s", "b").cached is False


def test_make_ranking_client_caches_unless_disabled(make_settings):
    assert isinstance(make_ranking_client(make_settings()), CachingRankingClient)
    plain = make_ranking_client(make_settings(llm_cache_ttl_seconds=0))
    assert isinstance(plain, GroqRankingClient)


class BlockingLLM:
    """Holds every call until released, so two identical requests overlap."""

    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail
        self.started = threading.Event()
        self.release = threading.Event()

    def rank(self, system_prompt: str, user_message: str) -> LLMResult:
        self.calls += 1
        self.started.set()
        assert self.release.wait(5)
        if self.fail:
            raise LLMUnavailableError("API error 429 (rate_limit_exceeded)")
        return LLMResult(
            ranking=RANKING, model="m", request_id="req_1", stop_reason="stop", latency_ms=1
        )


def overlapping_calls(cache: CachingRankingClient, inner: BlockingLLM) -> list:
    """Start one request, send an identical one while it waits, then let both finish."""
    outcomes = [None, None]

    def call(i):
        try:
            outcomes[i] = cache.rank("s", "u")
        except LLMUnavailableError as exc:
            outcomes[i] = exc

    first = threading.Thread(target=call, args=(0,))
    first.start()
    assert inner.started.wait(5)
    second = threading.Thread(target=call, args=(1,))
    second.start()
    second.join(0.2)  # the second request is now waiting on the first
    inner.release.set()
    first.join(5)
    second.join(5)
    return outcomes


def test_identical_requests_in_flight_share_one_llm_call():  # a double-click
    inner = BlockingLLM()
    first, second = overlapping_calls(CachingRankingClient(inner, ttl_seconds=60), inner)
    assert inner.calls == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.ranking == first.ranking


def test_a_shared_failure_reaches_both_and_is_not_cached():
    inner = BlockingLLM(fail=True)
    cache = CachingRankingClient(inner, ttl_seconds=60)
    first, second = overlapping_calls(cache, inner)
    assert inner.calls == 1
    assert isinstance(first, LLMUnavailableError) and isinstance(second, LLMUnavailableError)
    inner.fail = False
    assert cache.rank("s", "u").cached is False  # the next search tries the LLM again
    assert inner.calls == 2
