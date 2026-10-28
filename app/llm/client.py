"""Groq client that ranks candidates and explains them (architecture §5.7).

Any failure (no API key, timeout, rate limit, cut-off or unparseable output) surfaces
as LLMUnavailableError so the recommender can fall back to the deterministic ranking.
Case IDs refer to docs/edge-case.md §6.
"""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import Protocol

import groq
from pydantic import BaseModel, ValidationError

from app.config import Settings, settings

logger = logging.getLogger(__name__)

# Room for the model's reasoning plus the JSON answer. Groq's default (1024) is too low
# for a reasoning model.
MAX_COMPLETION_TOKENS = 8192
SCHEMA_NAME = "restaurant_ranking"
CACHE_MAX_ENTRIES = 256  # a few KB each
# One search sends exactly one request to Groq. The SDK's default (2 retries, waiting out
# a 429's retry-after) turned one rate-limited search into up to 3 requests and 20+ s
# of waiting; failing fast to the fallback ranking is the better experience (EC-LLM-02).
MAX_RETRIES = 0
USAGE_KEYS = ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens")
# Statuses that need someone to fix the configuration (a bad key, an unknown model, a
# request the model doesn't support), unlike passing outages such as 429 or 503.
CONFIG_ERROR_STATUSES = frozenset({400, 401, 403, 404})
KEY_REJECTED_STATUSES = frozenset({401, 403})


class LLMRecommendation(BaseModel):
    restaurant_id: str
    rank: int
    explanation: str


class LLMRanking(BaseModel):
    recommendations: list[LLMRecommendation]


@dataclass(frozen=True)
class LLMResult:
    ranking: LLMRanking
    model: str  # the model that actually served the request
    request_id: str | None
    stop_reason: str | None
    latency_ms: int
    usage: dict[str, int] = field(default_factory=dict)
    cached: bool = False  # served from CachingRankingClient; no tokens were spent


class LLMUnavailableError(RuntimeError):
    def __init__(
        self,
        reason: str,
        *,
        request_id: str | None = None,
        status_code: int | None = None,
        finish_reason: str | None = None,
        usage: dict[str, int] | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(reason)
        self.reason = reason
        self.request_id = request_id
        self.status_code = status_code  # set for API errors
        self.finish_reason = finish_reason  # set when a response arrived but was unusable
        self.usage = usage or {}  # tokens spent anyway, e.g. on a cut-off answer
        # Seconds Groq asked to wait (its retry-after header), e.g. on a 429. The app never
        # waits; the evaluation runner (tests/eval) does.
        self.retry_after = retry_after

    @property
    def is_config_error(self) -> bool:
        """True when retrying won't help until the configuration is fixed (EC-CFG-05, 06)."""
        return self.status_code in CONFIG_ERROR_STATUSES


class RankingClient(Protocol):
    """Anything that can rank candidates; tests substitute a fake."""

    def rank(self, system_prompt: str, user_message: str) -> LLMResult: ...


def strict_json_schema(model: type[BaseModel]) -> dict:
    """Pydantic's JSON schema, adapted for Groq's strict mode.

    Strict mode guarantees the output matches the schema, but requires every object to
    list all of its properties as required and to set additionalProperties to false.
    """
    schema = model.model_json_schema()

    def visit(node) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(schema)
    return schema


def _error_code(exc: groq.APIStatusError) -> str | None:
    """Groq's machine-readable error code, e.g. "rate_limit_exceeded" on a 413 or 429.

    Only the code is kept: it says why a request failed without copying the provider's
    full error message into logs.
    """
    error = exc.body.get("error") if isinstance(exc.body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else None


def _retry_after(exc: groq.APIStatusError) -> float | None:
    try:
        return float(exc.response.headers["retry-after"])
    except (KeyError, ValueError):
        return None


def _usage(response) -> dict[str, int]:
    """Token counts, keyed by USAGE_KEYS. Reasoning tokens are part of completion_tokens."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "prompt_tokens": usage.prompt_tokens or 0,
        "completion_tokens": usage.completion_tokens or 0,
        "cached_tokens": getattr(prompt_details, "cached_tokens", None) or 0,
        "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None) or 0,
    }


class GroqRankingClient:
    def __init__(self, cfg: Settings = settings, client: groq.Groq | None = None):
        self._cfg = cfg
        if client is None and cfg.groq_api_key:
            # Settings already read GROQ_API_KEY from the environment and .env. Without a
            # key there is no client and every request uses the fallback ranking (D-03);
            # the SDK isn't asked to look again, as it would accept a blank variable.
            client = groq.Groq(
                api_key=cfg.groq_api_key.get_secret_value(),
                timeout=cfg.llm_timeout_seconds,
                max_retries=MAX_RETRIES,
            )
        self._client = client
        self._key_rejected = False
        self._response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": SCHEMA_NAME,
                "strict": True,
                "schema": strict_json_schema(LLMRanking),
            },
        }

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    @property
    def key_rejected(self) -> bool:
        """Groq refused the key (401/403) on the last call; /health reports it (EC-CFG-05)."""
        return self._key_rejected

    def rank(self, system_prompt: str, user_message: str) -> LLMResult:
        if self._client is None:
            raise LLMUnavailableError("no Groq API key configured")  # EC-CFG-04

        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._cfg.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format=self._response_format,
                reasoning_effort=self._cfg.llm_effort,
                include_reasoning=False,  # the app only needs the answer
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
        except groq.APIConnectionError as exc:  # includes timeouts (EC-LLM-01, 04)
            raise LLMUnavailableError(f"connection failed: {type(exc).__name__}") from exc
        except groq.APIStatusError as exc:  # any 4xx/5xx, including 429 (EC-LLM-02, 03, 05)
            self._key_rejected = exc.status_code in KEY_REJECTED_STATUSES
            request_id = getattr(exc, "request_id", None)
            code = _error_code(exc)
            reason = f"API error {exc.status_code}" + (f" ({code})" if code else "")
            raise LLMUnavailableError(
                reason,
                request_id=request_id,
                status_code=exc.status_code,
                retry_after=_retry_after(exc),
            ) from exc
        self._key_rejected = False
        latency_ms = round((time.perf_counter() - start) * 1000)

        x_groq = getattr(response, "x_groq", None)
        request_id = getattr(x_groq, "id", None)
        usage = _usage(response)  # billed even when the answer can't be used
        if not response.choices:
            raise LLMUnavailableError("response had no choices", request_id=request_id, usage=usage)
        choice = response.choices[0]
        if choice.finish_reason != "stop":  # "length" means cut off (EC-LLM-11)
            raise LLMUnavailableError(
                f"finish_reason {choice.finish_reason}",
                request_id=request_id,
                finish_reason=choice.finish_reason,
                usage=usage,
            )
        content = choice.message.content
        if not content:  # EC-LLM-13
            raise LLMUnavailableError(
                "response had no content",
                request_id=request_id,
                finish_reason=choice.finish_reason,
                usage=usage,
            )
        try:
            # Strict mode should guarantee this; validate anyway rather than trust it.
            ranking = LLMRanking.model_validate_json(content)
        except ValidationError as exc:  # EC-LLM-13
            raise LLMUnavailableError(
                "response did not match the expected schema",
                request_id=request_id,
                finish_reason=choice.finish_reason,
                usage=usage,
            ) from exc

        return LLMResult(
            ranking=ranking,
            model=response.model,
            request_id=request_id,
            stop_reason=choice.finish_reason,
            latency_ms=latency_ms,
            usage=usage,
        )


class CachingRankingClient:
    """Reuses rankings so a repeated request spends no tokens (D-10).

    The key is the exact prompt, so a different preference, candidate list or prompt
    version is a miss. An identical request that arrives while the first is still
    waiting on the LLM (a double-click, a rerun) waits for that answer instead of
    sending a second request. Failures are never cached: after a fallback, the next
    identical request tries the LLM again. Thread-safe, as FastAPI serves requests from
    a thread pool.
    """

    def __init__(
        self,
        inner: RankingClient,
        ttl_seconds: float,
        max_entries: int = CACHE_MAX_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._inner = inner
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, LLMResult]] = OrderedDict()
        self._in_flight: dict[str, Future] = {}
        self._lock = threading.Lock()

    @property
    def is_configured(self) -> bool:
        return getattr(self._inner, "is_configured", True)

    @property
    def key_rejected(self) -> bool:
        return getattr(self._inner, "key_rejected", False)

    def rank(self, system_prompt: str, user_message: str) -> LLMResult:
        key = hashlib.sha256(f"{system_prompt}\0{user_message}".encode()).hexdigest()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and self._clock() - entry[0] < self._ttl:
                self._entries.move_to_end(key)
                return replace(entry[1], cached=True)
            pending = self._in_flight.get(key)
            owner = pending is None
            if owner:
                pending = self._in_flight[key] = Future()

        if not owner:
            # Raises the same LLMUnavailableError if that request failed.
            return replace(pending.result(), cached=True)

        try:
            result = self._inner.rank(system_prompt, user_message)  # outside the lock
        except BaseException as exc:
            with self._lock:
                del self._in_flight[key]
            pending.set_exception(exc)
            raise
        with self._lock:
            del self._in_flight[key]
            self._entries[key] = (self._clock(), result)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)  # least recently used
        pending.set_result(result)
        return result


def make_ranking_client(cfg: Settings = settings) -> RankingClient:
    """The Groq client, behind the response cache unless LLM_CACHE_TTL_SECONDS is 0."""
    client = GroqRankingClient(cfg)
    if cfg.llm_cache_ttl_seconds > 0:
        return CachingRankingClient(client, cfg.llm_cache_ttl_seconds)
    return client
