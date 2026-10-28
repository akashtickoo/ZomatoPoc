"""Preferences in, recommendations out: filter, ask the LLM to rank, validate, enrich.

Implements architecture §5.8. The LLM only chooses which candidates to show, in what
order, and why; every fact shown comes from the repository. Case IDs refer to
docs/edge-case.md §6–7.

Manual smoke test against the real API (needs GROQ_API_KEY):
    python -m app.services.recommender --location Indiranagar --cuisine Italian --budget medium
"""

import argparse
import json
import logging
import sys
import time
import uuid

from app.config import Settings, settings
from app.data.preprocess import truncate_text
from app.data.repository import RestaurantRepository
from app.llm.client import (
    USAGE_KEYS,
    LLMRanking,
    LLMResult,
    LLMUnavailableError,
    RankingClient,
    make_ranking_client,
)
from app.llm.prompt_builder import MAX_DISHES, approx_tokens, system_prompt, user_message
from app.models.preferences import UserPreferences
from app.models.recommendation import RecommendationItem, RecommendationResponse, ResponseMeta
from app.models.restaurant import Restaurant
from app.observability import annotate, configure_logging, llm_cost_usd, request_log
from app.services.explanations import template_explanation
from app.services.filtering import filter_candidates

logger = logging.getLogger(__name__)

MAX_EXPLANATION_CHARS = 500  # EC-LLM-29
# The LLM sees at least this many candidates per recommendation it returns, so a large
# top_n still leaves it a real choice even with a small CANDIDATE_TOP_K.
MIN_CANDIDATES_PER_PICK = 2
NOTICE_FALLBACK = (
    "AI-written explanations are unavailable right now, so these are ranked by rating "
    "and popularity."
)
NOTICE_NO_RESULTS = (
    "No restaurants match these preferences, even after widening the search. "
    "Try removing a cuisine or lowering the minimum rating."
)


def shorten(text: str, max_chars: int = MAX_EXPLANATION_CHARS) -> str:
    """Cut an over-long explanation at a sentence end if possible, else a word boundary."""
    if len(text) <= max_chars:
        return text
    end = text.rfind(". ", 0, max_chars)
    if end > max_chars // 2:
        return text[: end + 1]
    return truncate_text(text, max_chars)


def _item(rank: int, r: Restaurant, explanation: str, source: str) -> RecommendationItem:
    return RecommendationItem(
        rank=rank,
        restaurant_id=r.restaurant_id,
        name=r.name,
        location=r.location,
        cuisines=r.cuisines,
        rating=r.rating,
        votes=r.votes,
        cost_for_two=r.cost_for_two,
        rest_type=r.rest_type,
        online_order=r.online_order,
        book_table=r.book_table,
        popular_dishes=r.dish_liked[:MAX_DISHES],  # the same dishes the LLM saw
        explanation=explanation,
        source=source,
    )


def select_items(
    ranking: LLMRanking | None, candidates: list[Restaurant], prefs: UserPreferences
) -> list[RecommendationItem]:
    """Validate the LLM's picks against the candidates and fill any gaps.

    - IDs not among the candidates, and repeats, are dropped (EC-LLM-24, 25, 26).
    - Ranks follow the LLM's list order, renumbered from 1 (EC-LLM-27).
    - Extra picks are cut; missing ones are filled from the pre-ranked list with
      template explanations, after the LLM's picks (EC-LLM-21, 22, 23, EC-VAL-03).
    """
    by_id = {r.restaurant_id: r for r in candidates}
    target = min(prefs.top_n, len(candidates))
    picks: list[tuple[Restaurant, str, str]] = []
    seen: set[str] = set()

    for rec in ranking.recommendations if ranking else []:
        if len(picks) == target:
            break
        rid = rec.restaurant_id.strip().lower()
        restaurant = by_id.get(rid)
        if restaurant is None or rid in seen:
            continue
        seen.add(rid)
        explanation = " ".join(rec.explanation.split())
        if explanation:
            picks.append((restaurant, shorten(explanation), "llm"))
        else:  # EC-LLM-28
            picks.append((restaurant, template_explanation(restaurant, prefs), "fallback"))

    for restaurant in candidates:
        if len(picks) == target:
            break
        if restaurant.restaurant_id not in seen:
            seen.add(restaurant.restaurant_id)
            picks.append((restaurant, template_explanation(restaurant, prefs), "fallback"))

    return [_item(i, r, text, source) for i, (r, text, source) in enumerate(picks, start=1)]


def loggable_preferences(prefs: UserPreferences) -> dict:
    """Preferences for the request log. The free text is logged as its length only: it is
    the one field that could hold personal details (architecture §10)."""
    data = prefs.model_dump(exclude={"additional_preferences"})
    data["additional_preferences_chars"] = len(prefs.additional_preferences or "")
    return data


def llm_log(
    latency_ms: int,
    cfg: Settings,
    result: LLMResult | None = None,
    error: LLMUnavailableError | None = None,
) -> dict:
    """The LLM part of the request log line: what was called, how it went, what it cost."""
    if result is not None:
        usage = {} if result.cached else result.usage  # a cache hit spends no tokens
        fields = {
            "model": result.model,
            "finish_reason": result.stop_reason,
            "groq_request_id": result.request_id,
            "cache_hit": result.cached,
        }
    else:
        usage = error.usage
        fields = {
            "model": cfg.llm_model,
            "error": error.reason,
            "status_code": error.status_code,
            "config_error": error.is_config_error,
            "finish_reason": error.finish_reason,
            "groq_request_id": error.request_id,
        }
    usage = {key: usage.get(key, 0) for key in USAGE_KEYS}
    return {
        "latency_ms": latency_ms,
        **fields,
        "usage": usage,
        "cost_usd": llm_cost_usd(usage, cfg),
    }


class RecommendationService:
    def __init__(
        self,
        repo: RestaurantRepository,
        llm: RankingClient | None,
        cfg: Settings = settings,
    ):
        self._repo = repo
        self._llm = llm  # None disables the LLM; every response uses the fallback path
        self._cfg = cfg

    def recommend(self, prefs: UserPreferences) -> RecommendationResponse:
        """Raises UnknownValueError for a location or cuisine not in the data.

        Adds what happened to the current request's log line (architecture §10).
        """
        start = time.perf_counter()
        annotate(preferences=loggable_preferences(prefs))
        result = filter_candidates(
            self._repo,
            prefs,
            top_k=max(self._cfg.candidate_top_k, MIN_CANDIDATES_PER_PICK * prefs.top_n),
            min_candidates=self._cfg.min_candidates,
        )
        candidates = result.candidates
        annotate(
            candidates=len(candidates),
            total_matched=result.total_matched,
            relaxations=result.relaxations,
        )

        def respond(items, status, model=None, notice=None):
            annotate(
                llm_status=status,
                fallback=status == "fallback",
                results=len(items),
                fallback_items=sum(i.source == "fallback" for i in items),
            )
            return RecommendationResponse(
                recommendations=items,
                relaxations=result.relaxations,
                notice=notice,
                meta=ResponseMeta(
                    llm_status=status,
                    model=model,
                    candidates_considered=len(candidates),
                    total_matched=result.total_matched,
                    latency_ms=round((time.perf_counter() - start) * 1000),
                ),
            )

        if not candidates:  # nothing to rank, so no LLM call (EC-FIL-17)
            return respond([], "not_called", notice=NOTICE_NO_RESULTS)

        llm_result = None
        if self._llm is not None:
            message = user_message(prefs, candidates, result.relaxations)
            logger.debug("Prompt is about %d tokens", approx_tokens(system_prompt() + message))
            llm_start = time.perf_counter()
            try:
                llm_result = self._llm.rank(system_prompt(), message)
            except LLMUnavailableError as exc:
                latency_ms = round((time.perf_counter() - llm_start) * 1000)
                annotate(llm=llm_log(latency_ms, self._cfg, error=exc), fallback_reason=exc.reason)
            else:
                latency_ms = round((time.perf_counter() - llm_start) * 1000)
                annotate(llm=llm_log(latency_ms, self._cfg, result=llm_result))
        else:
            annotate(fallback_reason="LLM disabled")

        ranking = llm_result.ranking if llm_result else None
        items = select_items(ranking, candidates, prefs)
        no_llm_items = not any(i.source == "llm" for i in items)  # EC-VAL-04
        return respond(
            items,
            "ok" if llm_result else "fallback",
            model=llm_result.model if llm_result else None,
            notice=NOTICE_FALLBACK if no_llm_items else None,
        )


def main(argv: list[str] | None = None) -> int:
    from app.services.filtering import UnknownValueError

    parser = argparse.ArgumentParser(description="Get recommendations, calling the real API.")
    parser.add_argument("--location", required=True)
    parser.add_argument("--cuisine", action="append", default=[], dest="cuisines")
    parser.add_argument("--budget", choices=["low", "medium", "high"])
    parser.add_argument("--min-rating", type=float, default=0.0)
    parser.add_argument("--prefs", dest="additional_preferences")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="print the raw response")
    args = parser.parse_args(argv)

    configure_logging()  # the request's JSON log line goes to stderr, as from the API
    service = RecommendationService(RestaurantRepository.load(), make_ranking_client())
    prefs = UserPreferences(**{k: v for k, v in vars(args).items() if k != "json"})
    try:
        with request_log(request_id=uuid.uuid4().hex[:12], source="cli"):
            response = service.recommend(prefs)
    except UnknownValueError as exc:
        print(exc)
        return 2

    if args.json:
        print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
        return 0
    meta = response.meta
    print(f"\nllm_status={meta.llm_status} model={meta.model} latency={meta.latency_ms} ms")
    for line in ([response.notice] if response.notice else []) + response.relaxations:
        print(f"  note: {line}")
    for item in response.recommendations:
        rating = f"{item.rating} ★" if item.rating is not None else "unrated"
        cost = f"₹{item.cost_for_two}" if item.cost_for_two is not None else "cost n/a"
        print(f"\n{item.rank}. {item.name} ({item.location}) {rating}, {cost} [{item.source}]")
        print(f"   {item.explanation}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
