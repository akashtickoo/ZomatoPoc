"""What the recommender returns (architecture §5.9)."""

from typing import Literal

from pydantic import BaseModel


class RecommendationItem(BaseModel):
    """One recommendation. Every fact comes from the dataset; only `explanation` may be
    written by the LLM."""

    rank: int
    restaurant_id: str
    name: str
    location: str
    cuisines: list[str]
    rating: float | None
    votes: int
    cost_for_two: int | None
    rest_type: list[str]
    online_order: bool
    book_table: bool
    popular_dishes: list[str]  # up to 5 dishes diners liked here (dish_liked)
    explanation: str
    source: Literal["llm", "fallback"]  # who chose this item and wrote its explanation


class ResponseMeta(BaseModel):
    # "ok": the LLM ranked the results; "fallback": it was unavailable, so the
    # deterministic ranking was used; "not_called": there was nothing to rank.
    llm_status: Literal["ok", "fallback", "not_called"]
    model: str | None  # the model that served the request, if any
    candidates_considered: int
    total_matched: int
    latency_ms: int


class RecommendationResponse(BaseModel):
    recommendations: list[RecommendationItem]
    relaxations: list[str]  # filters loosened to find enough restaurants
    notice: str | None = None  # e.g. AI unavailable; the UI shows it only for an empty result
    meta: ResponseMeta
