"""Build the prompt that asks the LLM to rank candidates (architecture §5.6).

The system prompt is static and comes first, so the provider can reuse it across
requests (prompt caching); everything specific to a request goes in the user message
as tagged JSON.
"""

import json
from functools import cache
from pathlib import Path

from app.models.preferences import UserPreferences
from app.models.restaurant import Restaurant

PROMPTS_DIR = Path(__file__).parent / "prompts"
MAX_DISHES = 5


@cache
def system_prompt() -> str:
    return (PROMPTS_DIR / "system.md").read_text(encoding="utf-8")


def _to_json(data) -> str:
    """Compact JSON with sorted keys, so identical requests give identical prompts.

    '<' and '>' are escaped (still valid JSON) so text from reviews or the user can
    never close or open one of the prompt's tags.
    """
    text = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e")


def candidate_payload(r: Restaurant) -> dict:
    """The fields the model needs, and no more (architecture §5.6)."""
    return {
        "restaurant_id": r.restaurant_id,
        "name": r.name,
        "location": r.location,
        "cuisines": r.cuisines,
        "cost_for_two": r.cost_for_two,
        "budget": r.budget,
        "rating": r.rating,
        "votes": r.votes,
        "rest_type": r.rest_type,
        "online_order": r.online_order,
        "book_table": r.book_table,
        # Dishes diners liked at this restaurant. Named "dish_liked", the model read them
        # as the user's own preferences ("matches your preferred items").
        "popular_dishes": r.dish_liked[:MAX_DISHES],
        "review_snippets": r.review_snippets,
    }


def user_message(
    prefs: UserPreferences, candidates: list[Restaurant], relaxations: list[str]
) -> str:
    preferences = {
        "location": prefs.location,
        "budget": prefs.budget or "any",
        "cuisines": prefs.cuisines or "any",
        "min_rating": prefs.min_rating,
        "additional_preferences": prefs.additional_preferences,
        "top_n": prefs.top_n,
    }
    parts = [f"<preferences>\n{_to_json(preferences)}\n</preferences>"]
    if relaxations:
        parts.append(f"<relaxations>\n{_to_json(relaxations)}\n</relaxations>")
    rows = "\n".join(_to_json(candidate_payload(r)) for r in candidates)
    parts.append(f"<candidates>\n{rows}\n</candidates>")
    count = min(prefs.top_n, len(candidates))
    parts.append(f"Recommend {count} of these {len(candidates)} candidates.")
    return "\n\n".join(parts)


def approx_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token) for development logging."""
    return len(text) // 4
