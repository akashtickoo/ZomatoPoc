"""Template explanations, used when the LLM is unavailable or returns too few picks.

They state only facts from the restaurant record and leave out anything missing,
so they never show "None ★" or "₹None" (EC-VAL-07).
"""

from app.models.preferences import UserPreferences
from app.models.restaurant import Restaurant


def format_votes(votes: int) -> str:
    """775 -> '775'; 1234 -> '1.2k'; 1000 -> '1k'."""
    if votes < 1000:
        return str(votes)
    return f"{votes / 1000:.1f}".removesuffix(".0") + "k"


def _rating_phrase(r: Restaurant) -> str:
    if r.rating is None:
        return "New, not yet rated"
    if r.rating >= 4.5:
        label = "Top rated"
    elif r.rating >= 4.0:
        label = "Highly rated"
    elif r.rating >= 3.5:
        label = "Well rated"
    else:
        label = "Rated"
    return f"{label} ({r.rating:.1f} ★, {format_votes(r.votes)} votes)"


def _kind_phrase(r: Restaurant, prefs: UserPreferences) -> str:
    wanted = {c.casefold() for c in prefs.cuisines}
    matched = [c for c in r.cuisines if c.casefold() in wanted]
    shown = matched or r.cuisines[:2]
    venue = r.rest_type[0].lower() if r.rest_type else ""
    if venue in {c.casefold() for c in shown}:
        venue = ""  # avoid "Cafe cafe"
    return " ".join(part for part in (", ".join(shown), venue) if part) or "restaurant"


def _cost_phrase(r: Restaurant, prefs: UserPreferences) -> str:
    if r.cost_for_two is None:
        return ""
    phrase = f", ₹{r.cost_for_two:,} for two"
    if prefs.budget is not None and r.budget is not None:
        fit = "within your budget" if r.budget == prefs.budget else f"{r.budget} budget"
        phrase += f" ({fit})"
    return phrase


def template_explanation(r: Restaurant, prefs: UserPreferences) -> str:
    """E.g. 'Highly rated (4.1 ★, 775 votes): North Indian casual dining in Banashankari,
    ₹800 for two (within your budget).'"""
    return f"{_rating_phrase(r)}: {_kind_phrase(r, prefs)} in {r.location}{_cost_phrase(r, prefs)}."
