"""Deterministic candidate selection: hard filters, progressive relaxation and pre-ranking.

Implements architecture §5.5. Case IDs refer to docs/edge-case.md §5; D-xx to its §10.

Manual check:
    python -m app.services.filtering --location Indiranagar --cuisine Italian --budget medium
"""

import argparse
import difflib
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from app.config import settings
from app.data.repository import RestaurantRepository
from app.models.preferences import UserPreferences
from app.models.restaurant import Budget, Restaurant

# Budget sets tried in order when there are too few matches. A medium budget widens to
# cheaper places before pricier ones (D-05).
BUDGET_WIDENING: dict[str, list[tuple[Budget, ...]]] = {
    "low": [("low", "medium")],
    "high": [("medium", "high")],
    "medium": [("low", "medium"), ("low", "medium", "high")],
}
RATING_STEP = 0.5
MAX_OUTLETS_PER_BRAND = 2  # D-08
CITY_NAMES = {"bangalore", "bengaluru"}  # as a location, means the whole city (D-04)
_EPSILON = 1e-9


class UnknownValueError(ValueError):
    """A location or cuisine that isn't in the data, with close matches (EC-IN-03, 21)."""

    def __init__(self, field: str, value: str, suggestions: list[str]):
        self.field = field
        self.value = value
        self.suggestions = suggestions
        message = f"Unknown {field} '{value}'."
        if suggestions:
            message += f" Did you mean: {', '.join(suggestions)}?"
        elif field == "location":
            message += " Locations are Bangalore neighbourhoods; the data covers Bangalore only."
        super().__init__(message)


@dataclass(frozen=True)
class Constraints:
    """The filters actually applied, after any relaxation."""

    location: str | None  # canonical spelling; None means the whole city
    area: str | None  # listing area, set once the location has been widened
    budgets: tuple[Budget, ...] | None  # None means any budget
    cuisines: frozenset[str]  # case-folded; empty means any cuisine
    min_rating: float


@dataclass(frozen=True)
class FilterResult:
    candidates: list[Restaurant]  # best first, at most top_k
    relaxations: list[str]  # one readable note per relaxation that was applied
    total_matched: int  # matches before the top-K cut
    constraints: Constraints


def suggest(value: str, choices: list[str], n: int = 3) -> list[str]:
    by_key = {choice.casefold(): choice for choice in choices}
    matches = difflib.get_close_matches(value.casefold(), list(by_key), n=n, cutoff=0.6)
    return [by_key[m] for m in matches]


def build_constraints(repo: RestaurantRepository, prefs: UserPreferences) -> Constraints:
    """Resolve the user's text to known values; raises UnknownValueError otherwise."""
    if prefs.location.casefold() in CITY_NAMES:
        location = None
    else:
        location = repo.resolve_location(prefs.location)
        if location is None:
            raise UnknownValueError(
                "location", prefs.location, suggest(prefs.location, repo.get_locations())
            )

    cuisines = set()
    for cuisine in prefs.cuisines:
        if repo.resolve_cuisine(cuisine) is None:
            raise UnknownValueError("cuisine", cuisine, suggest(cuisine, repo.get_cuisines()))
        cuisines.add(cuisine.casefold())

    return Constraints(
        location=location,
        area=None,
        budgets=(prefs.budget,) if prefs.budget else None,
        cuisines=frozenset(cuisines),
        min_rating=prefs.min_rating,
    )


def matches(repo: RestaurantRepository, c: Constraints) -> np.ndarray:
    """Boolean mask over repo.frame of the rows that satisfy every constraint."""
    frame = repo.frame
    mask = np.ones(len(frame), dtype=bool)
    if c.location is not None:
        in_location = repo.location_mask(c.location)  # exact, not substring (EC-FIL-08)
        if c.area is not None:
            in_location = in_location | repo.area_mask(c.area)
        mask &= in_location
    if c.budgets is not None:
        # Restaurants without a cost have no budget and never match a budget filter.
        mask &= frame["budget"].isin(c.budgets).to_numpy()
    if c.cuisines:
        mask &= np.logical_or.reduce([repo.cuisine_mask(k) for k in c.cuisines])  # any-of
    if c.min_rating > 0:
        # NaN compares false, so unrated restaurants drop out only here (EC-FIL-04, 05)
        mask &= frame["rating"].to_numpy() >= c.min_rating - _EPSILON
    return mask


@dataclass(frozen=True)
class Relaxation:
    field: str  # the Constraints field this step loosens
    value: object  # its loosened value
    note: str


Step = Callable[[Constraints], Relaxation | None]


def _describe_budgets(budgets: tuple[Budget, ...]) -> str:
    return "any budget" if len(budgets) == 3 else " or ".join(budgets)


def _format_rating(value: float) -> str:
    """4.0 -> '4.0', 4.25 -> '4.25'."""
    text = f"{value:.2f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _relaxation_steps(original: Constraints, repo: RestaurantRepository) -> list[Step]:
    """Budget first, then rating, then location (architecture §5.5).

    Cuisine is never relaxed: it is the user's most explicit choice (EC-FIL-19).
    """
    steps: list[Step] = []

    if original.budgets is not None:  # no budget means nothing to widen (EC-FIL-11)
        (budget,) = original.budgets
        for widened in BUDGET_WIDENING[budget]:
            note = f"Budget widened from {budget} to {_describe_budgets(widened)}."
            steps.append(lambda c, w=widened, n=note: Relaxation("budgets", w, n))

    def lower_rating(c: Constraints) -> Relaxation | None:
        if c.min_rating <= 0:
            return None
        lowered = max(0.0, c.min_rating - RATING_STEP)  # never below 0 (EC-FIL-12)
        note = (
            f"Minimum rating lowered from {_format_rating(c.min_rating)} "
            f"to {_format_rating(lowered)}."
        )
        return Relaxation("min_rating", lowered, note)

    def widen_location(c: Constraints) -> Relaxation | None:
        area = repo.primary_area(c.location) if c.location is not None else None
        if area is None:
            return None
        # Zomato's listing areas are not always next door, so the note doesn't say "nearby".
        note = f"Search widened from {c.location} to places Zomato lists under {area}."
        return Relaxation("area", area, note)

    return steps + [lower_rating, widen_location]


def rank(matched: pd.DataFrame, top_k: int) -> list[int]:
    """Row positions of the best `top_k` matches, best first.

    Rated restaurants come first by Bayesian score, unrated ones after them by votes (D-07).
    Ties break on votes, then name, then ID, so the order is reproducible (EC-FIL-33).
    No brand fills more than MAX_OUTLETS_PER_BRAND slots unless there is nothing else (D-08).
    """
    ordered = matched.assign(unrated=matched["rating"].isna()).sort_values(
        ["unrated", "score", "votes", "name", "restaurant_id"],
        ascending=[True, False, False, True, True],
        kind="stable",
    )
    picked: list[int] = []
    overflow: list[int] = []
    per_brand: Counter = Counter()
    for position, brand in zip(ordered.index, ordered["name_key"], strict=True):
        if len(picked) == top_k:
            break
        if per_brand[brand] < MAX_OUTLETS_PER_BRAND:
            per_brand[brand] += 1
            picked.append(position)
        else:
            overflow.append(position)
    return picked + overflow[: top_k - len(picked)]


def filter_candidates(
    repo: RestaurantRepository,
    prefs: UserPreferences,
    *,
    top_k: int | None = None,
    min_candidates: int | None = None,
) -> FilterResult:
    top_k = top_k or settings.candidate_top_k
    min_candidates = min_candidates or settings.min_candidates
    frame = repo.frame

    original = build_constraints(repo, prefs)
    current = original
    count = int(matches(repo, current).sum())

    # Loosen one step at a time, on top of the previous steps, until there are enough
    # matches. Steps stack because sometimes no single step helps but a combination does.
    applied: list[tuple[Relaxation, object]] = []  # (step, the value it replaced)
    for step in _relaxation_steps(original, repo):
        if count >= min_candidates:  # exactly MIN_CANDIDATES is enough (EC-FIL-02)
            break
        relaxation = step(current)
        if relaxation is None:
            continue
        applied.append((relaxation, getattr(current, relaxation.field)))
        current = replace(current, **{relaxation.field: relaxation.value})
        count = int(matches(repo, current).sum())

    # Undo, latest first, any step the final result doesn't depend on, so that every
    # note shown to the user changed what they see (EC-FIL-10, EC-FIL-16).
    relaxations: list[str] = []
    kept_fields: set[str] = set()
    for relaxation, previous in reversed(applied):
        if relaxation.field in kept_fields:
            continue  # superseded by a later step on the same filter, e.g. budget twice
        undone = replace(current, **{relaxation.field: previous})
        if count > 0 and int(matches(repo, undone).sum()) == count:
            current = undone
        else:
            kept_fields.add(relaxation.field)
            relaxations.insert(0, relaxation.note)
    if count == 0:
        relaxations = []  # nothing found even fully relaxed (EC-FIL-17)
        current = original

    positions = rank(frame[matches(repo, current)], top_k)
    return FilterResult(
        candidates=repo.at_positions(positions),
        relaxations=relaxations,
        total_matched=count,
        constraints=current,
    )


def main(argv: list[str] | None = None) -> int:
    from app.services.explanations import template_explanation

    parser = argparse.ArgumentParser(description="Check candidate filtering by hand.")
    parser.add_argument("--location", required=True)
    parser.add_argument("--cuisine", action="append", default=[], dest="cuisines")
    parser.add_argument("--budget", choices=["low", "medium", "high"])
    parser.add_argument("--min-rating", type=float, default=0.0)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args(argv)

    repo = RestaurantRepository.load()
    prefs = UserPreferences(
        location=args.location,
        cuisines=args.cuisines,
        budget=args.budget,
        min_rating=args.min_rating,
        top_n=args.top_n,
    )
    start = time.perf_counter()
    try:
        result = filter_candidates(repo, prefs)
    except UnknownValueError as exc:
        print(exc)
        return 2
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(
        f"{result.total_matched} matches, {len(result.candidates)} candidates, {elapsed_ms:.1f} ms"
    )
    for note in result.relaxations:
        print(f"  relaxed: {note}")
    for i, r in enumerate(result.candidates[: prefs.top_n], start=1):
        print(f"{i}. {r.name} [{r.restaurant_id}]")
        print(f"   {template_explanation(r, prefs)}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
