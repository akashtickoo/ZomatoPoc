"""Tests for template explanations (EC-VAL-07)."""

import pytest

from app.models.preferences import UserPreferences
from app.models.restaurant import Restaurant
from app.services.explanations import format_votes, template_explanation


def restaurant(make_record, **overrides) -> Restaurant:
    record = make_record("Jalsa", **({"location": "Banashankari", "budget": "high"} | overrides))
    return Restaurant.model_validate(record)


def prefs(**fields) -> UserPreferences:
    return UserPreferences(**({"location": "Banashankari"} | fields))


@pytest.mark.parametrize(
    ("votes", "expected"), [(0, "0"), (775, "775"), (1000, "1k"), (1234, "1.2k"), (14700, "14.7k")]
)
def test_format_votes(votes, expected):
    assert format_votes(votes) == expected


def test_full_explanation(make_record):
    r = restaurant(
        make_record,
        cuisines=["North Indian", "Mughlai"],
        cost_for_two=800,
        rating=4.1,
        votes=775,
    )
    assert template_explanation(r, prefs(budget="high")) == (
        "Highly rated (4.1 ★, 775 votes): North Indian, Mughlai casual dining in Banashankari, "
        "₹800 for two (within your budget)."
    )


def test_mentions_a_different_budget_band(make_record):
    r = restaurant(make_record, cost_for_two=800)
    assert "(high budget)" in template_explanation(r, prefs(budget="medium"))


def test_no_budget_note_without_a_budget_preference(make_record):
    assert "budget" not in template_explanation(restaurant(make_record), prefs())


def test_matching_cuisines_come_first(make_record):
    r = restaurant(make_record, cuisines=["North Indian", "Chinese", "Biryani"])
    assert ": Biryani casual dining" in template_explanation(r, prefs(cuisines=["biryani"]))


def test_unrated_restaurant(make_record):
    text = template_explanation(restaurant(make_record, rating=None), prefs())
    assert text.startswith("New, not yet rated:")
    assert "None" not in text


def test_missing_cost_is_left_out(make_record):
    r = restaurant(make_record, cost_for_two=None, budget=None)
    text = template_explanation(r, prefs(budget="low"))
    assert "₹" not in text
    assert "None" not in text


def test_cuisine_and_venue_are_not_repeated(make_record):
    r = restaurant(make_record, cuisines=["Cafe"], rest_type=["Cafe"])
    assert ": Cafe in Banashankari" in template_explanation(r, prefs())


def test_falls_back_to_restaurant_when_type_and_cuisine_are_unknown(make_record):
    r = restaurant(make_record, cuisines=[], rest_type=[])
    assert ": restaurant in Banashankari" in template_explanation(r, prefs())
