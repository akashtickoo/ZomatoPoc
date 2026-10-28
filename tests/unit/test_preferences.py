"""Tests for UserPreferences. Case IDs refer to docs/edge-case.md §4."""

import pytest
from pydantic import ValidationError

from app.models.preferences import UserPreferences


def prefs(**fields) -> UserPreferences:
    return UserPreferences(**({"location": "Indiranagar"} | fields))


def test_defaults():
    p = prefs()
    assert (p.budget, p.cuisines, p.min_rating, p.additional_preferences, p.top_n) == (
        None,
        [],
        0.0,
        None,
        5,
    )


@pytest.mark.parametrize("location", ["", "   "])
def test_location_is_required(location):  # EC-IN-01
    with pytest.raises(ValidationError):
        UserPreferences(location=location)


def test_location_whitespace_is_normalized():  # EC-IN-02
    assert prefs(location="  Koramangala   5th Block ").location == "Koramangala 5th Block"


@pytest.mark.parametrize(
    ("raw", "expected"), [("LOW", "low"), (" Medium ", "medium"), (None, None)]
)
def test_budget_is_case_insensitive(raw, expected):  # EC-IN-11
    assert prefs(budget=raw).budget == expected


def test_unknown_budget_is_rejected():  # EC-IN-11
    with pytest.raises(ValidationError):
        prefs(budget="cheap")


def test_cuisines_drop_blanks_and_repeats():  # EC-IN-22, EC-IN-24
    assert prefs(cuisines=["chinese", "", " Chinese ", "Thai"]).cuisines == ["chinese", "Thai"]


def test_too_many_cuisines_are_rejected():  # EC-IN-23
    with pytest.raises(ValidationError):
        prefs(cuisines=[f"Cuisine {i}" for i in range(11)])


def test_cuisines_must_be_a_list():  # EC-IN-55
    with pytest.raises(ValidationError):
        prefs(cuisines="Chinese")


@pytest.mark.parametrize("value", [-1, 5.1, "four"])
def test_min_rating_out_of_range_or_non_numeric(value):  # EC-IN-31, EC-IN-32
    with pytest.raises(ValidationError):
        prefs(min_rating=value)


@pytest.mark.parametrize(("value", "expected"), [("4", 4.0), (5.0, 5.0), (4.25, 4.25)])
def test_min_rating_accepted_values(value, expected):  # EC-IN-33, EC-IN-34
    assert prefs(min_rating=value).min_rating == expected


@pytest.mark.parametrize("text", ["", "   \n "])
def test_blank_free_text_becomes_none(text):  # EC-IN-40
    assert prefs(additional_preferences=text).additional_preferences is None


def test_free_text_length_limit():  # EC-IN-41
    assert len(prefs(additional_preferences="x" * 300).additional_preferences) == 300
    with pytest.raises(ValidationError):
        prefs(additional_preferences="x" * 301)


@pytest.mark.parametrize("top_n", [0, 11])
def test_top_n_range(top_n):  # EC-IN-50
    with pytest.raises(ValidationError):
        prefs(top_n=top_n)


def test_unknown_fields_are_rejected():  # EC-IN-52
    with pytest.raises(ValidationError):
        prefs(cuisine=["Chinese"])
