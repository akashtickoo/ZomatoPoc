"""Tests for the ingest helpers that don't need the network."""

import pandas as pd

from app.data.ingest import build_metadata, validate_records


def record(**overrides):
    base = {
        "restaurant_id": "abc123def456",
        "name": "Jalsa",
        "address": "942, 21st Main Road",
        "city": "Bangalore",
        "location": "Banashankari",
        "areas": ["Banashankari"],
        "cuisines": ["North Indian"],
        "cost_for_two": 800,
        "budget": "high",
        "rating": 4.1,
        "votes": 775,
        "rest_type": ["Casual Dining"],
        "online_order": True,
        "book_table": True,
        "dish_liked": [],
        "listing_types": ["Buffet"],
        "review_snippets": [],
    }
    return base | overrides


def test_valid_records_pass():
    assert validate_records([record(), record(restaurant_id="other", address="5 MG Road")]) == []


def test_reports_schema_violations():
    problems = validate_records([record(rating=7.5)])
    assert len(problems) == 1
    assert "abc123def456" in problems[0]


def test_reports_duplicate_ids_and_pairs():
    problems = validate_records([record(), record()])
    assert any("duplicate restaurant_id" in p for p in problems)
    assert any("duplicate (name, address)" in p for p in problems)


def test_reports_budget_without_cost():
    problems = validate_records([record(cost_for_two=None, budget="high")])
    assert any("budget and cost_for_two disagree" in p for p in problems)


def test_metadata_lists_are_sorted_and_unique(make_settings):
    df = pd.DataFrame(
        {
            "location": ["BTM", "Indiranagar", "BTM"],
            "areas": [["BTM", "HSR"], ["Indiranagar"], ["BTM"]],
            "cuisines": [["thai", "Chinese"], ["Chinese"], ["Biryani"]],
        }
    )
    meta = build_metadata(df, make_settings())
    assert meta["locations"] == ["BTM", "Indiranagar"]
    assert meta["areas"] == ["BTM", "HSR", "Indiranagar"]
    assert meta["cuisines"] == ["Biryani", "Chinese", "thai"]
    assert meta["dataset_revision"] == "5738e9eda2fad49ad51c6e0ed26e761d9b947133"
    assert meta["restaurants"] == 3
