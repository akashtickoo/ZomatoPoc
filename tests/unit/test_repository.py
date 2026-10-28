"""Tests for RestaurantRepository."""

import math

import pytest

from app.data.repository import DataLoadError, RestaurantRepository


def test_missing_file_says_to_run_ingest(make_settings, tmp_path):  # EC-CFG-01
    cfg = make_settings(data_path=tmp_path / "missing.parquet")
    with pytest.raises(DataLoadError, match="app.data.ingest"):
        RestaurantRepository.load(cfg)


def test_corrupt_file_is_reported(make_settings, tmp_path):  # EC-CFG-02
    path = tmp_path / "restaurants.parquet"
    path.write_bytes(b"not parquet")
    with pytest.raises(DataLoadError, match="Could not read"):
        RestaurantRepository.load(make_settings(data_path=path))


def test_records_that_dont_match_the_schema_are_reported(make_settings, make_record):
    record = make_record()
    del record["cuisines"]
    with pytest.raises(DataLoadError, match="schema"):
        RestaurantRepository.from_records([record], make_settings())


def test_empty_data_is_reported(make_settings):
    with pytest.raises(DataLoadError):
        RestaurantRepository.from_records([], make_settings())


def test_loads_the_fixture(fixture_repo):
    assert len(fixture_repo) == 50


def test_budget_is_derived_from_current_config(make_repo, make_record):  # D-01
    record = make_record(cost_for_two=400, budget="low")  # stored band is ignored
    assert make_repo([record]).get_by_ids([record["restaurant_id"]])[0].budget == "medium"
    repo = make_repo([record], budget_low_max=100, budget_medium_max=200)
    assert repo.get_by_ids([record["restaurant_id"]])[0].budget == "high"


def test_get_by_ids_keeps_order_and_skips_unknown(make_repo, make_record):
    a, b = make_record("A"), make_record("B")
    repo = make_repo([a, b])
    found = repo.get_by_ids([b["restaurant_id"], "unknown", a["restaurant_id"]])
    assert [r.name for r in found] == ["B", "A"]


def test_lookups_are_case_insensitive(make_repo, make_record):
    repo = make_repo([make_record(location="Koramangala 5th Block", cuisines=["North Indian"])])
    assert repo.resolve_location("  koramangala 5th   block") == "Koramangala 5th Block"
    assert repo.resolve_cuisine("NORTH INDIAN") == "North Indian"
    assert repo.resolve_location("Koramangala") is None


def test_locations_and_cuisines_are_sorted(make_repo, make_record):
    repo = make_repo(
        [
            make_record("A", location="indiranagar", cuisines=["thai", "Chinese"]),
            make_record("B", location="BTM", cuisines=["Biryani"]),
        ]
    )
    assert repo.get_locations() == ["BTM", "indiranagar"]
    assert repo.get_cuisines() == ["Biryani", "Chinese", "thai"]


def test_primary_area_is_the_most_common_listing_area(make_repo, make_record):
    repo = make_repo(
        [
            make_record("A", location="Ulsoor", areas=["MG Road", "Indiranagar"]),
            make_record("B", location="Ulsoor", areas=["MG Road"]),
            make_record("C", location="Domlur", areas=["Old Airport Road", "Indiranagar"]),
        ]
    )
    assert repo.primary_area("ulsoor") == "MG Road"
    assert repo.primary_area("Domlur") == "Indiranagar"  # tie: alphabetically first
    assert repo.primary_area("Nowhere") is None


def test_prerank_score(make_repo, make_record):
    repo = make_repo(
        [
            make_record("Rated", rating=4.0, votes=0),
            make_record("Other", rating=3.0, votes=50),
            make_record("Unrated", rating=None, votes=10),
        ]
    )
    scores = dict(zip(repo.frame["name"], repo.frame["score"], strict=True))
    assert repo.mean_rating == pytest.approx(3.5)
    assert scores["Rated"] == pytest.approx(3.5)  # no votes: exactly the mean (EC-FIL-31)
    assert scores["Other"] == pytest.approx(0.5 * 3.0 + 0.5 * 3.5)  # 50 votes = half weight
    assert math.isnan(scores["Unrated"])
