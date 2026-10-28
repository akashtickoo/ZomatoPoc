"""Tests for candidate filtering. Case IDs refer to docs/edge-case.md §5; D-xx to its §10."""

import statistics
import time

import pytest

from app.config import settings
from app.data.repository import RestaurantRepository
from app.models.preferences import UserPreferences
from app.services.filtering import UnknownValueError, filter_candidates


def run(repo, top_k=25, min_candidates=1, **prefs):
    """Filter with explicit limits so tests don't depend on the developer's .env."""
    preferences = UserPreferences(**({"location": "Indiranagar"} | prefs))
    return filter_candidates(repo, preferences, top_k=top_k, min_candidates=min_candidates)


def names(result) -> list[str]:
    return [r.name for r in result.candidates]


class TestHardFilters:
    def test_location_is_an_exact_match(self, make_repo, make_record):  # EC-FIL-08
        repo = make_repo(
            [
                make_record("In", location="Koramangala"),
                make_record("Out", location="Koramangala 5th Block"),
            ]
        )
        assert names(run(repo, location="koramangala")) == ["In"]

    def test_city_name_means_all_locations(self, make_repo, make_record):  # D-04
        repo = make_repo([make_record("A", location="BTM"), make_record("B", location="HSR")])
        assert sorted(names(run(repo, location="Bengaluru"))) == ["A", "B"]

    def test_unknown_location_suggests_close_matches(self, make_repo, make_record):  # EC-IN-03
        repo = make_repo([make_record(location="Indiranagar")])
        with pytest.raises(UnknownValueError) as exc:
            run(repo, location="Indranagar")
        assert exc.value.suggestions == ["Indiranagar"]

    def test_other_city_says_bangalore_only(self, make_repo, make_record):  # EC-IN-05
        repo = make_repo([make_record(location="Indiranagar")])
        with pytest.raises(UnknownValueError, match="Bangalore only"):
            run(repo, location="Delhi")

    def test_unknown_cuisine_suggests_close_matches(self, make_repo, make_record):  # EC-IN-21
        repo = make_repo([make_record(cuisines=["Chinese"])])
        with pytest.raises(UnknownValueError) as exc:
            run(repo, cuisines=["Chinesee"])
        assert exc.value.field == "cuisine"
        assert exc.value.suggestions == ["Chinese"]

    def test_budget(self, make_repo, make_record):  # EC-FIL-03, EC-DATA-24
        repo = make_repo(
            [
                make_record("Low", cost_for_two=300),
                make_record("Medium", cost_for_two=301),
                make_record("No cost", cost_for_two=None),
            ]
        )
        assert names(run(repo, budget="low")) == ["Low"]
        assert sorted(names(run(repo))) == ["Low", "Medium", "No cost"]

    def test_cuisine_any_of_and_case_insensitive(self, make_repo, make_record):  # EC-FIL-07
        repo = make_repo(
            [
                make_record("Both", cuisines=["Chinese", "Thai"]),
                make_record("Thai", cuisines=["Thai"]),
                make_record("Other", cuisines=["Biryani"]),
            ]
        )
        assert sorted(names(run(repo, cuisines=["thai"]))) == ["Both", "Thai"]

    def test_min_rating(self, make_repo, make_record):  # EC-FIL-04, 05, 06
        repo = make_repo(
            [
                make_record("Exact", rating=4.0),
                make_record("Below", rating=3.9),
                make_record("Unrated", rating=None),
            ]
        )
        assert sorted(names(run(repo))) == ["Below", "Exact", "Unrated"]
        assert names(run(repo, min_rating=4.0)) == ["Exact"]


class TestRelaxation:
    def test_exactly_min_candidates_is_enough(self, make_repo, make_record):  # EC-FIL-02
        repo = make_repo(
            [make_record(f"Low {i}", cost_for_two=200) for i in range(5)]
            + [make_record("Medium", cost_for_two=400)]
        )
        result = run(repo, budget="low", min_candidates=5)
        assert result.relaxations == []
        assert result.total_matched == 5

    def test_low_budget_widens_to_medium(self, make_repo, make_record):  # EC-FIL-13
        repo = make_repo([make_record("Medium", cost_for_two=400)])
        result = run(repo, budget="low", min_candidates=1)
        assert names(result) == ["Medium"]
        assert result.relaxations == ["Budget widened from low to low or medium."]

    def test_high_budget_widens_to_medium(self, make_repo, make_record):  # EC-FIL-14
        repo = make_repo([make_record("Medium", cost_for_two=400)])
        result = run(repo, budget="high", min_candidates=1)
        assert result.relaxations == ["Budget widened from high to medium or high."]

    def test_medium_budget_widens_to_low_first(self, make_repo, make_record):  # D-05
        repo = make_repo(
            [make_record("Low", cost_for_two=200), make_record("High", cost_for_two=900)]
        )
        result = run(repo, budget="medium", min_candidates=1)
        assert names(result) == ["Low"]
        assert result.relaxations == ["Budget widened from medium to low or medium."]

    def test_medium_budget_then_widens_to_any(self, make_repo, make_record):  # D-05
        repo = make_repo([make_record("High", cost_for_two=900)])
        result = run(repo, budget="medium", min_candidates=1)
        assert names(result) == ["High"]
        # The low-or-medium step is superseded, so only one budget note is shown.
        assert result.relaxations == ["Budget widened from medium to any budget."]

    def test_rating_relaxed_when_there_is_no_budget(self, make_repo, make_record):  # EC-FIL-11
        repo = make_repo([make_record("Good", rating=4.2)])
        result = run(repo, min_rating=4.5, min_candidates=1)
        assert result.relaxations == ["Minimum rating lowered from 4.5 to 4.0."]

    def test_rating_never_goes_below_zero(self, make_repo, make_record):  # EC-FIL-12
        repo = make_repo([make_record("Unrated", rating=None)])
        result = run(repo, min_rating=0.3, min_candidates=1)
        assert names(result) == ["Unrated"]
        assert result.relaxations == ["Minimum rating lowered from 0.3 to 0.0."]

    def test_location_widens_to_primary_listing_area(self, make_repo, make_record):
        repo = make_repo(
            [
                make_record(
                    "Ulsoor place", location="Ulsoor", areas=["MG Road"], cuisines=["Thai"]
                ),
                make_record("MG Road place", location="MG Road", areas=["MG Road"]),
                make_record("Far away", location="Whitefield", areas=["Whitefield"]),
            ]
        )
        result = run(repo, location="Ulsoor", cuisines=["North Indian"], min_candidates=1)
        assert names(result) == ["MG Road place"]
        assert result.relaxations == [
            "Search widened from Ulsoor to places Zomato lists under MG Road."
        ]

    def test_stops_after_the_first_step_that_is_enough(self, make_repo, make_record):  # EC-FIL-10
        repo = make_repo([make_record("Medium", cost_for_two=400, rating=4.6)])
        result = run(repo, budget="low", min_rating=4.5, min_candidates=1)
        assert len(result.relaxations) == 1
        assert result.constraints.min_rating == 4.5

    def test_combines_steps_when_none_is_enough_alone(self, make_repo, make_record):
        # Only reachable with a wider budget, a lower rating AND a wider location.
        repo = make_repo(
            [
                make_record("Anchor", location="Sadashiv Nagar", areas=["Malleshwaram"]),
                make_record(
                    "Target",
                    location="Malleshwaram",
                    areas=["Malleshwaram"],
                    cuisines=["Italian"],
                    cost_for_two=500,
                    rating=4.1,
                ),
            ]
        )
        result = run(
            repo,
            location="Sadashiv Nagar",
            cuisines=["Italian"],
            budget="low",
            min_rating=4.5,
            min_candidates=1,
        )
        assert names(result) == ["Target"]
        assert result.relaxations == [
            "Budget widened from low to low or medium.",
            "Minimum rating lowered from 4.5 to 4.0.",
            "Search widened from Sadashiv Nagar to places Zomato lists under Malleshwaram.",
        ]

    def test_steps_that_did_not_help_are_undone(self, make_repo, make_record):  # EC-FIL-16
        # Widening the budget adds nothing (no medium places); lowering the rating is enough.
        repo = make_repo([make_record(f"Low {i}", cost_for_two=200, rating=4.2) for i in range(3)])
        result = run(repo, budget="low", min_rating=4.5, min_candidates=3)
        assert result.relaxations == ["Minimum rating lowered from 4.5 to 4.0."]
        assert result.constraints.budgets == ("low",)

    def test_nothing_found_even_fully_relaxed(self, make_repo, make_record):  # EC-FIL-17
        repo = make_repo([make_record(cuisines=["Chinese"])])
        result = run(repo, cuisines=["Chinese"], location="Indiranagar", min_rating=5.0)
        assert result.candidates == []
        assert result.relaxations == []
        assert result.total_matched == 0

    def test_returns_the_few_found_after_relaxing(self, make_repo, make_record):  # EC-FIL-18
        repo = make_repo([make_record("Only", cost_for_two=400)])
        result = run(repo, budget="low", min_candidates=5)
        assert names(result) == ["Only"]
        assert result.relaxations

    def test_cuisine_is_never_relaxed(self, make_repo, make_record):  # EC-FIL-19
        repo = make_repo([make_record(cuisines=["Chinese"]), make_record("X", cuisines=["Thai"])])
        assert run(repo, cuisines=["Thai"], min_candidates=5).total_matched == 1


class TestRanking:
    def test_many_votes_beat_a_perfect_score_from_few(self, make_repo, make_record):  # EC-FIL-30
        # Fillers elsewhere bring the dataset mean C down to a realistic 3.29.
        fillers = [make_record(f"Filler {i}", location="Whitefield", rating=3.0) for i in range(10)]
        repo = make_repo(
            [
                make_record("Few votes", rating=5.0, votes=3),
                make_record("Many votes", rating=4.3, votes=3000),
                make_record("Average", rating=3.5, votes=500),
            ]
            + fillers
        )
        # Scores: Many 4.30, Average 3.48, Few 3.39. Three 5-star votes are pulled
        # almost all the way to the mean, below a solidly reviewed 3.5.
        assert names(run(repo)) == ["Many votes", "Average", "Few votes"]

    def test_unrated_come_after_rated_sorted_by_votes(self, make_repo, make_record):  # D-07
        repo = make_repo(
            [
                make_record("Unrated quiet", rating=None, votes=1),
                make_record("Unrated popular", rating=None, votes=40),
                make_record("Poorly rated", rating=2.0, votes=10),
            ]
        )
        assert names(run(repo)) == ["Poorly rated", "Unrated popular", "Unrated quiet"]

    def test_ties_break_by_name(self, make_repo, make_record):  # EC-FIL-33
        repo = make_repo([make_record("Bravo"), make_record("Alpha"), make_record("Charlie")])
        assert names(run(repo)) == ["Alpha", "Bravo", "Charlie"]

    def test_at_most_two_outlets_per_brand(self, make_repo, make_record):  # D-08
        chain = [
            make_record("Cafe Coffee Day", address=f"{i} Road", rating=4.5, votes=1000)
            for i in range(4)
        ]
        others = [make_record("Local A", rating=3.5), make_record("Local B", rating=3.4)]
        repo = make_repo(chain + others)
        assert names(run(repo, top_k=4)) == [
            "Cafe Coffee Day",
            "Cafe Coffee Day",
            "Local A",
            "Local B",
        ]
        # With nothing else left, the remaining outlets fill the list.
        assert names(run(repo, top_k=6)).count("Cafe Coffee Day") == 4

    def test_top_k_cut(self, make_repo, make_record):
        repo = make_repo([make_record(f"R{i}", votes=i) for i in range(10)])
        result = run(repo, top_k=3)
        assert len(result.candidates) == 3
        assert result.total_matched == 10

    def test_fewer_matches_than_top_k(self, make_repo, make_record):  # EC-FIL-35
        repo = make_repo([make_record("A"), make_record("B")])
        assert len(run(repo, top_k=25).candidates) == 2


class TestAgainstFixture:
    """Compare filtering on 50 real restaurants with a plain-Python reference."""

    @staticmethod
    def expected_ids(repo, location=None, budget=None, cuisines=(), min_rating=0.0):
        wanted = {c.casefold() for c in cuisines}
        return {
            r.restaurant_id
            for r in repo.get_by_ids(list(repo.frame["restaurant_id"]))
            if (location is None or r.location == location)
            and (budget is None or r.budget == budget)
            and (not wanted or wanted & {c.casefold() for c in r.cuisines})
            and (min_rating == 0 or (r.rating is not None and r.rating >= min_rating))
        }

    @pytest.mark.parametrize(
        "query",
        [
            {"location": "Indiranagar"},
            {"location": "BTM", "budget": "low"},
            {"location": "Koramangala 5th Block", "cuisines": ["Cafe", "Italian"]},
            {"location": "Jayanagar", "min_rating": 3.8},
            {"location": "Indiranagar", "budget": "medium", "cuisines": ["North Indian"]},
            {"location": "BTM", "budget": "low", "cuisines": ["South Indian"], "min_rating": 3.5},
        ],
    )
    def test_matches_reference(self, fixture_repo, query):
        result = run(fixture_repo, top_k=100, min_candidates=1, **query)
        assert result.relaxations == []
        assert {r.restaurant_id for r in result.candidates} == self.expected_ids(
            fixture_repo, **query
        )


@pytest.mark.skipif(not settings.data_path.exists(), reason="needs the full processed dataset")
def test_filtering_is_fast_on_the_full_dataset():  # Phase 2 exit criterion: < 100 ms
    repo = RestaurantRepository.load()
    queries = [
        {"location": "BTM"},
        {"location": "Bangalore", "min_rating": 4.0},
        {"location": "Indiranagar", "cuisines": ["Italian"], "budget": "medium"},
        # Needs every relaxation step plus the undo pass: the slowest path.
        {"location": "Kengeri", "cuisines": ["Japanese"], "budget": "medium", "min_rating": 4.8},
    ]
    for query in queries:
        prefs = UserPreferences(**query)
        timings = []
        for _ in range(5):
            start = time.perf_counter()
            filter_candidates(repo, prefs, top_k=25, min_candidates=5)
            timings.append((time.perf_counter() - start) * 1000)
        assert statistics.median(timings) < 100, (query, timings)
