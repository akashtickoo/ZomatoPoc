"""UI tests: the pure helpers, then the Streamlit app driven by AppTest against a live API.

The API runs in a background thread over the fixture data with a fake LLM, so the real
HTTP contract between the UI and the API is exercised without network access or an API
key. Case IDs refer to docs/edge-case.md §9.
"""

import socket
import threading
import time
from pathlib import Path

import pytest
import streamlit as st
import uvicorn
from streamlit.testing.v1 import AppTest

from app.api.routes import get_service
from app.config import settings
from app.llm.client import LLMUnavailableError
from app.main import create_app
from app.services.recommender import NOTICE_FALLBACK, NOTICE_NO_RESULTS
from tests.fakes import FakeLLM
from ui.streamlit_app import (
    ALL_OF_BANGALORE,
    QUICK_SEARCHES,
    available_quick_searches,
    budget_label,
    build_payload,
    card_html,
    count,
    describe_request,
    format_cost,
    format_rating,
    h,
    notice_html,
)

APP_FILE = str(Path(__file__).resolve().parents[2] / "ui" / "streamlit_app.py")
HOSTILE = "[click me](http://evil.example) <img src=x onerror=alert(1)> **bold** $5"


def item(**overrides) -> dict:
    """A recommendation as the API returns it."""
    return {
        "rank": 1,
        "restaurant_id": "abc123def456",
        "name": "Glen's Bakehouse",
        "location": "Indiranagar",
        "cuisines": ["Bakery", "Cafe", "Italian"],
        "rating": 4.3,
        "votes": 5352,
        "cost_for_two": 600,
        "rest_type": ["Bakery", "Cafe"],
        "online_order": True,
        "book_table": False,
        "popular_dishes": ["Pizza", "Pasta"],
        "explanation": "Rated 4.3 with 5,352 votes.",
        "source": "llm",
    } | overrides


# --- Helpers ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rating", "votes", "expected"),
    [(None, 0, "New"), (4.3, 5352, "4.3 ★ (5,352)"), (3.0, 7, "3.0 ★ (7)")],
)
def test_format_rating(rating, votes, expected):  # EC-UI-10
    assert format_rating(rating, votes) == expected


def test_format_cost():  # EC-UI-10
    assert format_cost(None) == "Cost N/A"
    assert format_cost(1200) == "₹1,200 for two"


def test_budget_labels_show_the_configured_thresholds():
    low, medium = settings.budget_low_max, settings.budget_medium_max
    assert budget_label("any") == "Any"
    assert budget_label("low") == f"Low · ≤ ₹{low}"
    assert budget_label("medium") == f"Medium · ₹{low + 1}–{medium}"
    assert budget_label("high") == f"High · ₹{medium + 1}+"


@pytest.mark.parametrize(("n", "expected"), [(0, "0 matches"), (1, "1 match"), (12, "12 matches")])
def test_count(n, expected):
    assert count(n, "match", "matches") == expected
    assert count(n, "candidate").endswith("candidate" if n == 1 else "candidates")


def test_build_payload():
    assert build_payload(ALL_OF_BANGALORE, "any", [], 0.0, "   ", 5) == {
        "location": "Bangalore",  # the whole city (D-04)
        "budget": None,
        "cuisines": [],
        "min_rating": 0.0,
        "additional_preferences": None,
        "top_n": 5,
    }
    payload = build_payload("BTM", "medium", ["Chinese"], 4.0, " rooftop ", 3)
    assert (payload["budget"], payload["additional_preferences"]) == ("medium", "rooftop")
    assert build_payload("BTM", None, [], 0.0, "", 5)["budget"] is None


def test_describe_request():
    payload = build_payload("BTM", "low", ["Chinese", "Thai"], 4.0, "", 5)
    assert describe_request(payload) == "Chinese, Thai · in BTM · low budget · rated 4★ or more"
    everywhere = build_payload(ALL_OF_BANGALORE, "any", [], 0.0, "", 5)
    assert describe_request(everywhere) == "Any cuisine · anywhere in Bangalore"
    with_text = build_payload("HSR", "any", ["Cafe"], 0.0, "quick service", 4)
    assert describe_request(with_text) == "Cafe · in HSR · “quick service”"


def test_card_shows_the_five_required_fields():
    html = card_html(item())
    assert "Glen&#x27;s Bakehouse" in html  # name
    assert '<span class="cc-chip">Italian</span>' in html  # cuisine
    assert "4.3 ★ (5,352)" in html  # rating
    assert "₹600 for two" in html  # cost
    assert "Rated 4.3 with 5,352 votes." in html  # AI explanation
    assert "✦ Why it fits" in html
    assert ">01<" in html
    assert "Known for</b> Pizza · Pasta" in html


def test_card_tags_and_missing_values():  # EC-UI-09, 10
    html = card_html(
        item(
            book_table=True,
            online_order=False,
            rating=None,
            cost_for_two=None,
            cuisines=[f"C{i}" for i in range(7)],
            popular_dishes=[],
            source="fallback",
        )
    )
    assert "Table booking" in html and "Online ordering" not in html
    assert ">New<" in html and "Cost N/A" in html
    assert html.count('class="cc-chip"') == 5 and "+2 more" in html
    assert "Known for" not in html
    assert "Ranked by rating and popularity" in html


def test_markup_from_data_or_the_llm_is_escaped():  # EC-UI-20, EC-IN-48
    pieces = [
        card_html(
            item(name=HOSTILE, explanation=HOSTILE, cuisines=[HOSTILE], popular_dishes=[HOSTILE])
        ),
        notice_html("warn", HOSTILE, [HOSTILE, HOSTILE]),
    ]
    for html in pieces:
        assert "<img" not in html and "<b>bold" not in html
        assert h(HOSTILE) in html


def test_quick_searches_only_offer_values_in_the_data():
    assert available_quick_searches(["Indiranagar"], ["Italian"]) == ["Italian in Indiranagar"]
    assert available_quick_searches([], []) == []
    every_value = available_quick_searches(
        ["HSR", "Indiranagar", "Jayanagar"], ["Biryani", "Cafe", "Italian", "South Indian"]
    )
    assert set(every_value) == set(QUICK_SEARCHES)


# --- The app, against a live API ------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_cache():
    st.cache_data.clear()  # /meta and /health are cached across runs
    yield
    st.cache_data.clear()


@pytest.fixture
def start_api(make_settings, fixture_repo, monkeypatch):
    """Serve the API on a free port in a background thread and point the UI at it."""
    servers = []
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    def _start(llm):
        app = create_app(
            make_settings(), load_repository=lambda: fixture_repo, make_llm=lambda: llm
        )
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", access_log=False)
        )
        threading.Thread(target=server.run, daemon=True).start()
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "API did not start"
            time.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]
        monkeypatch.setattr(settings, "api_base_url", f"http://127.0.0.1:{port}")
        servers.append(server)
        return app

    yield _start
    for server in servers:
        server.should_exit = True


@pytest.fixture
def llm(start_api):
    fake = FakeLLM()
    start_api(fake)
    return fake


def dead_url() -> str:
    """A localhost URL nothing listens on."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def load() -> AppTest:
    at = AppTest.from_file(APP_FILE, default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def search(at: AppTest, location: str, cuisines=(), top_n: int = 5) -> AppTest:
    at.selectbox(key="location").set_value(location)
    at.multiselect(key="cuisines").set_value(list(cuisines))
    at.selectbox(key="top_n").set_value(top_n)
    at.button(key="find").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def page(at: AppTest) -> str:
    """All the HTML the app rendered, as one string."""
    return "\n".join(x.value for x in at.html)


def cards(at: AppTest) -> list[str]:
    return [x.value for x in at.html if x.value.startswith('<article class="cc-card">')]


def test_page_loads_with_filters_on_the_main_screen(llm, fixture_repo):
    at = load()
    html = page(at)
    assert 'Curated <span class="cc-accent">Cravings</span>' in html  # the title
    assert f"{len(fixture_repo.get_locations())} neighbourhoods" in html
    assert not at.sidebar.selectbox  # the filters are no longer in the sidebar
    assert at.selectbox(key="location").options == [
        ALL_OF_BANGALORE,
        *fixture_repo.get_locations(),
    ]
    assert at.multiselect(key="cuisines").options == fixture_repo.get_cuisines()
    assert at.segmented_control(key="budget").value == "any"
    assert "Where are we eating today?" in html  # empty state
    assert "AI ranking is off" not in html
    assert llm.calls == 0


def test_submitting_without_a_location_asks_for_one(llm):
    at = load()
    at.button(key="find").click().run()
    assert "Choose a neighbourhood first" in page(at)
    assert llm.calls == 0


def test_one_search_is_one_llm_call(llm):
    """A search calls the API, and the LLM, once; later reruns only redraw."""
    at = search(load(), "Indiranagar", top_n=3)
    assert llm.calls == 1
    found = cards(at)
    assert len(found) == 3
    for i, card in enumerate(found, start=1):
        assert f">{i:02d}<" in card
    assert "AI ranked top 3" in page(at)
    at.selectbox(key="top_n").set_value(5).run()  # changing a filter doesn't search
    at.run()
    assert llm.calls == 1
    assert len(cards(at)) == 3  # the last result stays on screen


def test_cards_show_the_five_required_fields(llm, fixture_repo):
    """Name, cuisine, rating, cost and the AI explanation (Phase 5 exit criterion)."""
    at = search(load(), "Indiranagar", top_n=3)
    by_name = {h(r.name): r for r in fixture_repo.at_positions(range(len(fixture_repo)))}
    for card in cards(at):
        restaurant = by_name[card.split('<h3 class="cc-name">')[1].split("</h3>")[0]]
        assert all(f'<span class="cc-chip">{h(c)}</span>' in card for c in restaurant.cuisines[:5])
        assert h(format_rating(restaurant.rating, restaurant.votes)) in card
        assert h(format_cost(restaurant.cost_for_two)) in card
        assert f"Picked {restaurant.restaurant_id}." in card


def test_widened_search_shows_only_the_results(llm):  # EC-UI-08, EC-UI-11
    # Only one Tibetan place in the fixture, reached by widening the search from BTM.
    at = search(load(), "BTM", cuisines=["Tibetan"])
    html = page(at)
    assert "We widened the search" not in html and "Search widened from BTM" not in html
    assert "Showing 1 of 5: only 1 matched" in html
    assert len(cards(at)) == 1


def test_no_results_message(llm):  # EC-UI-07
    at = search(load(), "Indiranagar", cuisines=["Tibetan"])
    assert "No restaurants found" in page(at)
    assert h(NOTICE_NO_RESULTS) in page(at)
    assert not cards(at)


def test_fallback_shows_only_the_results_with_labels(llm):  # EC-UI-11
    llm.error = LLMUnavailableError("API error 429 (rate_limit_exceeded)")
    at = search(load(), "Indiranagar", top_n=3)
    html = page(at)
    assert "AI explanations are unavailable" not in html and h(NOTICE_FALLBACK) not in html
    assert all("Ranked by rating and popularity" in c for c in cards(at))
    assert "Ranked without AI" in html and "ranked by rating" in html


def test_llm_text_is_shown_as_plain_text(llm):  # EC-UI-20
    llm.explanation = HOSTILE
    html = page(search(load(), "Indiranagar", top_n=3))
    assert html.count(h(HOSTILE)) == 3  # one per card
    assert "<img" not in html


def test_quick_search_fills_the_form_and_searches(llm):
    at = load()
    at.pills(key="quick").set_value("South Indian in Jayanagar").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="location").value == "Jayanagar"
    assert at.multiselect(key="cuisines").value == ["South Indian"]
    assert at.segmented_control(key="budget").value == "low"
    assert llm.calls == 1
    assert cards(at)


def test_notice_when_the_llm_is_not_configured(start_api):
    start_api(FakeLLM(is_configured=False))
    html = page(load())
    assert "AI ranking is off" in html and "No Groq API key is set" in html


def test_notice_when_the_llm_is_turned_off(start_api):
    start_api(None)  # what LLM_ENABLED=false builds
    html = page(load())
    assert "AI ranking is off" in html and "LLM_ENABLED=false" in html


def test_notice_when_groq_rejects_the_key(start_api):  # EC-CFG-05
    fake = FakeLLM()
    fake.key_rejected = True
    start_api(fake)
    html = page(load())
    assert "AI ranking is off" in html and "Groq rejected the API key" in html


def test_api_unreachable_at_load(monkeypatch):  # EC-UI-01
    monkeypatch.setattr(settings, "api_base_url", dead_url())
    at = load()
    assert "We can&#x27;t reach the kitchen" in page(at)
    assert [b.label for b in at.button] == ["Retry"]
    assert not at.selectbox  # no form to fill in


def test_api_unreachable_after_load(llm, monkeypatch):  # EC-UI-02
    at = load()
    monkeypatch.setattr(settings, "api_base_url", dead_url())
    at = search(at, "Indiranagar")
    assert "We can&#x27;t reach the kitchen" in page(at)
    assert not cards(at)


def test_server_error_shows_a_reference(start_api):
    class Exploding:
        def recommend(self, prefs):
            raise RuntimeError("boom")

    app = start_api(FakeLLM())
    app.dependency_overrides[get_service] = Exploding
    html = page(search(load(), "Indiranagar"))
    assert "unexpected error" in html and "Reference:" in html
    assert "boom" not in html


def test_validation_error_offers_suggestions(llm):  # EC-UI-06
    # The form only offers known values, so a 422 needs stale /meta data; inject one.
    at = load()
    at.session_state["result"] = {
        "kind": "invalid",
        "payload": build_payload("Indranagar", "any", [], 0.0, "", 5),
        "errors": [
            {
                "type": "unknown_value",
                "loc": ["body", "location"],
                "msg": "Unknown location 'Indranagar'. Did you mean: Indiranagar?",
                "input": "Indranagar",
                "ctx": {"suggestions": ["Indiranagar"]},
            }
        ],
    }
    at.run()
    assert "Did you mean: Indiranagar?" in page(at)
    at.button(key="suggest-0-Indiranagar").click().run()
    assert at.selectbox(key="location").value == "Indiranagar"
