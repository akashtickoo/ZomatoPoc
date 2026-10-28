"""Curated Cravings: the Streamlit front end for the recommendation API (architecture §5.10).

Run the API first, then, from the project root:
    streamlit run ui/streamlit_app.py

The UI only collects preferences and renders the API's response. Cards and notices are
HTML built here; every value from the data or the LLM is HTML-escaped first, so markup in
a review or an explanation shows as text and never as a link or tag (EC-UI-20). The
theme (colours, font, corner radius) is in .streamlit/config.toml. Case IDs refer to
docs/edge-case.md §9.
"""

import html
import sys
from pathlib import Path

import requests
import streamlit as st

# `streamlit run` puts only this folder on sys.path; the project root is needed for `app`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

TITLE = "Curated Cravings"
ALL_OF_BANGALORE = "All of Bangalore"  # sent as "Bangalore": no location filter (D-04)
BUDGETS = ["any", "low", "medium", "high"]
RESULT_COUNTS = [3, 5, 8, 10]
MAX_CHIPS = 5  # EC-UI-09
META_TTL_SECONDS = 3600  # picks up a re-ingest within the hour (EC-UI-05)
# The API makes one LLM call of at most LLM_TIMEOUT_SECONDS per search; wait a little
# longer than that before giving up (EC-UI-03).
REQUEST_TIMEOUT = (5, settings.llm_timeout_seconds + 30)
# Filter values live in session state under these keys, so quick searches can fill the
# form and results survive reruns.
DEFAULTS = {
    "location": None,
    "cuisines": [],
    "budget": "any",
    "min_rating": 0.0,
    "hints": "",
    "top_n": 5,
}
QUICK_SEARCHES = {
    "Italian in Indiranagar": {
        "location": "Indiranagar",
        "cuisines": ["Italian"],
        "budget": "medium",
    },
    "Biryani across the city": {
        "location": ALL_OF_BANGALORE,
        "cuisines": ["Biryani"],
        "min_rating": 4.0,
    },
    "Cosy cafés in HSR": {
        "location": "HSR",
        "cuisines": ["Cafe"],
        "hints": "quiet, good for working",
    },
    "South Indian in Jayanagar": {
        "location": "Jayanagar",
        "cuisines": ["South Indian"],
        "budget": "low",
    },
}

STYLESHEET = Path(__file__).with_name("style.css")


class ApiUnavailable(Exception):
    """The API could not be reached or did not answer in time."""


# --- Pure helpers (unit-tested in tests/ui) -------------------------------------------------


def h(value) -> str:
    """HTML-escape a value from the data or the LLM before it goes into markup (EC-UI-20)."""
    return html.escape(str(value), quote=True)


def count(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n} {singular if n == 1 else plural or singular + 's'}"


def budget_label(key: str) -> str:
    low, medium = settings.budget_low_max, settings.budget_medium_max
    return {
        "any": "Any",
        "low": f"Low · ≤ ₹{low}",
        "medium": f"Medium · ₹{low + 1}–{medium}",
        "high": f"High · ₹{medium + 1}+",
    }[key]


def format_rating(rating: float | None, votes: int) -> str:
    if rating is None:  # EC-UI-10
        return "New"
    return f"{rating:.1f} ★ ({votes:,})"


def format_cost(cost: int | None) -> str:
    return "Cost N/A" if cost is None else f"₹{cost:,} for two"  # EC-UI-10


def build_payload(
    location: str | None,
    budget: str | None,
    cuisines: list[str],
    min_rating: float,
    hints: str,
    top_n: int,
) -> dict:
    return {
        "location": "Bangalore" if location == ALL_OF_BANGALORE else location,
        "budget": None if budget in (None, "any") else budget,
        "cuisines": list(cuisines),
        "min_rating": min_rating,
        "additional_preferences": hints.strip() or None,
        "top_n": top_n,
    }


def describe_request(payload: dict) -> str:
    """One line restating what was asked, shown above the results."""
    location = payload["location"]
    parts = [", ".join(payload["cuisines"]) or "Any cuisine"]
    parts.append("anywhere in Bangalore" if location == "Bangalore" else f"in {location}")
    if payload["budget"]:
        parts.append(f"{payload['budget']} budget")
    if payload["min_rating"]:
        parts.append(f"rated {payload['min_rating']:g}★ or more")
    if payload["additional_preferences"]:
        parts.append(f"“{payload['additional_preferences']}”")
    return " · ".join(parts)


def notice_html(kind: str, title: str, lines: list[str] = ()) -> str:
    """A soft banner. `kind` is warn or error; every text is escaped here."""
    items = "".join(f"<li>{h(line)}</li>" for line in lines)
    body = f"<ul>{items}</ul>" if len(lines) > 1 else (h(lines[0]) if lines else "")
    return f'<div class="cc-notice cc-notice-{kind}"><b>{h(title)}</b>{body}</div>'


def card_html(item: dict) -> str:
    """One restaurant: rank, name, booking tags, rating, cost, place, cuisines, why, dishes."""
    tags = []
    if item["book_table"]:
        tags.append('<span class="cc-tag cc-tag-green">Table booking</span>')
    if item["online_order"]:
        tags.append('<span class="cc-tag">Online ordering</span>')

    rating_class = "cc-pill-new" if item["rating"] is None else "cc-pill-rating"
    meta = [
        f'<span class="cc-pill {rating_class}">'
        f"{h(format_rating(item['rating'], item['votes']))}</span>",
        f'<span class="cc-pill">{h(format_cost(item["cost_for_two"]))}</span>',
        f'<span class="cc-pill">📍 {h(item["location"])}</span>',
    ]
    if item["rest_type"]:
        meta.append(f'<span class="cc-pill">{h(", ".join(item["rest_type"]))}</span>')

    cuisines = item["cuisines"]
    chips = [f'<span class="cc-chip">{h(c)}</span>' for c in cuisines[:MAX_CHIPS]]
    if len(cuisines) > MAX_CHIPS:
        chips.append(f'<span class="cc-chip cc-chip-more">+{len(cuisines) - MAX_CHIPS} more</span>')

    why = "✦ Why it fits" if item["source"] == "llm" else "Ranked by rating and popularity"
    known = ""
    if item["popular_dishes"]:
        known = (
            f'<div class="cc-known"><b>Known for</b> {h(" · ".join(item["popular_dishes"]))}</div>'
        )

    return (
        '<article class="cc-card">'
        f'<div class="cc-rank">{item["rank"]:02d}</div>'
        '<div class="cc-body">'
        f'<div class="cc-head"><h3 class="cc-name">{h(item["name"])}</h3>'
        f'<div class="cc-tags">{"".join(tags)}</div></div>'
        f'<div class="cc-row">{"".join(meta)}</div>'
        + (f'<div class="cc-row">{"".join(chips)}</div>' if chips else "")
        + f'<p class="cc-why"><span class="cc-why-label">{why}</span>{h(item["explanation"])}</p>'
        f"{known}</div></article>"
    )


# --- API calls ----------------------------------------------------------------------------


def _get(path: str) -> dict:
    try:
        response = requests.get(f"{settings.api_base_url}{path}", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ApiUnavailable(str(exc)) from exc
    return response.json()


@st.cache_data(ttl=META_TTL_SECONDS, show_spinner=False)
def load_options() -> tuple[list[str], list[str]]:
    return _get("/meta/locations")["locations"], _get("/meta/cuisines")["cuisines"]


@st.cache_data(ttl=30, show_spinner=False)
def load_health() -> dict:
    return _get("/health")


def request_recommendations(payload: dict) -> dict:
    """Call the API once and describe the outcome, so rendering never handles exceptions."""
    try:
        response = requests.post(
            f"{settings.api_base_url}/recommendations", json=payload, timeout=REQUEST_TIMEOUT
        )
    except requests.Timeout:
        return {
            "kind": "error",
            "message": "The service took too long to answer. Please try again.",
        }
    except requests.RequestException:
        return {"kind": "unavailable"}
    if response.status_code == 200:
        return {"kind": "ok", "response": response.json()}
    if response.status_code == 422:
        return {"kind": "invalid", "errors": response.json().get("detail", [])}
    request_id = response.json().get("request_id") if response.content else None
    message = "The service hit an unexpected error. Please try again."
    if request_id:
        message += f" (Reference: {request_id})"
    return {"kind": "error", "message": message}


# --- Callbacks ----------------------------------------------------------------------------


def use_suggestion(field: str, wrong: str, suggestion: str) -> None:
    """Put a suggested value from a 422 into the form (EC-UI-06)."""
    if field == "location":
        st.session_state["location"] = suggestion
    else:
        st.session_state["cuisines"] = [
            suggestion if c == wrong else c for c in st.session_state.get("cuisines", [])
        ]


def apply_quick_search() -> None:
    """Fill the form from a quick-search pill and run it."""
    choice = st.session_state.get("quick")
    if choice in QUICK_SEARCHES:
        st.session_state.update(DEFAULTS | QUICK_SEARCHES[choice])
        st.session_state["run_search"] = True
    st.session_state["quick"] = None


# --- Rendering ----------------------------------------------------------------------------


def render_header(localities: int | None) -> None:
    count_pill = (
        f'<span class="cc-pill-nav cc-pill-live"><span class="cc-dot"></span>'
        f"{localities} neighbourhoods</span>"
        if localities
        else ""
    )
    st.html(
        '<nav class="cc-nav"><div class="cc-nav-left"><span class="cc-logo">✦ <b>cc</b></span>'
        '<span class="cc-pill-nav">📍 Bangalore</span></div>'
        '<span class="cc-nav-mid">Real Zomato data · AI-ranked picks</span>'
        f"{count_pill}</nav>"
        '<section class="cc-hero"><span class="cc-badge">✦ Your AI dining concierge</span>'
        f'<h1 class="cc-title">Curated <span class="cc-accent">Cravings</span></h1>'
        '<p class="cc-sub">Tell us what you are in the mood for. We narrow Bangalore\'s '
        "restaurants to your must-haves, then AI ranks the shortlist and tells you why each "
        "place fits.</p></section>"
    )


def search_card(locations: list[str], cuisines: list[str]):
    """The filter card. Returns (submitted, placeholder for the footer status line)."""
    with st.container(key="search_card"):
        with st.form("search", border=False):
            first = st.columns([1.15, 1.45, 1], gap="medium")
            first[0].selectbox(
                "Locality",
                [ALL_OF_BANGALORE, *locations],
                key="location",
                placeholder="Search neighbourhoods",
            )
            first[1].multiselect(
                "Cuisines",
                cuisines,
                key="cuisines",
                placeholder="e.g. Italian, Chinese",
                max_selections=10,
            )
            first[2].slider("Minimum rating", 0.0, 5.0, step=0.5, key="min_rating", format="%.1f ★")

            # The budget needs room for all four segments on one line.
            second = st.columns([1.8, 1.3, 0.6], gap="medium")
            second[0].segmented_control(
                "Budget for two",
                BUDGETS,
                format_func=budget_label,
                key="budget",
                required=True,
                wrap=True,  # onto a second row on narrow screens, instead of scrolling
            )
            second[1].text_input(
                "Anything else?",
                key="hints",
                max_chars=300,  # the API enforces it too (EC-UI-21)
                placeholder="quiet seating, quick service, good for groups",
            )
            second[2].selectbox(
                "Show", RESULT_COUNTS, key="top_n", format_func=lambda n: f"Top {n}"
            )

            foot = st.columns([3, 1.1], vertical_alignment="center")
            status = foot[0].empty()
            submitted = foot[1].form_submit_button(
                "Find restaurants →", type="primary", key="find", width="stretch"
            )
    return submitted, status


def render_status(status, result: dict | None) -> None:
    """The line beside the search button: how the last search was ranked."""
    if result and result["kind"] == "ok":
        meta = result["response"]["meta"]
        shown = len(result["response"]["recommendations"])
        ranked = f"AI ranked top {shown}" if meta["llm_status"] == "ok" else "ranked by rating"
        text = f"⚡ Screened <b>{count(meta['candidates_considered'], 'candidate')}</b> · {ranked}"
    else:
        text = "⚡ Your must-haves filter first, then AI ranks the shortlist"
    status.html(f'<div class="cc-foot">{text}</div>')


def available_quick_searches(locations: list[str], cuisines: list[str]) -> list[str]:
    """Quick searches whose values all exist in the current data, so the form accepts them."""
    places = {ALL_OF_BANGALORE, *locations}
    return [
        label
        for label, values in QUICK_SEARCHES.items()
        if values["location"] in places and set(values.get("cuisines", [])) <= set(cuisines)
    ]


def render_empty_state(locations: list[str], cuisines: list[str]) -> None:
    st.html(
        '<section class="cc-empty"><div class="cc-empty-title">Where are we eating today?</div>'
        "<p>Pick a neighbourhood above, or start from a popular search.</p></section>"
    )
    quick = available_quick_searches(locations, cuisines)
    if quick:
        with st.container(horizontal_alignment="center"):
            st.pills(
                "Popular searches",
                quick,
                key="quick",
                on_change=apply_quick_search,
                label_visibility="collapsed",
                width="content",
            )


def render_invalid(errors: list[dict]) -> None:
    for i, error in enumerate(errors):
        st.html(
            notice_html("error", "Please check your search", [error.get("msg", "Invalid request.")])
        )
        suggestions = (error.get("ctx") or {}).get("suggestions") or []
        field = error.get("loc", ["", ""])[-1]
        if suggestions and field in ("location", "cuisines"):
            columns = st.columns(len(suggestions))
            for column, suggestion in zip(columns, suggestions, strict=True):
                column.button(
                    f"Use {suggestion}",
                    key=f"suggest-{i}-{suggestion}",
                    on_click=use_suggestion,
                    args=(field, error.get("input"), suggestion),
                )


def render_response(payload: dict, data: dict) -> None:
    items = data["recommendations"]
    meta = data["meta"]
    asked = describe_request(payload)
    if 0 < len(items) < payload["top_n"]:  # EC-UI-08
        asked = f"Showing {len(items)} of {payload['top_n']}: only {len(items)} matched · {asked}"
    st.html(
        '<div class="cc-section"><h2>Your shortlist</h2>'
        f'<span class="cc-muted">{h(asked)}</span></div>'
    )

    if not items:  # EC-UI-07
        st.html(
            notice_html("warn", "No restaurants found", [data["notice"] or "Try fewer filters."])
        )
        return

    # Only the results here: no fallback or relaxation banners (EC-UI-11). The cards and
    # the footer still say when the ranking came without AI.
    for item in items:
        st.html(card_html(item))

    source = f"Ranked by {meta['model']}" if meta["llm_status"] == "ok" else "Ranked without AI"
    st.html(
        f'<div class="cc-muted" style="text-align:right">{h(source)} · '
        f"{count(meta['candidates_considered'], 'candidate')} considered from "
        f"{count(meta['total_matched'], 'match', 'matches')} · "
        f"{meta['latency_ms'] / 1000:.1f} s</div>"
    )


def render_result(result: dict) -> None:
    kind = result["kind"]
    if kind == "ok":
        render_response(result["payload"], result["response"])
    elif kind == "invalid":
        render_invalid(result["errors"])
    elif kind == "unavailable":  # EC-UI-02
        st.html(notice_html("error", "We can't reach the kitchen", [unavailable_text()]))
    else:
        st.html(notice_html("error", "Something went wrong", [result["message"]]))


def unavailable_text() -> str:
    return (
        f"The recommendation service isn't reachable at {settings.api_base_url}. "
        "Start it with `uvicorn app.main:app` and try again."
    )


def main() -> None:
    st.set_page_config(page_title=TITLE, page_icon="🍽️", layout="wide")
    st.html(f"<style>{STYLESHEET.read_text(encoding='utf-8')}</style>")
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)

    try:
        locations, cuisines = load_options()
        health = load_health()
    except ApiUnavailable:  # EC-UI-01: a message and a retry, not a crash
        render_header(None)
        st.html(notice_html("error", "We can't reach the kitchen", [unavailable_text()]))
        if st.button("Retry"):
            st.cache_data.clear()
            st.rerun()
        st.stop()

    render_header(len(locations))
    if health.get("llm") != "configured":
        reason = {
            "rejected": "Groq rejected the API key (check GROQ_API_KEY)",
            "disabled": "The LLM is turned off (LLM_ENABLED=false)",
        }.get(health.get("llm"), "No Groq API key is set")
        st.html(
            notice_html(
                "warn",
                "AI ranking is off",
                [f"{reason}, so results are ranked by rating and popularity."],
            )
        )

    submitted, status = search_card(locations, cuisines)
    run = submitted or st.session_state.pop("run_search", False)
    if run:
        state = st.session_state
        if state["location"] is None:
            st.html(
                notice_html(
                    "warn",
                    "Choose a neighbourhood first",
                    ["Pick a locality, or All of Bangalore."],
                )
            )
        else:
            payload = build_payload(
                state["location"],
                state["budget"],
                state["cuisines"],
                state["min_rating"],
                state["hints"],
                state["top_n"],
            )
            # A new search replaces the old result, so nothing stale stays on screen (EC-UI-02).
            with st.spinner("Curating your shortlist…"):
                result = request_recommendations(payload)
            st.session_state["result"] = result | {"payload": payload}

    result = st.session_state.get("result")
    render_status(status, result)
    if result is None:
        render_empty_state(locations, cuisines)
    else:
        render_result(result)


if __name__ == "__main__":
    main()
