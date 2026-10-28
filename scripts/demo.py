"""Run the demo scenes (docs/demo.md) against the running API and print what the UI shows.

Each scene also prints the API's relaxation notes and notice, which the UI doesn't display.

    python -m scripts.demo                     # the three AI scenes
    python -m scripts.demo --scene fallback    # after restarting the API with LLM_ENABLED=false
    python -m scripts.demo --api http://localhost:8001

Use it to rehearse, or to check the API before a demo. Each search the LLM ranks uses about
5,000 Groq tokens, so the three AI scenes use about 15,000 of the free tier's 200,000 a day.
A repeat within a day is served from the API's cache and costs nothing.
"""

import argparse
import sys
import textwrap
from dataclasses import dataclass

import requests

from app.config import settings


@dataclass(frozen=True)
class Scene:
    key: str
    title: str
    preferences: dict
    point_out: str


STRICT = {"location": "Indiranagar", "cuisines": ["Italian"], "budget": "medium", "top_n": 5}

SCENES = [
    Scene(
        "strict",
        "A strict match",
        STRICT,
        "Ten places match every filter, so nothing is widened. Each explanation cites that "
        "restaurant's own rating, votes, cost and dishes; the facts on the card come from the "
        "data, not the LLM.",
    ),
    Scene(
        "relaxation",
        "A widened search",
        {
            "location": "Koramangala 5th Block",
            "cuisines": ["North Indian", "Chinese"],
            "budget": "low",
            "min_rating": 4.0,
            "top_n": 5,
        },
        "No low-budget North Indian or Chinese place here is rated 4.0 or more, so the budget "
        "is widened to low or medium rather than returning nothing. Explanations should say "
        "when a pick costs more than the original budget.",
    ),
    Scene(
        "free-text",
        "A free-text wish",
        {
            "location": "Indiranagar",
            "budget": "high",
            "min_rating": 4.0,
            "additional_preferences": "rooftop with live music",
            "top_n": 5,
        },
        "Rooftops and live music are not in the data. Explanations should say the data "
        "doesn't confirm them, unless a review mentions one (a Barbeque Nation review mentions "
        "its rooftop).",
    ),
    Scene(
        "fallback",
        "The fallback path",
        STRICT,
        "With the LLM off, the first scene's search comes back at once, ranked by rating and "
        'popularity, with template explanations; the cards say "Ranked by rating and '
        'popularity". The same happens on a Groq outage or a spent rate limit.',
    ),
]
AI_SCENES = [s.key for s in SCENES if s.key != "fallback"]


def describe(prefs: dict) -> str:
    parts = [prefs["location"]]
    if prefs.get("cuisines"):
        parts.append(" + ".join(prefs["cuisines"]))
    if prefs.get("budget"):
        parts.append(f"{prefs['budget']} budget")
    if prefs.get("min_rating"):
        parts.append(f"rated {prefs['min_rating']:.1f}+")
    if prefs.get("additional_preferences"):
        parts.append(f"“{prefs['additional_preferences']}”")
    parts.append(f"top {prefs.get('top_n', 5)}")
    return " · ".join(parts)


def render(n: int, scene: Scene, body: dict) -> str:
    meta = body["meta"]
    ranked = f"ranked by {meta['model']}" if meta["llm_status"] == "ok" else "ranked without AI"
    seconds = meta["latency_ms"] / 1000
    lines = [
        f"\n━━ {n}. {scene.title} ━━",
        describe(scene.preferences),
        *textwrap.wrap(f"Point out: {scene.point_out}", 92),
        f"{ranked} · {meta['candidates_considered']} candidates · {seconds:.1f} s",
    ]
    lines += [f"  Widened: {note}" for note in body["relaxations"]]
    if body["notice"]:
        lines.append(f"  Notice: {body['notice']}")
    for item in body["recommendations"]:
        rating = (
            f"{item['rating']:.1f} ★ ({item['votes']:,} votes)"
            if item["rating"] is not None
            else "unrated"
        )
        cost = (
            f"₹{item['cost_for_two']:,} for two"
            if item["cost_for_two"] is not None
            else "cost unknown"
        )
        source = "AI" if item["source"] == "llm" else "template"
        lines.append(f"  {item['rank']}. {item['name']} · {rating} · {cost} · {source}")
        lines += textwrap.wrap(
            item["explanation"], 88, initial_indent="     ", subsequent_indent="     "
        )
    return "\n".join(lines)


def run(keys: list[str], api: str, http=requests) -> int:
    """Run the chosen scenes; returns the exit code. `http` is anything with requests'
    get and post (tests pass a client for an in-process app)."""
    try:
        health = http.get(f"{api}/health", timeout=10).json()
    except requests.RequestException:
        print(f"Can't reach the API at {api}. Start it with: uvicorn app.main:app")
        return 2
    llm = health["llm"]
    print(f"API at {api}: {health['restaurants']:,} restaurants, LLM {llm}")
    if "fallback" in keys and llm == "configured":
        print(
            "The fallback scene needs the LLM off. Restart the API with LLM_ENABLED=false "
            '(PowerShell: $env:LLM_ENABLED="false"; uvicorn app.main:app), then run '
            "python -m scripts.demo --scene fallback"
        )
        return 1
    if llm != "configured" and any(k in AI_SCENES for k in keys):
        print(f"Note: the LLM is {llm}, so these scenes will show the fallback ranking.")

    timeout = settings.llm_timeout_seconds + 15
    for n, scene in enumerate(SCENES, start=1):
        if scene.key not in keys:
            continue
        response = http.post(f"{api}/recommendations", json=scene.preferences, timeout=timeout)
        if response.status_code != 200:
            print(f"\n{n}. {scene.title}: HTTP {response.status_code} {response.text[:300]}")
            return 1
        print(render(n, scene, response.json()))
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the demo scenes against the API.")
    parser.add_argument(
        "--scene",
        action="append",
        choices=[s.key for s in SCENES],
        help="scene to run (repeatable); default: the three AI scenes",
    )
    parser.add_argument("--api", default=settings.api_base_url, help="API base URL")
    args = parser.parse_args(argv)
    return run(args.scene or AI_SCENES, args.api.rstrip("/"))


if __name__ == "__main__":
    sys.exit(main())
