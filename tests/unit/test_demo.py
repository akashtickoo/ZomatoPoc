"""The demo runner (scripts/demo.py) and the task shortcuts (tasks.py)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import tasks
from app.main import create_app
from app.models.preferences import UserPreferences
from scripts import demo
from tests.fakes import FakeLLM

METADATA = Path(__file__).parents[2] / "data" / "processed" / "metadata.json"


def test_scenes_cover_the_four_planned_cases():
    assert [s.key for s in demo.SCENES] == ["strict", "relaxation", "free-text", "fallback"]
    for scene in demo.SCENES:
        UserPreferences.model_validate(scene.preferences)
    fallback = demo.SCENES[-1]
    assert fallback.preferences == demo.SCENES[0].preferences  # same search, LLM off


@pytest.mark.skipif(not METADATA.exists(), reason="needs the processed data (app.data.ingest)")
def test_scene_values_exist_in_the_data():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    for scene in demo.SCENES:
        assert scene.preferences["location"] in metadata["locations"]
        assert set(scene.preferences.get("cuisines", [])) <= set(metadata["cuisines"])


def test_describe():
    prefs = demo.SCENES[2].preferences
    assert demo.describe(prefs) == (
        "Indiranagar · high budget · rated 4.0+ · “rooftop with live music” · top 5"
    )


@pytest.fixture
def api(make_settings, fixture_repo, monkeypatch):
    """The real app over the fixture data, served in process; scenes use a fixture location."""
    location = fixture_repo.get_locations()[0]
    scene = demo.Scene("strict", "A strict match", {"location": location, "top_n": 2}, "Look.")
    fallback = demo.Scene("fallback", "The fallback path", scene.preferences, "Look again.")
    monkeypatch.setattr(demo, "SCENES", [scene, fallback])

    def _make(llm):
        app = create_app(
            make_settings(), load_repository=lambda: fixture_repo, make_llm=lambda: llm
        )
        client = TestClient(app)
        client.__enter__()
        # requests-style calls; the in-process client has no use for a timeout
        return SimpleNamespace(
            get=lambda url, timeout: client.get(url),
            post=lambda url, json, timeout: client.post(url, json=json),
        )

    return _make


def test_scenes_print_what_the_ui_shows(api, capsys):
    assert demo.run(["strict"], "http://testserver", http=api(FakeLLM())) == 0
    out = capsys.readouterr().out
    assert "LLM configured" in out and "━━ 1. A strict match ━━" in out
    assert "ranked by fake-model" in out and " · AI" in out and "Point out: Look." in out


def test_fallback_scene_needs_the_llm_off(api, capsys):
    assert demo.run(["fallback"], "http://testserver", http=api(FakeLLM())) == 1
    assert "LLM_ENABLED=false" in capsys.readouterr().out

    assert demo.run(["fallback"], "http://testserver", http=api(None)) == 0
    out = capsys.readouterr().out
    assert "LLM disabled" in out and "ranked without AI" in out and " · template" in out
    assert "Notice: AI-written explanations are unavailable" in out


def test_unreachable_api(capsys):
    assert demo.run(["strict"], "http://127.0.0.1:9") == 2
    assert "Can't reach the API" in capsys.readouterr().out


# --- tasks.py -------------------------------------------------------------------------------


@pytest.fixture
def ran(monkeypatch):
    commands: list[list[str]] = []
    codes: list[int] = []

    def fake_run(args, cwd):
        commands.append(args[1:])  # without the Python executable
        return SimpleNamespace(returncode=codes.pop(0) if codes else 0)

    monkeypatch.setattr(tasks.subprocess, "run", fake_run)
    return SimpleNamespace(commands=commands, codes=codes)


def test_tasks_run_their_commands_with_extra_arguments(ran):
    assert tasks.main(["test", "-k", "filtering"]) == 0
    assert ran.commands == [["-m", "pytest", "-k", "filtering"], ["-m", "ruff", "check", "."]]


def test_a_failing_step_stops_the_task(ran):
    ran.codes.append(1)
    assert tasks.main(["test"]) == 1
    assert ran.commands == [["-m", "pytest"]]  # the linter didn't run


def test_unknown_task_prints_the_list(capsys):
    assert tasks.main(["deploy"]) == 2
    assert "python tasks.py ingest" in capsys.readouterr().out
