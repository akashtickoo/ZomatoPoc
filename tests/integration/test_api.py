"""API integration tests: TestClient over the fixture data with a fake LLM client.

No network calls are made. Case IDs refer to docs/edge-case.md §4 and §8.
"""

import json
import logging

import groq
import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.middleware import MAX_BODY_BYTES
from app.api.routes import get_service
from app.data.repository import DataLoadError
from app.llm.client import CachingRankingClient, LLMUnavailableError, make_ranking_client
from app.main import create_app
from app.observability import REDACTED, REQUEST_LOGGER, llm_cost_usd
from app.services.recommender import NOTICE_FALLBACK, NOTICE_NO_RESULTS
from tests.fakes import FakeLLM


@pytest.fixture
def make_client(make_settings, fixture_repo):
    """Start the app (lifespan included) over the fixture data with the given LLM client."""
    clients = []

    def _make(llm, **settings_overrides) -> TestClient:
        app = create_app(
            make_settings(**settings_overrides),
            load_repository=lambda: fixture_repo,
            make_llm=lambda: llm,
        )
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def client(make_client, llm):
    return make_client(llm)


def recommend(client, **body):
    return client.post("/recommendations", json=body)


def request_records(caplog) -> list[logging.LogRecord]:
    """The per-request log lines (architecture §10)."""
    return [r for r in caplog.records if r.name == REQUEST_LOGGER]


# --- /health and /meta ------------------------------------------------------------------


def test_health(client, fixture_repo):
    assert client.get("/health").json() == {
        "status": "ok",
        "data_loaded": True,
        "restaurants": len(fixture_repo),
        "llm": "configured",
    }


def test_health_and_fallback_without_an_api_key(make_client, make_settings, caplog):  # D-03
    # The real client factory, with no GROQ_API_KEY (conftest clears it): no network call.
    with caplog.at_level(logging.WARNING, logger="app.main"):
        client = make_client(make_ranking_client(make_settings()))
    assert "GROQ_API_KEY is not set" in caplog.text
    assert client.get("/health").json()["llm"] == "unconfigured"
    body = recommend(client, location="Indiranagar", top_n=2).json()
    assert body["meta"]["llm_status"] == "fallback"
    assert body["notice"] == NOTICE_FALLBACK


def test_health_without_an_llm_client(make_client):
    assert make_client(None).get("/health").json()["llm"] == "disabled"


def test_llm_enabled_false_serves_the_fallback(make_settings, fixture_repo, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")  # a key that must never be used
    app = create_app(make_settings(llm_enabled=False), load_repository=lambda: fixture_repo)
    with TestClient(app) as client:
        assert client.get("/health").json()["llm"] == "disabled"
        body = recommend(client, location="Indiranagar", top_n=2).json()
    assert body["meta"]["llm_status"] == "fallback"
    assert body["notice"] == NOTICE_FALLBACK
    assert all(i["source"] == "fallback" for i in body["recommendations"])


@pytest.mark.parametrize(
    ("path", "key"), [("/meta/locations", "locations"), ("/meta/cuisines", "cuisines")]
)
def test_meta_lists_are_sorted_and_unique(client, fixture_repo, path, key):  # EC-API-05
    values = client.get(path).json()[key]
    assert values == sorted(set(values), key=str.casefold)
    expected = fixture_repo.get_locations() if key == "locations" else fixture_repo.get_cuisines()
    assert values == expected


# --- POST /recommendations: success paths -------------------------------------------------


def test_happy_path(client, llm, fixture_repo):
    response = recommend(client, location="Indiranagar", top_n=3)
    assert response.status_code == 200
    body = response.json()
    items = body["recommendations"]
    assert body["meta"]["llm_status"] == "ok"
    assert body["meta"]["model"] == "fake-model"
    assert [i["rank"] for i in items] == [1, 2, 3]
    assert all(i["source"] == "llm" for i in items)
    assert items[0]["explanation"] == f"Picked {items[0]['restaurant_id']}."
    assert "summary" not in body
    # Facts come from the data, not from the LLM.
    for item in items:
        (restaurant,) = fixture_repo.get_by_ids([item["restaurant_id"]])
        assert (item["name"], item["rating"], item["cost_for_two"]) == (
            restaurant.name,
            restaurant.rating,
            restaurant.cost_for_two,
        )
    assert llm.calls == 1


def test_input_is_normalized(client):  # EC-IN-02, 11, 22
    response = recommend(
        client, location="  indiranagar ", budget=" HIGH ", cuisines=["chinese", "Chinese"]
    )
    assert response.status_code == 200
    assert all(i["location"] == "Indiranagar" for i in response.json()["recommendations"])


def test_city_name_means_all_of_bangalore(client):  # EC-IN-06, D-04
    items = recommend(client, location="Bengaluru", top_n=10).json()["recommendations"]
    assert len({i["location"] for i in items}) > 1


def test_no_results(client, llm):  # EC-FIL-17
    # Tibetan food exists in the fixture, but not in or near Indiranagar.
    body = recommend(client, location="Indiranagar", cuisines=["Tibetan"]).json()
    assert body["recommendations"] == []
    assert body["notice"] == NOTICE_NO_RESULTS
    assert body["meta"]["llm_status"] == "not_called"
    assert llm.calls == 0


def test_llm_failure_falls_back(client, llm):  # §8
    llm.error = LLMUnavailableError("API error 503")
    response = recommend(client, location="Indiranagar", top_n=3)
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["llm_status"] == "fallback"
    assert body["notice"] == NOTICE_FALLBACK
    assert [i["source"] for i in body["recommendations"]] == ["fallback"] * 3


def test_identical_requests_are_served_from_the_cache(make_client):  # D-10
    fake = FakeLLM()
    client = make_client(CachingRankingClient(fake, ttl_seconds=60))
    first = recommend(client, location="Indiranagar", top_n=2).json()
    second = recommend(client, location="Indiranagar", top_n=2).json()
    assert fake.calls == 1
    assert second["recommendations"] == first["recommendations"]


def test_non_ascii_round_trips(client):  # EC-API-09
    response = recommend(client, location="New BEL Road")
    assert response.headers["content-type"].startswith("application/json")
    assert "Andy's Café" in [i["name"] for i in response.json()["recommendations"]]


# --- POST /recommendations: validation errors -----------------------------------------------


def test_unknown_location_suggests_close_matches(client, llm):  # EC-IN-03, EC-API-06
    response = recommend(client, location="Indranagar")
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert error["type"] == "unknown_value"
    assert error["loc"] == ["body", "location"]
    assert error["input"] == "Indranagar"
    assert error["ctx"]["suggestions"][0] == "Indiranagar"
    assert len(error["ctx"]["suggestions"]) <= 3
    assert llm.calls == 0


def test_another_city_says_the_data_is_bangalore_only(client):  # EC-IN-05
    (error,) = recommend(client, location="Delhi").json()["detail"]
    assert error["ctx"]["suggestions"] == []
    assert "Bangalore only" in error["msg"]


@pytest.mark.parametrize(("cuisine", "suggestion"), [("Chineese", "Chinese"), ("Klingon", None)])
def test_unknown_cuisine(client, cuisine, suggestion):  # EC-IN-21
    response = recommend(client, location="Indiranagar", cuisines=["North Indian", cuisine])
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert (error["loc"], error["input"]) == (["body", "cuisines"], cuisine)
    suggestions = error["ctx"]["suggestions"]
    assert suggestions[:1] == ([suggestion] if suggestion else [])


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({}, "location"),  # EC-IN-01
        ({"location": ""}, "location"),
        ({"location": "BTM", "budget": "cheap"}, "budget"),  # EC-IN-11
        ({"location": "BTM", "min_rating": 5.1}, "min_rating"),  # EC-IN-31
        ({"location": "BTM", "min_rating": -1}, "min_rating"),
        ({"location": "BTM", "min_rating": "four"}, "min_rating"),  # EC-IN-32
        ({"location": "BTM", "additional_preferences": "x" * 301}, "additional_preferences"),
        ({"location": "BTM", "top_n": 0}, "top_n"),  # EC-IN-50
        ({"location": "BTM", "top_n": 11}, "top_n"),
        ({"location": "BTM", "cuisine": ["Chinese"]}, "cuisine"),  # EC-IN-52: typo'd field
        ({"location": "BTM", "cuisines": "Chinese"}, "cuisines"),  # EC-IN-55
        ({"location": "BTM", "cuisines": ["x"] * 11}, "cuisines"),  # EC-IN-23
    ],
)
def test_invalid_fields_return_422(client, llm, body, field):
    response = client.post("/recommendations", json=body)
    assert response.status_code == 422
    assert field in response.json()["detail"][0]["loc"]
    assert llm.calls == 0


def test_limits_are_inclusive(client):  # EC-IN-41, EC-IN-33
    response = recommend(
        client, location="BTM", additional_preferences="x" * 300, min_rating=5.0, top_n=10
    )
    assert response.status_code == 200


def test_malformed_json(client):  # EC-IN-53
    response = client.post(
        "/recommendations", content="{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert "Traceback" not in response.text


# --- Errors and routing -------------------------------------------------------------------


class ExplodingService:
    def recommend(self, prefs):
        raise RuntimeError("secret internal detail")


def test_unhandled_error_returns_a_generic_500(client, caplog):  # EC-API-04
    client.app.dependency_overrides[get_service] = ExplodingService
    with caplog.at_level(logging.ERROR, logger="app.request"):
        response = recommend(client, location="BTM")
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error"
    assert "secret internal detail" not in response.text
    # The client gets an id it can quote; the log line has the same id and the traceback.
    (record,) = request_records(caplog)
    assert record.levelno == logging.ERROR
    assert record.fields["request_id"] == body["request_id"] == response.headers["X-Request-ID"]
    assert record.fields["status"] == 500
    assert "secret internal detail" in caplog.text


def test_unknown_path_and_method(client):  # EC-API-07
    assert client.get("/nope").status_code == 404
    assert client.get("/recommendations").status_code == 405


def test_missing_data_stops_startup(make_settings, tmp_path):  # EC-CFG-01
    app = create_app(make_settings(data_path=tmp_path / "missing.parquet"), make_llm=lambda: None)
    with pytest.raises(DataLoadError, match="python -m app.data.ingest"):
        with TestClient(app):
            pass


def test_docs_list_the_four_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", "/meta/locations", "/meta/cuisines", "/recommendations"}
    assert set(paths["/recommendations"]) == {"post"}
    assert client.get("/docs").status_code == 200


# --- Request log lines and response metadata (architecture §10) ---------------------------


def json_lines(text: str, logger: str = REQUEST_LOGGER) -> list[dict]:
    """The JSON log lines from one logger in captured stderr or a log file."""
    entries = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    return [e for e in entries if e["logger"] == logger]


def test_each_request_writes_one_complete_json_line(client, llm, capsys, make_settings):
    capsys.readouterr()  # drop the startup lines
    response = recommend(client, location="Indiranagar", budget="high", top_n=3)
    (line,) = json_lines(capsys.readouterr().err)
    assert line["level"] == "INFO"
    assert line["message"] == "POST /recommendations 200"
    assert line["request_id"] == response.headers["X-Request-ID"]
    assert (line["method"], line["path"], line["status"]) == ("POST", "/recommendations", 200)
    assert line["duration_ms"] >= 0
    assert line["preferences"]["location"] == "Indiranagar"
    assert line["preferences"]["budget"] == "high"
    body = response.json()
    assert line["candidates"] == body["meta"]["candidates_considered"]
    assert line["total_matched"] == body["meta"]["total_matched"]
    assert line["relaxations"] == body["relaxations"]
    assert (line["llm_status"], line["fallback"], line["results"]) == ("ok", False, 3)
    assert line["llm"]["model"] == "fake-model"
    assert line["llm"]["finish_reason"] == "stop"
    assert line["llm"]["groq_request_id"] == "req_fake"
    assert line["llm"]["usage"] == llm.usage
    assert line["llm"]["cost_usd"] == pytest.approx(llm_cost_usd(llm.usage, make_settings()))


def test_response_metadata(client):
    meta = recommend(client, location="Indiranagar", top_n=3).json()["meta"]
    assert meta["model"] == "fake-model"
    assert meta["candidates_considered"] > 0
    assert meta["latency_ms"] >= 0


@pytest.mark.parametrize(
    ("method", "path", "body", "status"),
    [
        ("GET", "/health", None, 200),
        ("GET", "/meta/cuisines", None, 200),
        ("POST", "/recommendations", {"location": "BTM", "top_n": 11}, 422),
        ("POST", "/recommendations", {"location": "Indranagar"}, 422),
        ("GET", "/nope", None, 404),
    ],
)
def test_every_request_gets_one_line_and_an_id(client, caplog, method, path, body, status):
    caplog.set_level(logging.INFO, logger=REQUEST_LOGGER)
    response = client.request(method, path, json=body)
    assert response.status_code == status
    (record,) = request_records(caplog)
    assert record.fields["status"] == status
    assert record.fields["request_id"] == response.headers["X-Request-ID"]


def test_unknown_location_line_keeps_the_preferences(client, caplog):
    caplog.set_level(logging.INFO, logger=REQUEST_LOGGER)
    recommend(client, location="Indranagar", cuisines=["Chinese"])
    (record,) = request_records(caplog)
    assert record.fields["preferences"]["location"] == "Indranagar"
    assert "llm" not in record.fields


def test_fallback_line_is_a_warning(client, llm, caplog):  # §8
    llm.error = LLMUnavailableError("API error 503", status_code=503)
    caplog.set_level(logging.INFO, logger=REQUEST_LOGGER)
    recommend(client, location="Indiranagar", top_n=2)
    (record,) = request_records(caplog)
    assert record.levelno == logging.WARNING
    assert record.fields["fallback"] is True
    assert record.fields["fallback_reason"] == "API error 503"


# --- Request size limit (EC-IN-54) --------------------------------------------------------


def test_oversized_body_is_rejected_before_parsing(client, llm, caplog):
    caplog.set_level(logging.INFO, logger=REQUEST_LOGGER)
    body = json.dumps({"location": "BTM", "additional_preferences": "x" * MAX_BODY_BYTES})
    response = client.post(
        "/recommendations", content=body, headers={"content-type": "application/json"}
    )
    assert response.status_code == 413
    assert str(MAX_BODY_BYTES) in response.json()["detail"]
    assert response.headers["X-Request-ID"]
    (record,) = request_records(caplog)
    assert record.fields["status"] == 413
    assert llm.calls == 0


def test_oversized_body_without_a_length_is_rejected(client, llm):
    def chunks():  # sent chunked, with no Content-Length to check up front
        yield b'{"location": "BTM", "additional_preferences": "'
        for _ in range(MAX_BODY_BYTES // 1024 + 1):
            yield b"x" * 1024
        yield b'"}'

    response = client.post(
        "/recommendations", content=chunks(), headers={"content-type": "application/json"}
    )
    assert response.status_code == 413
    assert llm.calls == 0


def test_a_full_request_is_far_below_the_limit():
    body = {
        "location": "Koramangala 5th Block",
        "cuisines": ["North Indian"] * 10,
        "additional_preferences": "x" * 300,
        "budget": "medium",
        "min_rating": 4.0,
        "top_n": 10,
    }
    assert len(json.dumps(body)) < MAX_BODY_BYTES / 4


# --- The API key never leaks (EC-CFG-11); a rejected key is reported (EC-CFG-05) ----------

KEY = "gsk_" + "Zz9Yy8Xx" * 6 + "qq"  # shaped like a real Groq key


def test_api_key_never_reaches_logs_or_responses(
    make_settings, fixture_repo, monkeypatch, capsys, tmp_path
):
    def handler(request: httpx.Request) -> httpx.Response:
        # Worst case: the provider echoes the key back in its error message.
        error = {"message": f"Invalid API Key {KEY}", "code": "invalid_api_key"}
        return httpx.Response(401, json={"error": error})

    real_groq = groq.Groq

    def groq_over_fake_transport(**kwargs):
        return real_groq(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(groq, "Groq", groq_over_fake_transport)
    log_file = tmp_path / "api.jsonl"
    cfg = make_settings(groq_api_key=KEY, log_level="DEBUG", log_file=log_file)
    app = create_app(cfg, load_repository=lambda: fixture_repo)  # the real Groq client
    with TestClient(app) as client:
        assert client.get("/health").json()["llm"] == "configured"
        response = recommend(client, location="Indiranagar", top_n=2)
        health = client.get("/health").json()
        logging.getLogger("groq").error("request failed for key %s", KEY)  # a library error

    assert response.status_code == 200
    assert response.json()["meta"]["llm_status"] == "fallback"
    assert health["llm"] == "rejected"
    assert KEY not in response.text + json.dumps(health)
    for written in (capsys.readouterr().err, log_file.read_text(encoding="utf-8")):
        assert KEY not in written
        assert REDACTED in written  # the library line was scrubbed, not dropped
        (line,) = [e for e in json_lines(written) if e["path"] == "/recommendations"]
        assert line["level"] == "ERROR"  # a configuration problem, not a passing outage
        assert line["llm"]["error"] == "API error 401 (invalid_api_key)"
        assert line["llm"]["config_error"] is True


def test_health_reports_a_rejected_key(make_client):
    fake = FakeLLM()
    fake.key_rejected = True
    assert make_client(fake).get("/health").json()["llm"] == "rejected"
