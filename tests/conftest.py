"""Shared pytest fixtures."""

import logging
from pathlib import Path

import pytest

from app.config import Settings
from app.data.preprocess import make_restaurant_id
from app.data.repository import RestaurantRepository

FIXTURES = Path(__file__).parent / "fixtures"

SETTINGS_ENV_VARS = [
    "GROQ_API_KEY",
    "LLM_MODEL",
    "LLM_EFFORT",
    "LLM_TIMEOUT_SECONDS",
    "LLM_ENABLED",
    "LLM_CACHE_TTL_SECONDS",
    "LLM_PRICE_INPUT",
    "LLM_PRICE_CACHED_INPUT",
    "LLM_PRICE_OUTPUT",
    "CANDIDATE_TOP_K",
    "MIN_CANDIDATES",
    "BUDGET_LOW_MAX",
    "BUDGET_MEDIUM_MAX",
    "DATASET_REPO",
    "DATASET_REVISION",
    "DATA_PATH",
    "API_BASE_URL",
    "CORS_ORIGINS",
    "CORS_ORIGIN_REGEX",
    "LOG_LEVEL",
    "LOG_FILE",
]


@pytest.fixture(autouse=True)
def _undo_configure_logging():
    """The app's startup calls configure_logging; keep its handlers from leaking into the
    next test. Only the app's own handlers are touched, not pytest's."""
    app_logger = logging.getLogger("app")
    level = app_logger.level
    yield
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, "app_json", False)]:
        root.removeHandler(handler)
        handler.close()
    app_logger.setLevel(level)


@pytest.fixture
def make_settings(monkeypatch):
    """Build Settings isolated from the developer's `.env` and shell environment."""
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    def _make(**overrides) -> Settings:
        return Settings(_env_file=None, **overrides)

    return _make


@pytest.fixture
def make_record():
    """Build a valid processed-restaurant dict; override any field by keyword."""

    def _make(name: str = "Test Kitchen", **overrides) -> dict:
        address = overrides.pop("address", f"1 {name} Street")
        location = overrides.get("location", "Indiranagar")
        record = {
            "restaurant_id": make_restaurant_id(name, address),
            "name": name,
            "address": address,
            "city": "Bangalore",
            "location": location,
            "areas": [location],
            "cuisines": ["North Indian"],
            "cost_for_two": 400,
            "budget": None,  # the repository derives it from cost_for_two
            "rating": 4.0,
            "votes": 100,
            "rest_type": ["Casual Dining"],
            "online_order": True,
            "book_table": False,
            "dish_liked": [],
            "listing_types": ["Delivery"],
            "review_snippets": [],
        }
        return record | overrides

    return _make


@pytest.fixture
def make_repo(make_settings):
    """Repository over the given records, with default budget bands (≤300 / ≤600)."""

    def _make(records: list[dict], **settings_overrides) -> RestaurantRepository:
        return RestaurantRepository.from_records(records, make_settings(**settings_overrides))

    return _make


@pytest.fixture
def fixture_repo(make_settings):
    """Repository over the 50 real restaurants in tests/fixtures/sample_restaurants.parquet."""
    cfg = make_settings(data_path=FIXTURES / "sample_restaurants.parquet")
    return RestaurantRepository.load(cfg)
