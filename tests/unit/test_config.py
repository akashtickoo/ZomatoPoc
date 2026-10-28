import pytest
from pydantic import ValidationError

from app.config import PROJECT_ROOT, Settings
from app.observability import GROQ_KEY


def test_defaults_match_architecture(make_settings):
    s = make_settings()
    assert s.llm_model == "openai/gpt-oss-120b"
    assert s.llm_effort == "medium"
    assert s.llm_timeout_seconds == 60
    assert s.llm_enabled is True
    assert s.llm_cache_ttl_seconds == 86400
    assert s.candidate_top_k == 15
    assert s.min_candidates == 5
    assert (s.budget_low_max, s.budget_medium_max) == (300, 600)
    assert s.dataset_revision == "5738e9eda2fad49ad51c6e0ed26e761d9b947133"
    assert s.data_path == PROJECT_ROOT / "data" / "processed" / "restaurants.parquet"
    assert s.api_base_url == "http://localhost:8000"
    assert s.groq_api_key is None
    assert (s.llm_price_input, s.llm_price_cached_input, s.llm_price_output) == (0.15, 0.075, 0.6)
    assert (s.log_level, s.log_file) == ("INFO", None)


def test_env_example_documents_every_setting_without_a_real_key():  # EC-CFG-11
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert not GROQ_KEY.search(text), ".env.example must hold a placeholder, never a real key"
    missing = [name.upper() for name in Settings.model_fields if name.upper() not in text]
    assert missing == []


def test_env_vars_override_defaults(make_settings, monkeypatch):
    monkeypatch.setenv("LLM_EFFORT", "low")
    monkeypatch.setenv("CANDIDATE_TOP_K", "40")
    s = make_settings()
    assert s.llm_effort == "low"
    assert s.candidate_top_k == 40


def test_api_key_is_not_exposed_in_repr(make_settings):
    s = make_settings(groq_api_key="gsk_secret")
    assert "gsk_secret" not in repr(s)
    assert s.groq_api_key.get_secret_value() == "gsk_secret"


@pytest.mark.parametrize(
    ("overrides", "case_id"),
    [
        ({"llm_effort": "extreme"}, "EC-CFG-07"),
        ({"budget_low_max": 1200, "budget_medium_max": 1200}, "EC-CFG-08"),
        ({"budget_low_max": 1500, "budget_medium_max": 1200}, "EC-CFG-08"),
        ({"candidate_top_k": 9}, "EC-CFG-09"),
        ({"min_candidates": 30, "candidate_top_k": 25}, "EC-CFG-10"),
        ({"min_candidates": 0}, "EC-CFG-10"),
        ({"llm_timeout_seconds": 0}, "timeout must be positive"),
        ({"llm_cache_ttl_seconds": -1}, "cache TTL must not be negative"),
        ({"llm_price_output": -0.6}, "prices must not be negative"),
        ({"log_level": "VERBOSE"}, "unknown log level"),
    ],
)
def test_invalid_config_is_rejected(make_settings, overrides, case_id):
    with pytest.raises(ValidationError):
        make_settings(**overrides)
