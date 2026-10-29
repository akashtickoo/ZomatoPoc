"""Application settings, loaded from environment variables and `.env` (architecture §9)."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Upper bound for UserPreferences.top_n; CANDIDATE_TOP_K must be at least this (EC-CFG-09).
MAX_TOP_N = 10


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Optional: without a key the app still starts and serves the fallback ranking (D-03).
    groq_api_key: SecretStr | None = None

    # Must support Groq's strict structured outputs and reasoning_effort (gpt-oss models).
    llm_model: str = "openai/gpt-oss-120b"
    llm_effort: Literal["low", "medium", "high"] = "medium"  # sent as reasoning_effort
    llm_timeout_seconds: float = Field(60.0, gt=0)
    # false turns the LLM off: every request gets the deterministic ranking and no tokens
    # are spent, e.g. to demo the fallback path or when the daily allowance is used up.
    llm_enabled: bool = True
    # How long a successful ranking is reused for an identical request; 0 disables (D-10).
    llm_cache_ttl_seconds: float = Field(86400.0, ge=0)
    # US dollars per million tokens, used to log what each request cost. Groq's prices for
    # openai/gpt-oss-120b; cached input is billed at half the input price.
    llm_price_input: float = Field(0.15, ge=0)
    llm_price_cached_input: float = Field(0.075, ge=0)
    llm_price_output: float = Field(0.60, ge=0)

    # 15 keeps a request well inside Groq's free-tier 8K tokens/minute; in live runs the
    # LLM never picked a candidate below position 12. The recommender raises it to
    # 2 x top_n when needed.
    candidate_top_k: int = Field(15, ge=MAX_TOP_N)
    min_candidates: int = Field(5, ge=1)

    # Cost-for-two thresholds in rupees: low <= budget_low_max < medium <= budget_medium_max < high.
    # Chosen near the dataset's 33rd/66th percentiles: about 41% low, 39% medium, 19% high.
    budget_low_max: int = Field(300, gt=0)
    budget_medium_max: int = Field(600, gt=0)

    # Pinned so re-ingesting always produces the same data (EC-ING-04).
    dataset_repo: str = "ManikaSaini/zomato-restaurant-recommendation"
    dataset_revision: str = "5738e9eda2fad49ad51c6e0ed26e761d9b947133"

    data_path: Path = PROJECT_ROOT / "data" / "processed" / "restaurants.parquet"
    api_base_url: str = "http://localhost:8000"  # the API scripts/demo.py calls

    # Browser origins allowed to call the API: the web front end in web/, locally on Vite's
    # dev server and in production on Vercel (docs/deployment-plan.md). A JSON list in the
    # environment, e.g. CORS_ORIGINS=["https://curated-cravings.vercel.app"]; empty allows
    # none. Server-side callers such as scripts/demo.py need no entry.
    cors_origins: list[str] = []
    # Optional: a pattern for this project's Vercel preview URLs, matched against the whole
    # origin.
    cors_origin_regex: str | None = None

    # App logs are JSON lines on stderr, and also appended to LOG_FILE when it is set.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: Path | None = None

    @field_validator("groq_api_key", mode="before")
    @classmethod
    def _blank_key_is_no_key(cls, value):
        # "GROQ_API_KEY=" in .env or the shell means no key (EC-CFG-04), not a key that
        # Groq will reject on every request.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("cors_origins")
    @classmethod
    def _origins_without_trailing_slash(cls, value: list[str]) -> list[str]:
        # A browser's Origin header never ends in "/", so "https://x.vercel.app/" would
        # silently match nothing and every search would fail with a CORS error.
        return [origin.strip().rstrip("/") for origin in value]

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.budget_low_max >= self.budget_medium_max:
            raise ValueError("BUDGET_LOW_MAX must be less than BUDGET_MEDIUM_MAX")
        if self.min_candidates > self.candidate_top_k:
            raise ValueError("MIN_CANDIDATES must not exceed CANDIDATE_TOP_K")
        return self


settings = Settings()
