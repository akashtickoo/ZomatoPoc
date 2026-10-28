"""What the user asks for (architecture §5.4). Case IDs refer to docs/edge-case.md §4."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import MAX_TOP_N
from app.models.restaurant import Budget

MAX_CUISINES = 10
MAX_PREFERENCES_CHARS = 300


class UserPreferences(BaseModel):
    # Reject unknown fields so a client typo such as "cuisine" fails loudly (EC-IN-52).
    # The example pre-fills "Try it out" in the API docs at /docs.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "location": "Indiranagar",
                    "budget": "medium",
                    "cuisines": ["Italian"],
                    "min_rating": 4.0,
                    "additional_preferences": "good for a quiet dinner",
                    "top_n": 5,
                }
            ]
        },
    )

    location: str = Field(min_length=1)
    budget: Budget | None = None
    cuisines: list[str] = Field(default_factory=list, max_length=MAX_CUISINES)
    min_rating: float = Field(0.0, ge=0, le=5)
    additional_preferences: str | None = Field(None, max_length=MAX_PREFERENCES_CHARS)
    top_n: int = Field(5, ge=1, le=MAX_TOP_N)

    @field_validator("location", mode="before")
    @classmethod
    def _strip_location(cls, value):
        return " ".join(value.split()) if isinstance(value, str) else value

    @field_validator("budget", mode="before")
    @classmethod
    def _normalize_budget(cls, value):
        # "LOW" and " Low " are accepted; unknown words like "cheap" still fail (EC-IN-11).
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("cuisines")
    @classmethod
    def _clean_cuisines(cls, value: list[str]) -> list[str]:
        # Drop blanks and case-insensitive repeats, keeping the first spelling (EC-IN-22, 24).
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            item = " ".join(item.split())
            if item and item.casefold() not in seen:
                seen.add(item.casefold())
                cleaned.append(item)
        return cleaned

    @field_validator("additional_preferences")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None  # EC-IN-40
