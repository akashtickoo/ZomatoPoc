"""API-only response models (architecture §5.9).

The request body is `UserPreferences` and the recommendation response is
`RecommendationResponse`, both in app/models; this module holds the rest.
"""

from typing import Any, Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    data_loaded: bool
    restaurants: int
    # "unconfigured": no GROQ_API_KEY, so every request gets the fallback ranking (D-03);
    # "rejected": Groq refused the key (401/403) on the last call (EC-CFG-05);
    # "disabled": the service was started without an LLM client.
    llm: Literal["configured", "unconfigured", "rejected", "disabled"]


class LocationsResponse(BaseModel):
    locations: list[str]


class CuisinesResponse(BaseModel):
    cuisines: list[str]


class ErrorDetail(BaseModel):
    """One problem with the request, in the shape FastAPI uses for its own 422s.

    Unknown locations and cuisines use type "unknown_value" and carry up to 3 close
    matches in `ctx.suggestions` (EC-API-06).
    """

    type: str
    loc: list[str | int]
    msg: str
    input: Any = None
    ctx: dict[str, Any] | None = None


class ValidationErrorResponse(BaseModel):
    detail: list[ErrorDetail]


class ServerErrorResponse(BaseModel):
    detail: str
    request_id: str  # also in the server log, next to the full traceback (EC-API-04)
