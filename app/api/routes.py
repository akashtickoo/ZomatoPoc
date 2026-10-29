"""HTTP endpoints (architecture §5.9). Case IDs refer to docs/edge-case.md §4 and §8.

Endpoints are plain `def`, so FastAPI runs them in a thread pool and a slow LLM call
doesn't block other requests (EC-API-03).
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import RedirectResponse

from app.api.schemas import (
    BudgetsResponse,
    CuisinesResponse,
    HealthResponse,
    LocationsResponse,
    ServerErrorResponse,
    ValidationErrorResponse,
)
from app.config import Settings
from app.data.repository import RestaurantRepository
from app.models.preferences import UserPreferences
from app.models.recommendation import RecommendationResponse
from app.services.filtering import UnknownValueError
from app.services.recommender import RecommendationService

router = APIRouter()


# Dependencies read what the lifespan hook loaded; tests can replace any of them through
# app.dependency_overrides.
def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_repository(request: Request) -> RestaurantRepository:
    return request.app.state.repository


def get_service(request: Request) -> RecommendationService:
    return request.app.state.service


AppSettings = Annotated[Settings, Depends(get_settings)]
Repository = Annotated[RestaurantRepository, Depends(get_repository)]
Service = Annotated[RecommendationService, Depends(get_service)]


@router.get("/", include_in_schema=False)
def root():
    """The API has no page of its own; someone opening its address gets the docs."""
    return RedirectResponse("/docs")


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request, repo: Repository):
    """Liveness check, plus whether data is loaded and the LLM key is set and accepted."""
    llm = request.app.state.llm
    if llm is None:
        llm_status = "disabled"
    elif not getattr(llm, "is_configured", True):
        llm_status = "unconfigured"
    elif getattr(llm, "key_rejected", False):
        llm_status = "rejected"
    else:
        llm_status = "configured"
    return HealthResponse(status="ok", data_loaded=True, restaurants=len(repo), llm=llm_status)


@router.get("/meta/locations", response_model=LocationsResponse, tags=["meta"])
def locations(repo: Repository):
    """Bangalore neighbourhoods, sorted case-insensitively (EC-API-05)."""
    return LocationsResponse(locations=repo.get_locations())


@router.get("/meta/cuisines", response_model=CuisinesResponse, tags=["meta"])
def cuisines(repo: Repository):
    """Cuisines, sorted case-insensitively (EC-API-05)."""
    return CuisinesResponse(cuisines=repo.get_cuisines())


@router.get("/meta/budgets", response_model=BudgetsResponse, tags=["meta"])
def budgets(cfg: AppSettings):
    """The budget bands in rupees for two, so a front end can label them without its own
    copy of BUDGET_LOW_MAX and BUDGET_MEDIUM_MAX."""
    return BudgetsResponse(low_max=cfg.budget_low_max, medium_max=cfg.budget_medium_max)


@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
    responses={
        422: {"model": ValidationErrorResponse, "description": "Invalid preferences"},
        500: {"model": ServerErrorResponse, "description": "Unexpected server error"},
    },
    tags=["recommendations"],
)
def recommendations(prefs: UserPreferences, service: Service):
    """Ranked, explained recommendations for the given preferences.

    An unknown location or cuisine returns 422 with close matches in
    `detail[].ctx.suggestions` (EC-IN-03, 05, 21).
    """
    try:
        return service.recommend(prefs)
    except UnknownValueError as exc:
        field = "location" if exc.field == "location" else "cuisines"
        # Raised as a validation error so every 422 has the same shape.
        raise RequestValidationError(
            [
                {
                    "type": "unknown_value",
                    "loc": ("body", field),
                    "msg": str(exc),
                    "input": exc.value,
                    "ctx": {"suggestions": exc.suggestions},
                }
            ]
        ) from exc
