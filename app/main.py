"""FastAPI application (architecture §5.9).

Run with:
    uvicorn app.main:app --reload
then open http://localhost:8000/docs.
"""

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import BodySizeLimitMiddleware, RequestLogMiddleware
from app.api.routes import router
from app.config import Settings, settings
from app.data.repository import RestaurantRepository
from app.llm.client import RankingClient, make_ranking_client
from app.observability import configure_logging
from app.services.recommender import RecommendationService

logger = logging.getLogger(__name__)


def create_app(
    cfg: Settings = settings,
    load_repository: Callable[[], RestaurantRepository] | None = None,
    make_llm: Callable[[], RankingClient | None] | None = None,
) -> FastAPI:
    """Build the app. Tests pass isolated settings, fixture data and a fake LLM client."""
    load_repository = load_repository or (lambda: RestaurantRepository.load(cfg))
    # No client at all when LLM_ENABLED is false: /health reports "disabled".
    make_llm = make_llm or (lambda: make_ranking_client(cfg) if cfg.llm_enabled else None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # JSON log lines from here on (architecture §10). Done at startup rather than at
        # import, so importing the app (as tests do) leaves logging alone.
        configure_logging(cfg)
        # Everything is loaded before the first request is served, so no request sees a
        # half-loaded repository (EC-API-01). A missing data file stops startup with a
        # hint to run the ingest (EC-CFG-01).
        repository = load_repository()
        llm = make_llm()
        if llm is None:
            logger.info("The LLM is disabled; every request will use the fallback ranking")
        elif not getattr(llm, "is_configured", True):
            logger.warning("GROQ_API_KEY is not set; every request will use the fallback ranking")
        app.state.settings = cfg
        app.state.repository = repository
        app.state.llm = llm
        # One repository and one LLM client, shared by all requests (EC-API-02).
        app.state.service = RecommendationService(repository, llm, cfg)
        logger.info("Loaded %d restaurants", len(repository))
        yield

    app = FastAPI(
        title="Restaurant Recommender",
        version="0.1.0",
        summary="AI-ranked, explained restaurant recommendations for Bangalore.",
        lifespan=lifespan,
    )
    app.include_router(router)
    # The last one added runs first: every request, even one rejected for its size, gets
    # an id and a log line, and an unhandled error becomes a 500 (EC-API-04).
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(RequestLogMiddleware)
    # Added last, so it runs first: the 413s and 500s above carry CORS headers too, or the
    # browser would report a CORS failure instead of the real error. It answers preflight
    # requests itself, so those get no request id or log line (docs/deployment-plan.md).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_origin_regex=cfg.cors_origin_regex,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Request-ID"],  # so the web UI can quote it on an error
    )
    return app


app = create_app()
