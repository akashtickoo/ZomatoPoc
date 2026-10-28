"""In-memory access to the processed restaurant data (architecture §5.3).

Loaded once at startup. Everything downstream works with `Restaurant` objects; the
DataFrame in `frame` holds pre-computed, normalized columns for fast filtering.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pydantic import ValidationError

from app.config import Settings, settings
from app.data.preprocess import derive_budget, normalize_key
from app.models.restaurant import Restaurant

# Bayesian pre-rank prior: a restaurant needs about this many votes before its own rating
# outweighs the dataset mean. Close to the median vote count of rated restaurants (53).
PRIOR_VOTES = 50


class DataLoadError(RuntimeError):
    """The processed data file is missing or unreadable (EC-CFG-01, EC-CFG-02)."""


class RestaurantRepository:
    def __init__(self, restaurants: list[Restaurant]):
        if not restaurants:
            raise DataLoadError("No restaurants loaded")
        self._restaurants = restaurants
        self._by_id = {r.restaurant_id: r for r in restaurants}

        rated = [r.rating for r in restaurants if r.rating is not None]
        self.mean_rating = sum(rated) / len(rated) if rated else 0.0

        self.frame = pd.DataFrame(
            {
                "restaurant_id": [r.restaurant_id for r in restaurants],
                "name": [r.name for r in restaurants],
                "name_key": [normalize_key(r.name) for r in restaurants],
                "location_key": [r.location.casefold() for r in restaurants],
                "areas": [frozenset(r.areas) for r in restaurants],
                "cuisine_keys": [frozenset(c.casefold() for c in r.cuisines) for r in restaurants],
                "budget": [r.budget for r in restaurants],
                "rating": pd.array([r.rating for r in restaurants], dtype="float64"),
                "votes": [r.votes for r in restaurants],
            }
        )
        votes = self.frame["votes"]
        weight = votes / (votes + PRIOR_VOTES)
        # NaN for unrated restaurants; filtering ranks them after every rated one (D-07).
        self.frame["score"] = weight * self.frame["rating"] + (1 - weight) * self.mean_rating

        self._locations = {r.location.casefold(): r.location for r in restaurants}
        self._cuisines = {c.casefold(): c for r in restaurants for c in r.cuisines}
        self._primary_area = self._compute_primary_areas(restaurants)
        # Boolean row masks, built on first use and reused. Membership tests over list
        # columns are per-row Python work, too slow to repeat for every relaxation step.
        self._mask_cache: dict[tuple[str, str], np.ndarray] = {}

    @classmethod
    def load(cls, cfg: Settings = settings) -> "RestaurantRepository":
        path: Path = cfg.data_path
        if not path.exists():
            raise DataLoadError(
                f"Processed data not found at {path}. Run `python -m app.data.ingest` first."
            )
        try:
            records = pq.read_table(path).to_pylist()
        except Exception as exc:
            raise DataLoadError(f"Could not read {path}: {exc}") from exc
        return cls.from_records(records, cfg)

    @classmethod
    def from_records(cls, records: list[dict], cfg: Settings = settings) -> "RestaurantRepository":
        """Build from plain dicts, re-deriving each budget band from the current config.

        Deriving the band here rather than trusting the stored value means changing
        BUDGET_*_MAX takes effect on restart, without re-running the ingest (D-01).
        """
        try:
            restaurants = [
                Restaurant.model_validate(
                    record
                    | {
                        "budget": derive_budget(
                            record["cost_for_two"], cfg.budget_low_max, cfg.budget_medium_max
                        )
                    }
                )
                for record in records
            ]
        except (KeyError, ValidationError) as exc:
            raise DataLoadError(
                f"Processed data does not match the Restaurant schema; re-run the ingest. {exc}"
            ) from exc
        return cls(restaurants)

    @staticmethod
    def _compute_primary_areas(restaurants: list[Restaurant]) -> dict[str, str]:
        """For each location, the listing area most of its restaurants appear under.

        Ties go to the alphabetically first area so the result is deterministic.
        """
        counts: dict[str, Counter] = {}
        for r in restaurants:
            counts.setdefault(r.location.casefold(), Counter()).update(r.areas)
        return {
            location: min(counter.items(), key=lambda item: (-item[1], item[0]))[0]
            for location, counter in counts.items()
            if counter
        }

    def __len__(self) -> int:
        return len(self._restaurants)

    def get_locations(self) -> list[str]:
        return sorted(self._locations.values(), key=str.casefold)

    def get_cuisines(self) -> list[str]:
        return sorted(self._cuisines.values(), key=str.casefold)

    def resolve_location(self, text: str) -> str | None:
        """Canonical spelling of a known location, matched case-insensitively."""
        return self._locations.get(" ".join(text.split()).casefold())

    def resolve_cuisine(self, text: str) -> str | None:
        return self._cuisines.get(" ".join(text.split()).casefold())

    def primary_area(self, location: str) -> str | None:
        return self._primary_area.get(location.casefold())

    def _cached_mask(self, kind: str, key: str, build) -> np.ndarray:
        mask = self._mask_cache.get((kind, key))
        if mask is None:
            mask = self._mask_cache[(kind, key)] = build()
        return mask

    def location_mask(self, location: str) -> np.ndarray:
        """Rows whose location equals `location`, case-insensitively."""
        key = location.casefold()
        return self._cached_mask(
            "location", key, lambda: (self.frame["location_key"] == key).to_numpy()
        )

    def area_mask(self, area: str) -> np.ndarray:
        """Rows that Zomato lists under listing area `area`."""
        column = self.frame["areas"]
        return self._cached_mask(
            "area", area, lambda: np.fromiter((area in a for a in column), bool, len(column))
        )

    def cuisine_mask(self, cuisine: str) -> np.ndarray:
        """Rows serving `cuisine`, case-insensitively."""
        key = cuisine.casefold()
        column = self.frame["cuisine_keys"]
        return self._cached_mask(
            "cuisine", key, lambda: np.fromiter((key in c for c in column), bool, len(column))
        )

    def get_by_ids(self, ids: list[str]) -> list[Restaurant]:
        """Restaurants for the known IDs, in the order given; unknown IDs are skipped."""
        return [self._by_id[i] for i in ids if i in self._by_id]

    def at_positions(self, positions) -> list[Restaurant]:
        """Restaurants for row positions in `frame`."""
        return [self._restaurants[p] for p in positions]
