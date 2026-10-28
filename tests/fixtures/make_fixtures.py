"""Rebuild the test fixtures from the real dataset.

Usage (after `python -m app.data.ingest`):
    python -m tests.fixtures.make_fixtures

Writes:
  sample_raw.parquet          raw rows for a few restaurants, including their duplicate rows
                              and known quirks, for end-to-end tests of clean_dataframe
  sample_restaurants.parquet  ~50 processed restaurants for filtering and API tests
"""

from pathlib import Path

import pandas as pd

from app.config import settings
from app.data.ingest import load_raw
from app.data.preprocess import RAW_COLUMNS

FIXTURES = Path(__file__).parent
SEED = 7
MAX_REVIEWS_CHARS = 3000


def build_raw_sample(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw[list(RAW_COLUMNS)]
    picks = [
        # Two outlets with many duplicate rows across listing types and areas
        raw[raw["name"] == "Jalsa"],
        # Name stored with multi-layer mojibake ("CafÃ\x83Â\x83... Down The Alley")
        raw[raw["name"].str.contains("Down The Alley", regex=False)],
        raw[raw["rate"] == "NEW"].head(2),
        raw[raw["approx_cost(for two people)"].isna()].head(2),
        raw[raw["location"].isna()].head(1),
        raw[raw["reviews_list"] == "[]"].head(1),
        raw.sample(8, random_state=SEED),
    ]
    sample = pd.concat(picks).drop_duplicates()
    # Keep the file small; the regex-based review parser copes with a cut-off final tuple.
    sample["reviews_list"] = sample["reviews_list"].str.slice(0, MAX_REVIEWS_CHARS)
    return sample.reset_index(drop=True)


def build_processed_sample(df: pd.DataFrame) -> pd.DataFrame:
    by_location = {
        "Indiranagar": 12,
        "Koramangala 5th Block": 12,
        "BTM": 10,
        "Jayanagar": 8,
    }
    picks = [
        df[df["location"] == loc].sample(n, random_state=SEED) for loc, n in by_location.items()
    ]
    picks += [
        df[df["rating"].isna()].sample(4, random_state=SEED),
        df[df["cost_for_two"].isna()].sample(2, random_state=SEED),
        df[df["name"].str.contains("é", regex=False)].head(2),
    ]
    sample = pd.concat(picks).drop_duplicates("restaurant_id")
    return sample.sort_values(["name", "restaurant_id"]).reset_index(drop=True)


def main() -> None:
    raw = load_raw(settings.dataset_repo, settings.dataset_revision)
    raw_sample = build_raw_sample(raw)
    raw_sample.to_parquet(FIXTURES / "sample_raw.parquet", index=False)

    processed = pd.read_parquet(settings.data_path)
    processed_sample = build_processed_sample(processed)
    processed_sample.to_parquet(FIXTURES / "sample_restaurants.parquet", index=False)

    print(f"sample_raw.parquet: {len(raw_sample)} raw rows")
    print(
        f"sample_restaurants.parquet: {len(processed_sample)} restaurants, "
        f"budgets {processed_sample['budget'].value_counts(dropna=False).to_dict()}"
    )


if __name__ == "__main__":
    main()
