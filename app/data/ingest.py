"""Download the Zomato dataset, clean it and write the processed files (architecture §5.1).

Usage:
    python -m app.data.ingest

Writes `restaurants.parquet` and `metadata.json` next to settings.data_path. Both files are
written to temporary paths first and only moved into place once every check has passed,
so an interrupted or failed run never leaves a half-written file behind (EC-ING-05).
"""

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from pydantic import ValidationError

from app.config import Settings, settings
from app.data.preprocess import clean_dataframe
from app.models.restaurant import Restaurant

SCHEMA_VERSION = 1


def load_raw(repo: str, revision: str) -> pd.DataFrame:
    from datasets import load_dataset

    try:
        dataset = load_dataset(repo, split="train", revision=revision)
    except Exception as exc:  # network, auth or missing revision (EC-ING-01)
        raise SystemExit(f"Could not load dataset {repo}@{revision}: {exc}") from exc
    return dataset.to_pandas()


def validate_records(records: list[dict]) -> list[str]:
    """Check every record against Restaurant and the de-duplication invariants.

    Returns human-readable problems; an empty list means the data is valid.
    """
    problems = []
    for record in records:
        try:
            Restaurant.model_validate(record)
        except ValidationError as exc:
            problems.append(f"{record.get('restaurant_id')}: {exc.errors()[0]['msg']}")
            if len(problems) >= 10:
                break

    ids = [r["restaurant_id"] for r in records]
    if len(ids) != len(set(ids)):
        problems.append(f"{len(ids) - len(set(ids))} duplicate restaurant_id values")
    pairs = [(r["name"], r["address"]) for r in records]
    if len(pairs) != len(set(pairs)):
        problems.append(f"{len(pairs) - len(set(pairs))} duplicate (name, address) pairs")
    mismatched = sum((r["cost_for_two"] is None) != (r["budget"] is None) for r in records)
    if mismatched:
        problems.append(f"{mismatched} records where budget and cost_for_two disagree on missing")
    return problems


def build_metadata(df: pd.DataFrame, cfg: Settings) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_repo": cfg.dataset_repo,
        "dataset_revision": cfg.dataset_revision,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "restaurants": len(df),
        "budget_low_max": cfg.budget_low_max,
        "budget_medium_max": cfg.budget_medium_max,
        "locations": sorted(df["location"].unique().tolist()),
        "areas": sorted({a for areas in df["areas"] for a in areas}),
        "cuisines": sorted({c for cuisines in df["cuisines"] for c in cuisines}, key=str.casefold),
    }


def summarize(df: pd.DataFrame, stats: dict[str, int], metadata: dict) -> str:
    n = len(df)
    costs = df["cost_for_two"].dropna()
    budgets = df["budget"].fillna("missing").value_counts()
    lines = [
        "Ingest summary",
        f"  Raw rows:                    {stats['raw_rows']:,}",
        f"  Dropped (missing name):      {stats['dropped_missing_name']:,}",
        f"  Dropped (missing location):  {stats['dropped_missing_location']:,}",
        f"  Merged listing variants:     {stats['merged_listing_variants']:,}",
        f"  Restaurants after de-dup:    {n:,}",
        f"  Unrated:                     {df['rating'].isna().mean():.1%}",
        f"  Missing cost:                {df['cost_for_two'].isna().mean():.1%}",
        f"  No review snippets:          {(df['review_snippets'].map(len) == 0).mean():.1%}",
        f"  Locations / areas / cuisines: {len(metadata['locations'])} / "
        f"{len(metadata['areas'])} / {len(metadata['cuisines'])}",
        "  Cost for two percentiles:    "
        + ", ".join(f"p{int(q * 100)}=₹{costs.quantile(q):,.0f}" for q in (0.33, 0.5, 0.66, 0.9)),
        f"  Budget bands (low <= ₹{metadata['budget_low_max']}, "
        f"medium <= ₹{metadata['budget_medium_max']}): "
        + ", ".join(f"{band}={count / n:.0%}" for band, count in budgets.items()),
    ]
    return "\n".join(lines)


def ingest(cfg: Settings = settings) -> None:
    raw = load_raw(cfg.dataset_repo, cfg.dataset_revision)
    df, stats = clean_dataframe(
        raw, budget_low_max=cfg.budget_low_max, budget_medium_max=cfg.budget_medium_max
    )

    out_path: Path = cfg.data_path
    meta_path = out_path.with_name("metadata.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)  # EC-ING-06
    tmp_out = out_path.with_name(out_path.name + ".tmp")
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")

    df.to_parquet(tmp_out, index=False)
    # Validate what was actually written, read back the way later phases will read it.
    problems = validate_records(pq.read_table(tmp_out).to_pylist())
    if problems:
        tmp_out.unlink()
        raise SystemExit("Processed data failed validation:\n  " + "\n  ".join(problems))

    metadata = build_metadata(df, cfg)
    tmp_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_out, out_path)
    os.replace(tmp_meta, meta_path)

    print(summarize(df, stats, metadata))
    print(f"Wrote {out_path} and {meta_path.name}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # the summary prints ₹ on Windows consoles
    ingest()
