"""Pure functions that turn raw Zomato rows into clean restaurant records.

Implements architecture §5.2. Case IDs (EC-DATA-xx) refer to docs/edge-case.md §2.
Nothing here does I/O, so every rule can be unit-tested without the real dataset.
"""

import ast
import hashlib
import html
import math
import re
from difflib import SequenceMatcher

import pandas as pd

from app.models.restaurant import Budget

CITY = "Bangalore"

# Raw column -> working column. A missing raw column fails ingestion (EC-ING-03).
RAW_COLUMNS = {
    "name": "name",
    "address": "address",
    "location": "location",
    "listed_in(city)": "area",
    "cuisines": "cuisines",
    "approx_cost(for two people)": "cost_for_two",
    "rate": "rating",
    "votes": "votes",
    "rest_type": "rest_type",
    "online_order": "online_order",
    "book_table": "book_table",
    "dish_liked": "dish_liked",
    "reviews_list": "reviews_list",
    "listed_in(type)": "listing_type",
}

OUTPUT_COLUMNS = [
    "restaurant_id",
    "name",
    "address",
    "city",
    "location",
    "areas",
    "cuisines",
    "cost_for_two",
    "budget",
    "rating",
    "votes",
    "rest_type",
    "online_order",
    "book_table",
    "dish_liked",
    "listing_types",
    "review_snippets",
]

# Columns whose values are merged across duplicate rows instead of taken from the kept row.
MERGED_LIST_COLUMNS = ["areas", "listing_types"]

_MOJIBAKE_RE = re.compile("[ÃÂ]|â€")
_MOJIBAKE_RESIDUE_RE = re.compile("[\x80-\x9fÃÂ]|â€")
_RATING_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:/\s*5)?")
# One ('Rated x.x', '<text>') tuple; the text literal may use either quote style.
_REVIEW_TUPLE_RE = re.compile(
    r"""\(\s*'Rated\s*[\d.]*'\s*,\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')\s*\)"""
)
_RATED_PREFIX_RE = re.compile(r"^\s*RATED\b")

# Listing variants (EC-DATA-67): the same restaurant listed twice under one name and
# location, with addresses that differ only in wording ("Plot 7" vs "Plot No 7").
_ADDRESS_FILLER = {"no", "number", "bangalore", "bengaluru"}
_ADDRESS_ABBREVIATIONS = {"rd": "road", "opp": "opposite"}
_ORDINAL_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
ADDRESS_SIMILARITY = 0.8
# Reviews are written about one listing, so the same long text means the same place.
# Short ones ("good", "worst food") recur across unrelated restaurants.
SHARED_REVIEW_MIN_CHARS = 100


def _text(value) -> str:
    """Stripped string for any raw cell; '' for None, NaN or blank."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def fix_mojibake(text: str) -> str:
    """Undo repeated UTF-8-read-as-Latin-1 corruption, e.g. 'CafÃ\\x83Â\\x83...©' -> 'Café'.

    The dataset has names encoded several layers deep (EC-DATA-40), so the repair repeats
    until the text stops changing or no longer round-trips.
    """
    for _ in range(10):
        if not _MOJIBAKE_RE.search(text):
            return text
        try:
            repaired = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
        if repaired == text:
            return text
        text = repaired
    return text


def clean_text(value) -> str:
    """Trim, repair encoding and collapse internal whitespace."""
    return " ".join(fix_mojibake(_text(value)).split())


def normalize_key(text: str) -> str:
    """Case-, punctuation- and whitespace-insensitive form used for matching (EC-DATA-62)."""
    return " ".join(re.sub(r"[\W_]+", " ", text.casefold()).split())


def parse_rating(value) -> float | None:
    """'4.1/5' or '4.1 /5' -> 4.1; 'NEW', '-', blank and out-of-range -> None (EC-DATA-01..07)."""
    match = _RATING_RE.fullmatch(_text(value))
    if not match:
        return None
    rating = float(match.group(1))
    return rating if 0 <= rating <= 5 else None


def parse_cost(value) -> int | None:
    """'1,200' -> 1200; blank, non-numeric and zero -> None (EC-DATA-10..15)."""
    text = _text(value).replace("₹", "").replace(",", "").strip()
    try:
        cost = float(text)
    except ValueError:
        return None
    if not math.isfinite(cost) or cost <= 0:
        return None
    return round(cost)


def derive_budget(cost: int | None, low_max: int, medium_max: int) -> Budget | None:
    """Budget band from cost for two; both upper bounds are inclusive (EC-DATA-20..24)."""
    # pd.isna, not `is None`: pandas turns missing ints into NaN or pd.NA.
    if pd.isna(cost):
        return None
    if cost <= low_max:
        return "low"
    if cost <= medium_max:
        return "medium"
    return "high"


def split_list(value) -> list[str]:
    """'North Indian, Chinese' -> ['North Indian', 'Chinese'], without blanks or repeats."""
    items: list[str] = []
    seen: set[str] = set()
    for part in _text(value).split(","):
        item = clean_text(part)
        if item and item.casefold() not in seen:
            seen.add(item.casefold())
            items.append(item)
    return items


def parse_yes_no(value) -> bool:
    """'Yes' (any case, padded) -> True; anything else -> False (EC-DATA-65)."""
    return _text(value).casefold() == "yes"


def truncate_text(text: str, max_chars: int) -> str:
    """Cut at a word boundary and add an ellipsis when longer than max_chars."""
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    space = cut.rfind(" ")
    if space > max_chars // 2:
        cut = cut[:space]
    return cut.rstrip(" ,.;:-") + "…"


def _clean_review(text: str) -> str:
    text = _RATED_PREFIX_RE.sub("", text)
    text = fix_mojibake(html.unescape(text))
    # Some reviews are corrupted beyond repair; drop the leftover junk characters.
    text = _MOJIBAKE_RESIDUE_RE.sub("", text)
    return " ".join(text.split())


def extract_review_snippets(raw, max_reviews: int = 3, max_chars: int = 200) -> list[str]:
    """First few review texts from the stringified list of ('Rated x', 'text') tuples.

    Matches tuples one at a time instead of parsing the whole list, so a multi-megabyte
    entry costs no more than a short one (EC-DATA-56). Malformed entries are skipped.
    """
    snippets: list[str] = []
    for match in _REVIEW_TUPLE_RE.finditer(_text(raw)):
        try:
            text = ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            continue
        text = _clean_review(text) if isinstance(text, str) else ""
        if text:
            snippets.append(truncate_text(text, max_chars))
            if len(snippets) == max_reviews:
                break
    return snippets


def make_restaurant_id(name: str, address: str) -> str:
    """Stable 12-hex-character ID from the normalized name and address (EC-DATA-64)."""
    key = f"{normalize_key(name)}|{normalize_key(address)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def check_columns(raw: pd.DataFrame) -> None:
    missing = sorted(set(RAW_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {', '.join(missing)}")


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per restaurant_id (EC-DATA-60, 61).

    Keeps the most-voted row (first in input order on ties) and replaces each column in
    MERGED_LIST_COLUMNS with the sorted union of its values across the duplicates.
    """
    merged = {
        column: df.groupby("restaurant_id", sort=False)[column].agg(
            lambda lists: sorted(set().union(*lists))
        )
        for column in MERGED_LIST_COLUMNS
    }
    best = df.sort_values("votes", ascending=False, kind="stable").drop_duplicates("restaurant_id")
    for column, values in merged.items():
        best[column] = best["restaurant_id"].map(values)
    return best


def canonical_address(address: str) -> str:
    """Address reduced to what identifies a place: no filler words, no ordinal suffixes."""
    tokens = []
    for token in normalize_key(address).split():
        if token not in _ADDRESS_FILLER:
            token = _ADDRESS_ABBREVIATIONS.get(token, token)
            tokens.append(_ORDINAL_RE.sub(r"\1", token))
    return " ".join(tokens)


def is_listing_variant(
    address_a: str, address_b: str, reviews_a: list[str], reviews_b: list[str]
) -> bool:
    """Whether two listings with the same name and location are one restaurant (EC-DATA-67).

    Either the addresses carry the same numbers and are otherwise near-identical, or
    the listings share a long review. Matching numbers are required because "8, 23rd
    Main" and "23, 23rd Main" are different doors on the same road.
    """
    a, b = canonical_address(address_a), canonical_address(address_b)
    numbers_a = {t for t in a.split() if any(c.isdigit() for c in t)}
    numbers_b = {t for t in b.split() if any(c.isdigit() for c in t)}
    if numbers_a == numbers_b and SequenceMatcher(None, a, b).ratio() >= ADDRESS_SIMILARITY:
        return True
    shared = set(reviews_a) & set(reviews_b)
    return any(len(review) >= SHARED_REVIEW_MIN_CHARS for review in shared)


def merge_listing_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Fold listing variants of one restaurant into its most-voted listing (EC-DATA-67).

    Runs on de-duplicated rows with review_snippets parsed. Within each group sharing a
    normalized name and location, every listing is compared with the kept listings before
    it (most votes first). A match takes the kept listing's restaurant_id, and
    `deduplicate` then merges the two. Chain outlets that don't match stay separate.
    """
    keys = df["name"].map(normalize_key) + "|" + df["location"]
    grouped = df[keys.duplicated(keep=False)].assign(_key=keys)
    target: dict[str, str] = {}
    ordered = grouped.sort_values("votes", ascending=False, kind="stable")
    for _, group in ordered.groupby("_key", sort=False):
        kept = []
        for row in group.itertuples(index=False):
            match = next(
                (
                    k
                    for k in kept
                    if is_listing_variant(
                        k.address, row.address, k.review_snippets, row.review_snippets
                    )
                ),
                None,
            )
            if match is None:
                kept.append(row)
            else:
                target[row.restaurant_id] = match.restaurant_id
    if not target:
        return df
    return deduplicate(df.assign(restaurant_id=df["restaurant_id"].replace(target)))


def clean_dataframe(
    raw: pd.DataFrame, *, budget_low_max: int, budget_medium_max: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Raw dataset -> one clean row per restaurant, with columns in OUTPUT_COLUMNS order.

    Returns the cleaned frame and counts of rows dropped at each step.
    """
    check_columns(raw)
    df = raw[list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS)
    df["name"] = df["name"].map(clean_text)
    df["address"] = df["address"].map(clean_text)
    df["location"] = df["location"].map(clean_text)

    missing_name = df["name"] == ""
    missing_location = ~missing_name & (df["location"] == "")
    df = df[~missing_name & ~missing_location].copy()

    # Rows without an address are identified by name + location instead (EC-DATA-44).
    identity = df["address"].where(df["address"] != "", df["location"])
    df["restaurant_id"] = [
        make_restaurant_id(n, a) for n, a in zip(df["name"], identity, strict=True)
    ]
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    df["areas"] = df["area"].map(lambda v: [a] if (a := clean_text(v)) else [])
    df["listing_types"] = df["listing_type"].map(lambda v: [t] if (t := clean_text(v)) else [])

    # Parse the remaining fields only for the rows that survive de-duplication.
    df = deduplicate(df)

    df["city"] = CITY
    df["cuisines"] = df["cuisines"].map(split_list)
    df["rest_type"] = df["rest_type"].map(split_list)
    df["dish_liked"] = df["dish_liked"].map(split_list)
    df["online_order"] = df["online_order"].map(parse_yes_no)
    df["book_table"] = df["book_table"].map(parse_yes_no)
    df["rating"] = df["rating"].map(parse_rating).astype("Float64")
    df["cost_for_two"] = df["cost_for_two"].map(parse_cost).astype("Int64")
    df["budget"] = [derive_budget(c, budget_low_max, budget_medium_max) for c in df["cost_for_two"]]
    df["review_snippets"] = df["reviews_list"].map(extract_review_snippets)

    exact_unique = len(df)
    df = merge_listing_variants(df)

    df = df[OUTPUT_COLUMNS].sort_values(["name", "restaurant_id"]).reset_index(drop=True)
    stats = {
        "raw_rows": len(raw),
        "dropped_missing_name": int(missing_name.sum()),
        "dropped_missing_location": int(missing_location.sum()),
        "merged_listing_variants": exact_unique - len(df),
        "restaurants": len(df),
    }
    return df, stats
