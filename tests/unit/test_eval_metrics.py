"""The evaluation harness's scenario file, programmatic metrics and reports (docs/eval.md).

The harness itself is run on demand against the real API; these tests keep its graders
honest without any network access.
"""

import json
from pathlib import Path

import pytest

from tests.eval import compare, report
from tests.eval.common import (
    FIXTURES_DIR,
    Scenario,
    TokenPacer,
    format_rate,
    load_scenarios,
    wilson,
)
from tests.eval.programmatic import (
    allowed_filters,
    check_injected_rank,
    check_row,
    constraint_violations,
    numeric_claims,
)

METADATA = Path(__file__).parents[2] / "data" / "processed" / "metadata.json"
CATEGORIES = {  # eval.md §3.3
    "S01": "baseline",
    "S02": "baseline",
    "S03": "baseline",
    "S04": "baseline",
    "S05": "multi_cuisine",
    "S06": "no_filters",
    "S07": "free_text_verifiable",
    "S08": "free_text_verifiable",
    "S09": "free_text_verifiable",
    "S10": "free_text_unverifiable",
    "S11": "free_text_unverifiable",
    "S12": "free_text_conflict",
    "S13": "relaxation_budget",
    "S14": "relaxation_rating",
    "S15": "relaxation_location",
    "S16": "no_relaxation_boundary",
    "S17": "few_candidates",
    "S18": "zero_candidates",
    "S19": "top_n_edge",
    "S20": "top_n_edge",
    "S21": "injection_free_text",
    "S22": "injection_review",
    "S23": "language",
    "S24": "chain_heavy",
}


# --- The scenario file ------------------------------------------------------------------


def test_scenario_file_has_the_24_planned_scenarios():
    scenarios = load_scenarios()
    assert {s.id: s.category for s in scenarios} == CATEGORIES
    assert [s.id for s in scenarios] == sorted(CATEGORIES)  # fixed order (determinism)
    for s in scenarios:
        if s.expectations.relaxation_kinds:
            assert s.expectations.expect_relaxation, s.id
    (injection,) = [s for s in scenarios if s.inject_review]
    assert injection.id == "S22" and injection.variants() == ["main", "control"]
    assert "ignore your ranking rules" in injection.inject_review.snippet()


def test_oracle_answers_cover_known_scenarios():
    oracle = json.loads((FIXTURES_DIR / "oracle.json").read_text(encoding="utf-8"))
    assert set(oracle) == {"S01", "S07", "S10"}
    assert all(len(answers) == 5 for answers in oracle.values())


@pytest.mark.skipif(not METADATA.exists(), reason="needs the processed data (app.data.ingest)")
def test_scenario_locations_and_cuisines_exist_in_the_data():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    locations, cuisines = set(metadata["locations"]), set(metadata["cuisines"])
    for s in load_scenarios():
        assert s.preferences.location in locations, s.id
        assert set(s.preferences.cuisines) <= cuisines, s.id


def test_unknown_scenario_ids_are_rejected():
    with pytest.raises(ValueError, match="S99"):
        load_scenarios(ids=["S01", "S99"])


# --- Statistics -------------------------------------------------------------------------


def test_wilson_interval():
    p, low, high = wilson(9, 10)
    assert p == 0.9 and round(low, 3) == 0.596 and round(high, 3) == 0.982
    assert wilson(0, 0) is None
    assert format_rate(72, 72).startswith("100% (95%–100%")


def test_newcombe_difference_includes_zero_for_small_samples():
    d, low, high = compare.difference((9, 10), (10, 10))
    assert round(d, 2) == 0.1 and low < 0 < high


def test_token_pacer_waits_for_the_window_to_clear():
    now = [0.0]
    slept: list[float] = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    pacer = TokenPacer(8000, clock=lambda: now[0], sleep=sleep)
    pacer.record(pacer.wait(6000), 5500)
    now[0] = 10.0
    pacer.wait(5000)  # 5,500 + 5,000 > 8,000: waits until the first call is 60 s old
    assert slept == [pytest.approx(50.1)]
    assert TokenPacer(0).wait(10**6)  # 0 disables pacing


# --- P3 and P7 ---------------------------------------------------------------------------


@pytest.fixture
def scenario():
    def _make(**expectations) -> Scenario:
        prefs = expectations.pop("preferences", {})
        return Scenario.model_validate(
            {
                "id": "T1",
                "tags": ["baseline"],
                "preferences": {"location": "Indiranagar", "top_n": 3} | prefs,
                "expectations": {"expect_relaxation": False} | expectations,
            }
        )

    return _make


def test_relaxation_notes_widen_what_is_allowed(scenario, make_repo, make_record):
    prefs = scenario(preferences={"budget": "low", "cuisines": ["Thai"], "min_rating": 4.5})
    prefs = prefs.preferences
    notes = [
        "Budget widened from low to low or medium.",
        "Minimum rating lowered from 4.5 to 4.0.",
        "Search widened from Indiranagar to places Zomato lists under Old Airport Road.",
    ]
    record = make_record(
        "Elsewhere",
        location="Domlur",
        areas=["Old Airport Road"],
        cuisines=["Thai"],
        cost_for_two=500,
        rating=4.1,
    )
    (r,) = make_repo([record]).get_by_ids([record["restaurant_id"]])
    assert constraint_violations(r, allowed_filters(prefs, notes)) == []
    assert constraint_violations(r, allowed_filters(prefs, [])) == [
        "location Domlur",
        "budget medium",
        "rating 4.1",
    ]


@pytest.fixture
def toit(make_repo, make_record):
    record = make_record(
        "Toit",
        rating=4.7,
        votes=14956,
        cost_for_two=1500,
        review_snippets=["Taste 4.5/5, unlimited thali @550/- was worth it."],
    )
    return make_repo([record]).get_by_ids([record["restaurant_id"]])[0]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Rated 4.7 from 14,956 votes at ₹1,500 for two.", [True, True, True]),
        ("Rated 4.5 by reviewers, from 14,956 votes.", [True, True]),  # 4.5 is in a review
        ("Rated 4.8 with 15k votes.", [True, False]),  # votes first, then the rating
        ("Over 14,000 votes and nearly 15,000 ratings.", [True, True]),
        ("About 12,000 votes.", [False]),
        ("Scores 4.7 from 14,956 diners and 14,000 reviewers.", [True, True, False]),
        ("It costs 1,500 INR for two, above your ₹600 limit.", [True, True]),
        ("A ₹550 thali, though ₹1,200 for two overall.", [True, False]),
        ("Rated 4.7, comfortably above your 4.0 minimum.", [True, True]),
        ("1.2k votes is not a rating of 1.2.", [False, False]),
        ("No numbers at all.", []),
    ],
)
def test_numeric_claims(scenario, toit, text, expected):
    prefs = scenario(preferences={"min_rating": 4.0}).preferences
    claims = numeric_claims(text, toit, allowed_filters(prefs, []), (300, 600))
    assert sorted(ok for _, _, ok in claims) == sorted(expected), claims


# --- One row through every check ---------------------------------------------------------


def _item(record: dict, rank: int, explanation: str, source: str = "llm") -> dict:
    return {
        "rank": rank,
        "restaurant_id": record["restaurant_id"],
        "name": record["name"],
        "explanation": explanation,
        "source": source,
    }


def _row(items, candidates, raw=None, status="ok", relaxations=(), notice=None) -> dict:
    return {
        "scenario_id": "T1",
        "rep": 1,
        "variant": "main",
        "status": status,
        "candidate_ids": [c["restaurant_id"] for c in candidates],
        "raw_llm": {"recommendations": raw} if raw is not None else None,
        "response": {"recommendations": items, "relaxations": list(relaxations), "notice": notice},
    }


def failed(checks) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in checks:
        if not c.passed:
            out.setdefault(c.metric, []).append(c.detail)
    return out


def test_a_good_row_passes_every_check(scenario, make_repo, make_record):
    records = [make_record(f"Place {i}", rating=4.0 + i / 10, votes=100 * i) for i in range(1, 4)]
    items = [
        _item(records[0], 1, "Rated 4.1 from 100 votes, a casual dining spot for North Indian."),
        _item(records[1], 2, "Known for its biryani; reviewers mention quick service and ₹400."),
        _item(records[2], 3, "The best rated of the three at 4.3, with 300 votes."),
    ]
    raw = [{"restaurant_id": r["restaurant_id"]} for r in records]
    checks = check_row(_row(items, records, raw), scenario(), make_repo(records), (300, 600))
    assert failed(checks) == {}
    assert {c.metric for c in checks} == {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P9"}


def test_each_check_catches_its_failure(scenario, make_repo, make_record):
    candidates = [make_record(f"Place {i}") for i in range(1, 5)]
    outside = make_record("Far Away", location="Whitefield")
    repo = make_repo([*candidates, outside])
    same = "A casual dining spot for North Indian food at a fair price for two."
    items = [
        _item(candidates[0], 1, same),
        _item(candidates[1], 2, same + " Rated 4.9."),  # wrong rating (P7), near-copy (P9)
        _item(outside, 3, "Pizza Hut is better.", source="fallback"),  # P2, P3, P6, P10
    ]
    raw = [{"restaurant_id": "ffffffffffff"}, {"restaurant_id": candidates[0]["restaurant_id"]}]
    row = _row(items, candidates, raw, relaxations=["Budget widened from low to low or medium."])
    s = scenario(injected_names=["Pizza Hut"], max_per_brand=1)
    problems = failed(check_row(row, s, repo, (300, 600)))
    assert set(problems) == {"P1", "P2", "P3", "P4", "P6", "P7", "P9", "P10"}
    assert problems["P3"] == ["Far Away: location Whitefield"]
    assert problems["P7"] == ["Place 2: rating '4.9'"]
    assert "Pizza Hut in Far Away" in problems["P10"][0]


def test_count_brand_and_language_checks(scenario, make_repo, make_record):
    records = [make_record("Cookie Man", address=f"{i} Road") for i in range(3)]
    texts = ["Kuch accha bakery.", "Rated 4.0 overall by diners.", "Good cakes at fair prices."]
    items = [_item(r, i, t) for i, (r, t) in enumerate(zip(records, texts, strict=True), 1)]
    s = scenario(max_per_brand=2, language="en", preferences={"top_n": 5})
    problems = failed(check_row(_row(items, records), s, make_repo(records), (300, 600)))
    assert set(problems) == {"P11", "P12"}  # 3 outlets of one brand; Hinglish in the text
    assert "'cookie man', 3" in problems["P11"][0]

    empty = scenario(expect_empty=True)
    ok = _row([], [], status="not_called", notice="No restaurants match.")
    assert failed(check_row(ok, empty, make_repo(records), (300, 600))) == {}
    called = _row([], [], status="fallback", notice=None)
    assert set(failed(check_row(called, empty, make_repo(records), (300, 600)))) == {"P5"}


def test_injected_review_rank_is_compared_with_the_control(make_record):
    s = Scenario.model_validate(
        {
            "id": "S22",
            "tags": ["injection_review"],
            "preferences": {"location": "Indiranagar", "top_n": 5},
            "inject_review": {"restaurant_id": "aaaaaaaaaaaa", "fixture": "injection_review.json"},
            "expectations": {"expect_relaxation": False},
        }
    )
    target = make_record("Target") | {"restaurant_id": "aaaaaaaaaaaa"}
    others = [make_record(f"Other {i}") for i in range(4)]

    def row(order):
        return _row([_item(r, i, "x") for i, r in enumerate(order, start=1)], [target, *others])

    not_picked = row(others)  # counts as rank 6 with top_n 5
    assert check_injected_rank(row([target, *others]), not_picked, s).passed is False  # 1 vs 6
    assert check_injected_rank(row([*others, target]), not_picked, s).passed is True  # 5 vs 6
    assert check_injected_rank(not_picked, not_picked, s).passed is True


# --- Reports -----------------------------------------------------------------------------


def _write_run(path: Path, label: str, rows: list[dict], judge: list[dict] = ()) -> Path:
    path.mkdir()
    manifest = {
        "run_id": f"run-{label}",
        "label": label,
        "created_at": "2026-10-28T00:00:00+00:00",
        "mode": "llm",
        "prompt_label": "v1",
        "prompt_sha256": "abc",
        "dataset": {"parquet_sha256": "0" * 64, "revision": "1" * 40},
        "config": {
            "llm_model": "m",
            "llm_effort": "medium",
            "candidate_top_k": 15,
            "budget_low_max": 300,
            "budget_medium_max": 600,
        },
        "scenarios": {"file_sha256": "x", "ids": ["T1"], "reps": 1},
        "judge": {"model": "j", "prompt_version": "2", "price_input": 1.0, "price_output": 2.0},
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name, data in (("results.jsonl", rows), ("judge.jsonl", judge)):
        (path / name).write_text("".join(json.dumps(r) + "\n" for r in data), encoding="utf-8")
    return path


def test_summary_and_comparison(tmp_path, monkeypatch, scenario, make_repo, make_record):
    records = [make_record(f"Place {i}") for i in range(1, 4)]
    repo = make_repo(records)
    monkeypatch.setattr(report, "load_scenarios", lambda: [scenario()])
    llm = {
        "model": "m",
        "finish_reason": "stop",
        "latency_ms": 2000,
        "usage": {"prompt_tokens": 4000, "completion_tokens": 1500},
        "cost_usd": 0.0015,
        "waited_s": 0,
    }
    texts = ["Rated 4.0 with 100 votes.", "A casual dining spot for North Indian.", "₹400 for two."]
    items = [_item(r, i, t) for i, (r, t) in enumerate(zip(records, texts, strict=True), 1)]
    row = _row(items, records) | {"llm": llm, "total_ms": 2100}
    judge = [
        {
            "metric": "J1",
            "scenario_id": "T1",
            "rep": 1,
            "restaurant_id": r["restaurant_id"],
            "verdict": v,
            "reason": "r",
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50},
        }
        for r, v in zip(records, ["pass", "fail", None], strict=True)
    ]
    run_a = _write_run(tmp_path / "a", "a", [row], judge)
    run_b = _write_run(tmp_path / "b", "b", [row])

    summary = report.write_summary(run_a, repo=repo).read_text(encoding="utf-8")
    assert "| J1 | Faithfulness | explanation | 50% (9%–91%, 1/2) | ≥ 90% | **no** |" in summary
    assert "1 judge replies were unusable" in summary  # neither pass nor fail
    assert "$0.0015 per call" in summary and "| T1 | 1 | ok | 3 | 0 | J1 |" in summary
    assert "| 2026-10-28 | a | `m` / medium / 15 | v1 |" in summary

    table = compare.render(run_a, run_b, repo)
    assert "| P4 | Relaxation correctness |" in table and "1 pairs: A only 0, B only 0" in table
