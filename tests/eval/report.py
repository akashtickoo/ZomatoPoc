"""Summaries of a run: every metric with its 95% interval (docs/eval.md §7).

python -m tests.eval.report tests/eval/results/<run_id>    # rewrite summary.md
"""

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.data.repository import RestaurantRepository
from tests.eval.common import (
    SCENARIOS_PATH,
    file_sha256,
    format_rate,
    load_scenarios,
    percentile,
    read_jsonl,
    wilson,
)
from tests.eval.programmatic import Check, check_rows

# id: (name, unit, target)
METRIC_INFO = {
    "P1": ("Raw ID validity", "item", 0.99),
    "P2": ("Final ID validity", "item", 1.0),
    "P3": ("Constraint compliance", "item", 1.0),
    "P4": ("Relaxation correctness", "scenario", 1.0),
    "P5": ("Count correctness", "scenario", 1.0),
    "P6": ("LLM coverage", "item", 0.95),
    "P7": ("Numeric fact accuracy", "explanation", 0.98),
    "P9": ("Distinct explanations", "scenario", 1.0),
    "P10": ("Injection resistance", "scenario", 1.0),
    "P11": ("Brand variety", "scenario", 1.0),
    "P12": ("English output", "scenario", 1.0),
    "J1": ("Faithfulness", "explanation", 0.90),
    "J2": ("Free-text handling", "scenario", 0.85),
    "J3": ("Specificity", "explanation", 0.90),
    "J4": ("LLM wins or ties vs baseline", "scenario", 0.70),
}
CALLED = {"ok", "fallback", "truncated"}  # statuses where the LLM was called


@dataclass
class Run:
    path: Path
    manifest: dict
    rows: list[dict]
    errors: list[dict]
    judge: list[dict]


def load_run(run_dir: Path) -> Run:
    return Run(
        path=run_dir,
        manifest=json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")),
        rows=read_jsonl(run_dir / "results.jsonl"),
        errors=read_jsonl(run_dir / "errors.jsonl"),
        judge=read_jsonl(run_dir / "judge.jsonl"),
    )


def judge_checks(judge_rows: list[dict]) -> list[Check]:
    """One check per judged item; the latest verdict wins, and judge errors are left out
    (they are neither passes nor fails, eval.md §5)."""
    latest: dict[tuple, dict] = {}
    for row in judge_rows:
        if row["verdict"] is None:
            continue
        key = (row["metric"], row["scenario_id"], row["rep"], row.get("restaurant_id"))
        latest[key] = row
    checks = []
    for (metric, scenario_id, rep, rid), row in latest.items():
        key = (scenario_id, rep) if rid is None else (scenario_id, rep, rid)
        if metric == "J4":
            passed = row["verdict"] in ("llm", "tie")
            detail = f"{row['verdict']}: {row['reason']}"
        else:
            passed = row["verdict"] == "pass"
            detail = row["reason"] or ""
        checks.append(Check(metric, key, passed, detail))
    return checks


def all_checks(run: Run, repo: RestaurantRepository) -> list[Check]:
    scenarios = {s.id: s for s in load_scenarios()}
    config = run.manifest["config"]
    thresholds = (config["budget_low_max"], config["budget_medium_max"])
    rows = [r for r in run.rows if r["scenario_id"] in scenarios]
    return check_rows(rows, scenarios, repo, thresholds) + judge_checks(run.judge)


def tally(checks: list[Check]) -> dict[str, tuple[int, int, list[Check]]]:
    by_metric: dict[str, list[Check]] = defaultdict(list)
    for check in checks:
        by_metric[check.metric].append(check)
    return {
        metric: (sum(c.passed for c in cs), len(cs), [c for c in cs if not c.passed])
        for metric, cs in by_metric.items()
    }


def operational(run: Run) -> dict:
    called = [r for r in run.rows if r["status"] in CALLED and r["llm"]]
    ok = [r for r in called if r["status"] == "ok"]
    usage_keys = ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens")
    tokens = {
        k: (sum(r["llm"]["usage"].get(k, 0) for r in called) / len(called) if called else None)
        for k in usage_keys
    }
    costs = [r["llm"]["cost_usd"] for r in called]
    judge = run.manifest.get("judge") or {}
    judge_usage = [r["usage"] for r in run.judge if r.get("usage")]
    judge_cost = (
        sum(
            u.get("prompt_tokens", 0) * judge.get("price_input", 0)
            + u.get("completion_tokens", 0) * judge.get("price_output", 0)
            for u in judge_usage
        )
        / 1_000_000
    )
    return {
        "llm_calls": len(called),
        "latency_p50_ms": percentile([r["llm"]["latency_ms"] for r in ok], 0.5),
        "latency_p90_ms": percentile([r["llm"]["latency_ms"] for r in ok], 0.9),
        "total_p50_ms": percentile([r["total_ms"] for r in run.rows], 0.5),
        "tokens": tokens,
        "cost_per_call": sum(costs) / len(costs) if costs else None,
        "cost_total": sum(costs),
        "finish_reasons": Counter(r["llm"]["finish_reason"] for r in called),
        "fallbacks": sum(r["status"] in ("fallback", "truncated") for r in called),
        "truncated": sum(r["status"] == "truncated" for r in called),
        "served_models": Counter(r["llm"]["model"] for r in called),
        "waited_s": sum(r["llm"].get("waited_s") or 0 for r in called),
        "judge_calls": len(run.judge),
        "judge_errors": sum(1 for r in run.judge if r["verdict"] is None),
        "judge_tokens": sum(sum(u.values()) for u in judge_usage),
        "judge_cost": judge_cost,
    }


def _ms(value: float | None) -> str:
    return "–" if value is None else f"{value / 1000:.1f} s"


def _metric_rows(results: dict) -> list[str]:
    lines = ["| ID | Metric | Unit | Result (95% CI) | Target | Met |", "|---|---|---|---|---|---|"]
    for metric, (name, unit, target) in METRIC_INFO.items():
        if metric not in results:
            continue
        passes, total, _ = results[metric]
        rate = wilson(passes, total)
        met = "–" if rate is None else ("yes" if rate[0] >= target else "**no**")
        cells = [metric, name, unit, format_rate(passes, total), f"≥ {target:.0%}", met]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def results_log_row(run: Run, results: dict, ops: dict) -> str:
    """A row for eval.md §10."""
    m, c = run.manifest, run.manifest["config"]

    def rate(metric: str) -> str:
        if metric not in results:
            return "–"
        passes, total, _ = results[metric]
        return format_rate(passes, total)

    calls = ops["llm_calls"]
    fallback = format_rate(ops["fallbacks"], calls) if calls else "–"
    cost = f"${ops['cost_per_call']:.4f}" if ops["cost_per_call"] is not None else "–"
    setup = f"`{c['llm_model']}` / {c['llm_effort']} / {c['candidate_top_k']}"
    if m["mode"] != "llm":
        setup = m["mode"]
    cells = [
        m["created_at"][:10],
        m["label"],
        setup,
        m["prompt_label"],
        rate("P2"),
        rate("P3"),
        rate("J1"),
        rate("J2"),
        rate("J3"),
        rate("J4"),
        fallback,
        _ms(ops["latency_p50_ms"]),
        cost,
        "",
    ]
    return "| " + " | ".join(cells) + " |"


def render(run: Run, repo: RestaurantRepository) -> str:
    m, c = run.manifest, run.manifest["config"]
    checks = all_checks(run, repo)
    results = tally(checks)
    ops = operational(run)
    statuses = Counter(r["status"] for r in run.rows)
    judge = m.get("judge")
    judge_text = f"`{judge['model']}`, prompt v{judge['prompt_version']}" if judge else "not judged"
    status_text = ", ".join(f"{n} {s}" for s, n in statuses.most_common())

    lines = [f"# Evaluation run `{m['run_id']}`", ""]
    if m["scenarios"]["file_sha256"] != file_sha256(SCENARIOS_PATH):
        lines += ["> ⚠️ scenarios.json has changed since this run; expectations may differ.", ""]
    lines += [
        "| Setting | Value |",
        "|---|---|",
        f"| Mode | {m['mode']} |",
        f"| Model / effort / top-K | `{c['llm_model']}` / {c['llm_effort']} "
        f"/ {c['candidate_top_k']} |",
        f"| Prompt | {m['prompt_label']} (`{m['prompt_sha256']}`) |",
        f"| Scenarios × reps | {len(m['scenarios']['ids'])} × {m['scenarios']['reps']} |",
        f"| Data | `{m['dataset']['parquet_sha256'][:12]}` "
        f"(revision `{m['dataset']['revision'][:12]}`) |",
        f"| Judge | {judge_text} |",
        "",
        f"**Rows:** {len(run.rows)} results ({status_text}); "
        f"{len(run.errors)} infrastructure errors (not scored).",
        "",
        "## Metrics",
        "",
        *_metric_rows(results),
        "",
    ]
    if ops["judge_errors"]:
        lines += [
            f"{ops['judge_errors']} judge replies were unusable and are left out of J1–J4.",
            "",
        ]

    tokens = ops["tokens"]
    lines += ["## Operational", ""]
    if ops["llm_calls"]:
        lines += [
            f"- **LLM calls:** {ops['llm_calls']}; fallbacks {ops['fallbacks']} "
            f"(of which cut off: {ops['truncated']}); "
            f"finish reasons {dict(ops['finish_reasons'])}; "
            f"served models {dict(ops['served_models'])}",
            f"- **Latency of the LLM call:** p50 {_ms(ops['latency_p50_ms'])}, "
            f"p90 {_ms(ops['latency_p90_ms'])}; whole scenario run p50 {_ms(ops['total_p50_ms'])} "
            f"(rate-limit waits: {ops['waited_s']:.0f} s in total)",
            f"- **Tokens per call:** {tokens['prompt_tokens']:,.0f} prompt "
            f"({tokens['cached_tokens']:,.0f} cached), "
            f"{tokens['completion_tokens']:,.0f} completion "
            f"({tokens['reasoning_tokens']:,.0f} reasoning)",
            f"- **Cost:** ${ops['cost_per_call']:.4f} per call, "
            f"${ops['cost_total']:.3f} for the run",
        ]
    else:
        lines.append("- No LLM calls in this run.")
    if ops["judge_calls"]:
        lines.append(
            f"- **Judge:** {ops['judge_calls']} calls, {ops['judge_tokens']:,} tokens, "
            f"${ops['judge_cost']:.3f} (separate from the system's cost)"
        )
    lines.append("")

    failed_by_run: dict[tuple, list[str]] = defaultdict(list)
    for check in checks:
        if not check.passed:
            failed_by_run[check.key[:2]].append(check.metric)
    lines += [
        "## Scenario runs",
        "",
        "| Scenario | Rep | Status | Items | Relaxations | Failed checks |",
        "|---|---|---|---|---|---|",
    ]
    for row in sorted(run.rows, key=lambda r: (r["scenario_id"], r["rep"], r["variant"])):
        if row["variant"] != "main":
            continue
        failed = sorted(set(failed_by_run[(row["scenario_id"], row["rep"])]))
        lines.append(
            f"| {row['scenario_id']} | {row['rep']} | {row['status']} | "
            f"{len(row['response']['recommendations'])} | {len(row['response']['relaxations'])} | "
            f"{', '.join(failed) or '–'} |"
        )
    lines.append("")

    failures = [(metric, f) for metric in METRIC_INFO for f in results.get(metric, (0, 0, []))[2]]
    if failures:
        lines += ["## Failed checks", ""]
        for metric, check in failures:
            where = " ".join(str(k) for k in check.key)
            lines.append(f"- **{metric}** {where}: {check.detail}")
        lines.append("")
    if run.errors:
        lines += ["## Infrastructure errors (re-run with `--resume`)", ""]
        lines += [f"- {e['scenario_id']} rep {e['rep']}: {e['error']}" for e in run.errors]
        lines.append("")
    lines += ["## Results-log row (eval.md §10)", "", results_log_row(run, results, ops), ""]
    return "\n".join(lines)


def write_summary(run_dir: Path, repo: RestaurantRepository | None = None) -> Path:
    run = load_run(run_dir)
    repo = repo or RestaurantRepository.load()
    path = run_dir / "summary.md"
    path.write_text(render(run, repo), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m tests.eval.report <run folder>")
        return 2
    print(write_summary(Path(args[0])).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
