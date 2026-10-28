"""Compare two runs on the same scenarios (docs/eval.md §7).

    python -m tests.eval.compare tests/eval/results/<run_a> tests/eval/results/<run_b>

Per-scenario metrics are paired on (scenario, rep); per-item metrics compare rates, as the
items differ between runs. Differences come with a 95% interval: one that includes 0 is
noise at this sample size.
"""

import math
import sys
from pathlib import Path

from app.data.repository import RestaurantRepository
from tests.eval.common import format_rate, wilson
from tests.eval.report import METRIC_INFO, all_checks, load_run, operational, tally


def difference(a: tuple[int, int], b: tuple[int, int]) -> tuple[float, float, float] | None:
    """B's rate minus A's, with Newcombe's hybrid score interval (independent samples)."""
    wa, wb = wilson(*a), wilson(*b)
    if wa is None or wb is None:
        return None
    (pa, la, ua), (pb, lb, ub) = wa, wb
    d = pb - pa
    low = d - math.sqrt((pb - lb) ** 2 + (ua - pa) ** 2)
    high = d + math.sqrt((ub - pb) ** 2 + (pa - la) ** 2)
    return d, low, high


def _signed(value: float) -> str:
    return f"{value * 100:+.0f}"


def render(run_a_dir: Path, run_b_dir: Path, repo: RestaurantRepository) -> str:
    run_a, run_b = load_run(run_a_dir), load_run(run_b_dir)
    checks_a, checks_b = all_checks(run_a, repo), all_checks(run_b, repo)
    tally_a, tally_b = tally(checks_a), tally(checks_b)
    label_a, label_b = run_a.manifest["label"], run_b.manifest["label"]

    lines = [
        f"# `{label_a}` (A) vs `{label_b}` (B)",
        "",
        f"A: `{run_a.manifest['run_id']}`  ",
        f"B: `{run_b.manifest['run_id']}`",
        "",
        "| ID | Metric | A | B | B − A, points (95% CI) | Paired on the same scenario runs |",
        "|---|---|---|---|---|---|",
    ]
    for metric, (name, unit, _) in METRIC_INFO.items():
        if metric not in tally_a and metric not in tally_b:
            continue
        a = tally_a.get(metric, (0, 0, []))
        b = tally_b.get(metric, (0, 0, []))
        diff = difference(a[:2], b[:2])
        diff_text = (
            "–"
            if diff is None
            else f"{_signed(diff[0])} ({_signed(diff[1])} to {_signed(diff[2])})"
        )
        paired = "–"
        if unit == "scenario":
            by_key_a = {c.key: c.passed for c in checks_a if c.metric == metric}
            by_key_b = {c.key: c.passed for c in checks_b if c.metric == metric}
            common = by_key_a.keys() & by_key_b.keys()
            if common:
                only_a = sum(by_key_a[k] and not by_key_b[k] for k in common)
                only_b = sum(by_key_b[k] and not by_key_a[k] for k in common)
                paired = f"{len(common)} pairs: A only {only_a}, B only {only_b}"
        cells = [metric, name, format_rate(*a[:2]), format_rate(*b[:2]), diff_text, paired]
        lines.append("| " + " | ".join(cells) + " |")

    ops_a, ops_b = operational(run_a), operational(run_b)

    def ops_line(name: str, fmt, key) -> str:
        values = [key(ops) for ops in (ops_a, ops_b)]
        cells = ["–" if v is None else fmt(v) for v in values]
        return f"| {name} | {cells[0]} | {cells[1]} |"

    lines += [
        "",
        "| Operational | A | B |",
        "|---|---|---|",
        ops_line("LLM calls", str, lambda o: o["llm_calls"]),
        ops_line("Fallbacks", str, lambda o: o["fallbacks"]),
        ops_line("Latency p50", lambda v: f"{v / 1000:.1f} s", lambda o: o["latency_p50_ms"]),
        ops_line("Latency p90", lambda v: f"{v / 1000:.1f} s", lambda o: o["latency_p90_ms"]),
        ops_line(
            "Prompt tokens per call", lambda v: f"{v:,.0f}", lambda o: o["tokens"]["prompt_tokens"]
        ),
        ops_line(
            "Completion tokens per call",
            lambda v: f"{v:,.0f}",
            lambda o: o["tokens"]["completion_tokens"],
        ),
        ops_line("Cost per call", lambda v: f"${v:.4f}", lambda o: o["cost_per_call"]),
        "",
        "Per-scenario differences under about 12 points, and per-item ones under about 5, are "
        "within noise for a full 72-run set (eval.md §7.1); smaller runs need larger differences.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: python -m tests.eval.compare <run A folder> <run B folder>")
        return 2
    print(render(Path(args[0]), Path(args[1]), RestaurantRepository.load()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
