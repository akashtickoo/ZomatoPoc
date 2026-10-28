"""Run the evaluation scenarios through the real recommender (docs/eval.md §6).

Calls `RecommendationService.recommend()`, the same path the API uses, and saves both
the raw LLM answer (before the validator) and the final response. It costs tokens, so it
runs on demand, never in CI.

    python -m tests.eval.run_eval --scenarios S01,S07,S10 --reps 1        # smoke run
    python -m tests.eval.run_eval --reps 3 --label baseline
    python -m tests.eval.run_eval --reps 1 --top-k 25 --label topk25      # one variable
    python -m tests.eval.run_eval --no-llm --label deterministic          # E5, free
    python -m tests.eval.run_eval --oracle --label oracle                 # hand-written answers
    python -m tests.eval.run_eval --resume tests/eval/results/<run_id>

Results go to tests/eval/results/<run_id>/; grade them with `python -m tests.eval.judge`.
"""

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.config import PROJECT_ROOT, Settings, settings
from app.data.repository import RestaurantRepository
from app.llm.client import (
    GroqRankingClient,
    LLMRanking,
    LLMRecommendation,
    LLMResult,
    LLMUnavailableError,
    RankingClient,
)
from app.llm.prompt_builder import approx_tokens, system_prompt, user_message
from app.models.preferences import UserPreferences
from app.observability import llm_cost_usd
from app.services.filtering import filter_candidates
from app.services.recommender import MIN_CANDIDATES_PER_PICK, RecommendationService
from tests.eval.common import (
    FIXTURES_DIR,
    RESULTS_DIR,
    SCENARIOS_PATH,
    InjectReview,
    RunStopped,
    Scenario,
    TokenPacer,
    append_jsonl,
    file_sha256,
    load_scenarios,
    read_jsonl,
    row_key,
    text_sha256,
)

CANDIDATE_ID = re.compile(r'"restaurant_id":"([0-9a-f]{12})"')
EXPECTED_COMPLETION_TOKENS = 1500  # measured average in Phase 6, reasoning included
FREE_TIER_TPM = 8000
MAX_ATTEMPTS = 6
MAX_ATTEMPTS_TOO_LARGE = 2  # a 413 can mean the request alone exceeds the per-minute limit
CONFIG_KEYS = (
    "llm_model",
    "llm_effort",
    "candidate_top_k",
    "min_candidates",
    "budget_low_max",
    "budget_medium_max",
    "llm_timeout_seconds",
    "llm_price_input",
    "llm_price_cached_input",
    "llm_price_output",
)


class InfraError(RuntimeError):
    """A failure that says nothing about quality: a rate limit that outlasted the retries,
    an outage, or the wrong model answering. It goes to errors.jsonl, never into the scores
    (eval.md §6); `--resume` retries it."""


def _spent(usage: dict[str, int]) -> int:
    """Tokens that count against a per-minute limit (reasoning is part of completion)."""
    return usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)


def _retryable(exc: LLMUnavailableError) -> bool:
    status = exc.status_code
    if status is None:
        return exc.reason.startswith("connection failed")  # includes timeouts
    return status in (408, 409, 413, 429) or status >= 500


class RecordingClient:
    """Wraps the real ranking client for one scenario run.

    Records the prompt, the raw answer, its usage and how many attempts it took. Waits out
    rate limits (Groq's retry-after) so they never show up as fallbacks. Answers the model
    got wrong (cut off, invalid JSON) are re-raised: the service falls back, as it would for
    a user, and the row records that outcome.
    """

    def __init__(
        self,
        inner: RankingClient,
        *,
        expected_model: str | None = None,
        system_prompt_override: str | None = None,
        pacer: TokenPacer | None = None,
        max_wait: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._inner = inner
        self._expected_model = expected_model
        self._system_prompt_override = system_prompt_override
        self._pacer = pacer
        self._max_wait = max_wait
        self._sleep = sleep
        self.system_prompt: str | None = None
        self.user_message: str | None = None
        self.result: LLMResult | None = None
        self.error: LLMUnavailableError | None = None
        self.attempts = 0
        self.waited_s = 0.0

    def rank(self, system_prompt: str, user_message: str) -> LLMResult:
        self.system_prompt = self._system_prompt_override or system_prompt
        self.user_message = user_message
        while True:
            self.attempts += 1
            entry = None
            if self._pacer is not None:
                estimate = approx_tokens(self.system_prompt + user_message)
                entry = self._pacer.wait(estimate + EXPECTED_COMPLETION_TOKENS)
            try:
                result = self._inner.rank(self.system_prompt, user_message)
            except LLMUnavailableError as exc:
                if entry is not None:  # a rejected request spends nothing
                    self._pacer.record(entry, _spent(exc.usage))
                if exc.is_config_error or exc.reason.startswith("no Groq API key"):
                    # Every other call would fail the same way.
                    raise RunStopped(f"LLM configuration problem: {exc.reason}") from exc
                if not _retryable(exc):
                    self.error = exc  # a real outcome: the service serves the fallback
                    raise
                limit = MAX_ATTEMPTS_TOO_LARGE if exc.status_code == 413 else MAX_ATTEMPTS
                if self.attempts >= limit:
                    raise InfraError(f"{exc.reason} after {self.attempts} attempts") from exc
                wait = exc.retry_after
                if wait is None:
                    wait = 60.0 if exc.status_code == 413 else min(60.0, 5.0 * 2**self.attempts)
                if wait > self._max_wait:
                    raise RunStopped(
                        f"Groq asked to wait {wait:.0f} s ({exc.reason}): the daily token "
                        "limit is probably spent. Continue later with --resume."
                    ) from exc
                self.waited_s += wait
                self._sleep(wait + 0.5)
                continue
            if entry is not None:
                self._pacer.record(entry, _spent(result.usage))
            if self._expected_model and result.model != self._expected_model:
                raise InfraError(f"served model {result.model}, expected {self._expected_model}")
            self.result = result
            return result


class OracleClient:
    """Returns hand-written ideal answers instead of calling the LLM (eval.md §6, oracle
    check). They pass through the same validator and enricher as a real answer."""

    def __init__(self, answers: list[dict]):
        self._answers = answers

    def rank(self, system_prompt: str, user_message: str) -> LLMResult:
        recs = [
            LLMRecommendation(
                restaurant_id=a["restaurant_id"], rank=i, explanation=a["explanation"]
            )
            for i, a in enumerate(self._answers, start=1)
        ]
        return LLMResult(
            ranking=LLMRanking(recommendations=recs),
            model="oracle",
            request_id=None,
            stop_reason="stop",
            latency_ms=0,
        )


def with_injected_review(repo: RestaurantRepository, inject: InjectReview) -> RestaurantRepository:
    """A copy of the repository where one restaurant's first review is the hostile one.
    The dataset on disk is never edited."""
    restaurants = repo.at_positions(range(len(repo)))
    if not any(r.restaurant_id == inject.restaurant_id for r in restaurants):
        raise ValueError(f"inject_review: no restaurant {inject.restaurant_id}")
    snippet = inject.snippet()
    return RestaurantRepository(
        [
            r.model_copy(update={"review_snippets": [snippet, *r.review_snippets][:3]})
            if r.restaurant_id == inject.restaurant_id
            else r
            for r in restaurants
        ]
    )


@dataclass
class RunContext:
    """Everything one run needs, shared by its scenario runs."""

    repo: RestaurantRepository
    cfg: Settings
    mode: str  # "llm", "no_llm" or "oracle"
    system_prompt: str
    llm: RankingClient | None = None
    oracle: dict[str, list[dict]] = field(default_factory=dict)
    pacer: TokenPacer | None = None
    max_wait: float = 120.0
    sleep: Callable[[float], None] = time.sleep
    _injected: dict[str, RestaurantRepository] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def repo_for(self, scenario: Scenario, variant: str) -> RestaurantRepository:
        if scenario.inject_review is None or variant == "control":
            return self.repo
        with self._lock:
            if scenario.id not in self._injected:
                self._injected[scenario.id] = with_injected_review(
                    self.repo, scenario.inject_review
                )
            return self._injected[scenario.id]

    def client_for(self, scenario: Scenario) -> RecordingClient | None:
        if self.mode == "no_llm":
            return None
        if self.mode == "oracle":
            return RecordingClient(OracleClient(self.oracle[scenario.id]))
        return RecordingClient(
            self.llm,
            expected_model=self.cfg.llm_model,
            system_prompt_override=self.system_prompt,
            pacer=self.pacer,
            max_wait=self.max_wait,
            sleep=self.sleep,
        )

    def candidates(self, scenario: Scenario, variant: str):
        prefs = scenario.preferences
        return filter_candidates(
            self.repo_for(scenario, variant),
            prefs,
            top_k=max(self.cfg.candidate_top_k, MIN_CANDIDATES_PER_PICK * prefs.top_n),
            min_candidates=self.cfg.min_candidates,
        )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _status(client: RecordingClient | None, has_candidates: bool) -> str:
    if not has_candidates:
        return "not_called"  # nothing to rank, so no LLM call in any mode (EC-FIL-17)
    if client is None:
        return "no_llm"
    if client.user_message is None:
        return "not_called"
    if client.result is not None:
        return "ok"
    if client.error is not None and client.error.finish_reason == "length":
        return "truncated"
    return "fallback"


def _llm_fields(client: RecordingClient, cfg: Settings) -> dict | None:
    if client.result is not None:
        result = client.result
        usage = result.usage
        fields = {
            "model": result.model,
            "request_id": result.request_id,
            "finish_reason": result.stop_reason,
            "latency_ms": result.latency_ms,
            "error": None,
        }
    elif client.error is not None:
        usage = client.error.usage
        fields = {
            "model": cfg.llm_model,
            "request_id": client.error.request_id,
            "finish_reason": client.error.finish_reason,
            "latency_ms": None,
            "error": client.error.reason,
        }
    else:
        return None
    return {
        **fields,
        "usage": usage,
        "cost_usd": llm_cost_usd(usage, cfg),
        "attempts": client.attempts,
        "waited_s": round(client.waited_s, 1),
    }


def run_one(scenario: Scenario, rep: int, variant: str, ctx: RunContext) -> tuple[str, dict]:
    """One scenario run. Returns ("result", row) or ("error", row)."""
    prefs: UserPreferences = scenario.preferences
    repo = ctx.repo_for(scenario, variant)
    base = {"scenario_id": scenario.id, "rep": rep, "variant": variant}
    start = time.perf_counter()
    filtered = ctx.candidates(scenario, variant)
    candidate_ids = [r.restaurant_id for r in filtered.candidates]

    client = ctx.client_for(scenario)
    try:
        response = RecommendationService(repo, client, ctx.cfg).recommend(prefs)
    except InfraError as exc:
        return "error", {
            **base,
            "error": str(exc),
            "attempts": client.attempts if client else 0,
            "at": _now(),
        }

    if client is not None and client.user_message is not None:
        sent = CANDIDATE_ID.findall(client.user_message)
        if sent != candidate_ids:  # the harness must see exactly what the model saw
            raise AssertionError(f"{scenario.id}: candidate list differs from the prompt's")
    # The deterministic ranking for the same request, for the pairwise judge (J4). Free.
    baseline = None
    if client is not None and filtered.candidates:
        baseline = RecommendationService(repo, None, ctx.cfg).recommend(prefs).model_dump()

    row = {
        **base,
        "status": _status(client, bool(filtered.candidates)),
        "category": scenario.category,
        "preferences": prefs.model_dump(),
        "candidate_ids": candidate_ids,
        "total_matched": filtered.total_matched,
        "prompt": None,
        "raw_llm": None,
        "llm": None,
        "response": response.model_dump(),
        "baseline_response": baseline,
    }
    if client is not None and client.user_message is not None:
        row["prompt"] = {
            "system_sha256": text_sha256(client.system_prompt)[:12],
            "user_message": client.user_message,
        }
        row["raw_llm"] = client.result.ranking.model_dump() if client.result else None
        row["llm"] = _llm_fields(client, ctx.cfg)
    row["total_ms"] = round((time.perf_counter() - start) * 1000)
    row["finished_at"] = _now()
    return "result", row


def git_commit() -> dict:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    prompt_status = git("status", "--porcelain", "app/llm/prompts/system.md")
    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "prompt_uncommitted": bool(prompt_status),
    }


def build_manifest(run_id: str, args, cfg: Settings, prompt_text: str, ids: list[str]) -> dict:
    mode = "no_llm" if args.no_llm else "oracle" if args.oracle else "llm"
    return {
        "run_id": run_id,
        "label": args.label,
        "created_at": _now(),
        "mode": mode,
        "git": git_commit(),
        "prompt_label": args.prompt_label,
        "prompt_sha256": text_sha256(prompt_text)[:12],
        "system_prompt": prompt_text,
        "dataset": {
            "repo": cfg.dataset_repo,
            "revision": cfg.dataset_revision,
            "parquet_sha256": file_sha256(cfg.data_path),
        },
        "config": {key: getattr(cfg, key) for key in CONFIG_KEYS},
        "scenarios": {"file_sha256": file_sha256(SCENARIOS_PATH), "ids": ids, "reps": args.reps},
        "judge": None,
    }


def estimate(tasks, ctx: RunContext) -> tuple[int, int]:
    """LLM calls and approximate prompt tokens for the tasks still to run."""
    calls = prompt_tokens = 0
    sizes: dict[tuple[str, str], int] = {}
    for scenario, _, variant in tasks:
        key = (scenario.id, variant)
        if key not in sizes:
            filtered = ctx.candidates(scenario, variant)
            sizes[key] = (
                approx_tokens(
                    ctx.system_prompt
                    + user_message(scenario.preferences, filtered.candidates, filtered.relaxations)
                )
                if filtered.candidates
                else 0
            )
        if sizes[key]:
            calls += 1
            prompt_tokens += sizes[key]
    return calls, prompt_tokens


def confirm_cost(tasks, ctx: RunContext, assume_yes: bool) -> bool:
    if ctx.mode != "llm":
        return True
    calls, prompt_tokens = estimate(tasks, ctx)
    completion = calls * EXPECTED_COMPLETION_TOKENS
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion}
    cost = llm_cost_usd(usage, ctx.cfg)
    minutes = (prompt_tokens + completion) / FREE_TIER_TPM
    print(
        f"{len(tasks)} scenario runs, {calls} LLM calls on {ctx.cfg.llm_model}: about "
        f"{prompt_tokens:,} prompt + {completion:,} completion tokens, ${cost:.3f}. "
        f"At Groq's free-tier {FREE_TIER_TPM:,} tokens/minute that is at least "
        f"{minutes:.0f} minutes."
    )
    if assume_yes:
        return True
    return input("Proceed? [y/N] ").strip().lower() in ("y", "yes")


def run(tasks, ctx: RunContext, run_dir: Path, concurrency: int = 1) -> None:
    """Run every task, appending rows as they finish. Raises RunStopped to stop early."""
    results, errors = run_dir / "results.jsonl", run_dir / "errors.jsonl"
    stop = threading.Event()

    def work(task):
        scenario, rep, variant = task
        if stop.is_set():
            return
        try:
            kind, row = run_one(scenario, rep, variant, ctx)
        except RunStopped:
            stop.set()
            raise
        append_jsonl(results if kind == "result" else errors, row)
        print(_progress(kind, row), flush=True)

    if concurrency <= 1:
        for task in tasks:
            work(task)
        return
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(work, task) for task in tasks]
        for future in futures:
            future.result()  # re-raises RunStopped from a worker


def _progress(kind: str, row: dict) -> str:
    label = f"{row['scenario_id']} rep {row['rep']}" + (
        f" ({row['variant']})" if row["variant"] != "main" else ""
    )
    if kind == "error":
        return f"{label}: ERROR {row['error']}"
    llm = row["llm"] or {}
    usage = llm.get("usage") or {}
    tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    parts = [row["status"], f"{len(row['response']['recommendations'])} items"]
    if llm.get("latency_ms") is not None:
        parts.append(f"{llm['latency_ms'] / 1000:.1f} s")
    if tokens:
        parts.append(f"{tokens:,} tokens, ${llm['cost_usd']:.4f}")
    if llm.get("waited_s"):
        parts.append(f"waited {llm['waited_s']:.0f} s for rate limits")
    return f"{label}: " + ", ".join(parts)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the evaluation scenarios (docs/eval.md).")
    parser.add_argument("--scenarios", help="comma-separated ids, e.g. S01,S07 (default: all)")
    parser.add_argument("--reps", type=int, default=3, help="runs per scenario (default 3)")
    parser.add_argument("--label", default="run", help="name for the run, e.g. baseline")
    parser.add_argument("--top-k", type=int, help="override CANDIDATE_TOP_K (10 or more)")
    parser.add_argument("--effort", choices=["low", "medium", "high"], help="override LLM_EFFORT")
    parser.add_argument("--model", help="override LLM_MODEL")
    parser.add_argument("--prompt", type=Path, help="system prompt file to test instead")
    parser.add_argument("--prompt-label", default="v1", help="version name for the prompt")
    parser.add_argument("--no-llm", action="store_true", help="deterministic ranking only (E5)")
    parser.add_argument(
        "--oracle",
        type=Path,
        nargs="?",
        const=FIXTURES_DIR / "oracle.json",
        help="use hand-written answers (default fixtures/oracle.json)",
    )
    parser.add_argument("--concurrency", type=int, default=1, help="parallel requests (default 1)")
    parser.add_argument(
        "--tpm",
        type=int,
        default=FREE_TIER_TPM,
        help=f"tokens per minute to stay under (default {FREE_TIER_TPM:,}: Groq's free tier)",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=120.0,
        help="stop if Groq asks to wait longer than this many seconds (default 120)",
    )
    parser.add_argument("--resume", type=Path, help="continue an earlier run in this folder")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = parser.parse_args(argv)
    if args.top_k is not None and args.top_k < 10:
        parser.error("--top-k must be 10 or more (the largest top_n)")
    if args.no_llm and args.oracle:
        parser.error("--no-llm and --oracle are different modes; pick one")
    return args


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)

    if args.resume:
        run_dir = args.resume
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["scenarios"]["file_sha256"] != file_sha256(SCENARIOS_PATH):
            print("scenarios.json changed since this run started; start a new run instead.")
            return 2
        cfg = settings.model_copy(update=manifest["config"] | {"llm_cache_ttl_seconds": 0})
        prompt_text = manifest["system_prompt"]
        mode = manifest["mode"]
        ids, reps = manifest["scenarios"]["ids"], manifest["scenarios"]["reps"]
        oracle_path = FIXTURES_DIR / "oracle.json"
    else:
        overrides = {"llm_cache_ttl_seconds": 0}  # every rep must reach the LLM
        for key, value in (
            ("candidate_top_k", args.top_k),
            ("llm_effort", args.effort),
            ("llm_model", args.model),
        ):
            if value is not None:
                overrides[key] = value
        cfg = settings.model_copy(update=overrides)
        prompt_text = args.prompt.read_text(encoding="utf-8") if args.prompt else system_prompt()
        mode = "no_llm" if args.no_llm else "oracle" if args.oracle else "llm"
        oracle_path = args.oracle
        reps = args.reps
        ids = None

    oracle = {}
    if mode == "oracle":
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        ids = ids or list(oracle)
    elif ids is None:
        ids = args.scenarios.split(",") if args.scenarios else None
    scenarios = load_scenarios(ids=ids)
    ids = [s.id for s in scenarios]

    if mode == "llm" and not cfg.groq_api_key:
        print("GROQ_API_KEY is not set; use --no-llm for the deterministic ranking only.")
        return 2

    print("Loading the restaurant data…", flush=True)
    ctx = RunContext(
        repo=RestaurantRepository.load(cfg),
        cfg=cfg,
        mode=mode,
        system_prompt=prompt_text,
        llm=GroqRankingClient(cfg) if mode == "llm" else None,
        oracle=oracle,
        pacer=TokenPacer(args.tpm),
        max_wait=args.max_wait,
    )

    if not args.resume:
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{args.label}"
        run_dir = RESULTS_DIR / run_id
        run_dir.mkdir(parents=True)
        manifest = build_manifest(run_id, args, cfg, prompt_text, ids)
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    done = {row_key(row) for row in read_jsonl(run_dir / "results.jsonl")}
    tasks = [
        (s, rep, variant)
        for rep in range(1, reps + 1)
        for s in scenarios
        for variant in s.variants()
        if (s.id, rep, variant) not in done
    ]
    print(f"Run folder: {run_dir}")
    if not tasks:
        print("Nothing left to run.")
    elif not confirm_cost(tasks, ctx, args.yes):
        return 1

    from tests.eval.report import write_summary  # loads the data again; only needed here

    exit_code = 0
    try:
        run(tasks, ctx, run_dir, concurrency=args.concurrency)
    except RunStopped as exc:
        print(f"Stopped: {exc}")
        exit_code = 3
    summary = write_summary(run_dir, repo=ctx.repo)
    print(f"Summary: {summary}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
