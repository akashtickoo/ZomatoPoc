"""Pieces shared by the evaluation runner, graders and reports (docs/eval.md)."""

import hashlib
import json
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.preferences import UserPreferences

EVAL_DIR = Path(__file__).parent
SCENARIOS_PATH = EVAL_DIR / "scenarios.json"
FIXTURES_DIR = EVAL_DIR / "fixtures"
RESULTS_DIR = EVAL_DIR / "results"

RelaxationKind = Literal["budget", "rating", "location"]


class RunStopped(RuntimeError):
    """Groq asked to wait far longer than a per-minute limit needs: the daily token limit
    is spent. The run stops; `--resume` continues it later."""


class TokenPacer:
    """Keeps the tokens sent in any 60-second window under a limit.

    On Groq's free tier (8K tokens per minute per model) a run then waits a little before
    a call instead of collecting 429s, each of which still counts against the daily
    request limit. Thread-safe; a limit of 0 disables it.
    """

    WINDOW = 60.0

    def __init__(
        self,
        tokens_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.tokens_per_minute = tokens_per_minute
        self._clock = clock
        self._sleep = sleep
        self._log: list[list[float]] = []  # [sent_at, tokens]
        self._lock = threading.Lock()

    def wait(self, estimate: int) -> list[float]:
        """Block until `estimate` tokens fit, then reserve them. Pass the returned entry to
        `record` with the real count once the call is done."""
        while True:
            with self._lock:
                now = self._clock()
                self._log = [e for e in self._log if now - e[0] < self.WINDOW]
                used = sum(e[1] for e in self._log)
                if not self.tokens_per_minute or not self._log:
                    fits = True
                else:
                    fits = used + estimate <= self.tokens_per_minute
                if fits:
                    entry = [now, float(estimate)]
                    self._log.append(entry)
                    return entry
                delay = self.WINDOW - (now - self._log[0][0]) + 0.1
            self._sleep(delay)

    def record(self, entry: list[float], tokens: int) -> None:
        with self._lock:
            entry[1] = float(tokens)


class Expectations(BaseModel):
    """What must hold for a scenario, written by people (eval.md §3.1)."""

    model_config = ConfigDict(extra="forbid")

    expect_relaxation: bool
    relaxation_kinds: list[RelaxationKind] | None = None  # when set, exactly these kinds
    expect_empty: bool = False  # nothing matches: empty result, a notice, no LLM call
    free_text_verifiable: list[str] = []  # must be reflected in the picks or explanations
    free_text_unverifiable: list[str] = []  # must be flagged as not confirmed by the data
    free_text_conflicting: list[str] = []  # conflicts with a filter; must be acknowledged
    must_not_recommend_ids: list[str] = []
    injected_names: list[str] = []  # restaurants an injection asks for (P10)
    max_per_brand: int | None = None  # outlets of one brand allowed in the results (P11)
    language: Literal["en"] | None = None  # explanations must be in this language (P12)
    notes: str = ""

    @property
    def free_text_checks(self) -> bool:
        return bool(
            self.free_text_verifiable or self.free_text_unverifiable or self.free_text_conflicting
        )


class InjectReview(BaseModel):
    """Prepend a hostile review to one candidate, in memory, for this scenario only."""

    restaurant_id: str
    fixture: str  # file in tests/eval/fixtures holding {"snippet": ...}

    def snippet(self) -> str:
        return json.loads((FIXTURES_DIR / self.fixture).read_text(encoding="utf-8"))["snippet"]


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    tags: list[str] = Field(min_length=1)  # tags[0] is the category used in reports
    preferences: UserPreferences
    expectations: Expectations
    inject_review: InjectReview | None = None

    @property
    def category(self) -> str:
        return self.tags[0]

    def variants(self) -> list[str]:
        """'main' is the scenario itself. An injection scenario also runs a 'control'
        without the injected review, to compare the target's rank against (P10)."""
        return ["main", "control"] if self.inject_review else ["main"]


def load_scenarios(path: Path = SCENARIOS_PATH, ids: list[str] | None = None) -> list[Scenario]:
    scenarios = [Scenario.model_validate(s) for s in json.loads(path.read_text(encoding="utf-8"))]
    by_id = {s.id: s for s in scenarios}
    if len(by_id) != len(scenarios):
        raise ValueError(f"{path} has duplicate scenario ids")
    if ids is None:
        return scenarios
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise ValueError(f"Unknown scenario ids: {', '.join(unknown)}")
    return [s for s in scenarios if s.id in ids]  # keep the file's order (determinism)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_write_lock = threading.Lock()


def append_jsonl(path: Path, row: dict) -> None:
    line = json.dumps(row, ensure_ascii=False, default=str)
    with _write_lock, path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def row_key(row: dict) -> tuple[str, int, str]:
    return row["scenario_id"], row["rep"], row["variant"]


def wilson(passes: int, total: int, z: float = 1.96) -> tuple[float, float, float] | None:
    """The pass rate and its 95% Wilson score interval; None when there is nothing to rate."""
    if total == 0:
        return None
    p = passes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def format_rate(passes: int, total: int) -> str:
    """'92% (85–96%, 69/75)': the rate, its 95% interval and the counts."""
    interval = wilson(passes, total)
    if interval is None:
        return "–"
    p, low, high = interval
    return f"{p:.0%} ({low:.0%}–{high:.0%}, {passes}/{total})"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]
