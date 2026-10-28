"""The repository holds everything a clone needs (implementation plan, Phase 8).

A bare `data/` in .gitignore once also ignored the `app/data/` package, which a fresh clone
would have been missing.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SOURCE_DIRS = ["app", "ui", "scripts", "tests"]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


@pytest.mark.skipif(
    shutil.which("git") is None or _git("rev-parse").returncode != 0,
    reason="needs git and a repository",
)
def test_no_source_file_is_git_ignored():
    sources = [
        str(p.relative_to(ROOT))
        for d in SOURCE_DIRS
        for p in (ROOT / d).rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".json", ".md", ".css", ".parquet"}
        and "__pycache__" not in p.parts
        and "results" not in p.parts  # tests/eval/results: run outputs, ignored on purpose
    ]
    ignored = _git("check-ignore", *sources).stdout.split()
    assert ignored == []
    assert _git("check-ignore", "-q", "data/processed/restaurants.parquet").returncode == 0
