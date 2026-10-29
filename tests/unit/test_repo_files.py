"""The repository holds everything a clone needs (implementation plan, Phase 8).

A bare `data/` in .gitignore once also ignored the `app/data/` package, which a fresh clone
would have been missing. The processed data is part of the repository too, so a deploy can
start the API without running the ingest (deployment plan, task 1.3).
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SOURCE_DIRS = ["app", "ui", "scripts", "tests", "data/processed", "web/src", "web/public"]
# The deploy files: the Render Blueprint and what Vercel builds web/ from.
DEPLOY_FILES = [
    "render.yaml",
    ".python-version",
    "requirements-api.txt",
    "web/package.json",
    "web/package-lock.json",
    "web/index.html",
    "web/vite.config.ts",
    "web/vercel.json",
    "Dockerfile",
    ".dockerignore",
    "docker-compose.yml",
    "web/Dockerfile",
    "web/.dockerignore",
    "web/nginx.conf",
]


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
        and p.suffix in {".py", ".json", ".md", ".css", ".parquet", ".ts", ".tsx", ".svg"}
        and "__pycache__" not in p.parts
        and "results" not in p.parts  # tests/eval/results: run outputs, ignored on purpose
    ]
    assert "data/processed/restaurants.parquet" in [Path(s).as_posix() for s in sources]
    assert all((ROOT / f).is_file() for f in DEPLOY_FILES)
    ignored = _git("check-ignore", *sources, *DEPLOY_FILES).stdout.split()
    assert ignored == []
    # Anything else under data/ stays out of the repository, as do the web front end's
    # dependencies and build output.
    for generated in ["data/raw/download.csv", "web/node_modules/react/index.js", "web/dist/x.js"]:
        assert _git("check-ignore", "-q", generated).returncode == 0, generated
