"""Shortcuts for the common commands, the same on Windows, macOS and Linux.

    python tasks.py ingest      download and clean the dataset into data/processed/
    python tasks.py api         start the API on http://localhost:8000 (reloads on changes)
    python tasks.py ui          start the UI on http://localhost:8501
    python tasks.py test        run the tests, then the linter
    python tasks.py demo        run the demo scenes against the running API (docs/demo.md)
    python tasks.py eval ...    run the quality evaluation (docs/eval.md)

Extra arguments go to the underlying command, e.g. `python tasks.py test -k filtering` or
`python tasks.py eval --no-llm --reps 1`. Run it with the virtual environment active, so
the project's packages are used.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
TASKS: dict[str, list[list[str]]] = {
    "ingest": [[PY, "-m", "app.data.ingest"]],
    "api": [[PY, "-m", "uvicorn", "app.main:app", "--reload"]],
    "ui": [[PY, "-m", "streamlit", "run", "ui/streamlit_app.py"]],
    "test": [[PY, "-m", "pytest"], [PY, "-m", "ruff", "check", "."]],
    "demo": [[PY, "-m", "scripts.demo"]],
    "eval": [[PY, "-m", "tests.eval.run_eval"]],
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in TASKS:
        print(__doc__)
        return 2
    name, extra = argv[0], argv[1:]
    for n, command in enumerate(TASKS[name]):
        args = command + extra if n == 0 else command  # extra arguments go to the first
        try:
            code = subprocess.run(args, cwd=ROOT).returncode
        except KeyboardInterrupt:  # Ctrl+C stops a server; that's how api and ui end
            return 130
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
