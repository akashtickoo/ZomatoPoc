"""The Docker setup stays in step with the rest of the deployment (docs/deployment-plan.md).

These checks read the files; they don't need Docker. Building the images is the real test.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
API_DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
WEB_DOCKERFILE = (ROOT / "web" / "Dockerfile").read_text(encoding="utf-8")


def copied_sources(dockerfile: str) -> list[str]:
    """The local sources of each COPY, leaving out copies from another build stage."""
    return [
        source
        for line in re.findall(r"^COPY (?!--from)(.+)$", dockerfile, re.MULTILINE)
        for source in line.split()[:-1]
    ]


def test_api_image_copies_only_what_exists():
    sources = copied_sources(API_DOCKERFILE)
    assert sources == ["requirements-api.txt", "app/", "data/processed/"]
    assert all((ROOT / s).exists() for s in sources)
    assert "pip install -r requirements-api.txt" in API_DOCKERFILE


def test_api_build_context_has_the_data_and_no_secrets():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert {".env", ".env.*", ".venv/", ".git/", "web/"} <= set(ignored)
    # The data folder is excluded except for the processed files, which the image needs.
    assert ignored.index("/data/*") < ignored.index("!/data/processed/")


def test_api_image_uses_the_projects_python():
    base = re.search(r"^FROM python:(\d+\.\d+)\.\d+-slim", API_DOCKERFILE, re.MULTILINE)
    pinned = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    assert base and base.group(1) == pinned


def test_api_image_starts_the_app_like_render():
    render = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "uvicorn app.main:app --host 0.0.0.0" in render
    assert "exec uvicorn app.main:app --host 0.0.0.0 --port" in API_DOCKERFILE
    assert "--workers" not in API_DOCKERFILE  # one worker: one copy of the data, one LLM cache
    assert re.search(r"^USER (?!root)\w+", API_DOCKERFILE, re.MULTILINE)


def test_web_image_uses_the_node_vercel_uses():
    engines = json.loads((ROOT / "web" / "package.json").read_text(encoding="utf-8"))["engines"]
    base = re.search(r"^FROM node:(\d+)\.\d+\.\d+-alpine", WEB_DOCKERFILE, re.MULTILINE)
    assert base and f"{base.group(1)}.x" == engines["node"]
    assert copied_sources(WEB_DOCKERFILE)[0:2] == ["package.json", "package-lock.json"]
    assert "npm ci" in WEB_DOCKERFILE


def test_compose_lets_the_page_call_the_api():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    # The browser loads the page from the web port and calls the API's published port.
    assert '"8080:8080"' in compose and '"8000:8000"' in compose
    assert """CORS_ORIGINS: '["http://localhost:8080"]'""" in compose
    assert "VITE_API_BASE_URL: http://localhost:8000" in compose
    assert "listen 8080;" in (ROOT / "web" / "nginx.conf").read_text(encoding="utf-8")
