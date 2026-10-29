# The API as a container image (docs/deployment-plan.md, "Docker"). It runs the same way as
# the native Render service in render.yaml: only requirements-api.txt, the committed data in
# data/processed/, one uvicorn worker.
#
#   docker build -t curated-cravings-api .
#   docker run --rm -p 8000:8000 -e GROQ_API_KEY=... curated-cravings-api
#
# Settings come from environment variables only: .env is not copied into the image.

# The Python the project is developed on (3.11–3.13; pyarrow has no newer wheels yet).
FROM python:3.13.17-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /srv

# Dependencies first, so a code change doesn't reinstall them.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY app/ app/
COPY data/processed/ data/processed/

# Nothing is written at runtime (logs go to stderr), so the app runs as an unprivileged
# user that can't change its own code or data.
RUN useradd --system --no-create-home --uid 10001 app
USER app

EXPOSE 8000

# Loading the data takes 10–15 s on a laptop, longer on a small instance.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

# One worker: each would load its own copy of the data and keep its own LLM cache. PORT is
# set by Render; elsewhere it defaults to 8000.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"$PORT\""]
