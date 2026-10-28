# ZomatoPoc

An AI-powered restaurant recommender for Bangalore, built as a proof of concept for the NextLeap program. It filters the [Zomato dataset](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation) by location, budget, cuisine and rating, then asks an LLM on [Groq](https://console.groq.com) (`openai/gpt-oss-120b`) to rank the matches and explain each choice. Every fact shown comes from the dataset. If the LLM is unavailable, a deterministic ranking with template explanations is returned instead.

![Curated Cravings: Italian restaurants in Indiranagar, ranked and explained by the LLM](docs/images/curated-cravings.png)

> **Status:** Phases 0–6 and 8 of the [implementation plan](docs/implementation-plan.md) are done. Phase 7 (quality evaluation and tuning) is in progress: the evaluation harness is built and checked, and the full baseline and tuning runs are waiting on Groq's free-tier daily token limit.

## How it works

1. **Filter.** The request (neighbourhood, budget, cuisines, minimum rating, free text) is matched against 12,415 restaurants. If fewer than 5 match, the search is widened one step at a time (budget, then rating, then neighbourhood; never cuisine), and each step is listed in the API response.
2. **Pre-rank.** Matches are sorted by a score that balances rating against the number of votes, and the top 15 become candidates.
3. **Rank and explain.** The candidates and the preferences go to the LLM, which picks the best few, orders them, and writes why each one fits, citing that restaurant's data. It is told to say when a wish (such as "rooftop") isn't confirmed by the data.
4. **Check.** Any pick that wasn't a candidate is dropped, and every fact on a card (name, rating, cost, cuisines) is filled in from the dataset, never from the LLM.
5. **Fall back.** If the LLM is down, slow, rate-limited or turned off, the pre-ranked candidates are returned with template explanations and a notice.

The Streamlit UI calls a FastAPI service, which does all of this. [docs/architecture.md](docs/architecture.md) has the design.

## Prerequisites

- **Python 3.11–3.13** (pyarrow has no wheels for newer versions yet) and Git
- **Internet access and about 1.5 GB of disk** for the first setup: the Python packages, and a one-time download of the dataset (a 574 MB CSV, which the Hugging Face cache keeps along with a converted copy of about the same size)
- **A Groq API key**, free at https://console.groq.com/keys. It's optional: without one everything runs, using the fallback ranking.

## Setup

```bash
git clone <repository-url>
cd ZomatoPoc
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then set GROQ_API_KEY in .env (Windows cmd: copy .env.example .env)
python -m app.data.ingest       # downloads the dataset, writes data/processed/
```

The ingest prints a summary (12,415 restaurants after de-duplication; 93 locations and 107 cuisines) and writes `data/processed/restaurants.parquet` and `metadata.json`. Its warning about unauthenticated requests to the Hugging Face Hub is harmless. Run every command from the project folder with the virtual environment active.

## Run the app

Start the API, then the UI, each in its own terminal (with the virtual environment active):

```bash
uvicorn app.main:app --reload            # API on http://localhost:8000
streamlit run ui/streamlit_app.py        # UI on http://localhost:8501
```

Open http://localhost:8501 (**Curated Cravings**), pick a locality and press **Find restaurants**, or try one of the quick searches. The UI finds the API through `API_BASE_URL` (default `http://localhost:8000`). The first time you run Streamlit it asks for an email address; press Enter to skip.

**Shortcuts.** `tasks.py` wraps the common commands so they're the same on every OS:

```bash
python tasks.py ingest      # download and clean the dataset
python tasks.py api         # start the API
python tasks.py ui          # start the UI
python tasks.py test        # tests, then the linter
python tasks.py demo        # run the demo scenes against the running API
python tasks.py eval ...    # quality evaluation (see below)
```

## Demo

[docs/demo.md](docs/demo.md) is a 10-minute script with four scenes: a strict match, a widened search, a free-text wish the data can't confirm, and the fallback. `python tasks.py demo` runs the scenes against the API and prints what the UI will show, plus the API's notes on any widening. It also caches the answers, so the live demo is instant. For the fallback scene, start the API with the LLM turned off:

```bash
LLM_ENABLED=false uvicorn app.main:app                  # PowerShell: $env:LLM_ENABLED="false"; uvicorn app.main:app
python tasks.py demo --scene fallback
```

## The API

Open http://localhost:8000/docs to try the endpoints in the browser, or call them directly:

```bash
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"location": "Indiranagar", "cuisines": ["Italian"], "budget": "medium", "top_n": 3}'
```

| Endpoint | Returns |
|---|---|
| `GET /health` | Status, restaurant count, and the LLM's state: `configured`, `unconfigured` (no key), `rejected` (Groq refused the key) or `disabled` (`LLM_ENABLED=false`) |
| `GET /meta/locations`, `GET /meta/cuisines` | Valid values for the request |
| `POST /recommendations` | Ranked recommendations; `422` with suggestions for an unknown location or cuisine |

## Try it from the command line

```bash
# Recommendations ranked and explained by the LLM
python -m app.services.recommender --location Indiranagar --cuisine Italian --budget medium
python -m app.services.recommender --location HSR --cuisine Cafe --prefs "family-friendly" --top-n 3

# Filtering only, no LLM call
python -m app.services.filtering --location Indiranagar --cuisine Italian --budget medium
```

Each search sends exactly one request to Groq; nothing is retried. Groq's free tier allows 8,000 tokens per minute and a full request uses about 4,000–6,000, so a second new search within a minute may come back at once with the plain ranking and a notice. The API caches identical queries, so a repeat costs nothing; the command line starts fresh each run.

## Logs and cost

The API writes JSON lines to stderr: one line per request, plus warnings and errors. Each request line holds the request id (also sent back in the `X-Request-ID` header), the preferences, the candidate count and relaxations, and the LLM call's latency, token usage, `finish_reason`, Groq request id and cost in US dollars. It also says whether the fallback ranking was used. API keys are redacted from every line. See [architecture §10](docs/architecture.md#10-security-cost-and-observability) for the full format.

To total the cost, set `LOG_FILE` and run the report on that file:

```bash
LOG_FILE=logs/api.jsonl uvicorn app.main:app      # PowerShell: $env:LOG_FILE="logs/api.jsonl"; uvicorn app.main:app
python -m app.observability logs/api.jsonl         # requests, tokens, average cost and latency
```

**Measured cost** (2026-10-28): 7 different searches on `openai/gpt-oss-120b`, effort `medium`, through the running API, at Groq's prices of $0.15 per million input tokens, $0.075 per million cached input tokens and $0.60 per million output tokens:

| Per search that reaches Groq | Average | Range |
|---|---|---|
| Prompt tokens (of which cached) | 3,931 (73) | 2,812–4,473 |
| Completion tokens (of which reasoning) | 1,498 (1,079) | 1,205–1,878 |
| Cost | **$0.0015** (about $1.48 per 1,000 searches) | $0.0012–$0.0018 |
| Latency | 3.9 s | 3.3–4.6 s |

These figures include the overall summary the LLM wrote until it was dropped later that day (roughly 60–100 completion tokens); the evaluation runs since measured **$0.0013** per search. A repeated search is answered from the cache in about 12 ms at no cost. On the free tier, the ~5,400 tokens per new search allow about one new search a minute and roughly 37 a day (200,000 tokens).

## Tests

```bash
pytest          # no network or API key needed; the LLM is faked, the UI tests start their own API
ruff check .
```

`pytest` includes unit tests for the evaluation harness, the demo runner and `tasks.py`; running the evaluation itself is described next.

## Quality evaluation

`tests/eval/` measures how good the recommendations are, not just whether the code works: 24 scenarios run through the real recommender, are checked by code (valid IDs, filters respected, numbers matching the data) and graded by an LLM judge from a different model family (`qwen/qwen3.8-27b`) for faithfulness, free-text handling and specificity. [docs/eval.md](docs/eval.md) has the design and the results log.

```bash
python -m tests.eval.run_eval --no-llm --reps 1 --label deterministic   # free: no LLM calls
python -m tests.eval.run_eval --reps 1 --label baseline                 # about 115K Groq tokens
python -m tests.eval.judge tests/eval/results/<run_id>                  # J1–J4, on the judge model
python -m tests.eval.compare tests/eval/results/<run_a> tests/eval/results/<run_b>
```

It costs tokens, so it runs on demand, never in CI. On Groq's free tier a full run uses more than half a day's 200,000 tokens, and **the app shares that allowance**: while the eval runs, or after it has spent the allowance, the app falls back to the plain ranking. A run stops cleanly when the allowance runs out; `--resume <run folder>` continues it.

## Known limitations

- **Bangalore only.** The dataset covers 93 Bangalore neighbourhoods; other cities, and searches by distance or street, aren't possible. Searching "Bangalore" covers the whole city.
- **A static snapshot.** The data is a one-time scrape of Zomato. Ratings, prices ("approximate cost for two" at the time) and whether a place is still open may have changed, and nothing updates it.
- **Free-text wishes are inferred from limited data.** The LLM sees each restaurant's cuisines, type, popular dishes, table booking, online ordering and at most three short review excerpts. There is nothing on amenities, opening hours, vegetarian menus or accessibility. It is told to say when the data doesn't confirm a wish, but it can still read too much into a review: the evaluation caught it calling cafés "laptop-friendly" because reviews called them calm.
- **The free Groq tier is tight.** About one new AI search a minute and 37 a day. Beyond that, searches fall back to the plain ranking until the allowance refills.
- **Answers vary.** The same search can return different picks or wording from one run to the next; identical searches are cached for a day per API process.
- **Quality is only partly measured.** The evaluation harness is complete, but the full baseline and tuning runs are pending (Phase 7).
- **POC scope.** No accounts or history, explanations in English only, and data held in memory in a single process.

## Project structure

```
app/
  main.py          FastAPI app (uvicorn app.main:app)
  api/             routes, API response models, request-id/logging and size-limit middleware
  config.py        settings from environment variables and .env
  observability.py JSON logging, per-request log line, cost, cost report
  data/            ingestion, preprocessing, in-memory repository
  models/          Restaurant, UserPreferences, response schemas
  services/        filtering and pre-ranking, recommender, template explanations
  llm/             Groq client, response cache, prompt builder, system prompt
ui/                Streamlit front end and its stylesheet (theme in .streamlit/config.toml)
scripts/demo.py    the demo scenes, run against the API
tasks.py           shortcuts: ingest, api, ui, test, demo, eval
tests/             unit, API and UI tests, fixtures
  eval/            quality evaluation: 24 scenarios, runner, programmatic checks, LLM judge
docs/              problem statement, architecture, implementation plan, edge cases, evaluation, demo
```

See [docs/architecture.md](docs/architecture.md) for the design and [docs/edge-case.md](docs/edge-case.md) for the decisions behind each rule.

## Contributing

1. Create a branch from `main`: `git checkout -b feature/<short-description>`
2. Commit your changes with clear messages.
3. Open a pull request against `main`.

## License

No license has been specified yet. All rights reserved by the author until one is added.
