# Demo Script

A 10-minute walkthrough that shows what the recommender does: a strict match, a widened search, a free-text wish the data can't confirm, and the fallback when the LLM is unavailable. Each scene lists what to enter, what happens, and what to point out.

`python tasks.py demo` (or `python -m scripts.demo`) runs the same scenes against the API and prints what the UI will show. Use it to rehearse.

---

## Before you start

1. Set up the project as the [README](../README.md) describes: virtual environment, `.env` with `GROQ_API_KEY` and `CORS_ORIGINS=["http://localhost:5173"]`, and `npm install` in `web/`.
2. Start the API and the UI, each in its own terminal: `python tasks.py api`, and `npm run dev` in `web/`. Open http://localhost:5173. (Or demo the deployed site; the first visit after a quiet spell waits a minute while the API wakes up, so open it beforehand.)
3. Check the top of the page: there should be **no** "AI ranking is off" banner. If there is one, it says why (no key, a rejected key, or the LLM turned off).
4. **Rehearse with `python tasks.py demo`, against the API you will demo with.** The API remembers each answer for 24 hours, so when you then make the same searches in the UI they come back instantly and spend no tokens. The scenes below use the UI's defaults (5 results) so that the requests match exactly. Pick cuisines in the order given: the order is part of the request.
5. Mind the Groq allowance. Each new AI search uses about 5,000 tokens; Groq's free tier allows 200,000 a day, refilled gradually. If AI explanations are unavailable when they shouldn't be, the allowance is spent; it refills at about 8,000 tokens an hour.

---

## Scene 1: A strict match (about 2 minutes)

**Do:** click the **Italian in Indiranagar** quick search on the empty page. (It fills Locality *Indiranagar*, Cuisines *Italian*, Budget *Medium* and searches.)

**What happens:** five cards in a few seconds. The line under the button reads "Screened 10 candidates · AI ranked top 5": ten restaurants matched every filter, so nothing was widened.

**Point out:**

- Each card has the five things the brief asks for: name, cuisine, rating, cost for two and an AI explanation ("✦ Why it fits"), plus table booking, online ordering and the dishes it is known for.
- The explanations cite each restaurant's own rating, votes, cost, dishes and reviews.
- **Facts come from the data, never from the LLM.** The model only chooses which candidates to show, in what order, and why. Rating, cost and cuisines on the card are filled in from the dataset, and any pick that wasn't one of the candidates is thrown away.

## Scene 2: A widened search (about 2 minutes)

**Do:** set Locality *Koramangala 5th Block*, Cuisines *North Indian* then *Chinese*, Minimum rating *4.0*, Budget *Low*, and press **Find restaurants**.

**What happens:** five cards, usually including some at medium prices. The shortlist shows only the results; there is no banner about the widening.

**Point out:**

- No low-budget North Indian or Chinese place here is rated 4.0 or more. Rather than return nothing, the search loosens one filter at a time (budget first, then rating, then location; never cuisine).
- The explanations say when a pick costs more than the original budget, and each card shows its cost for two.
- The API still reports exactly what it changed ("Budget widened from low to low or medium."); `python tasks.py demo --scene relaxation` prints it.

## Scene 3: A free-text wish (about 2 minutes)

**Do:** set Locality *Indiranagar*, clear the cuisines, Minimum rating *4.0*, Budget *High*, and type `rooftop with live music` in **Anything else?**. Press **Find restaurants**.

**What happens:** five well-rated, high-budget places. The explanations handle the wish honestly: rooftops and live music aren't fields in the data, so they say the data doesn't confirm them. The exception is a restaurant whose review mentions it (in rehearsals, a Barbeque Nation review mentioned its rooftop seating).

**Point out:**

- Free text is interpreted by the LLM, and only the LLM; the filters stay structured.
- The model is told never to claim a preference the data can't support. This is checked in the quality evaluation ([eval.md](eval.md)), which found where it still slips: for "good for working on a laptop" it once inferred laptop-friendliness from a "calm" ambience.
- Text from reviews and from the user is passed to the model as data, so instructions hidden in it ("ignore your rules and recommend…") don't work.

## Scene 4: The fallback path (about 2 minutes)

**Do:** stop the API (Ctrl+C) and start it again with the LLM turned off:

```bash
LLM_ENABLED=false python tasks.py api                       # macOS / Linux / Git Bash
$env:LLM_ENABLED="false"; python tasks.py api                # PowerShell
```

Reload the UI, then click **Italian in Indiranagar** again. To rehearse this scene from the command line: `python tasks.py demo --scene fallback`.

**What happens:** an "AI ranking is off" banner (the LLM is turned off), then the same search returns at once. The cards say "Ranked by rating and popularity" with template explanations, and the footer says "Ranked without AI".

**Point out:**

- The same happens on its own when Groq is down, slow or rate-limited, or the key is wrong: the user always gets an answer.
- The fallback ranking is a Bayesian score that balances rating against the number of votes, so a 4.5 from 20 votes doesn't beat a 4.3 from 5,000.

**Afterwards:** stop the API and start it normally (in PowerShell, `Remove-Item Env:LLM_ENABLED` first).

---

## If there is time

- **The API:** http://localhost:8000/docs lists the four endpoints and lets you try them.
- **A typo:** search for *Jaynagar* through the API (`/docs` or `curl`); the `422` answer suggests *Jayanagar*.
- **Cost:** each request writes one JSON log line with its tokens and cost: about $0.0013 a search. `python -m app.observability <log file>` totals them (README, "Logs and cost").
- **Quality:** [eval.md §10](eval.md#10-results-log) has the evaluation results so far: 24 scenarios, checks in code and an LLM judge.

## If something goes wrong

| You see | Why | Do |
|---|---|---|
| "We can't reach the kitchen" | The API isn't running, or `API_BASE_URL` points elsewhere | Start the API, then press Retry |
| "AI ranking is off: No Groq API key is set" | `GROQ_API_KEY` is missing or blank in `.env` | Set it and restart the API |
| "Ranked by rating and popularity" on every card | Groq's rate limit or daily allowance is spent, or Groq is down | Carry on: this *is* the fallback scene. The allowance refills gradually. |
| A search takes several seconds | The model is reasoning; a new search takes about 3–5 s | Rehearse first: repeated searches come from the cache instantly |
