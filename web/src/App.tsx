// Curated Cravings: the web front end for the recommendation API, a port of
// ui/streamlit_app.py that runs as a static site on Vercel (docs/deployment-plan.md,
// Phase 3). It only collects preferences and renders the API's response. Case IDs refer
// to docs/edge-case.md §9.

import { useCallback, useEffect, useRef, useState } from "react";

import {
  WAKE,
  loadOptions,
  requestRecommendations,
  wakeUp,
  type Options,
  type WakeOptions,
} from "./api";
import Header from "./components/Header";
import Notice, { Spinner } from "./components/Notice";
import Results, { UNAVAILABLE_TEXT, type SuggestionField } from "./components/Results";
import SearchForm from "./components/SearchForm";
import {
  DEFAULTS,
  QUICK_SEARCHES,
  availableQuickSearches,
  buildPayload,
  count,
  llmOffReason,
  type FormState,
} from "./format";
import type { Health, Payload, SearchResult } from "./types";

// Render's free instance sleeps after 15 minutes without traffic. A tab quiet for this
// long checks /health (and waits for a wake-up) before it searches.
const LIKELY_ASLEEP_MS = 14 * 60_000;
const WAKING_TEXT =
  "The free server sleeps when nobody has used it for a while, so this can take a minute or two.";

type Boot =
  | { phase: "loading"; waking: boolean }
  | { phase: "ready"; options: Options; health: Health }
  | { phase: "unreachable" };

interface Shown {
  result: SearchResult;
  payload: Payload;
}

export default function App({ wake = WAKE }: { wake?: WakeOptions }) {
  const [boot, setBoot] = useState<Boot>({ phase: "loading", waking: false });
  const [form, setForm] = useState<FormState>(DEFAULTS);
  const [shown, setShown] = useState<Shown | null>(null);
  const [busy, setBusy] = useState<false | "waking" | "searching">(false);
  const [needsLocation, setNeedsLocation] = useState(false);
  const lastContact = useRef(0);

  const connect = useCallback(
    async (isCurrent: () => boolean) => {
      setBoot({ phase: "loading", waking: false });
      try {
        const health = await wakeUp(wake, () => {
          if (isCurrent()) setBoot({ phase: "loading", waking: true });
        });
        const options = await loadOptions();
        lastContact.current = Date.now();
        if (isCurrent()) setBoot({ phase: "ready", options, health });
      } catch {
        if (isCurrent()) setBoot({ phase: "unreachable" }); // EC-UI-01
      }
    },
    [wake],
  );

  useEffect(() => {
    let current = true;
    void connect(() => current);
    return () => {
      current = false;
    };
  }, [connect]);

  async function search(values: FormState) {
    if (values.location === null) {
      setNeedsLocation(true);
      return;
    }
    setNeedsLocation(false);
    const payload = buildPayload({ ...values, location: values.location });
    // A new search replaces the old result, so nothing stale stays on screen (EC-UI-02).
    setShown(null);
    if (Date.now() - lastContact.current > LIKELY_ASLEEP_MS) {
      setBusy("waking");
      try {
        await wakeUp(wake);
      } catch {
        setShown({ result: { kind: "unavailable" }, payload });
        setBusy(false);
        return;
      }
    }
    setBusy("searching");
    const result = await requestRecommendations(payload);
    if (result.kind !== "unavailable") lastContact.current = Date.now();
    setShown({ result, payload });
    setBusy(false);
  }

  function quickSearch(label: string) {
    const values = { ...DEFAULTS, ...QUICK_SEARCHES[label] };
    setForm(values);
    void search(values);
  }

  // EC-UI-06: put a suggested value from a 422 into the form.
  function applySuggestion(field: SuggestionField, wrong: string, suggestion: string) {
    setForm((f) =>
      field === "location"
        ? { ...f, location: suggestion }
        : { ...f, cuisines: f.cuisines.map((c) => (c === wrong ? suggestion : c)) },
    );
  }

  if (boot.phase !== "ready") {
    return (
      <main className="cc-page">
        <Header localities={null} />
        {boot.phase === "unreachable" ? (
          <>
            <Notice kind="error" title="We can't reach the kitchen" lines={[UNAVAILABLE_TEXT]} />
            <button type="button" className="cc-button" onClick={() => void connect(() => true)}>
              Retry
            </button>
          </>
        ) : boot.waking ? (
          <Notice kind="warn" title="Waking up the kitchen…" lines={[WAKING_TEXT]} />
        ) : (
          <Spinner text="Setting the table…" />
        )}
      </main>
    );
  }

  const { options, health } = boot;
  const offReason = llmOffReason(health.llm);
  return (
    <main className="cc-page">
      <Header localities={options.locations.length} />
      {offReason && (
        <Notice
          kind="warn"
          title="AI ranking is off"
          lines={[`${offReason}, so results are ranked by rating and popularity.`]}
        />
      )}
      <SearchForm
        options={options}
        value={form}
        onChange={setForm}
        onSubmit={() => void search(form)}
        busy={busy !== false}
        status={<StatusLine shown={shown} />}
      />
      {needsLocation && (
        <Notice
          kind="warn"
          title="Choose a neighbourhood first"
          lines={["Pick a locality, or All of Bangalore."]}
        />
      )}
      {busy ? (
        <Spinner text={busy === "waking" ? "Waking up the kitchen…" : "Curating your shortlist…"} />
      ) : shown ? (
        <Results result={shown.result} payload={shown.payload} onSuggestion={applySuggestion} />
      ) : (
        <EmptyState
          quick={availableQuickSearches(options.locations, options.cuisines)}
          onPick={quickSearch}
        />
      )}
    </main>
  );
}

/** The line beside the search button: how the last search was ranked. */
function StatusLine({ shown }: { shown: Shown | null }) {
  if (shown?.result.kind !== "ok") {
    return <>⚡ Your must-haves filter first, then AI ranks the shortlist</>;
  }
  const { meta, recommendations } = shown.result.response;
  const ranked =
    meta.llm_status === "ok" ? `AI ranked top ${recommendations.length}` : "ranked by rating";
  return (
    <>
      ⚡ Screened <b>{count(meta.candidates_considered, "candidate")}</b> · {ranked}
    </>
  );
}

function EmptyState({ quick, onPick }: { quick: string[]; onPick: (label: string) => void }) {
  return (
    <>
      <section className="cc-empty">
        <div className="cc-empty-title">Where are we eating today?</div>
        <p>Pick a neighbourhood above, or start from a popular search.</p>
      </section>
      {quick.length > 0 && (
        <div className="cc-quick" role="group" aria-label="Popular searches">
          {quick.map((label) => (
            <button key={label} type="button" className="cc-quick-pill" onClick={() => onPick(label)}>
              {label}
            </button>
          ))}
        </div>
      )}
    </>
  );
}
