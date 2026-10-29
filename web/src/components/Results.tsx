import { count, describeRequest } from "../format";
import type { ErrorDetail, Payload, RecommendationResponse, SearchResult } from "../types";
import Notice from "./Notice";
import ResultCard from "./ResultCard";

export const UNAVAILABLE_TEXT =
  "The recommendation service isn't responding. Please try again in a minute.";

export type SuggestionField = "location" | "cuisines";

export interface ResultsProps {
  result: SearchResult;
  payload: Payload;
  onSuggestion: (field: SuggestionField, wrong: string, suggestion: string) => void;
}

/** What the last search returned. */
export default function Results({ result, payload, onSuggestion }: ResultsProps) {
  switch (result.kind) {
    case "ok":
      return <Shortlist payload={payload} data={result.response} />;
    case "invalid":
      return <Invalid errors={result.errors} onSuggestion={onSuggestion} />;
    case "unavailable": // EC-UI-02
      return <Notice kind="error" title="We can't reach the kitchen" lines={[UNAVAILABLE_TEXT]} />;
    case "error":
      return <Notice kind="error" title="Something went wrong" lines={[result.message]} />;
  }
}

function Shortlist({ payload, data }: { payload: Payload; data: RecommendationResponse }) {
  const items = data.recommendations;
  const { meta } = data;
  let asked = describeRequest(payload);
  if (items.length > 0 && items.length < payload.top_n) {
    // EC-UI-08
    asked = `Showing ${items.length} of ${payload.top_n}: only ${items.length} matched · ${asked}`;
  }
  const source = meta.llm_status === "ok" ? `Ranked by ${meta.model}` : "Ranked without AI";
  const footer =
    `${source} · ${count(meta.candidates_considered, "candidate")} considered from ` +
    `${count(meta.total_matched, "match", "matches")} · ${(meta.latency_ms / 1000).toFixed(1)} s`;
  return (
    <>
      <div className="cc-section">
        <h2>Your shortlist</h2>
        <span className="cc-muted">{asked}</span>
      </div>
      {items.length === 0 ? (
        // EC-UI-07
        <Notice kind="warn" title="No restaurants found" lines={[data.notice ?? "Try fewer filters."]} />
      ) : (
        <>
          {/* Only the results: no fallback or relaxation banners (EC-UI-11). The cards and
              the footer still say when the ranking came without AI. */}
          {items.map((item) => (
            <ResultCard key={item.restaurant_id} item={item} />
          ))}
          <div className="cc-muted cc-results-foot">{footer}</div>
        </>
      )}
    </>
  );
}

function Invalid({
  errors,
  onSuggestion,
}: {
  errors: ErrorDetail[];
  onSuggestion: ResultsProps["onSuggestion"];
}) {
  return (
    <>
      {errors.map((error, i) => {
        const field = error.loc.at(-1);
        const suggestions = error.ctx?.suggestions ?? [];
        return (
          <div key={i}>
            <Notice
              kind="error"
              title="Please check your search"
              lines={[error.msg || "Invalid request."]}
            />
            {/* EC-UI-06: a suggestion goes into the form; the next search uses it. */}
            {(field === "location" || field === "cuisines") && suggestions.length > 0 && (
              <div className="cc-suggestions">
                {suggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    className="cc-button"
                    onClick={() => onSuggestion(field, String(error.input), suggestion)}
                  >
                    Use {suggestion}
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </>
  );
}
