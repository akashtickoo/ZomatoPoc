import type { ReactNode } from "react";
import Select from "react-select";

import type { Options } from "../api";
import {
  ALL_OF_BANGALORE,
  BUDGETS,
  MAX_CUISINES,
  MAX_HINT_CHARS,
  RESULT_COUNTS,
  budgetLabel,
  type FormState,
} from "../format";

interface Choice {
  value: string;
  label: string;
}

const choice = (value: string): Choice => ({ value, label: value });

// react-select draws only its layout; every colour comes from the cc-select__* rules in
// styles.css.
const SELECT_PROPS = { unstyled: true, classNamePrefix: "cc-select" } as const;

export interface SearchFormProps {
  options: Options;
  value: FormState;
  onChange: (value: FormState) => void;
  onSubmit: () => void;
  busy: boolean;
  status: ReactNode;
}

/** The filter card: every value the API takes, with options read from /meta/*. */
export default function SearchForm({
  options,
  value,
  onChange,
  onSubmit,
  busy,
  status,
}: SearchFormProps) {
  const set = <K extends keyof FormState>(key: K, v: FormState[K]) =>
    onChange({ ...value, [key]: v });

  return (
    <section className="cc-search-card">
      <form
        aria-label="Search restaurants"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit();
        }}
      >
        <div className="cc-grid cc-grid-first">
          <div className="cc-field">
            <label htmlFor="location">Locality</label>
            <Select<Choice>
              {...SELECT_PROPS}
              inputId="location"
              options={[ALL_OF_BANGALORE, ...options.locations].map(choice)}
              value={value.location === null ? null : choice(value.location)}
              onChange={(picked) => set("location", picked?.value ?? null)}
              placeholder="Search neighbourhoods"
            />
          </div>
          <div className="cc-field">
            <label htmlFor="cuisines">Cuisines</label>
            <Select<Choice, true>
              {...SELECT_PROPS}
              isMulti
              inputId="cuisines"
              options={options.cuisines.map(choice)}
              value={value.cuisines.map(choice)}
              onChange={(picked) => set("cuisines", picked.map((c) => c.value))}
              isOptionDisabled={() => value.cuisines.length >= MAX_CUISINES}
              placeholder="e.g. Italian, Chinese"
            />
          </div>
          <div className="cc-field">
            <label htmlFor="min_rating">
              Minimum rating <output htmlFor="min_rating">{value.minRating.toFixed(1)} ★</output>
            </label>
            <input
              id="min_rating"
              type="range"
              min={0}
              max={5}
              step={0.5}
              value={value.minRating}
              onChange={(event) => set("minRating", Number(event.target.value))}
            />
          </div>
        </div>

        <div className="cc-grid cc-grid-second">
          <fieldset className="cc-field">
            <legend>Budget for two</legend>
            <div className="cc-segments">
              {BUDGETS.map((budget) => (
                <label key={budget} className="cc-segment">
                  <input
                    type="radio"
                    name="budget"
                    value={budget}
                    checked={value.budget === budget}
                    onChange={() => set("budget", budget)}
                  />
                  <span>{budgetLabel(budget, options.budgets)}</span>
                </label>
              ))}
            </div>
          </fieldset>
          <div className="cc-field">
            <label htmlFor="hints">Anything else?</label>
            <input
              id="hints"
              className="cc-input"
              type="text"
              maxLength={MAX_HINT_CHARS}
              placeholder="quiet seating, quick service, good for groups"
              value={value.hints}
              onChange={(event) => set("hints", event.target.value)}
            />
          </div>
          <div className="cc-field">
            <label htmlFor="top_n">Show</label>
            <select
              id="top_n"
              className="cc-input"
              value={value.topN}
              onChange={(event) => set("topN", Number(event.target.value))}
            >
              {RESULT_COUNTS.map((n) => (
                <option key={n} value={n}>
                  Top {n}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="cc-form-foot">
          <div className="cc-foot">{status}</div>
          {/* Disabled while a search runs, so a double click is one request (EC-UI-04). */}
          <button type="submit" className="cc-button cc-button-primary" disabled={busy}>
            Find restaurants →
          </button>
        </div>
      </form>
    </section>
  );
}
