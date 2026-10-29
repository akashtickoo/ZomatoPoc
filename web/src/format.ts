// Form constants and pure helpers, ported from ui/streamlit_app.py. format.test.ts checks
// them against the same cases as tests/ui/test_streamlit_app.py. Case IDs refer to
// docs/edge-case.md §9.

import type { BudgetBands, LlmState, Payload } from "./types";

export const ALL_OF_BANGALORE = "All of Bangalore"; // sent as "Bangalore": no location filter (D-04)
export const BUDGETS = ["any", "low", "medium", "high"] as const;
export type BudgetChoice = (typeof BUDGETS)[number];
export const RESULT_COUNTS = [3, 5, 8, 10];
export const MAX_CHIPS = 5; // EC-UI-09
export const MAX_CUISINES = 10; // the API's limit
export const MAX_HINT_CHARS = 300; // EC-UI-21; the API enforces it too

export interface FormState {
  location: string | null;
  cuisines: string[];
  budget: BudgetChoice;
  minRating: number;
  hints: string;
  topN: number;
}

export const DEFAULTS: FormState = {
  location: null,
  cuisines: [],
  budget: "any",
  minRating: 0,
  hints: "",
  topN: 5,
};

export const QUICK_SEARCHES: Record<string, Partial<FormState> & { location: string }> = {
  "Italian in Indiranagar": { location: "Indiranagar", cuisines: ["Italian"], budget: "medium" },
  "Biryani across the city": { location: ALL_OF_BANGALORE, cuisines: ["Biryani"], minRating: 4 },
  "Cosy cafés in HSR": { location: "HSR", cuisines: ["Cafe"], hints: "quiet, good for working" },
  "South Indian in Jayanagar": {
    location: "Jayanagar",
    cuisines: ["South Indian"],
    budget: "low",
  },
};

// Grouped like Python's f"{n:,}" in the Streamlit UI: 5,352.
export const formatNumber = (n: number): string => n.toLocaleString("en-US");

export function count(n: number, singular: string, plural?: string): string {
  return `${n} ${n === 1 ? singular : (plural ?? `${singular}s`)}`;
}

export function budgetLabel(key: BudgetChoice, bands: BudgetBands): string {
  const { low_max: low, medium_max: medium } = bands;
  return {
    any: "Any",
    low: `Low · ≤ ₹${low}`,
    medium: `Medium · ₹${low + 1}–${medium}`,
    high: `High · ₹${medium + 1}+`,
  }[key];
}

export function formatRating(rating: number | null, votes: number): string {
  return rating === null ? "New" : `${rating.toFixed(1)} ★ (${formatNumber(votes)})`; // EC-UI-10
}

export function formatCost(cost: number | null): string {
  return cost === null ? "Cost N/A" : `₹${formatNumber(cost)} for two`; // EC-UI-10
}

export function buildPayload(form: FormState & { location: string }): Payload {
  return {
    location: form.location === ALL_OF_BANGALORE ? "Bangalore" : form.location,
    budget: form.budget === "any" ? null : form.budget,
    cuisines: [...form.cuisines],
    min_rating: form.minRating,
    additional_preferences: form.hints.trim() || null,
    top_n: form.topN,
  };
}

/** One line restating what was asked, shown above the results. */
export function describeRequest(payload: Payload): string {
  const parts = [payload.cuisines.join(", ") || "Any cuisine"];
  parts.push(
    payload.location === "Bangalore" ? "anywhere in Bangalore" : `in ${payload.location}`,
  );
  if (payload.budget) parts.push(`${payload.budget} budget`);
  if (payload.min_rating) parts.push(`rated ${payload.min_rating}★ or more`);
  if (payload.additional_preferences) parts.push(`“${payload.additional_preferences}”`);
  return parts.join(" · ");
}

/** Quick searches whose values all exist in the current data, so the API accepts them. */
export function availableQuickSearches(locations: string[], cuisines: string[]): string[] {
  const places = new Set([ALL_OF_BANGALORE, ...locations]);
  const known = new Set(cuisines);
  return Object.entries(QUICK_SEARCHES)
    .filter(([, v]) => places.has(v.location) && (v.cuisines ?? []).every((c) => known.has(c)))
    .map(([label]) => label);
}

/** Why AI ranking is off, or null when it's on. */
export function llmOffReason(llm: LlmState): string | null {
  if (llm === "configured") return null;
  if (llm === "rejected") return "Groq rejected the API key (check GROQ_API_KEY)";
  if (llm === "disabled") return "The LLM is turned off (LLM_ENABLED=false)";
  return "No Groq API key is set";
}
