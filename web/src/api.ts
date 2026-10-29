// Calls to the recommendation API (app/api/routes.py). Every outcome of a search becomes one
// of four result kinds (types.ts), so components never handle exceptions.

import type { BudgetBands, Health, Payload, SearchResult } from "./types";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000").replace(
  /\/+$/,
  "",
);

const META_TIMEOUT_MS = 10_000;
// The API makes one LLM call of at most LLM_TIMEOUT_SECONDS (60 s) per search; wait a
// little longer than that before giving up (EC-UI-03).
export const SEARCH_TIMEOUT_MS = 90_000;

export interface WakeOptions {
  retryEveryMs: number;
  giveUpAfterMs: number;
}

// A free Render instance sleeps after 15 idle minutes and takes a minute or more to wake
// (docs/deployment-plan.md, Phase 3). Meanwhile it may answer with its own HTML loading
// page, without CORS headers, so any failure counts as "still waking".
export const WAKE: WakeOptions = { retryEveryMs: 5_000, giveUpAfterMs: 180_000 };

export class ApiUnavailable extends Error {}

export interface Options {
  locations: string[];
  cuisines: string[];
  budgets: BudgetBands;
}

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs: number) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function getJson<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetchWithTimeout(`${API_BASE_URL}${path}`, {}, META_TIMEOUT_MS);
  } catch (error) {
    throw new ApiUnavailable(`${path}: ${error}`);
  }
  if (!response.ok) throw new ApiUnavailable(`${path} answered ${response.status}`);
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiUnavailable(`${path} did not answer with JSON`);
  }
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Poll /health until the API answers; `onWaiting` runs after each failed try. */
export async function wakeUp(options: WakeOptions = WAKE, onWaiting?: () => void) {
  const started = Date.now();
  for (;;) {
    try {
      return await getJson<Health>("/health");
    } catch (error) {
      if (Date.now() - started + options.retryEveryMs > options.giveUpAfterMs) throw error;
      onWaiting?.();
      await sleep(options.retryEveryMs);
    }
  }
}

/** The values the form offers, read from the data so the API accepts them. */
export async function loadOptions(): Promise<Options> {
  const [locations, cuisines, budgets] = await Promise.all([
    getJson<{ locations: string[] }>("/meta/locations"),
    getJson<{ cuisines: string[] }>("/meta/cuisines"),
    getJson<BudgetBands>("/meta/budgets"),
  ]);
  return { locations: locations.locations, cuisines: cuisines.cuisines, budgets };
}

/** Call the API once and describe the outcome. */
export async function requestRecommendations(
  payload: Payload,
  timeoutMs = SEARCH_TIMEOUT_MS,
): Promise<SearchResult> {
  let response: Response;
  try {
    response = await fetchWithTimeout(
      `${API_BASE_URL}/recommendations`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
      timeoutMs,
    );
  } catch (error) {
    if ((error as Error)?.name === "AbortError") {
      return { kind: "error", message: "The service took too long to answer. Please try again." };
    }
    return { kind: "unavailable" }; // EC-UI-02
  }
  // Render's own error pages (a 502 while it restarts, say) aren't JSON.
  const body = await response.json().catch(() => null);
  if (response.status === 200 && body) return { kind: "ok", response: body };
  if (response.status === 422) {
    return { kind: "invalid", errors: Array.isArray(body?.detail) ? body.detail : [] };
  }
  const requestId = body?.request_id ?? response.headers.get("X-Request-ID");
  let message = "The service hit an unexpected error. Please try again.";
  if (requestId) message += ` (Reference: ${requestId})`;
  return { kind: "error", message };
}
