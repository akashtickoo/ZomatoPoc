// A stand-in for the API, installed as the global fetch, so tests run without a server.
// Its answers follow app/api/routes.py.

import { vi } from "vitest";

import type { LlmState, Payload, Recommendation, RecommendationResponse } from "../types";

export const LOCATIONS = ["BTM", "HSR", "Indiranagar", "Jayanagar"];
export const CUISINES = ["Biryani", "Cafe", "Italian", "South Indian", "Tibetan"];
export const HOSTILE = "[click me](http://evil.example) <img src=x onerror=alert(1)> **bold** $5";

/** A recommendation as the API returns it. */
export function item(overrides: Partial<Recommendation> = {}): Recommendation {
  return {
    rank: 1,
    restaurant_id: "abc123def456",
    name: "Glen's Bakehouse",
    location: "Indiranagar",
    cuisines: ["Bakery", "Cafe", "Italian"],
    rating: 4.3,
    votes: 5352,
    cost_for_two: 600,
    rest_type: ["Bakery", "Cafe"],
    online_order: true,
    book_table: false,
    popular_dishes: ["Pizza", "Pasta"],
    explanation: "Rated 4.3 with 5,352 votes.",
    source: "llm",
    ...overrides,
  };
}

/** A successful response with `n` LLM-ranked items. */
export function ranked(n: number, overrides: Partial<Recommendation> = {}): RecommendationResponse {
  return {
    recommendations: Array.from({ length: n }, (_, i) =>
      item({ rank: i + 1, restaurant_id: `id${i}`, name: `Place ${i + 1}`, ...overrides }),
    ),
    relaxations: [],
    notice: null,
    meta: {
      llm_status: "ok",
      model: "fake-model",
      candidates_considered: 10,
      total_matched: 12,
      latency_ms: 4300,
    },
  };
}

export const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

export interface FakeApiOptions {
  llm?: LlmState;
  /** Answers POST /recommendations; defaults to 3 ranked items. */
  recommend?: (payload: Payload) => Response | Promise<Response>;
}

export function fakeApi({ llm = "configured", recommend }: FakeApiOptions = {}) {
  const api = {
    down: false, // true: every request fails as if the server were unreachable
    payloads: [] as Payload[],
    healthChecks: 0,
  };
  const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
    if (api.down) throw new TypeError("Failed to fetch");
    const path = new URL(String(url)).pathname;
    if (path === "/health") {
      api.healthChecks += 1;
      return json({ status: "ok", data_loaded: true, restaurants: 50, llm });
    }
    if (path === "/meta/locations") return json({ locations: LOCATIONS });
    if (path === "/meta/cuisines") return json({ cuisines: CUISINES });
    if (path === "/meta/budgets") return json({ low_max: 300, medium_max: 600 });
    if (path === "/recommendations" && init?.method === "POST") {
      const payload = JSON.parse(String(init.body)) as Payload;
      api.payloads.push(payload);
      return recommend ? recommend(payload) : json(ranked(3));
    }
    return json({ detail: "Not Found" }, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return api;
}
