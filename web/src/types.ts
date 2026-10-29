// The API's request and response shapes (app/models, app/api/schemas.py); /docs on the API
// shows the same schemas.

export type Budget = "low" | "medium" | "high";

// "unconfigured": no GROQ_API_KEY; "rejected": Groq refused the key; "disabled":
// LLM_ENABLED=false. In every case but "configured", searches get the fallback ranking.
export type LlmState = "configured" | "unconfigured" | "rejected" | "disabled";

export interface Health {
  status: "ok";
  data_loaded: boolean;
  restaurants: number;
  llm: LlmState;
}

// Rupees for two: low is up to low_max, medium up to medium_max, high above it.
export interface BudgetBands {
  low_max: number;
  medium_max: number;
}

export interface Payload {
  location: string;
  budget: Budget | null;
  cuisines: string[];
  min_rating: number;
  additional_preferences: string | null;
  top_n: number;
}

export interface Recommendation {
  rank: number;
  restaurant_id: string;
  name: string;
  location: string;
  cuisines: string[];
  rating: number | null;
  votes: number;
  cost_for_two: number | null;
  rest_type: string[];
  online_order: boolean;
  book_table: boolean;
  popular_dishes: string[];
  explanation: string;
  source: "llm" | "fallback";
}

export interface RecommendationResponse {
  recommendations: Recommendation[];
  relaxations: string[];
  notice: string | null;
  meta: {
    llm_status: "ok" | "fallback" | "not_called";
    model: string | null;
    candidates_considered: number;
    total_matched: number;
    latency_ms: number;
  };
}

// One problem with a request, in FastAPI's 422 shape. Unknown locations and cuisines carry
// up to 3 close matches in ctx.suggestions (EC-API-06).
export interface ErrorDetail {
  type: string;
  loc: (string | number)[];
  msg: string;
  input?: unknown;
  ctx?: { suggestions?: string[] } | null;
}

// Every outcome of a search, so components never handle exceptions.
export type SearchResult =
  | { kind: "ok"; response: RecommendationResponse }
  | { kind: "invalid"; errors: ErrorDetail[] }
  | { kind: "unavailable" }
  | { kind: "error"; message: string };
