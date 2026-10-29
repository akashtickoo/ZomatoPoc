import { describe, expect, it } from "vitest";

import {
  ALL_OF_BANGALORE,
  DEFAULTS,
  QUICK_SEARCHES,
  availableQuickSearches,
  budgetLabel,
  buildPayload,
  count,
  describeRequest,
  formatCost,
  formatRating,
  llmOffReason,
} from "./format";

const form = (overrides: Partial<typeof DEFAULTS> & { location: string }) => ({
  ...DEFAULTS,
  ...overrides,
});

describe("formatting", () => {
  it.each([
    [null, 0, "New"],
    [4.3, 5352, "4.3 ★ (5,352)"],
    [3.0, 7, "3.0 ★ (7)"],
  ])("formats rating %s with %s votes (EC-UI-10)", (rating, votes, expected) => {
    expect(formatRating(rating, votes)).toBe(expected);
  });

  it("formats the cost (EC-UI-10)", () => {
    expect(formatCost(null)).toBe("Cost N/A");
    expect(formatCost(1200)).toBe("₹1,200 for two");
  });

  it("labels the budgets with the API's thresholds", () => {
    const bands = { low_max: 300, medium_max: 600 };
    expect(budgetLabel("any", bands)).toBe("Any");
    expect(budgetLabel("low", bands)).toBe("Low · ≤ ₹300");
    expect(budgetLabel("medium", bands)).toBe("Medium · ₹301–600");
    expect(budgetLabel("high", bands)).toBe("High · ₹601+");
    expect(budgetLabel("medium", { low_max: 250, medium_max: 800 })).toBe("Medium · ₹251–800");
  });

  it.each([
    [0, "0 matches"],
    [1, "1 match"],
    [12, "12 matches"],
  ])("counts %s", (n, expected) => {
    expect(count(n, "match", "matches")).toBe(expected);
    expect(count(n, "candidate")).toBe(`${n} ${n === 1 ? "candidate" : "candidates"}`);
  });
});

describe("requests", () => {
  it("builds the payload", () => {
    expect(buildPayload(form({ location: ALL_OF_BANGALORE, hints: "   " }))).toEqual({
      location: "Bangalore", // the whole city (D-04)
      budget: null,
      cuisines: [],
      min_rating: 0,
      additional_preferences: null,
      top_n: 5,
    });
    const payload = buildPayload(
      form({ location: "BTM", budget: "medium", cuisines: ["Chinese"], minRating: 4, hints: " rooftop ", topN: 3 }),
    );
    expect([payload.budget, payload.additional_preferences, payload.top_n]).toEqual([
      "medium",
      "rooftop",
      3,
    ]);
  });

  it("describes the request", () => {
    const low = form({ location: "BTM", budget: "low", cuisines: ["Chinese", "Thai"], minRating: 4 });
    expect(describeRequest(buildPayload(low))).toBe(
      "Chinese, Thai · in BTM · low budget · rated 4★ or more",
    );
    expect(describeRequest(buildPayload(form({ location: ALL_OF_BANGALORE })))).toBe(
      "Any cuisine · anywhere in Bangalore",
    );
    const withText = form({ location: "HSR", cuisines: ["Cafe"], hints: "quick service", minRating: 3.5 });
    expect(describeRequest(buildPayload(withText))).toBe(
      "Cafe · in HSR · rated 3.5★ or more · “quick service”",
    );
  });

  it("offers only quick searches whose values are in the data", () => {
    expect(availableQuickSearches(["Indiranagar"], ["Italian"])).toEqual(["Italian in Indiranagar"]);
    expect(availableQuickSearches([], [])).toEqual([]);
    const everyValue = availableQuickSearches(
      ["HSR", "Indiranagar", "Jayanagar"],
      ["Biryani", "Cafe", "Italian", "South Indian"],
    );
    expect(new Set(everyValue)).toEqual(new Set(Object.keys(QUICK_SEARCHES)));
  });

  it("explains why AI ranking is off", () => {
    expect(llmOffReason("configured")).toBeNull();
    expect(llmOffReason("unconfigured")).toBe("No Groq API key is set");
    expect(llmOffReason("disabled")).toContain("LLM_ENABLED=false");
    expect(llmOffReason("rejected")).toContain("Groq rejected the API key");
  });
});
