import { describe, expect, it, vi } from "vitest";

import { ApiUnavailable, requestRecommendations, wakeUp } from "./api";
import { fakeApi, json, ranked } from "./test/fakeApi";
import type { Payload } from "./types";

const PAYLOAD: Payload = {
  location: "Indiranagar",
  budget: null,
  cuisines: [],
  min_rating: 0,
  additional_preferences: null,
  top_n: 3,
};

const answer = (response: Response | Promise<Response>) => fakeApi({ recommend: () => response });

describe("requestRecommendations", () => {
  it("returns the response", async () => {
    const api = answer(json(ranked(3)));
    const result = await requestRecommendations(PAYLOAD);
    expect(result).toEqual({ kind: "ok", response: ranked(3) });
    expect(api.payloads).toEqual([PAYLOAD]);
  });

  it("returns the 422 details", async () => {
    const detail = [{ type: "unknown_value", loc: ["body", "location"], msg: "Unknown location" }];
    answer(json({ detail }, 422));
    expect(await requestRecommendations(PAYLOAD)).toEqual({ kind: "invalid", errors: detail });
  });

  it("quotes the request id of a 500", async () => {
    answer(json({ detail: "Internal server error", request_id: "a1b2c3" }, 500));
    expect(await requestRecommendations(PAYLOAD)).toEqual({
      kind: "error",
      message: "The service hit an unexpected error. Please try again. (Reference: a1b2c3)",
    });
  });

  it("handles a response that isn't JSON, such as a proxy's 502 page", async () => {
    answer(new Response("<html>Bad Gateway</html>", { status: 502 }));
    expect(await requestRecommendations(PAYLOAD)).toEqual({
      kind: "error",
      message: "The service hit an unexpected error. Please try again.",
    });
  });

  it("reports an unreachable API (EC-UI-02)", async () => {
    fakeApi().down = true;
    expect(await requestRecommendations(PAYLOAD)).toEqual({ kind: "unavailable" });
  });

  it("gives up after its timeout (EC-UI-03)", async () => {
    vi.stubGlobal(
      "fetch",
      (_url: string, init: RequestInit) =>
        new Promise((_resolve, reject) =>
          init.signal?.addEventListener("abort", () =>
            reject(new DOMException("aborted", "AbortError")),
          ),
        ),
    );
    const result = await requestRecommendations(PAYLOAD, 20);
    expect(result).toEqual({
      kind: "error",
      message: "The service took too long to answer. Please try again.",
    });
  });
});

describe("wakeUp", () => {
  it("retries until the API answers", async () => {
    const api = fakeApi();
    api.down = true;
    const onWaiting = vi.fn(() => {
      if (onWaiting.mock.calls.length === 2) api.down = false;
    });
    const health = await wakeUp({ retryEveryMs: 1, giveUpAfterMs: 5_000 }, onWaiting);
    expect(health.llm).toBe("configured");
    expect(onWaiting).toHaveBeenCalledTimes(2);
  });

  it("gives up after the deadline", async () => {
    fakeApi().down = true;
    await expect(wakeUp({ retryEveryMs: 1, giveUpAfterMs: 20 })).rejects.toBeInstanceOf(
      ApiUnavailable,
    );
  });

  it("treats a page that isn't JSON as still waking", async () => {
    vi.stubGlobal("fetch", async () => new Response("<html>Waking up…</html>"));
    await expect(wakeUp({ retryEveryMs: 1, giveUpAfterMs: 0 })).rejects.toBeInstanceOf(
      ApiUnavailable,
    );
  });
});
