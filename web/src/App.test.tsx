// The app against a fake API: the web counterpart of the AppTest cases in
// tests/ui/test_streamlit_app.py. Case IDs refer to docs/edge-case.md §9.

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import App from "./App";
import { HOSTILE, LOCATIONS, fakeApi, json, ranked, type FakeApiOptions } from "./test/fakeApi";
import type { LlmState } from "./types";

const FAST_WAKE = { retryEveryMs: 1, giveUpAfterMs: 50 };

async function load(options: FakeApiOptions = {}) {
  const api = fakeApi(options);
  const user = userEvent.setup();
  render(<App wake={FAST_WAKE} />);
  await screen.findByRole("button", { name: "Find restaurants →" });
  return { api, user };
}

const cards = () => document.querySelectorAll("article.cc-card");

async function pickLocality(user: ReturnType<typeof userEvent.setup>, text: string) {
  await user.click(screen.getByLabelText("Locality"));
  await user.keyboard(`${text}{Enter}`);
}

describe("loading", () => {
  it("shows the form, the empty state and the quick searches", async () => {
    const { api } = await load();
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Curated Cravings");
    expect(screen.getByText(`${LOCATIONS.length} neighbourhoods`)).toBeInTheDocument();
    expect(screen.getByText("Where are we eating today?")).toBeInTheDocument();
    const quick = within(screen.getByRole("group", { name: "Popular searches" }));
    expect(quick.getAllByRole("button")).toHaveLength(4);
    expect(screen.getByRole("radio", { name: "Any" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Medium · ₹301–600" })).not.toBeChecked();
    expect(screen.queryByText("AI ranking is off")).not.toBeInTheDocument();
    expect(api.payloads).toEqual([]);
  });

  it.each<[LlmState, string]>([
    ["unconfigured", "No Groq API key is set"],
    ["disabled", "LLM_ENABLED=false"],
    ["rejected", "Groq rejected the API key"],
  ])("says when AI ranking is off (%s)", async (llm, reason) => {
    await load({ llm });
    expect(screen.getByText("AI ranking is off")).toBeInTheDocument();
    expect(screen.getByText(new RegExp(reason))).toBeInTheDocument();
  });

  it("says the API is waking up, then shows the form", async () => {
    const api = fakeApi();
    api.down = true;
    render(<App wake={{ retryEveryMs: 30, giveUpAfterMs: 5_000 }} />);
    expect(await screen.findByText("Waking up the kitchen…")).toBeInTheDocument();
    api.down = false;
    expect(await screen.findByRole("button", { name: "Find restaurants →" })).toBeInTheDocument();
  });

  it("offers a retry when the API can't be reached (EC-UI-01)", async () => {
    const api = fakeApi();
    api.down = true;
    const user = userEvent.setup();
    render(<App wake={FAST_WAKE} />);
    expect(await screen.findByText("We can't reach the kitchen")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Find restaurants →" })).not.toBeInTheDocument();
    api.down = false;
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("button", { name: "Find restaurants →" })).toBeInTheDocument();
  });
});

describe("searching", () => {
  it("asks for a neighbourhood first", async () => {
    const { api, user } = await load();
    await user.click(screen.getByRole("button", { name: "Find restaurants →" }));
    expect(screen.getByText("Choose a neighbourhood first")).toBeInTheDocument();
    expect(api.payloads).toEqual([]);
  });

  it("sends the form and shows the ranked cards", async () => {
    const { api, user } = await load();
    await pickLocality(user, "Indir");
    await user.click(screen.getByLabelText("Cuisines"));
    await user.keyboard("Ital{Enter}");
    await user.click(screen.getByRole("radio", { name: "Medium · ₹301–600" }));
    await user.type(screen.getByLabelText("Anything else?"), "  quiet  ");
    await user.selectOptions(screen.getByLabelText("Show"), "3");
    await user.click(screen.getByRole("button", { name: "Find restaurants →" }));

    await waitFor(() => expect(cards()).toHaveLength(3));
    expect(api.payloads).toEqual([
      {
        location: "Indiranagar",
        budget: "medium",
        cuisines: ["Italian"],
        min_rating: 0,
        additional_preferences: "quiet",
        top_n: 3,
      },
    ]);
    cards().forEach((card, i) => expect(card.querySelector(".cc-rank")).toHaveTextContent(`0${i + 1}`));
    expect(screen.getByText("Your shortlist")).toBeInTheDocument();
    expect(screen.getByText("Italian · in Indiranagar · medium budget · “quiet”")).toBeInTheDocument();
    expect(screen.getByText(/AI ranked top 3/)).toBeInTheDocument();
    expect(
      screen.getByText("Ranked by fake-model · 10 candidates considered from 12 matches · 4.3 s"),
    ).toBeInTheDocument();
  });

  it("fills the form from a quick search and runs it", async () => {
    const { api, user } = await load();
    await user.click(screen.getByRole("button", { name: "South Indian in Jayanagar" }));
    await waitFor(() => expect(cards()).toHaveLength(3));
    expect(api.payloads).toEqual([
      {
        location: "Jayanagar",
        budget: "low",
        cuisines: ["South Indian"],
        min_rating: 0,
        additional_preferences: null,
        top_n: 5,
      },
    ]);
    expect(screen.getByText("Jayanagar", { selector: ".cc-select__single-value" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Low · ≤ ₹300" })).toBeChecked();
  });

  it("labels a fallback ranking and shows no banner (EC-UI-11)", async () => {
    const response = ranked(3, { source: "fallback" });
    response.meta = { ...response.meta, llm_status: "fallback", model: null };
    response.notice = "AI explanations are unavailable right now.";
    response.relaxations = ["Search widened from BTM"];
    const { user } = await load({ recommend: () => json(response) });
    await user.click(screen.getByRole("button", { name: "Italian in Indiranagar" }));
    await waitFor(() => expect(cards()).toHaveLength(3));
    expect(screen.getAllByText("Ranked by rating and popularity")).toHaveLength(3);
    expect(screen.getByText(/^Ranked without AI/)).toBeInTheDocument();
    expect(screen.getByText(/ranked by rating$/)).toBeInTheDocument();
    expect(screen.queryByText(response.notice)).not.toBeInTheDocument();
    expect(screen.queryByText("Search widened from BTM")).not.toBeInTheDocument();
  });

  it("says when fewer restaurants matched than were asked for (EC-UI-08)", async () => {
    const { user } = await load({ recommend: () => json(ranked(1)) });
    await user.click(screen.getByRole("button", { name: "Biryani across the city" }));
    expect(
      await screen.findByText(/^Showing 1 of 5: only 1 matched · Biryani · anywhere in Bangalore/),
    ).toBeInTheDocument();
  });

  it("says when nothing matched (EC-UI-07)", async () => {
    const empty = { ...ranked(0), notice: "No restaurants match these filters." };
    empty.meta = { ...empty.meta, llm_status: "not_called" };
    const { user } = await load({ recommend: () => json(empty) });
    await user.click(screen.getByRole("button", { name: "Cosy cafés in HSR" }));
    expect(await screen.findByText("No restaurants found")).toBeInTheDocument();
    expect(screen.getByText("No restaurants match these filters.")).toBeInTheDocument();
    expect(cards()).toHaveLength(0);
  });

  it("shows LLM text as plain text (EC-UI-20)", async () => {
    const { user } = await load({ recommend: () => json(ranked(3, { explanation: HOSTILE })) });
    await user.click(screen.getByRole("button", { name: "Italian in Indiranagar" }));
    await waitFor(() => expect(cards()).toHaveLength(3));
    expect(screen.getAllByText(HOSTILE)).toHaveLength(3);
    expect(document.querySelector("main img, main a")).toBeNull();
  });

  it("disables the button while a search runs (EC-UI-03, 04)", async () => {
    let finish: (r: Response) => void = () => {};
    const { api, user } = await load({ recommend: () => new Promise((r) => (finish = r)) });
    await user.click(screen.getByRole("button", { name: "Italian in Indiranagar" }));
    expect(await screen.findByText("Curating your shortlist…")).toBeInTheDocument();
    const button = screen.getByRole("button", { name: "Find restaurants →" });
    expect(button).toBeDisabled();
    await user.click(button);
    expect(api.payloads).toHaveLength(1);
    finish(json(ranked(3)));
    await waitFor(() => expect(button).toBeEnabled());
  });
});

describe("errors", () => {
  it("puts a suggested location into the form (EC-UI-06)", async () => {
    const error = {
      type: "unknown_value",
      loc: ["body", "location"],
      msg: "Unknown location 'Indiranagar'. Did you mean: Indiranagar 2nd Stage?",
      input: "Indiranagar",
      ctx: { suggestions: ["Indiranagar 2nd Stage"] },
    };
    const { user } = await load({ recommend: () => json({ detail: [error] }, 422) });
    await user.click(screen.getByRole("button", { name: "Italian in Indiranagar" }));
    expect(await screen.findByText(error.msg)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Use Indiranagar 2nd Stage" }));
    expect(
      screen.getByText("Indiranagar 2nd Stage", { selector: ".cc-select__single-value" }),
    ).toBeInTheDocument();
  });

  it("puts a suggested cuisine into the form and searches with it (EC-UI-06)", async () => {
    const error = {
      type: "unknown_value",
      loc: ["body", "cuisines"],
      msg: "Unknown cuisine 'Italian'. Did you mean: Italiano?",
      input: "Italian",
      ctx: { suggestions: ["Italiano"] },
    };
    let first = true;
    const { api, user } = await load({
      recommend: () => (first ? ((first = false), json({ detail: [error] }, 422)) : json(ranked(2))),
    });
    await user.click(screen.getByRole("button", { name: "Italian in Indiranagar" }));
    await user.click(await screen.findByRole("button", { name: "Use Italiano" }));
    await user.click(screen.getByRole("button", { name: "Find restaurants →" }));
    await waitFor(() => expect(cards()).toHaveLength(2));
    expect(api.payloads.map((p) => p.cuisines)).toEqual([["Italian"], ["Italiano"]]);
  });

  it("quotes the reference of a server error", async () => {
    const { user } = await load({
      recommend: () => json({ detail: "Internal server error", request_id: "a1b2c3" }, 500),
    });
    await user.click(screen.getByRole("button", { name: "Italian in Indiranagar" }));
    expect(await screen.findByText("Something went wrong")).toBeInTheDocument();
    expect(screen.getByText(/\(Reference: a1b2c3\)/)).toBeInTheDocument();
  });

  it("clears the old results when the API goes away (EC-UI-02)", async () => {
    const { api, user } = await load();
    await user.click(screen.getByRole("button", { name: "Italian in Indiranagar" }));
    await waitFor(() => expect(cards()).toHaveLength(3));
    api.down = true;
    await user.click(screen.getByRole("button", { name: "Find restaurants →" }));
    expect(await screen.findByText("We can't reach the kitchen")).toBeInTheDocument();
    expect(cards()).toHaveLength(0);
  });
});
