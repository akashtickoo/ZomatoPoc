import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { HOSTILE, item } from "../test/fakeApi";
import ResultCard from "./ResultCard";

describe("ResultCard", () => {
  it("shows the five required fields", () => {
    const { container } = render(<ResultCard item={item()} />);
    expect(screen.getByRole("heading", { name: "Glen's Bakehouse" })).toBeInTheDocument();
    expect(screen.getByText("Italian")).toHaveClass("cc-chip");
    expect(screen.getByText("4.3 ★ (5,352)")).toBeInTheDocument();
    expect(screen.getByText("₹600 for two")).toBeInTheDocument();
    expect(screen.getByText("Rated 4.3 with 5,352 votes.")).toBeInTheDocument();
    expect(screen.getByText("✦ Why it fits")).toBeInTheDocument();
    expect(container.querySelector(".cc-rank")).toHaveTextContent(/^01$/);
    expect(container.querySelector(".cc-known")).toHaveTextContent("Known for Pizza · Pasta");
  });

  it("shows tags and handles missing values (EC-UI-09, 10)", () => {
    const { container } = render(
      <ResultCard
        item={item({
          book_table: true,
          online_order: false,
          rating: null,
          cost_for_two: null,
          cuisines: Array.from({ length: 7 }, (_, i) => `C${i}`),
          popular_dishes: [],
          source: "fallback",
        })}
      />,
    );
    expect(screen.getByText("Table booking")).toBeInTheDocument();
    expect(screen.queryByText("Online ordering")).not.toBeInTheDocument();
    expect(screen.getByText("New")).toHaveClass("cc-pill-new");
    expect(screen.getByText("Cost N/A")).toBeInTheDocument();
    expect(container.querySelectorAll(".cc-chip:not(.cc-chip-more)")).toHaveLength(5);
    expect(screen.getByText("+2 more")).toBeInTheDocument();
    expect(screen.queryByText("Known for")).not.toBeInTheDocument();
    expect(screen.getByText("Ranked by rating and popularity")).toBeInTheDocument();
  });

  it("shows markup from the data or the LLM as plain text (EC-UI-20)", () => {
    const { container } = render(
      <ResultCard
        item={item({ name: HOSTILE, explanation: HOSTILE, cuisines: [HOSTILE], popular_dishes: [HOSTILE] })}
      />,
    );
    expect(container.querySelector("img, a, strong")).toBeNull();
    expect(screen.getByRole("heading")).toHaveTextContent(HOSTILE);
    expect(container.textContent?.split(HOSTILE)).toHaveLength(5); // name, chip, why, dishes
  });
});
