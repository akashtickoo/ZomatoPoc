// EC-UI-20: text from the data or the LLM is rendered by React, which escapes it. Raw HTML
// injection would bypass that, so no source file may use it.

import { expect, it } from "vitest";

const sources = import.meta.glob(["./**/*.{ts,tsx}", "!./**/*.test.{ts,tsx}"], {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

it("never renders raw HTML", () => {
  expect(Object.keys(sources)).toContain("./App.tsx");
  const offenders = Object.entries(sources)
    .filter(([, text]) => /dangerouslySetInnerHTML|\.innerHTML\b|insertAdjacentHTML/.test(text))
    .map(([path]) => path);
  expect(offenders).toEqual([]);
});
