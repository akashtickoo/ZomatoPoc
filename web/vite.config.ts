/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The Curated Cravings web front end (docs/deployment-plan.md, Phase 3). `npm run dev`
// serves it on http://localhost:5173, the origin the API allows locally (CORS_ORIGINS in
// the project's .env); strictPort keeps it from moving to a port the API would refuse.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
  },
});
