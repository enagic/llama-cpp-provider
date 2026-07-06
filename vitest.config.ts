import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    testTimeout: 300_000,
    hookTimeout: 300_000,
    // The native llama.cpp addon is loaded per worker; keep everything in one fork so
    // model memory isn't duplicated and Metal teardown doesn't race across workers.
    pool: "forks",
    maxWorkers: 1,
    minWorkers: 1,
  },
});
