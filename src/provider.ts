import { LlamaCppLanguageModel } from "./language-model.js";
import { LlamaCppEmbeddingModel } from "./embedding-model.js";
import type { LlamaCppProviderConfig } from "./config.js";

export interface LlamaCppProvider {
  (config: LlamaCppProviderConfig): LlamaCppLanguageModel;
  languageModel(config: LlamaCppProviderConfig): LlamaCppLanguageModel;
  embedding(config: LlamaCppProviderConfig): LlamaCppEmbeddingModel;
}

function createLlamaCpp(): LlamaCppProvider {
  const provider = (config: LlamaCppProviderConfig) =>
    new LlamaCppLanguageModel(config);

  provider.languageModel = provider;
  provider.embedding = (config: LlamaCppProviderConfig) =>
    new LlamaCppEmbeddingModel(config);

  return provider as LlamaCppProvider;
}

/**
 * Creates a llama.cpp language model for the Vercel AI SDK, backed by node-llama-cpp.
 *
 * Tool calls are grammar-constrained at sampling time: the chat wrapper renders tool
 * definitions the way the model was trained to see them, and once the model starts a
 * tool call, every subsequent token is masked so that only a valid tool name and
 * schema-conforming JSON arguments can be sampled. Malformed tool calls cannot be
 * generated.
 *
 * @example
 * ```typescript
 * import { generateText, tool } from "ai";
 * import { llamaCpp } from "@enagic/llama-cpp-provider";
 * import { z } from "zod";
 *
 * const model = llamaCpp({ modelPath: "./models/qwen3-8b.Q4_K_M.gguf" });
 *
 * const result = await generateText({
 *   model,
 *   tools: {
 *     weather: tool({
 *       description: "Get the weather in a location",
 *       inputSchema: z.object({ location: z.string() }),
 *       execute: async ({ location }) => ({ location, temperature: 22 }),
 *     }),
 *   },
 *   prompt: "What is the weather in Tokyo?",
 * });
 *
 * await model.dispose();
 * ```
 */
export const llamaCpp = createLlamaCpp();

export default llamaCpp;
