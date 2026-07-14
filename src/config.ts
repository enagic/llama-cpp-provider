import type { ResolvableChatWrapperTypeName } from "node-llama-cpp";

export interface LlamaCppProviderConfig {
  /** Path to a local GGUF model file. */
  modelPath: string;
  /**
   * Context size in tokens.
   *
   * Defaults to node-llama-cpp's automatic sizing, which picks the largest
   * context that fits in available memory (up to the model's training context).
   */
  contextSize?: number;
  /**
   * Number of model layers to offload to the GPU.
   *
   * Defaults to 999 (offload all layers to GPU). Set to 0 to disable GPU offload.
   */
  gpuLayers?: number;
  /**
   * Number of generations that can run concurrently (context sequences).
   *
   * Each parallel request gets its own slot with an isolated KV cache.
   * Slots cache their conversation prefix, so multi-turn agent loops reuse
   * previously evaluated tokens. Default: 1.
   */
  parallel?: number;
  /**
   * Chat wrapper used to render messages and tool definitions.
   *
   * - "auto" (default): resolve from the GGUF's chat template. Known templates
   *   get a specialized wrapper whose tool-call syntax is grammar-constrained;
   *   unknown templates fall back to a Jinja wrapper that still constrains
   *   tool-call output.
   * - A node-llama-cpp wrapper name (e.g. "qwen", "llama3.1", "gemma",
   *   "chatML", "harmony") to force a specific wrapper.
   */
  chatWrapper?: "auto" | ResolvableChatWrapperTypeName;
  /**
   * Default budget for reasoning ("thought") tokens per generation.
   *
   * Defaults to node-llama-cpp's heuristic (50-75% of context size).
   * Set to 0 to disable thinking for models that support toggling it.
   */
  thoughtTokenBudget?: number;
  /** Print node-llama-cpp debug/log output. Default: false (warnings only). */
  debug?: boolean;
}

/** Options accepted under `providerOptions.llamaCpp` on individual AI SDK calls. */
export interface LlamaCppCallOptions {
  /** min-p sampling (0-1). */
  minP?: number;
  /** Maximum number of parallel tool calls in one generation. Default: model-dependent. */
  maxParallelToolCalls?: number;
  /** Per-call override of the reasoning token budget. */
  thoughtTokenBudget?: number;
}

export function parseLlamaCppCallOptions(
  providerOptions: Record<string, Record<string, unknown>> | undefined
): LlamaCppCallOptions {
  const raw = providerOptions?.llamaCpp ?? providerOptions?.["llama.cpp"] ?? {};
  const result: LlamaCppCallOptions = {};

  if (typeof raw.minP === "number") result.minP = raw.minP;
  if (typeof raw.maxParallelToolCalls === "number")
    result.maxParallelToolCalls = raw.maxParallelToolCalls;
  if (typeof raw.thoughtTokenBudget === "number")
    result.thoughtTokenBudget = raw.thoughtTokenBudget;

  return result;
}
