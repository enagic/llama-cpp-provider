import type {
  EmbeddingModelV4,
  EmbeddingModelV4CallOptions,
  EmbeddingModelV4Result,
} from "@ai-sdk/provider";
import {
  getLlama,
  LlamaLogLevel,
  type Llama,
  type LlamaModel,
  type LlamaEmbeddingContext,
} from "node-llama-cpp";
import { modelIdFromPath } from "./engine.js";
import type { LlamaCppProviderConfig } from "./config.js";

interface EmbeddingEngine {
  llama: Llama;
  model: LlamaModel;
  context: LlamaEmbeddingContext;
}

export class LlamaCppEmbeddingModel implements EmbeddingModelV4 {
  readonly specificationVersion = "v4" as const;
  readonly provider = "llama.cpp";
  readonly modelId: string;
  readonly maxEmbeddingsPerCall = 2048;
  readonly supportsParallelCalls = false;

  private readonly config: LlamaCppProviderConfig;
  private enginePromise: Promise<EmbeddingEngine> | null = null;

  constructor(config: LlamaCppProviderConfig) {
    this.config = config;
    this.modelId = modelIdFromPath(config.modelPath);
  }

  private ensureEngine(): Promise<EmbeddingEngine> {
    if (this.enginePromise == null) {
      const config = this.config;
      this.enginePromise = (async () => {
        const llama = await getLlama({
          logLevel: config.debug ? LlamaLogLevel.debug : LlamaLogLevel.warn,
        });
        const model = await llama.loadModel({
          modelPath: config.modelPath,
          gpuLayers: config.gpuLayers === "auto" ? undefined : config.gpuLayers,
        });
        const context = await model.createEmbeddingContext({
          contextSize: config.contextSize,
        });
        return { llama, model, context };
      })().catch((error) => {
        this.enginePromise = null;
        throw error;
      });
    }
    return this.enginePromise;
  }

  async dispose(): Promise<void> {
    if (this.enginePromise != null) {
      const engine = await this.enginePromise;
      this.enginePromise = null;
      await engine.context.dispose();
      await engine.model.dispose();
      await engine.llama.dispose();
    }
  }

  async doEmbed(
    options: EmbeddingModelV4CallOptions
  ): Promise<EmbeddingModelV4Result> {
    const engine = await this.ensureEngine();

    const embeddings: number[][] = [];
    let tokens = 0;
    for (const value of options.values) {
      if (options.abortSignal?.aborted)
        throw options.abortSignal.reason ?? new Error("Aborted");
      const embedding = await engine.context.getEmbeddingFor(value);
      embeddings.push([...embedding.vector]);
      tokens += engine.model.tokenize(value).length;
    }

    return {
      embeddings,
      usage: { tokens },
      warnings: [],
    };
  }
}
