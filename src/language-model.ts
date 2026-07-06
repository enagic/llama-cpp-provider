import crypto from "node:crypto";
import type {
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4Content,
  LanguageModelV4FunctionTool,
  LanguageModelV4GenerateResult,
  LanguageModelV4StreamPart,
  LanguageModelV4StreamResult,
  SharedV4Warning,
} from "@ai-sdk/provider";
import type {
  ChatHistoryItem,
  ChatModelFunctions,
  LlamaGrammar,
} from "node-llama-cpp";
import { createEngine, modelIdFromPath, type Engine } from "./engine.js";
import type { Slot } from "./slot-pool.js";
import { sharedPrefixLength } from "./slot-pool.js";
import {
  convertFinishReason,
  convertPrompt,
  convertTools,
  convertUsage,
  generateToolCallId,
  normalizeResponseFormatSchema,
  resolveForcedToolMode,
  splitTools,
  type ForcedToolMode,
} from "./convert.js";
import {
  parseLlamaCppCallOptions,
  type LlamaCppProviderConfig,
} from "./config.js";

interface PreparedCall {
  history: ChatHistoryItem[];
  functions: ChatModelFunctions | undefined;
  grammar: LlamaGrammar | undefined;
  forcedToolMode: ForcedToolMode | undefined;
  warnings: SharedV4Warning[];
  maxTokens: number | undefined;
  temperature: number | undefined;
  topP: number | undefined;
  topK: number | undefined;
  minP: number | undefined;
  seed: number | undefined;
  customStopTriggers: string[] | undefined;
  maxParallelFunctionCalls: number | undefined;
  thoughtTokenBudget: number | undefined;
}

interface ToolCallResult {
  id: string;
  name: string;
  /** JSON-encoded arguments (the AI SDK wire format requires a string) */
  input: string;
}

interface GenerationOutcome {
  text: string;
  reasoning: string;
  toolCalls: ToolCallResult[];
  finishReason: ReturnType<typeof convertFinishReason>;
  promptTokens: number;
  completionTokens: number;
}

interface GenerationEvents {
  onText?: (text: string) => void;
  onReasoning?: (text: string) => void;
  onToolCallStart?: (index: number, id: string, name: string) => void;
  onToolCallArgs?: (index: number, argsChunk: string) => void;
}

export class LlamaCppLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = "v4" as const;
  readonly provider = "llama.cpp";
  readonly modelId: string;
  readonly supportedUrls: Record<string, RegExp[]> = {};

  private readonly config: LlamaCppProviderConfig;
  private enginePromise: Promise<Engine> | null = null;

  constructor(config: LlamaCppProviderConfig) {
    this.config = config;
    this.modelId = modelIdFromPath(config.modelPath);
  }

  private ensureEngine(): Promise<Engine> {
    if (this.enginePromise == null) {
      this.enginePromise = createEngine(this.config).catch((error) => {
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
      await engine.dispose();
    }
  }

  private async prepare(
    options: LanguageModelV4CallOptions,
    engine: Engine
  ): Promise<PreparedCall> {
    const converted = convertPrompt(options.prompt);
    const warnings: SharedV4Warning[] = [...converted.warnings];
    const callOptions = parseLlamaCppCallOptions(
      options.providerOptions as Record<string, Record<string, unknown>> | undefined
    );

    for (const setting of ["presencePenalty", "frequencyPenalty"] as const)
      if (options[setting] != null && options[setting] !== 0)
        warnings.push({
          type: "unsupported",
          feature: setting,
          details: `${setting} is not supported and was ignored`,
        });

    const functionTools: LanguageModelV4FunctionTool[] = splitTools(
      options.tools,
      warnings
    );
    const toolChoice = options.toolChoice ?? { type: "auto" };
    const toolsActive = functionTools.length > 0 && toolChoice.type !== "none";

    const wantsJson = options.responseFormat?.type === "json";
    if (wantsJson && toolsActive)
      throw new Error(
        "responseFormat json cannot be combined with active tools. " +
          'Disable tools for this call or set toolChoice to "none".'
      );

    let functions: ChatModelFunctions | undefined;
    let grammar: LlamaGrammar | undefined;
    let forcedToolMode: ForcedToolMode | undefined;
    const history = converted.history;

    if (toolsActive && (toolChoice.type === "required" || toolChoice.type === "tool")) {
      forcedToolMode = resolveForcedToolMode(functionTools, toolChoice, warnings);
      grammar = await engine.llama.createGrammarForJsonSchema(
        forcedToolMode.schema as any
      );
      // The chat wrapper only documents tools passed as functions; in forced mode the
      // definitions travel in a system message instead (output stays grammar-enforced).
      const firstNonSystem = history.findIndex((item) => item.type !== "system");
      history.splice(firstNonSystem === -1 ? history.length : firstNonSystem, 0, {
        type: "system",
        text: forcedToolMode.systemPrompt,
      });
    } else if (toolsActive) {
      functions = convertTools(functionTools, warnings);
    } else if (wantsJson) {
      grammar =
        options.responseFormat?.type === "json" && options.responseFormat.schema != null
          ? await engine.llama.createGrammarForJsonSchema(
              normalizeResponseFormatSchema(options.responseFormat.schema, warnings)
            )
          : await engine.llama.getGrammarFor("json");
    }

    const thoughtTokenBudget =
      options.reasoning === "none"
        ? 0
        : (callOptions.thoughtTokenBudget ?? this.config.thoughtTokenBudget);

    return {
      history,
      functions,
      grammar,
      forcedToolMode,
      warnings,
      maxTokens: options.maxOutputTokens,
      temperature: options.temperature,
      topP: options.topP,
      topK: options.topK,
      minP: callOptions.minP,
      seed: options.seed,
      customStopTriggers: options.stopSequences,
      maxParallelFunctionCalls: callOptions.maxParallelToolCalls,
      thoughtTokenBudget,
    };
  }

  /**
   * Runs one generation on a slot.
   *
   * Tool calls come back as structured `functionCalls` from LlamaChat — the chat
   * wrapper's function-call syntax is enforced by GBNF grammars during sampling
   * (`FunctionCallNameGrammar` / `FunctionCallParamsGrammar`), so there is no output
   * parsing and no "model produced malformed tool call" failure mode.
   */
  private async runGeneration(
    slot: Slot,
    prepared: PreparedCall,
    signal: AbortSignal | undefined,
    events: GenerationEvents
  ): Promise<GenerationOutcome> {
    const meterBefore = slot.sequence.tokenMeter.getState();

    let reasoning = "";
    let mainTextStarted = false; // suppress whitespace-only lead-in after think blocks
    // Tool calls announced to streaming callbacks, by callIndex. pendingWs holds a
    // trailing-whitespace run that is only emitted if more non-whitespace args follow —
    // models pad args with newlines before the call-suffix tokens, and that padding
    // shouldn't reach the client.
    const announcedCalls = new Map<
      number,
      { id: string; name: string; args: string; pendingWs: string }
    >();

    const sharedOptions = {
      signal,
      stopOnAbortSignal: false,
      maxTokens: prepared.maxTokens,
      temperature: prepared.temperature,
      topK: prepared.topK,
      topP: prepared.topP,
      minP: prepared.minP,
      seed: prepared.seed,
      customStopTriggers: prepared.customStopTriggers,
      ...(prepared.thoughtTokenBudget != null
        ? { budgets: { thoughtTokens: prepared.thoughtTokenBudget } }
        : {}),
      onResponseChunk(chunk: {
        type: undefined | "segment";
        segmentType?: string;
        text: string;
      }) {
        if (chunk.type === "segment") {
          if (chunk.segmentType === "thought" && chunk.text.length > 0) {
            reasoning += chunk.text;
            events.onReasoning?.(chunk.text);
          }
          return;
        }
        let text = chunk.text;
        if (!mainTextStarted) {
          text = text.trimStart();
          if (text.length === 0) return;
          mainTextStarted = true;
        }
        events.onText?.(text);
      },
      // Only reuse the previous evaluation's context window when the incoming request
      // actually continues the conversation this slot last served (all prior items
      // match, allowing the trailing in-progress model item to differ). Passing it for
      // an unrelated conversation makes LlamaChat continue the old context.
      lastEvaluationContextWindow:
        slot.lastContextWindow != null &&
        slot.chatHistory.length > 0 &&
        sharedPrefixLength(slot.chatHistory, prepared.history) >=
          slot.chatHistory.length - 1
          ? { history: slot.lastContextWindow }
          : undefined,
    };

    const response =
      prepared.functions != null
        ? await slot.llamaChat.generateResponse(prepared.history, {
            ...sharedOptions,
            functions: prepared.functions,
            maxParallelFunctionCalls: prepared.maxParallelFunctionCalls,
            onFunctionCallParamsChunk({ callIndex, functionName, paramsChunk }) {
              let call = announcedCalls.get(callIndex);
              if (call == null) {
                call = {
                  id: generateToolCallId(),
                  name: functionName,
                  args: "",
                  pendingWs: "",
                };
                announcedCalls.set(callIndex, call);
                events.onToolCallStart?.(callIndex, call.id, functionName);
              }

              const combined = call.pendingWs + paramsChunk;
              const trimmed = combined.replace(/\s+$/, "");
              call.pendingWs = combined.slice(trimmed.length);
              if (trimmed.length > 0) {
                call.args += trimmed;
                events.onToolCallArgs?.(callIndex, trimmed);
              }
            },
          })
        : await slot.llamaChat.generateResponse(prepared.history, {
            ...sharedOptions,
            grammar: prepared.grammar,
          });

    // Persist evaluation state on the slot for KV-cache reuse on the next request
    slot.chatHistory = response.lastEvaluation.cleanHistory;
    slot.lastContextWindow = response.lastEvaluation.contextWindow;
    slot.contextShiftMetadata = response.lastEvaluation.contextShiftMetadata;

    // Build final tool calls from the structured result; announce any call the params
    // stream never surfaced (e.g. a call with empty arguments)
    const toolCalls: ToolCallResult[] = [];
    const functionCalls = response.functionCalls ?? [];
    for (let i = 0; i < functionCalls.length; i++) {
      const fc = functionCalls[i]!;
      const argsJson = fc.params === undefined ? "{}" : JSON.stringify(fc.params);
      const announced = announcedCalls.get(i);
      if (announced == null) {
        const id = generateToolCallId();
        events.onToolCallStart?.(i, id, fc.functionName);
        events.onToolCallArgs?.(i, argsJson);
        toolCalls.push({ id, name: fc.functionName, input: argsJson });
      } else {
        // the streamed text is what the client saw; keep it authoritative
        toolCalls.push({
          id: announced.id,
          name: fc.functionName,
          input: announced.args.length > 0 ? announced.args : argsJson,
        });
      }
    }

    let text = response.response.trimStart();

    // Forced tool mode: the entire (grammar-guaranteed) JSON response is the tool call.
    if (prepared.forcedToolMode != null && toolCalls.length === 0 && text.length > 0) {
      const parsed = JSON.parse(text) as { name: string; arguments?: unknown };
      const id = generateToolCallId();
      const input = JSON.stringify(parsed.arguments ?? {});
      events.onToolCallStart?.(0, id, parsed.name);
      events.onToolCallArgs?.(0, input);
      toolCalls.push({ id, name: parsed.name, input });
      text = "";
    }

    const meterAfter = slot.sequence.tokenMeter.getState();
    const promptTokens = Math.max(
      0,
      meterAfter.usedInputTokens - meterBefore.usedInputTokens
    );
    const completionTokens = Math.max(
      0,
      meterAfter.usedOutputTokens - meterBefore.usedOutputTokens
    );

    return {
      text,
      reasoning,
      toolCalls,
      finishReason: convertFinishReason(
        response.metadata.stopReason,
        toolCalls.length > 0
      ),
      promptTokens,
      completionTokens,
    };
  }

  private requestBody(prepared: PreparedCall): unknown {
    return {
      modelId: this.modelId,
      history: prepared.history,
      functions: prepared.functions ? Object.keys(prepared.functions) : undefined,
      grammarConstrained: prepared.grammar != null || prepared.functions != null,
      maxTokens: prepared.maxTokens,
      temperature: prepared.temperature,
      topP: prepared.topP,
      topK: prepared.topK,
      minP: prepared.minP,
      seed: prepared.seed,
    };
  }

  async doGenerate(
    options: LanguageModelV4CallOptions
  ): Promise<LanguageModelV4GenerateResult> {
    const engine = await this.ensureEngine();
    const prepared = await this.prepare(options, engine);

    const slot = await engine.slots.acquire(prepared.history, options.abortSignal);
    let outcome: GenerationOutcome;
    try {
      outcome = await this.runGeneration(slot, prepared, options.abortSignal, {});
    } finally {
      engine.slots.release(slot);
    }

    const content: LanguageModelV4Content[] = [];
    if (outcome.reasoning.length > 0)
      content.push({ type: "reasoning", text: outcome.reasoning });
    if (outcome.text.length > 0) content.push({ type: "text", text: outcome.text });
    for (const toolCall of outcome.toolCalls)
      content.push({
        type: "tool-call",
        toolCallId: toolCall.id,
        toolName: toolCall.name,
        input: toolCall.input,
      });

    return {
      content,
      finishReason: outcome.finishReason,
      usage: convertUsage(outcome.promptTokens, outcome.completionTokens),
      warnings: prepared.warnings,
      request: { body: this.requestBody(prepared) },
    };
  }

  async doStream(
    options: LanguageModelV4CallOptions
  ): Promise<LanguageModelV4StreamResult> {
    const engine = await this.ensureEngine();
    const prepared = await this.prepare(options, engine);
    const self = this;

    let cancelled = false;
    const abortController = new AbortController();
    if (options.abortSignal != null) {
      if (options.abortSignal.aborted) abortController.abort(options.abortSignal.reason);
      else
        options.abortSignal.addEventListener(
          "abort",
          () => abortController.abort(options.abortSignal!.reason),
          { once: true }
        );
    }

    const stream = new ReadableStream<LanguageModelV4StreamPart>({
      async start(controller) {
        const enqueue = (part: LanguageModelV4StreamPart) => {
          if (!cancelled) controller.enqueue(part);
        };

        let slot: Slot | undefined;
        try {
          enqueue({ type: "stream-start", warnings: prepared.warnings });
          enqueue({
            type: "response-metadata",
            id: crypto.randomUUID(),
            modelId: self.modelId,
            timestamp: new Date(),
          });

          slot = await engine.slots.acquire(prepared.history, abortController.signal);

          let textId: string | undefined;
          let reasoningId: string | undefined;
          const toolIdsByIndex = new Map<number, string>();
          const inputClosed = new Set<string>();

          const endReasoning = () => {
            if (reasoningId != null) {
              enqueue({ type: "reasoning-end", id: reasoningId });
              reasoningId = undefined;
            }
          };
          const endText = () => {
            if (textId != null) {
              enqueue({ type: "text-end", id: textId });
              textId = undefined;
            }
          };

          const outcome = await self.runGeneration(
            slot,
            prepared,
            abortController.signal,
            {
              onReasoning(text) {
                if (reasoningId == null) {
                  reasoningId = crypto.randomUUID();
                  enqueue({ type: "reasoning-start", id: reasoningId });
                }
                enqueue({ type: "reasoning-delta", id: reasoningId, delta: text });
              },
              onText(text) {
                endReasoning();
                if (textId == null) {
                  textId = crypto.randomUUID();
                  enqueue({ type: "text-start", id: textId });
                }
                enqueue({ type: "text-delta", id: textId, delta: text });
              },
              onToolCallStart(index, id, name) {
                endReasoning();
                endText();
                toolIdsByIndex.set(index, id);
                enqueue({ type: "tool-input-start", id, toolName: name });
              },
              onToolCallArgs(index, argsChunk) {
                const id = toolIdsByIndex.get(index);
                if (id != null)
                  enqueue({ type: "tool-input-delta", id, delta: argsChunk });
              },
            }
          );

          endReasoning();
          endText();

          for (const toolCall of outcome.toolCalls) {
            if (!inputClosed.has(toolCall.id)) {
              enqueue({ type: "tool-input-end", id: toolCall.id });
              inputClosed.add(toolCall.id);
            }
            enqueue({
              type: "tool-call",
              toolCallId: toolCall.id,
              toolName: toolCall.name,
              input: toolCall.input,
            });
          }

          enqueue({
            type: "finish",
            finishReason: outcome.finishReason,
            usage: convertUsage(outcome.promptTokens, outcome.completionTokens),
          });
          if (!cancelled) controller.close();
        } catch (error) {
          if (!cancelled) controller.error(error);
        } finally {
          if (slot != null) engine.slots.release(slot);
        }
      },
      cancel() {
        cancelled = true;
        abortController.abort();
      },
    });

    return {
      stream,
      request: { body: this.requestBody(prepared) },
    };
  }
}
