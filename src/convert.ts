import crypto from "node:crypto";
import { UnsupportedFunctionalityError } from "@ai-sdk/provider";
import type {
  JSONSchema7,
  LanguageModelV4FinishReason,
  LanguageModelV4FunctionTool,
  LanguageModelV4Message,
  LanguageModelV4Prompt,
  LanguageModelV4ToolChoice,
  LanguageModelV4Usage,
  SharedV4Warning,
} from "@ai-sdk/provider";
import type {
  ChatHistoryItem,
  ChatModelFunctionCall,
  ChatModelFunctions,
  ChatModelResponse,
  GbnfJsonSchema,
} from "node-llama-cpp";
import { normalizeToolParameters } from "./schema.js";

export function generateToolCallId(): string {
  return "call_" + crypto.randomBytes(8).toString("hex");
}

// ---------------------------------------------------------------------------
// Prompt conversion: AI SDK V4 prompt → node-llama-cpp ChatHistoryItem[]
// ---------------------------------------------------------------------------

function toolResultOutputToValue(
  output: Extract<LanguageModelV4Message, { role: "tool" }>["content"][number] extends infer P
    ? P extends { type: "tool-result"; output: infer O }
      ? O
      : never
    : never
): any {
  switch (output.type) {
    case "text":
      return output.value;
    case "json":
      return output.value;
    case "error-text":
      return { error: output.value };
    case "error-json":
      return { error: output.value };
    case "execution-denied":
      return { error: `Execution denied${output.reason ? `: ${output.reason}` : ""}` };
    case "content":
      return output.value
        .map((item) =>
          item.type === "text"
            ? item.text
            : item.type === "file"
              ? `[File: ${item.mediaType}]`
              : "[Unsupported content]"
        )
        .join("\n");
    default:
      return null;
  }
}

function parseToolCallInput(input: unknown): any {
  if (typeof input !== "string") return input ?? {};
  if (input.trim() === "") return {};
  try {
    return JSON.parse(input);
  } catch {
    return input;
  }
}

/**
 * Converts an AI SDK prompt into node-llama-cpp chat history.
 *
 * Assistant tool calls become `functionCall` entries inside a `model` response item;
 * the matching `tool` message results are attached to those entries (that is the shape
 * chat wrappers render back into the model's native tool-call syntax). Consecutive
 * model items are merged so a call → result → answer turn stays one model response.
 */
export function convertPrompt(prompt: LanguageModelV4Prompt): {
  history: ChatHistoryItem[];
  warnings: SharedV4Warning[];
} {
  const history: ChatHistoryItem[] = [];
  const warnings: SharedV4Warning[] = [];
  const pendingCallsById = new Map<string, ChatModelFunctionCall>();

  const lastModelItem = (): ChatModelResponse | undefined => {
    const last = history[history.length - 1];
    return last?.type === "model" ? last : undefined;
  };

  for (const message of prompt) {
    switch (message.role) {
      case "system":
        history.push({ type: "system", text: message.content });
        break;

      case "user": {
        let text = "";
        for (const part of message.content) {
          if (part.type === "text") text += part.text;
          else
            throw new UnsupportedFunctionalityError({
              functionality: `${part.type} parts in user messages (multimodal input is not supported by node-llama-cpp)`,
            });
        }
        history.push({ type: "user", text });
        break;
      }

      case "assistant": {
        const model = lastModelItem() ?? { type: "model" as const, response: [] };
        if (model.response.length === 0 && lastModelItem() === undefined)
          history.push(model);

        let firstCallInMessage = true;
        for (const part of message.content) {
          if (part.type === "text") {
            if (part.text.length > 0) model.response.push(part.text);
          } else if (part.type === "reasoning") {
            // Old reasoning is omitted from history; chat wrappers of reasoning
            // models expect prior turns without thought blocks.
          } else if (part.type === "tool-call") {
            const functionCall: ChatModelFunctionCall = {
              type: "functionCall",
              name: part.toolName,
              params: parseToolCallInput(part.input),
              result: undefined,
              ...(firstCallInMessage ? { startsNewChunk: true } : {}),
            };
            firstCallInMessage = false;
            model.response.push(functionCall);
            pendingCallsById.set(part.toolCallId, functionCall);
          } else if (part.type === "tool-result") {
            const call = pendingCallsById.get(part.toolCallId);
            if (call != null) call.result = toolResultOutputToValue(part.output);
          } else {
            warnings.push({
              type: "other",
              message: `Ignored unsupported assistant content part: ${part.type}`,
            });
          }
        }
        break;
      }

      case "tool": {
        for (const part of message.content) {
          if (part.type === "tool-result") {
            const call = pendingCallsById.get(part.toolCallId);
            if (call == null)
              throw new Error(
                `Tool result references unknown tool call id "${part.toolCallId}"`
              );
            call.result = toolResultOutputToValue(part.output);
          } else {
            warnings.push({
              type: "other",
              message: `Ignored unsupported tool content part: ${part.type}`,
            });
          }
        }
        break;
      }
    }
  }

  // A call that never received a result would render as still-pending; represent it
  // as an explicitly empty result instead.
  for (const call of pendingCallsById.values())
    if (call.result === undefined) call.result = null;

  return { history, warnings };
}

// ---------------------------------------------------------------------------
// Tool conversion: AI SDK function tools → node-llama-cpp ChatModelFunctions
// ---------------------------------------------------------------------------

export function convertTools(
  tools: LanguageModelV4FunctionTool[],
  warnings: SharedV4Warning[]
): ChatModelFunctions {
  const functions: Record<
    string,
    { description?: string; params?: Readonly<GbnfJsonSchema> }
  > = {};

  for (const tool of tools) {
    const normalized = normalizeToolParameters(tool.inputSchema, tool.name);
    for (const warning of normalized.warnings)
      warnings.push({
        type: "compatibility",
        feature: "tool input schema",
        details: warning,
      });

    functions[tool.name] = {
      ...(tool.description != null ? { description: tool.description } : {}),
      ...(normalized.schema != null
        ? { params: normalized.schema as GbnfJsonSchema }
        : {}),
    };
  }

  return functions;
}

/**
 * Normalizes a `responseFormat: { type: "json", schema }` schema into the GBNF-JSON
 * subset, so structured output is grammar-enforced the same way tool arguments are.
 */
export function normalizeResponseFormatSchema(
  schema: unknown,
  warnings: SharedV4Warning[]
): any {
  const normalized = normalizeToolParameters(schema, "responseFormat");
  for (const warning of normalized.warnings)
    warnings.push({
      type: "compatibility",
      feature: "responseFormat schema",
      details: warning,
    });
  // An empty object schema means "any JSON object"
  return normalized.schema ?? { type: "object", additionalProperties: true };
}

// ---------------------------------------------------------------------------
// Forced tool choice (toolChoice "required" / named tool)
//
// node-llama-cpp's lazy function-call grammar cannot force a call the way
// llama.cpp's `min_calls = 1` eager grammar does. Instead, the entire response
// is constrained to a JSON grammar of the allowed calls; the tool definitions
// are injected as a system message so the model knows their semantics. The
// output syntax is still sampler-enforced, so it always parses.
// ---------------------------------------------------------------------------

export interface ForcedToolMode {
  schema: GbnfJsonSchema;
  systemPrompt: string;
  tools: LanguageModelV4FunctionTool[];
}

export function resolveForcedToolMode(
  tools: LanguageModelV4FunctionTool[],
  toolChoice: LanguageModelV4ToolChoice,
  warnings: SharedV4Warning[]
): ForcedToolMode {
  const activeTools =
    toolChoice.type === "tool"
      ? tools.filter((tool) => tool.name === toolChoice.toolName)
      : tools;

  if (toolChoice.type === "tool" && activeTools.length === 0)
    throw new Error(`toolChoice references unknown tool: ${toolChoice.toolName}`);

  const callSchemas: GbnfJsonSchema[] = activeTools.map((tool) => {
    const normalized = normalizeToolParameters(tool.inputSchema, tool.name);
    for (const warning of normalized.warnings)
      warnings.push({
        type: "compatibility",
        feature: "tool input schema",
        details: warning,
      });

    return {
      type: "object",
      properties: {
        name: { const: tool.name },
        arguments: (normalized.schema ?? {
          type: "object",
          properties: {},
        }) as GbnfJsonSchema,
      },
    } as GbnfJsonSchema;
  });

  const schema: GbnfJsonSchema =
    callSchemas.length === 1
      ? callSchemas[0]!
      : ({ oneOf: callSchemas } as GbnfJsonSchema);

  const toolDocs = activeTools
    .map((tool) => {
      const params = JSON.stringify(tool.inputSchema ?? {});
      return `- ${tool.name}: ${tool.description ?? "No description"}\n  Input schema: ${params}`;
    })
    .join("\n");

  const systemPrompt =
    `You must call one of the following tools:\n\n${toolDocs}\n\n` +
    `Respond with a JSON object of the form {"name": "<tool name>", "arguments": {...}} and nothing else.`;

  return { schema, systemPrompt, tools: activeTools };
}

// ---------------------------------------------------------------------------
// Result mapping
// ---------------------------------------------------------------------------

export function convertFinishReason(
  stopReason: "eogToken" | "stopGenerationTrigger" | "functionCalls" | "maxTokens" | "abort" | "customStopTrigger",
  hasToolCalls: boolean
): LanguageModelV4FinishReason {
  if (hasToolCalls || stopReason === "functionCalls")
    return { unified: "tool-calls", raw: stopReason };
  if (stopReason === "maxTokens") return { unified: "length", raw: stopReason };
  if (stopReason === "abort") return { unified: "other", raw: stopReason };
  return { unified: "stop", raw: stopReason };
}

export function convertUsage(
  promptTokens: number,
  completionTokens: number
): LanguageModelV4Usage {
  return {
    inputTokens: {
      total: promptTokens,
      // Prefix-cached tokens are not re-evaluated, so the meter only counts
      // uncached prompt tokens; cache metrics are not reported separately.
      noCache: promptTokens,
      cacheRead: undefined,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: completionTokens,
      text: completionTokens,
      reasoning: undefined,
    },
  };
}

export function splitTools(
  tools: LanguageModelV4CallOptionsTools,
  warnings: SharedV4Warning[]
): LanguageModelV4FunctionTool[] {
  const functionTools: LanguageModelV4FunctionTool[] = [];
  for (const tool of tools ?? []) {
    if (tool.type === "function") functionTools.push(tool);
    else
      warnings.push({
        type: "unsupported",
        feature: `provider tool ${tool.name}`,
        details: "Provider-defined tools are not supported by llama.cpp",
      });
  }
  return functionTools;
}

type LanguageModelV4CallOptionsTools =
  | Array<LanguageModelV4FunctionTool | { type: "provider"; name: string }>
  | undefined;

export type { JSONSchema7 };
