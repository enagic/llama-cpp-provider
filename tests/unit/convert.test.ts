import { describe, expect, it } from "vitest";
import type { LanguageModelV4Prompt } from "@ai-sdk/provider";
import {
  convertFinishReason,
  convertPrompt,
  convertTools,
  convertUsage,
  resolveForcedToolMode,
} from "../../src/convert.js";

describe("convertPrompt", () => {
  it("converts system, user, and assistant text messages", () => {
    const prompt: LanguageModelV4Prompt = [
      { role: "system", content: "Be concise." },
      { role: "user", content: [{ type: "text", text: "Hi" }] },
      { role: "assistant", content: [{ type: "text", text: "Hello!" }] },
    ];

    const { history, warnings } = convertPrompt(prompt);
    expect(warnings).toEqual([]);
    expect(history).toEqual([
      { type: "system", text: "Be concise." },
      { type: "user", text: "Hi" },
      { type: "model", response: ["Hello!"] },
    ]);
  });

  it("attaches tool results to the matching function call", () => {
    const prompt: LanguageModelV4Prompt = [
      { role: "user", content: [{ type: "text", text: "Weather in Tokyo?" }] },
      {
        role: "assistant",
        content: [
          {
            type: "tool-call",
            toolCallId: "call_1",
            toolName: "weather",
            input: { location: "Tokyo" },
          },
        ],
      },
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "call_1",
            toolName: "weather",
            output: { type: "json", value: { temperature: 22 } },
          },
        ],
      },
    ];

    const { history } = convertPrompt(prompt);
    expect(history).toHaveLength(2);
    expect(history[1]).toEqual({
      type: "model",
      response: [
        {
          type: "functionCall",
          name: "weather",
          params: { location: "Tokyo" },
          result: { temperature: 22 },
          startsNewChunk: true,
        },
      ],
    });
  });

  it("merges a call → result → answer agent turn into one model item", () => {
    const prompt: LanguageModelV4Prompt = [
      { role: "user", content: [{ type: "text", text: "Weather?" }] },
      {
        role: "assistant",
        content: [
          {
            type: "tool-call",
            toolCallId: "call_1",
            toolName: "weather",
            input: '{"location":"Tokyo"}',
          },
        ],
      },
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "call_1",
            toolName: "weather",
            output: { type: "text", value: "22C, sunny" },
          },
        ],
      },
      { role: "assistant", content: [{ type: "text", text: "It is 22C." }] },
    ];

    const { history } = convertPrompt(prompt);
    expect(history).toHaveLength(2);
    const model = history[1]!;
    expect(model.type).toBe("model");
    expect((model as any).response).toEqual([
      {
        type: "functionCall",
        name: "weather",
        params: { location: "Tokyo" },
        result: "22C, sunny",
        startsNewChunk: true,
      },
      "It is 22C.",
    ]);
  });

  it("drops reasoning parts from assistant history", () => {
    const prompt: LanguageModelV4Prompt = [
      { role: "user", content: [{ type: "text", text: "Hi" }] },
      {
        role: "assistant",
        content: [
          { type: "reasoning", text: "thinking..." },
          { type: "text", text: "Hello!" },
        ],
      },
    ];

    const { history } = convertPrompt(prompt);
    expect(history[1]).toEqual({ type: "model", response: ["Hello!"] });
  });

  it("throws on tool results for unknown call ids", () => {
    const prompt: LanguageModelV4Prompt = [
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "missing",
            toolName: "weather",
            output: { type: "text", value: "x" },
          },
        ],
      },
    ];

    expect(() => convertPrompt(prompt)).toThrow(/unknown tool call id/);
  });

  it("rejects image inputs", () => {
    const prompt: LanguageModelV4Prompt = [
      {
        role: "user",
        content: [
          {
            type: "file",
            mediaType: "image/png",
            data: { type: "data", data: new Uint8Array([1]) },
          },
        ],
      },
    ];

    expect(() => convertPrompt(prompt)).toThrow(/multimodal/);
  });

  it("defaults a call without a result to null", () => {
    const prompt: LanguageModelV4Prompt = [
      {
        role: "assistant",
        content: [
          {
            type: "tool-call",
            toolCallId: "call_1",
            toolName: "weather",
            input: {},
          },
        ],
      },
    ];

    const { history } = convertPrompt(prompt);
    expect((history[0] as any).response[0].result).toBe(null);
  });
});

describe("convertTools", () => {
  it("builds ChatModelFunctions with normalized schemas", () => {
    const warnings: any[] = [];
    const functions = convertTools(
      [
        {
          type: "function",
          name: "weather",
          description: "Get weather",
          inputSchema: {
            type: "object",
            properties: { location: { type: "string", pattern: "^[A-Z]" } },
            required: ["location"],
          } as any,
        },
        {
          type: "function",
          name: "noop",
          inputSchema: { type: "object", properties: {} } as any,
        },
      ],
      warnings
    );

    expect(Object.keys(functions)).toEqual(["weather", "noop"]);
    expect((functions.weather!.params as any).properties.location).toEqual({
      type: "string",
    });
    expect(functions.noop!.params).toBeUndefined();
    expect(warnings).toHaveLength(1);
    expect(warnings[0].details).toMatch(/pattern/);
  });
});

describe("resolveForcedToolMode", () => {
  const tools = [
    {
      type: "function" as const,
      name: "a",
      inputSchema: { type: "object", properties: { x: { type: "number" } } } as any,
    },
    {
      type: "function" as const,
      name: "b",
      inputSchema: { type: "object", properties: {} } as any,
    },
  ];

  it("builds a oneOf schema for required tool choice", () => {
    const mode = resolveForcedToolMode(tools, { type: "required" }, []);
    expect((mode.schema as any).oneOf).toHaveLength(2);
    expect(mode.systemPrompt).toContain('"name"');
  });

  it("selects a single tool for named tool choice", () => {
    const mode = resolveForcedToolMode(tools, { type: "tool", toolName: "a" }, []);
    expect((mode.schema as any).properties.name).toEqual({ const: "a" });
    expect(mode.tools).toHaveLength(1);
  });

  it("throws for unknown named tools", () => {
    expect(() =>
      resolveForcedToolMode(tools, { type: "tool", toolName: "zzz" }, [])
    ).toThrow(/unknown tool/);
  });
});

describe("result mapping", () => {
  it("maps stop reasons", () => {
    expect(convertFinishReason("eogToken", false).unified).toBe("stop");
    expect(convertFinishReason("maxTokens", false).unified).toBe("length");
    expect(convertFinishReason("functionCalls", true).unified).toBe("tool-calls");
    expect(convertFinishReason("eogToken", true).unified).toBe("tool-calls");
  });

  it("maps usage", () => {
    const usage = convertUsage(10, 5);
    expect(usage.inputTokens.total).toBe(10);
    expect(usage.outputTokens.total).toBe(5);
  });
});
