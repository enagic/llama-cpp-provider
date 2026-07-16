import { describe, expect, it } from "vitest";
import type { ChatHistoryItem, ChatModelFunctions } from "node-llama-cpp";
import { MiniCpmChatWrapper } from "../../src/minicpm-chat-wrapper.js";

const weatherFunctions: ChatModelFunctions = {
  weather: {
    description: "Get the current weather in a city",
    params: {
      type: "object",
      properties: { city: { type: "string" } },
    },
  },
};

function render(history: ChatHistoryItem[], functions?: ChatModelFunctions): string {
  const wrapper = new MiniCpmChatWrapper();
  const { contextText } = wrapper.generateContextState({
    chatHistory: history,
    availableFunctions: functions,
  });
  return contextText.toString();
}

describe("MiniCpmChatWrapper", () => {
  it("documents tools in the MiniCPM system format", () => {
    const text = render(
      [
        { type: "user", text: "Weather in Tokyo?" },
        { type: "model", response: [] },
      ],
      weatherFunctions
    );

    expect(text).toContain(
      "You are provided with function signatures within <tools></tools> XML tags:"
    );
    expect(text).toContain(
      '<tools>\n{"type": "function", "function": {"name": "weather", "description": "Get the current weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}\n</tools>'
    );
    expect(text).toContain(
      '<function name="function-name"><param name="param-name">param-value</param></function>'
    );
  });

  it("renders history calls in per-key XML with the result in a tool_response user turn", () => {
    const text = render(
      [
        { type: "user", text: "Weather in Tokyo?" },
        {
          type: "model",
          response: [
            {
              type: "functionCall",
              name: "weather",
              params: { city: "Tokyo" },
              result: { temperature: 22 },
              startsNewChunk: true,
            },
            "It is 22 degrees.",
          ],
        },
      ],
      weatherFunctions
    );

    expect(text).toContain(
      '<function name="weather"><param name="city">Tokyo</param></function>'
    );
    expect(text).toContain(
      '<|im_start|>user\n<tool_response>\n{"temperature": 22}\n</tool_response><|im_end|>'
    );
  });

  it("renders string tool results verbatim", () => {
    const text = render([
      { type: "user", text: "Weather?" },
      {
        type: "model",
        response: [
          {
            type: "functionCall",
            name: "weather",
            params: { city: "Tokyo" },
            result: "sunny, 22 degrees",
            startsNewChunk: true,
          },
        ],
      },
    ]);

    expect(text).toContain("<tool_response>\nsunny, 22 degrees\n</tool_response>");
    expect(text).not.toContain('"sunny, 22 degrees"');
  });

  it("wraps param values containing <, & or newlines in CDATA", () => {
    const text = render([
      { type: "user", text: "Run it" },
      {
        type: "model",
        response: [
          {
            type: "functionCall",
            name: "run",
            params: { script: "line1\nline2 <b> & more" },
            result: null,
            startsNewChunk: true,
          },
        ],
      },
    ]);

    expect(text).toContain(
      '<param name="script"><![CDATA[line1\nline2 <b> & more]]></param>'
    );
  });

  it("renders non-string param values as JSON without CDATA", () => {
    const text = render([
      { type: "user", text: "Weather in Tokyo and Osaka?" },
      {
        type: "model",
        response: [
          {
            type: "functionCall",
            name: "weather",
            params: { cities: ["Tokyo", "Osaka"], detailed: true },
            result: null,
            startsNewChunk: true,
          },
        ],
      },
    ]);

    expect(text).toContain(
      '<function name="weather"><param name="cities">["Tokyo", "Osaka"]</param><param name="detailed">true</param></function>'
    );
  });

  it("prepends BOS (the chat template renders bos_token and the GGUF has add_bos_token=false)", () => {
    const wrapper = new MiniCpmChatWrapper();
    const { contextText } = wrapper.generateContextState({
      chatHistory: [
        { type: "user", text: "Hi" },
        { type: "model", response: [] },
      ],
    });
    expect(contextText.toJSON()[0]).toEqual({ type: "specialToken", value: "BOS" });
  });

  it("uses the native call prefix as the generation trigger", () => {
    const wrapper = new MiniCpmChatWrapper();
    expect(wrapper.settings.functions.call.prefix.toString()).toBe('<function name="');
    expect(wrapper.settings.functions.call.suffix.toString()).toBe(
      "</param></function>"
    );
  });
});
