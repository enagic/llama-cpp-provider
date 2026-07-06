import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { generateObject, generateText, stepCountIs, streamText, tool } from "ai";
import { z } from "zod";
import { llamaCpp } from "../../src/index.js";

const modelPath =
  process.env.LLAMA_TEST_MODEL ??
  path.join(os.homedir(), "models/ornith-1.0-9b/ornith-1.0-9b-Q4_K_M.gguf");

const hasModel = fs.existsSync(modelPath);

describe.skipIf(!hasModel)("e2e agent smoke", () => {
  const model = llamaCpp({ modelPath, contextSize: 8192 });

  afterAll(async () => {
    await model.dispose();
  });

  const weatherTool = tool({
    description: "Get the current weather in a city",
    inputSchema: z.object({ city: z.string().describe("Name of the city") }),
    execute: async ({ city }) => ({ city, temperature: 22, condition: "sunny" }),
  });

  it("generates plain text", async () => {
    const result = await generateText({
      model,
      prompt: "Reply with exactly the word: pong",
    });
    expect(result.text.toLowerCase()).toContain("pong");
    expect(result.usage.outputTokens).toBeGreaterThan(0);
    expect(result.usage.inputTokens).toBeGreaterThan(0);
  });

  it("runs a tool-calling agent loop", async () => {
    const result = await generateText({
      model,
      tools: { weather: weatherTool },
      stopWhen: stepCountIs(4),
      prompt: "What is the weather in Tokyo? Use the weather tool.",
    });

    const toolCalls = result.steps.flatMap((step) => step.toolCalls);
    expect(toolCalls.length).toBeGreaterThan(0);
    expect(toolCalls[0]!.toolName).toBe("weather");
    // Grammar-constrained: the input must be schema-valid JSON
    expect((toolCalls[0]!.input as any).city.toLowerCase()).toContain("tokyo");
    expect(result.text.toLowerCase()).toMatch(/22|sunny/);
  });

  it("streams a tool-calling agent loop", async () => {
    const result = streamText({
      model,
      tools: { weather: weatherTool },
      stopWhen: stepCountIs(4),
      prompt: "What is the weather in Paris? Use the weather tool.",
    });

    let streamedText = "";
    for await (const part of result.fullStream) {
      if (part.type === "text-delta") streamedText += part.text;
    }

    const toolCalls = (await result.steps).flatMap((step) => step.toolCalls);
    expect(toolCalls.length).toBeGreaterThan(0);
    expect(toolCalls[0]!.toolName).toBe("weather");
    expect(streamedText.toLowerCase()).toMatch(/22|sunny/);
  });

  it("forces a named tool call (toolChoice tool)", async () => {
    const result = await generateText({
      model,
      tools: { weather: weatherTool },
      toolChoice: { type: "tool", toolName: "weather" },
      stopWhen: stepCountIs(2),
      prompt: "Hello!",
    });

    const toolCalls = result.steps.flatMap((step) => step.toolCalls);
    expect(toolCalls.length).toBeGreaterThan(0);
    expect(toolCalls[0]!.toolName).toBe("weather");
  });

  it("generates structured output", async () => {
    const result = await generateObject({
      model,
      schema: z.object({
        name: z.string(),
        population: z.number(),
      }),
      prompt: "Give me the name and approximate population of Japan's capital.",
    });

    expect(typeof result.object.name).toBe("string");
    expect(typeof result.object.population).toBe("number");
  });
});

if (!hasModel)
  console.warn(
    `Skipping e2e tests: no GGUF model found at ${modelPath} (set LLAMA_TEST_MODEL)`
  );
