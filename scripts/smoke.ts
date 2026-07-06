// Quick interactive smoke test.
// Usage: npx tsx scripts/smoke.ts /path/to/model.gguf
import { generateText, stepCountIs, tool } from "ai";
import { z } from "zod";
import { llamaCpp } from "../src/index.js";

const modelPath = process.argv[2];
if (!modelPath) {
  console.error("Usage: npx tsx scripts/smoke.ts <model.gguf>");
  process.exit(1);
}

const model = llamaCpp({ modelPath });

const result = await generateText({
  model,
  tools: {
    weather: tool({
      description: "Get the current weather in a city",
      inputSchema: z.object({ city: z.string() }),
      execute: async ({ city }) => ({ city, temperature: 22, condition: "sunny" }),
    }),
  },
  stopWhen: stepCountIs(4),
  prompt: "What is the weather in Tokyo? Use the weather tool.",
});

for (const step of result.steps)
  for (const call of step.toolCalls)
    console.log(`tool call: ${call.toolName}(${JSON.stringify(call.input)})`);
console.log(`text: ${result.text}`);

await model.dispose();
