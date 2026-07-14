# @enagic/llama-cpp-provider

A [Vercel AI SDK](https://sdk.vercel.ai/) provider for local GGUF models, built on
[node-llama-cpp](https://node-llama-cpp.withcat.ai/). It implements `LanguageModelV4`
and `EmbeddingModelV4`, so it works with `generateText`, `streamText`, `generateObject`,
`ToolLoopAgent`, `embed`, and the rest of the AI SDK.

Inspired by [@lgrammel/llama-cpp-provider](https://github.com/lgrammel/llama-cpp-provider),
rebuilt around one idea: **tool calls must be enforced by the sampler, not requested by
the prompt.**

## Why tool calls are reliable here

llama.cpp's `llama-server` never produces a malformed tool call because it compiles the
tool JSON schemas into a GBNF grammar and masks the token sampler with it: once the
model starts a tool call, tokens that would produce an invalid tool name or
schema-violating arguments have -inf probability. Malformed output is *unsampleable*.

node-llama-cpp contains the same machinery in TypeScript (`FunctionCallNameGrammar`,
`FunctionCallParamsGrammar`, lazy trigger detection on the chat wrapper's function-call
prefix), and this provider routes every tool-enabled call through it:

1. The resolved chat wrapper renders your tool definitions in the exact syntax the model
   was trained on (Qwen/Hermes `<tool_call>` JSON, Qwen 3.5 XML parameters, Llama 3.x,
   Gemma, Functionary, Harmony/GPT-OSS, DeepSeek, or a Jinja-template fallback for
   unknown models).
2. During generation, a prefix detector watches for the wrapper's tool-call trigger.
   Text stays unconstrained; the moment a call starts, the grammar takes over.
3. Tool calls come back as **structured results** from the engine. There is no regex, no
   JSON.parse of model text, no XML fallback parser — and no "model produced fenced JSON
   instead of a tool call" failure mode.

By contrast, `@lgrammel/llama-cpp-provider` obtains llama.cpp's lazy tool-call grammar
but installs it with the *eager* grammar sampler and drops the trigger metadata
(`grammar_lazy`, `grammar_triggers`, `preserved_tokens` are unused in its native layer),
which is why it needs three layers of fallback output parsers — and why tool calls still
drift.

Structured output (`generateObject` / `responseFormat: "json"`) is grammar-enforced the
same way, including your JSON schema.

## Other improvements

- **Cross-platform prebuilt binaries** (macOS Metal, Linux/Windows CUDA & Vulkan) via
  node-llama-cpp — no CMake, no Xcode, no source build on install.
- **KV-cache reuse across agent steps**: generation slots remember the conversation
  prefix they last evaluated, so each step of a tool loop only evaluates the new tokens.
- **Parallel generations** with `parallel: N` (one context sequence per slot, FIFO queue
  when saturated).
- **Native reasoning support**: `<think>` blocks are segmented by the chat wrapper (not
  regex) and surfaced as AI SDK reasoning parts, with a configurable thought-token
  budget.
- **Schema hygiene**: tool schemas are normalized into the GBNF-JSON subset
  (`anyOf`→`oneOf`, unsupported keywords stripped with warnings) instead of failing or
  silently misbehaving.
- **Forced tool choice** (`toolChoice: "required"` / named tool) is emulated with a
  whole-response JSON grammar, so even forced calls always parse.

## Requirements

- Node.js >= 20.
- A local GGUF model file. (`npx --no node-llama-cpp pull <hf-url>` is a handy way to
  download one.)

```bash
npm install @enagic/llama-cpp-provider ai
```

## Usage

```typescript
import { generateText, stepCountIs, tool } from "ai";
import { llamaCpp } from "@enagic/llama-cpp-provider";
import { z } from "zod";

const model = llamaCpp({ modelPath: "./models/qwen3-8b.Q4_K_M.gguf" });

const result = await generateText({
  model,
  tools: {
    weather: tool({
      description: "Get the weather in a city",
      inputSchema: z.object({ city: z.string() }),
      execute: async ({ city }) => ({ city, temperature: 22 }),
    }),
  },
  stopWhen: stepCountIs(5),
  prompt: "What is the weather in Tokyo?",
});

console.log(result.text);
await model.dispose(); // release native GPU/CPU resources
```

### Configuration

```typescript
const model = llamaCpp({
  modelPath: "./models/model.gguf", // required
  contextSize: 8192,                // default: auto-fit to available memory
  gpuLayers: 999,                   // default: offload all layers; 0 disables
  parallel: 1,                      // concurrent generation slots
  chatWrapper: "auto",              // or a node-llama-cpp wrapper name, e.g. "qwen"
  thoughtTokenBudget: undefined,    // cap reasoning tokens; 0 disables thinking
  debug: false,
});
```

Per-call options via `providerOptions`:

```typescript
await generateText({
  model,
  prompt: "...",
  providerOptions: {
    llamaCpp: {
      minP: 0.05,
      maxParallelToolCalls: 2,
      thoughtTokenBudget: 1024,
    },
  },
});
```

### Embeddings

```typescript
import { embed } from "ai";
import { llamaCpp } from "@enagic/llama-cpp-provider";

const embeddingModel = llamaCpp.embedding({
  modelPath: "./models/nomic-embed-text-v1.5.Q8_0.gguf",
});

const { embedding } = await embed({
  model: embeddingModel,
  value: "sunny day at the beach",
});

await embeddingModel.dispose();
```

## Limitations

- No multimodal (image) input — node-llama-cpp does not support projector files yet.
  Use `@lgrammel/llama-cpp-provider` or llama-server if you need vision.
- `presencePenalty` / `frequencyPenalty` are ignored (warning emitted).
- Forced tool choice bypasses the model's native tool-call syntax (the response is a
  grammar-constrained JSON object instead); prefer `toolChoice: "auto"` for best
  results.
- node-llama-cpp 3.19 can segfault during native Metal teardown *after* the process has
  finished all work (observed on macOS even in a minimal upstream-only script). All
  output is flushed before it happens and long-running processes are unaffected, but
  short CLI scripts may exit 139 instead of 0.

## Testing

```bash
npm test          # unit tests (no model required)
npm run test:e2e  # agent smoke tests; set LLAMA_TEST_MODEL=/path/to/model.gguf
npm run smoke -- /path/to/model.gguf
```

## License

MIT
