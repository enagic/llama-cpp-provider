import path from "node:path";
import {
  getLlama,
  resolveChatWrapper,
  resolvableChatWrapperTypeNames,
  QwenChatWrapper,
  LlamaLogLevel,
  type Llama,
  type LlamaModel,
  type LlamaContext,
  type ChatWrapper,
} from "node-llama-cpp";
import type { LlamaCppProviderConfig } from "./config.js";
import { QwenXmlChatWrapper } from "./qwen-xml-chat-wrapper.js";
import { SlotPool } from "./slot-pool.js";

export interface Engine {
  llama: Llama;
  model: LlamaModel;
  context: LlamaContext;
  chatWrapper: ChatWrapper;
  slots: SlotPool;
  modelId: string;
  dispose(): Promise<void>;
}

export function modelIdFromPath(modelPath: string): string {
  return path.basename(modelPath).replace(/\.gguf$/i, "");
}

function resolveWrapper(
  model: LlamaModel,
  wrapperType: LlamaCppProviderConfig["chatWrapper"]
): ChatWrapper {
  if (wrapperType != null && wrapperType !== "auto") {
    if (!(resolvableChatWrapperTypeNames as readonly string[]).includes(wrapperType))
      throw new Error(
        `Unknown chatWrapper "${wrapperType}". Valid values: ${resolvableChatWrapperTypeNames.join(", ")}`
      );
    if (wrapperType === "qwen")
      return resolveChatWrapper(model, {
        type: "qwen",
        customWrapperSettings: { qwen: { variation: sniffQwenVariation(model) } },
      });
    return resolveChatWrapper(model, { type: wrapperType });
  }

  const auto = resolveChatWrapper(model);

  // The auto resolver falls back to the generic Jinja wrapper when a fine-tune's
  // template doesn't exactly match a specialized wrapper. The generic wrapper documents
  // node-llama-cpp's own function-call syntax, which tool-tuned models often ignore in
  // favor of the syntax they were trained on. Detect known native syntaxes in the
  // template and prefer the specialized wrapper that grammar-constrains that syntax.
  if (auto.constructor.name === "JinjaTemplateChatWrapper") {
    const template = model.fileInfo.metadata?.tokenizer?.chat_template ?? "";
    if (
      template.includes("<tool_call>") &&
      template.includes("<function=") &&
      template.includes("<parameter=")
    )
      return new QwenXmlChatWrapper();
    if (template.includes("<tool_call>") && template.includes("tojson"))
      return new QwenChatWrapper({ variation: "3" });
  }

  return auto;
}

function sniffQwenVariation(model: LlamaModel): "3" | "3.5" {
  const template = model.fileInfo.metadata?.tokenizer?.chat_template ?? "";
  return template.includes("<function=") ? "3.5" : "3";
}

export async function createEngine(config: LlamaCppProviderConfig): Promise<Engine> {
  const llama = await getLlama({
    logLevel: config.debug ? LlamaLogLevel.debug : LlamaLogLevel.warn,
  });

  const model = await llama.loadModel({
    modelPath: config.modelPath,
    gpuLayers: config.gpuLayers === "auto" ? undefined : config.gpuLayers,
  });

  const chatWrapper = resolveWrapper(model, config.chatWrapper);

  const context = await model.createContext({
    sequences: config.parallel ?? 1,
    contextSize: config.contextSize,
  });

  const slots = new SlotPool(context, chatWrapper);

  return {
    llama,
    model,
    context,
    chatWrapper,
    slots,
    modelId: modelIdFromPath(config.modelPath),
    async dispose() {
      await slots.dispose();
      await context.dispose();
      await model.dispose();
      await llama.dispose();
    },
  };
}
