import {
  QwenChatWrapper,
  LlamaText,
  SpecialToken,
  SpecialTokensText,
  jsonDumps,
  type ChatModelFunctions,
  type ChatWrapperGenerateContextStateOptions,
  type ChatWrapperGeneratedContextState,
  type ChatWrapperSettings,
} from "node-llama-cpp";

/**
 * Chat wrapper for MiniCPM5 models, whose tool-call syntax no node-llama-cpp wrapper
 * speaks (the auto resolver matches the template's ChatML skeleton and returns
 * `ChatMLChatWrapper`, which documents the generic `||call:` syntax these models were
 * never trained on):
 *
 *     <function name="get_weather"><param name="city">Tokyo</param></function>
 *
 * Param values containing `<`, `&` or newlines are CDATA-wrapped, tool results come
 * back as `<tool_response>` blocks inside a user turn, and thinking uses `<think>`
 * blocks — the same ChatML message structure, result rendering, and thought
 * segmentation as Qwen 3.5, so this extends QwenChatWrapper (variation 3.5) and swaps
 * the function-call syntax.
 *
 * Like the Qwen 3.5 wrapper, *new* calls are generated as grammar-constrained JSON in
 * a single `<param name="params">` block (schema-enforced, always parses), while calls
 * in *history* are rendered in the per-key XML format above, exactly as the model's
 * chat template renders its own past calls.
 */
export class MiniCpmChatWrapper extends QwenChatWrapper {
  public override readonly wrapperName: string = "MiniCPM";
  // `declare` so this type-only redeclaration doesn't shadow the value the parent
  // constructor assigns (a plain field would be re-defined as undefined after super())
  public declare readonly settings: ChatWrapperSettings;

  public constructor() {
    super({ variation: "3.5" });

    this.settings = {
      ...this.settings,
      functions: {
        ...this.settings.functions,
        call: {
          optionalPrefixSpace: true,
          prefix: LlamaText(new SpecialTokensText('<function name="')),
          paramsPrefix: LlamaText(new SpecialTokensText('"><param name="params">')),
          suffix: LlamaText(new SpecialTokensText("</param></function>")),
          emptyCallParamsPlaceholder: {},
        },
      },
    };
  }

  public override generateContextState(
    options: ChatWrapperGenerateContextStateOptions
  ): ChatWrapperGeneratedContextState {
    const state = super.generateContextState(options);

    // The MiniCPM5 chat template begins with `{{- bos_token }}` and the GGUF sets
    // add_bos_token=false, so BOS must come from the rendered context (without it the
    // model degenerates into repetitive garbage). The Qwen wrapper this extends never
    // prepends BOS because Qwen templates don't use one.
    return {
      ...state,
      contextText: LlamaText([new SpecialToken("BOS"), state.contextText]),
    };
  }

  public override generateAvailableFunctionsSystemText(
    availableFunctions: ChatModelFunctions,
    { documentParams = true }: { documentParams?: boolean }
  ): LlamaText {
    const functionsDocumentationGenerator = new FunctionsDocumentationGenerator(
      availableFunctions
    );

    if (!functionsDocumentationGenerator.hasAnyFunctions) return LlamaText([]);

    return LlamaText.joinValues("\n", [
      "# Tools",
      "",
      LlamaText(
        "You are provided with function signatures within ",
        new SpecialTokensText("<tools></tools>"),
        " XML tags:"
      ),
      LlamaText(new SpecialTokensText("<tools>")),
      functionsDocumentationGenerator.getSignatures({ documentParams }),
      LlamaText(new SpecialTokensText("</tools>")),
      "",
      "Tool usage guidelines:",
      LlamaText([
        "- You may call zero or more functions. If no function calls are needed, just answer normally and do not include any ",
        new SpecialTokensText("<function ... </function>"),
        ".",
      ]),
      LlamaText([
        "- When calling a function, return an XML object within ",
        new SpecialTokensText("<function ... </function>"),
        " using:",
      ]),
      LlamaText([
        new SpecialTokensText('<function name="'),
        "function-name",
        new SpecialTokensText('"><param name="'),
        "param-name",
        new SpecialTokensText('">'),
        "param-value",
        new SpecialTokensText("</param></function>"),
      ]),
      LlamaText([
        "- param-value may be multi-line. If it contains <, & or newline characters, wrap it in a CDATA block: ",
        new SpecialTokensText('<param name="'),
        "param-name",
        new SpecialTokensText('"><![CDATA['),
        "...multi-line value...",
        new SpecialTokensText("]]></param>"),
      ]),
    ]);
  }

  public override generateFunctionCall(name: string, params: any): LlamaText {
    const paramBlocks: LlamaText[] = [];

    if (params != null && typeof params === "object" && !Array.isArray(params)) {
      for (const [key, value] of Object.entries(params))
        paramBlocks.push(
          LlamaText([
            new SpecialTokensText('<param name="'),
            key,
            new SpecialTokensText('">'),
            renderParamValue(value),
            new SpecialTokensText("</param>"),
          ])
        );
    } else if (params !== undefined)
      paramBlocks.push(
        LlamaText([
          new SpecialTokensText('<param name="params">'),
          jsonDumps(params),
          new SpecialTokensText("</param>"),
        ])
      );

    return LlamaText([
      new SpecialTokensText('<function name="'),
      name,
      new SpecialTokensText('">'),
      LlamaText(paramBlocks),
      new SpecialTokensText("</function>"),
    ]);
  }

  public override generateFunctionCallResult(
    functionName: string,
    functionParams: any,
    result: any
  ): LlamaText {
    // The chat template renders string tool results verbatim (only non-strings go
    // through JSON serialization).
    if (typeof result === "string")
      return LlamaText([
        this.settings.functions.result.prefix,
        result,
        this.settings.functions.result.suffix,
      ]);
    return super.generateFunctionCallResult(functionName, functionParams, result);
  }
}

/**
 * String param values containing `<`, `&` or newlines are CDATA-wrapped, matching the
 * chat template (non-string values are serialized as JSON and never wrapped, also
 * matching the template).
 */
function renderParamValue(value: unknown): LlamaText {
  if (typeof value !== "string") return LlamaText(jsonDumps(value));

  if (!/[<&\n]/.test(value)) return LlamaText(value);

  return LlamaText([
    new SpecialTokensText("<![CDATA["),
    // A literal "]]>" would end the CDATA section early; split it across two sections
    value.replaceAll("]]>", "]]]]><![CDATA[>"),
    new SpecialTokensText("]]>"),
  ]);
}

/** Renders each tool as an OpenAI-style JSON object, one per line, like the chat template's `tojson` loop. */
class FunctionsDocumentationGenerator {
  public readonly hasAnyFunctions: boolean;
  private readonly functions: ChatModelFunctions;

  public constructor(functions: ChatModelFunctions) {
    this.functions = functions ?? {};
    this.hasAnyFunctions = Object.keys(this.functions).length > 0;
  }

  public getSignatures({ documentParams = true }: { documentParams?: boolean }): string {
    return Object.entries(this.functions)
      .map(([name, definition]) =>
        jsonDumps({
          type: "function",
          function: {
            name,
            ...(definition.description != null
              ? { description: definition.description }
              : {}),
            ...(documentParams && definition.params != null
              ? { parameters: definition.params }
              : {}),
          },
        })
      )
      .join("\n");
  }
}
