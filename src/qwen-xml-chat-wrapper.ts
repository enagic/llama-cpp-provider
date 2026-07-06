import {
  QwenChatWrapper,
  LlamaText,
  SpecialTokensText,
  jsonDumps,
} from "node-llama-cpp";

/**
 * QwenChatWrapper (variation 3.5) with function calls in *history* rendered in the
 * per-key XML parameter format these models are trained on:
 *
 *     <tool_call>
 *     <function=get_weather>
 *     <parameter=city>
 *     Tokyo
 *     </parameter>
 *     </function>
 *     </tool_call>
 *
 * The stock 3.5 variation renders past calls as a single `<parameter=params>` block
 * containing JSON. Models don't recognize that as their own completed call and tend to
 * re-issue the tool call instead of using its result (verified against llama.cpp's
 * llama-server, which renders per-key parameters and gets a proper final answer).
 *
 * Generation is unchanged: new calls are still produced as grammar-constrained JSON in a
 * `<parameter=params>` block, which is schema-enforced and reliably followed. The
 * provider returns those calls to the AI SDK immediately, so the two syntaxes never mix
 * within a single generation.
 */
export class QwenXmlChatWrapper extends QwenChatWrapper {
  public constructor() {
    super({ variation: "3.5" });
  }

  public override generateFunctionCall(name: string, params: any): LlamaText {
    const parameterBlocks: LlamaText[] = [];

    if (params != null && typeof params === "object" && !Array.isArray(params)) {
      for (const [key, value] of Object.entries(params))
        parameterBlocks.push(
          LlamaText([
            new SpecialTokensText("<parameter="), key, new SpecialTokensText(">\n"),
            typeof value === "string" ? value : jsonDumps(value),
            new SpecialTokensText("\n</parameter>\n"),
          ])
        );
    } else if (params !== undefined)
      parameterBlocks.push(
        LlamaText([
          new SpecialTokensText("<parameter=params>\n"),
          jsonDumps(params),
          new SpecialTokensText("\n</parameter>\n"),
        ])
      );

    return LlamaText([
      new SpecialTokensText("<tool_call>\n<function="), name, new SpecialTokensText(">\n"),
      LlamaText(parameterBlocks),
      new SpecialTokensText("</function>\n</tool_call>"),
    ]);
  }
}
