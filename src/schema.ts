// JSON Schema keywords node-llama-cpp's GBNF-JSON subset does not understand
// (see node-llama-cpp/src/utils/gbnfJson/types.ts). Structure keywords like
// type/enum/const/oneOf/$defs/properties/items/minItems/minLength ARE supported.
const UNSUPPORTED_KEYS = [
  "pattern", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
  "multipleOf", "patternProperties", "propertyNames", "dependencies",
  "dependentSchemas", "if", "then", "else", "not", "allOf",
  "minProperties", "maxProperties", "uniqueItems", "contains", "default", "examples",
] as const;

export class InvalidToolSchemaError extends Error {}

/**
 * Normalizes a JSON schema into the GBNF-JSON subset that node-llama-cpp can compile
 * into a sampling grammar.
 *
 * - `anyOf` is rewritten to `oneOf` (equivalent for generation purposes)
 * - unsupported constraint keywords are stripped (structure still constrained; the
 *   dropped predicate just isn't enforced) and reported in `warnings`
 * - object nodes missing `type` but having `properties` get `type: "object"` inferred
 * - structurally unusable nodes throw InvalidToolSchemaError
 *
 * Returns `undefined` when the schema is absent or an empty parameter-less object.
 */
export function normalizeToolParameters(
  schema: unknown,
  toolName: string
): { schema: Record<string, any> | undefined; warnings: string[] } {
  const warnings: string[] = [];

  if (schema == null) return { schema: undefined, warnings };

  if (typeof schema !== "object" || Array.isArray(schema))
    throw new InvalidToolSchemaError(
      `tools: parameters of tool "${toolName}" must be an object schema`
    );

  const cloned = structuredClone(schema) as Record<string, any>;

  // An empty {"type":"object","properties":{}} / {} means a parameter-less tool
  const props = cloned["properties"];
  if (
    (cloned["type"] === "object" || cloned["type"] == null) &&
    (props == null || Object.keys(props).length === 0) &&
    cloned["oneOf"] == null && cloned["anyOf"] == null && cloned["$ref"] == null
  )
    return { schema: undefined, warnings };

  walk(cloned, `tools["${toolName}"].parameters`, warnings);
  return { schema: cloned, warnings };
}

function walk(node: any, path: string, warnings: string[]) {
  if (node == null || typeof node !== "object" || Array.isArray(node)) return;

  if (node["anyOf"] != null && node["oneOf"] == null) {
    node["oneOf"] = node["anyOf"];
    delete node["anyOf"];
  }

  for (const key of UNSUPPORTED_KEYS) {
    if (key in node) {
      warnings.push(`${path}: stripped unsupported JSON Schema keyword "${key}"`);
      delete node[key];
    }
  }

  // nullable: true (OpenAPI style) → allow null in type
  if (node["nullable"] === true) {
    delete node["nullable"];
    if (typeof node["type"] === "string") node["type"] = [node["type"], "null"];
  }

  const hasStructure =
    node["type"] != null || node["enum"] != null || node["const"] !== undefined ||
    node["oneOf"] != null || node["$ref"] != null;
  if (!hasStructure) {
    if (node["properties"] != null) node["type"] = "object";
    else if (node["items"] != null || node["prefixItems"] != null) node["type"] = "array";
    else
      throw new InvalidToolSchemaError(
        `${path}: schema node has no usable structure (needs one of type/enum/const/oneOf/$ref)`
      );
  }

  if (node["properties"] != null)
    for (const [key, child] of Object.entries(node["properties"]))
      walk(child, `${path}.properties.${key}`, warnings);

  if (node["$defs"] != null)
    for (const [key, child] of Object.entries(node["$defs"]))
      walk(child, `${path}.$defs.${key}`, warnings);

  if (node["items"] != null && typeof node["items"] === "object" && !Array.isArray(node["items"]))
    walk(node["items"], `${path}.items`, warnings);

  if (Array.isArray(node["prefixItems"]))
    node["prefixItems"].forEach((child: any, i: number) =>
      walk(child, `${path}.prefixItems[${i}]`, warnings)
    );

  if (Array.isArray(node["oneOf"]))
    node["oneOf"].forEach((child: any, i: number) =>
      walk(child, `${path}.oneOf[${i}]`, warnings)
    );

  if (node["additionalProperties"] != null && typeof node["additionalProperties"] === "object")
    walk(node["additionalProperties"], `${path}.additionalProperties`, warnings);
}
