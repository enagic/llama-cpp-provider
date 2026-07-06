import { describe, expect, it } from "vitest";
import {
  InvalidToolSchemaError,
  normalizeToolParameters,
} from "../../src/schema.js";

describe("normalizeToolParameters", () => {
  it("returns undefined for parameter-less schemas", () => {
    expect(normalizeToolParameters(undefined, "t").schema).toBeUndefined();
    expect(normalizeToolParameters({}, "t").schema).toBeUndefined();
    expect(
      normalizeToolParameters({ type: "object", properties: {} }, "t").schema
    ).toBeUndefined();
  });

  it("rewrites anyOf to oneOf", () => {
    const { schema } = normalizeToolParameters(
      {
        type: "object",
        properties: {
          value: { anyOf: [{ type: "string" }, { type: "number" }] },
        },
      },
      "t"
    );
    expect(schema!.properties.value.oneOf).toHaveLength(2);
    expect(schema!.properties.value.anyOf).toBeUndefined();
  });

  it("strips unsupported keywords and reports them", () => {
    const { schema, warnings } = normalizeToolParameters(
      {
        type: "object",
        properties: {
          age: { type: "integer", minimum: 0, maximum: 150 },
        },
      },
      "t"
    );
    expect(schema!.properties.age).toEqual({ type: "integer" });
    expect(warnings).toHaveLength(2);
  });

  it("converts nullable to a null type union", () => {
    const { schema } = normalizeToolParameters(
      {
        type: "object",
        properties: { name: { type: "string", nullable: true } },
      },
      "t"
    );
    expect(schema!.properties.name.type).toEqual(["string", "null"]);
  });

  it("infers object/array types from structure", () => {
    const { schema } = normalizeToolParameters(
      {
        type: "object",
        properties: {
          obj: { properties: { a: { type: "string" } } },
          arr: { items: { type: "number" } },
        },
      },
      "t"
    );
    expect(schema!.properties.obj.type).toBe("object");
    expect(schema!.properties.arr.type).toBe("array");
  });

  it("throws on structurally unusable nodes", () => {
    expect(() =>
      normalizeToolParameters(
        { type: "object", properties: { bad: { description: "???" } } },
        "t"
      )
    ).toThrow(InvalidToolSchemaError);
  });

  it("does not mutate the input schema", () => {
    const input = {
      type: "object",
      properties: { x: { type: "string", pattern: "^a" } },
    };
    normalizeToolParameters(input, "t");
    expect(input.properties.x.pattern).toBe("^a");
  });
});
