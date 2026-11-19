import { strict as assert } from "node:assert";
import { createModelClient, modelSpec, modelCachePath } from "../src/model.ts";
import { OpenAIDecisions } from "../src/openai.ts";
import { OpenRouter } from "../src/openrouter.ts";
import { test } from "./harness.ts";

test("model: an explicit native prefix selects OpenRouter while the Jev default is preserved", () => {
  const selected = "openrouter:perplexity/pplx-decider-v1.3:free";
  assert.equal(modelSpec(selected).provider, "openrouter");
  assert.equal(modelSpec(selected).model, "perplexity/pplx-decider-v1.3:free");
  assert.equal(modelSpec(selected).inputUsdPerMillion, 0);
  assert.ok(createModelClient({ model: selected, apiKey: "fixture" }) instanceof OpenRouter);
  assert.equal(modelSpec("jev-1.14.0").provider, "jev");
});

test("model: native caches cannot read Jev verdicts or another model's cache", () => {
  const a = modelCachePath("cache.json", "openrouter:liquid/d1.2-ultra");
  assert.notEqual(a, "cache.json");
  assert.notEqual(a, modelCachePath("cache.json", "openrouter:perplexity/pplx-decider-v1.3:free"));
  assert.equal(modelCachePath("cache.json", "jev-1.14.0"), "cache.json");
  assert.equal(modelCachePath(null, "openrouter:liquid/d1.2-ultra"), null);
});

test("model: invalid native IDs and endpoint overrides fail before requests", () => {
  assert.throws(() => modelSpec("openrouter:unknown/model"), /unknown.*model/i);
  assert.throws(() => createModelClient({ model: "openrouter:liquid/d1.2-ultra", baseUrl: "https://example.com", apiKey: "fixture" }), /base.url|endpoint/i);
});

test("model: ambient router settings cannot redirect an explicitly supplied direct API key", () => {
  const previous = process.env.JEV_LINT_MODEL, routerKey = process.env.OPENROUTER_API_KEY;
  try {
    process.env.JEV_LINT_MODEL = "openrouter:typesafe/jev-1.14";
    process.env.OPENROUTER_API_KEY = "router-fixture";
    assert.ok(!(createModelClient({ apiKey: "direct-fixture" }) instanceof OpenRouter));
    assert.ok(createModelClient({ model: "openrouter:typesafe/jev-1.14", apiKey: "router-fixture" }) instanceof OpenRouter);
  } finally {
    if (previous === undefined) delete process.env.JEV_LINT_MODEL; else process.env.JEV_LINT_MODEL = previous;
    if (routerKey === undefined) delete process.env.OPENROUTER_API_KEY; else process.env.OPENROUTER_API_KEY = routerKey;
  }
});

test("model: openai:<id> selects OpenAI explicitly, and an OpenAI key alone never does", () => {
  const previous = process.env.OPENAI_API_KEY;
  try {
    process.env.OPENAI_API_KEY = "openai-fixture";
    assert.equal(modelSpec().provider, "jev");
    assert.ok(!(createModelClient({}) instanceof OpenAIDecisions));
    const spec = modelSpec("openai:gpt-6-luna");
    assert.deepEqual([spec.provider, spec.model, spec.inputUsdPerMillion, spec.keyAvailable, spec.implicitOpenRouter], ["openai", "gpt-6-luna", 0.1, true, false]);
    assert.ok(createModelClient({ model: "openai:gpt-6-luna" }) instanceof OpenAIDecisions);
  } finally {
    if (previous === undefined) delete process.env.OPENAI_API_KEY; else process.env.OPENAI_API_KEY = previous;
  }
});

test("model: OpenAI verdicts are cached apart from Jev's and OpenRouter's, whose paths do not move", () => {
  assert.equal(modelCachePath("cache.json", "openrouter:liquid/d1.2-ultra"), "cache.openrouter-1a5424cf3b38.json");
  const openai = modelCachePath("cache.json", "openai:gpt-6-luna");
  assert.match(openai!, /^cache\.openai-[0-9a-f]{12}\.json$/);
});

test("model: unknown OpenAI IDs, endpoint overrides and ambient OpenAI selection with a direct key fail safe", () => {
  assert.throws(() => modelSpec("openai:gpt-4o"), /unknown OpenAI decisions model/);
  assert.throws(() => createModelClient({ model: "openai:gpt-6-luna", baseUrl: "https://example.com", apiKey: "fixture" }), /OpenAI endpoint/);
  const previous = process.env.JEV_LINT_MODEL;
  try {
    process.env.JEV_LINT_MODEL = "openai:gpt-6-luna";
    assert.ok(!(createModelClient({ apiKey: "direct-fixture" }) instanceof OpenAIDecisions));
  } finally {
    if (previous === undefined) delete process.env.JEV_LINT_MODEL; else process.env.JEV_LINT_MODEL = previous;
  }
});
