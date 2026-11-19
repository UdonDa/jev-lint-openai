import { strict as assert } from "node:assert";
import { OpenAIDecisions, OPENAI_TOKENS_PER_SECOND, decisionsUsd, OPENAI_MODELS } from "../src/openai.ts";
import { JevError } from "../src/jev.ts";
import { readAnswer } from "../src/questions.ts";
import { testAsync, test } from "./harness.ts";
import type { Question } from "../src/types.ts";

const noul = { type: "noul", instructions: { ask: "The name contradicts the body." }, criteria: { true: "Contradiction.", false: "Consistent." } } satisfies Question;
interface Wire { model: string; input: string; questions: Array<{ type: string; name: string }> }
const answer = (body: Wire, usage: Record<string, unknown> = { input_tokens: 1_000_000, output_tokens: 0 }) =>
  Response.json({ model: "gpt-6-luna", answers: body.questions.map((q) => ({ type: "predicate", name: q.name, probability: 0.9 })), usage });

await testAsync("openai: requests reach /v1/decisions in OpenAI's format and answers come back in Jev's", async () => {
  const state = { source: "function pure() { write(); }" };
  let sent: Wire | null = null;
  const client = new OpenAIDecisions({ apiKey: "fixture", retries: 0, fetch: async (url, init) => {
    assert.equal(url, "https://api.openai.com/v1/decisions");
    assert.equal(new Headers(init?.headers).get("authorization"), "Bearer fixture");
    sent = JSON.parse(String(init?.body));
    return answer(sent!);
  } });
  const result = await client.askSplitting(state, { n: noul });
  assert.deepEqual(sent!.questions.map((q) => [q.type, q.name]), [["predicate", "n"]]);
  assert.deepEqual(JSON.parse(sent!.input), state);
  assert.equal(readAnswer(result.answers, "n", "noul")?.value, 0.9);
  assert.equal(client.servedModel, "gpt-6-luna");
  assert.equal(client.spent.inputTokens, 1_000_000); assert.equal(client.spent.usd, 0.1);
});

test("openai: cached and cache-written tokens are priced at their own rates, inside input_tokens", () => {
  const price = OPENAI_MODELS["gpt-6-luna"];
  // 600k uncached at 0.10, 300k cached at 0.01, 100k written at 0.125.
  assert.equal(decisionsUsd(price, { input_tokens: 1_000_000, input_tokens_details: { cached_tokens: 300_000, cache_write_tokens: 100_000 } }).toFixed(6), "0.075500");
  for (const usage of [{}, { input_tokens: -1 }, { input_tokens: 10, input_tokens_details: { cached_tokens: 11 } }, { input_tokens: 10, input_tokens_details: [] }]) {
    assert.throws(() => decisionsUsd(price, usage), JevError);
  }
});

await testAsync("openai: no OpenAI key sends nothing, and the TypeSafe key is never borrowed", async () => {
  const previous = process.env.TYPESAFE_API_KEY;
  try {
    process.env.TYPESAFE_API_KEY = "typesafe-fixture";
    const client = new OpenAIDecisions({ fetch: async () => { throw new Error("must not send"); } });
    assert.deepEqual((await client.askSplitting({}, {})).answers, {});
    await assert.rejects(() => client.askSplitting({}, { n: noul }), (e: unknown) => e instanceof JevError && e.kind === "auth" && /OPENAI_API_KEY/.test(e.message));
    assert.equal(client.spent.calls, 0);
  } finally {
    if (previous === undefined) delete process.env.TYPESAFE_API_KEY; else process.env.TYPESAFE_API_KEY = previous;
  }
});

await testAsync("openai: a malformed reply is no verdict for the batch, and is not paid for again by a retry", async () => {
  for (const reply of [{ answers: [], usage: { input_tokens: 1 } }, { answers: [{ type: "predicate", name: "other", probability: 1 }], usage: { input_tokens: 1 } },
    { answers: [{ type: "predicate", name: "n", probability: 1 }] }, { answers: [{ type: "predicate", name: "n", probability: 1 }], usage: { input_tokens: "1" } }]) {
    let calls = 0;
    const client = new OpenAIDecisions({ apiKey: "fixture", retries: 3, fetch: async () => { calls++; return Response.json(reply); } });
    await assert.rejects(() => client.askSplitting({}, { n: noul }), JevError);
    assert.equal(calls, 1, JSON.stringify(reply));
  }
});

await testAsync("openai: more questions than one request takes are split, keeping every ID", async () => {
  const sizes: number[] = [], all = Object.fromEntries(Array.from({ length: 500 }, (_, i) => [`q${i}`, noul]));
  const client = new OpenAIDecisions({ apiKey: "fixture", fetch: async (_url, init) => {
    const body = JSON.parse(String(init?.body)) as Wire; sizes.push(body.questions.length);
    return answer(body, { input_tokens: 10, output_tokens: 0 });
  } });
  const result = await client.askSplitting({ source: "context" }, all);
  assert.deepEqual(sizes, [250, 250]);
  assert.deepEqual(Object.keys(result.answers!), Object.keys(all));
  assert.equal(client.spent.splits, 1);
});

await testAsync("openai: a state past the standard-rate input limit is refused before sending, not billed at the long-context rate", async () => {
  const client = new OpenAIDecisions({ apiKey: "fixture", fetch: async () => { throw new Error("must not send"); } });
  await assert.rejects(() => client.askSplitting({ source: "x".repeat(2_000_000) }, { n: noul }), (e: unknown) => e instanceof JevError && e.kind === "too_big");
  assert.equal(client.spent.calls, 0);
});

await testAsync("openai: a request budget settles the priced usage, not the estimate", async () => {
  let calls = 0;
  const client = new OpenAIDecisions({ apiKey: "fixture", retries: 0, budget: { maxUsd: 0.09 }, fetch: async (_url, init) => { calls++; return answer(JSON.parse(String(init?.body))); } });
  await client.askSplitting({}, { n: noul });
  await assert.rejects(() => client.askSplitting({}, { n: noul }), /budget/);
  assert.equal(calls, 1); assert.equal(client.spent.usd, 0.1);
});

test("openai: the default pace is the lowest listed tier's, so a Build key is not driven into 429s", () => {
  assert.equal(new OpenAIDecisions({ apiKey: "fixture" }).tokensPerSecond, OPENAI_TOKENS_PER_SECOND);
  assert.throws(() => new OpenAIDecisions({ model: "gpt-4o" as "gpt-6-luna" }), /unknown/);
});

await testAsync("openai: a question past OpenAI's level limit gets no verdict without sinking the rest of its batch", async () => {
  const eleven = { type: "score", instructions: { ask: "Rate." }, criteria: Array.from({ length: 11 }, (_, i) => `Level ${i}.`) } satisfies Question;
  let names: string[] = [];
  const client = new OpenAIDecisions({ apiKey: "fixture", fetch: async (_url, init) => {
    const body = JSON.parse(String(init?.body)) as Wire; names = body.questions.map((q) => q.name);
    return answer(body, { input_tokens: 10, output_tokens: 0 });
  } });
  const result = await client.askSplitting({}, { n: noul, wide: eleven });
  assert.deepEqual(names, ["n"]);
  assert.equal(readAnswer(result.answers, "n", "noul")?.value, 0.9);
  assert.equal(readAnswer(result.answers, "wide", "score", 10), null);
  let calls = 0;
  const alone = new OpenAIDecisions({ apiKey: "fixture", fetch: async () => { calls++; return Response.json({}); } });
  assert.deepEqual((await alone.askSplitting({}, { wide: eleven })).answers, {}); assert.equal(calls, 0);
});

await testAsync("openai: a billed reply that answers the wrong questions is still counted in spend", async () => {
  const client = new OpenAIDecisions({ apiKey: "fixture", retries: 0, fetch: async () =>
    Response.json({ answers: [{ type: "predicate", name: "other", probability: 1 }], usage: { input_tokens: 1_000_000, output_tokens: 0 } }) });
  await assert.rejects(() => client.askSplitting({}, { n: noul }), JevError);
  assert.equal(client.spent.usd, 0.1);
});
