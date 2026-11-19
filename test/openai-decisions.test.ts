import { strict as assert } from "node:assert";
import { decodeDecisions, encodeDecisions } from "../src/openai-decisions.ts";
import { readAnswer, readChoice } from "../src/questions.ts";
import { test } from "./harness.ts";
import type { Question } from "../src/types.ts";

const questions = {
  n: { type: "noul", instructions: { ask: "The name contradicts the body." }, criteria: { true: "Contradiction.", false: { what: "Consistent.", examples: ["pure() that only reads"] } } },
  c: { type: "choice", instructions: { ask: "Choose a starter." }, criteria: { enable: "Enable.", defer: "Defer." } },
  s: { type: "score", instructions: { ask: "Count." }, criteria: ["None.", "One.", "Two."] },
} satisfies Record<string, Question>;
const names = Object.keys(questions);
const usage = { input_tokens: 100, output_tokens: 0 };
const reply = (answers: unknown[]) => ({ model: "gpt-6-luna-2026-11-01", answers, usage });
const good = [
  { type: "predicate", name: "n", probability: 0.9 },
  { type: "choice", name: "c", choice: "enable", confidence: 0.8, probabilities: [{ value: "enable", probability: 0.8 }, { value: "defer", probability: 0.2 }] },
  { type: "score", name: "s", score: 1.6, confidence: 0.7, probabilities: [{ value: 0, label: "0", probability: 0.1 }, { value: 1, label: "1", probability: 0.2 }, { value: 2, label: "2", probability: 0.7 }] },
];

test("openai-decisions: each Jev question becomes its OpenAI type, in order, with its ID as the echoed name", () => {
  const state = { source: "function pure() { write(); }" };
  const wire = encodeDecisions("gpt-6-luna", state, questions);
  assert.equal(wire.model, "gpt-6-luna");
  assert.deepEqual(JSON.parse(wire.input), state);
  assert.deepEqual(wire.questions.map((q) => [q.type, q.name]), [["predicate", "n"], ["choice", "c"], ["score", "s"]]);
  // A predicate has no slot for criteria, so both branches must reach the model through its instructions.
  assert.deepEqual(JSON.parse(wire.questions[0]!.instructions), { ...questions.n.instructions, criteria: questions.n.criteria });
  assert.deepEqual(wire.questions[1]!.choices, [{ value: "enable", description: "Enable." }, { value: "defer", description: "Defer." }]);
  assert.deepEqual(wire.questions[2]!.levels, [{ label: "0", description: "None." }, { label: "1", description: "One." }, { label: "2", description: "Two." }]);
});

test("openai-decisions: answers decode into what the shared readers accept, on Jev's scales", () => {
  const decoded = decodeDecisions(names, reply(good));
  assert.equal(decoded.model, "gpt-6-luna-2026-11-01");
  assert.equal(readAnswer(decoded.answers, "n", "noul")?.value, 0.9);
  assert.deepEqual(readChoice(decoded.answers, "c", ["enable", "defer"]), { choice: "enable", confidence: 0.8, probabilities: { enable: 0.8, defer: 0.2 } });
  const score = readAnswer(decoded.answers, "s", "score", 2);
  assert.equal(score?.value, 1.6); assert.equal(score?.confidence, 0.7);
  assert.deepEqual(score?.probabilities, { 0: 0.1, 1: 0.2, 2: 0.7 });
});

test("openai-decisions: a refusal is no verdict for that question only", () => {
  const decoded = decodeDecisions(names, reply([{ type: "refusal", name: "n" }, good[1], good[2]]));
  assert.equal(readAnswer(decoded.answers, "n", "noul"), null);
  assert.equal(readChoice(decoded.answers, "c", ["enable", "defer"])?.choice, "enable");
  assert.equal(readAnswer(decoded.answers, "s", "score", 2)?.value, 1.6);
});

test("openai-decisions: a reply that does not answer exactly the questions asked is refused whole", () => {
  for (const bad of [
    reply([good[1], good[0], good[2]]), // reordered: would attach one subject's verdict to another
    reply(good.slice(0, 2)),
    reply([...good, good[0]]),
    reply([{ ...good[0], name: null }, good[1], good[2]]),
    { ...reply(good), answers: Object.fromEntries(good.map((a) => [a.name, a])) },
    { answers: good },
    null,
  ]) assert.throws(() => decodeDecisions(names, bad), /OpenAI/);
});

test("openai-decisions: malformed answer values become no verdict rather than a passing value", () => {
  const decoded = decodeDecisions(names, reply([
    { type: "predicate", name: "n", probability: 1.5 },
    { type: "choice", name: "c", choice: "ship", confidence: 1, probabilities: [{ value: "ship", probability: 1 }] },
    { type: "score", name: "s", score: 3, confidence: 0.5, probabilities: [{ value: 2, label: "2", probability: 1 }] },
  ]));
  assert.equal(readAnswer(decoded.answers, "n", "noul"), null);
  assert.equal(readChoice(decoded.answers, "c", ["enable", "defer"]), null);
  assert.equal(readAnswer(decoded.answers, "s", "score", 2), null);
  const mistyped = decodeDecisions(names, reply([{ ...good[2], name: "n" }, good[1], { ...good[0], name: "s" }]));
  assert.equal(readAnswer(mistyped.answers, "n", "noul"), null);
  assert.equal(readAnswer(mistyped.answers, "s", "score", 2), null);
});
