/**
 * The OpenAI Decisions wire format, translated to and from Jev's.
 *
 * Pure: no network, no credentials, no pricing. The client in `openai.ts`
 * sends what `encodeDecisions` builds and hands `decodeDecisions` what comes
 * back, so everything above it -- questions, answer readers, cache -- keeps
 * speaking Jev's shape and never learns which provider answered.
 *
 * The two formats differ in three ways that matter:
 *
 *   - Jev keys questions by ID and answers by the same ID; OpenAI takes an
 *     array and answers in the same order, echoing each question's `name`.
 *     The ID travels as `name` and the echo is checked, so a reordered or
 *     short reply cannot attach one subject's verdict to another.
 *   - Jev's `state` is JSON; OpenAI's `input` is text or message items. The
 *     state is sent as its JSON text, which is what Jev itself reads.
 *   - Jev's noul is OpenAI's `predicate`, and the criteria that Jev keeps
 *     beside the instructions become `choices` and `levels` for the other two
 *     types. A predicate has no slot for them, so they ride in its
 *     instructions.
 */
import type { Criterion, Question, SystemOneResponse } from "./types.ts";

export interface DecisionsQuestion {
  type: "predicate" | "choice" | "score";
  name: string;
  instructions: string;
  choices?: Array<{ value: string; description: string }>;
  levels?: Array<{ label: string; description: string }>;
}

export interface DecisionsRequest {
  model: string;
  input: string;
  questions: DecisionsQuestion[];
}

const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);

const describe = (criterion: Criterion): string => typeof criterion === "string" ? criterion : JSON.stringify(criterion);

/**
 * Whether OpenAI accepts this question at all: a score takes 2 to 10 levels
 * and a choice 2 to 255 choices. A rule may declare more levels than that,
 * and one such question would make OpenAI refuse its whole batch.
 */
export function fitsDecisions(question: Question): boolean {
  const n = question.type === "score" ? question.criteria.length : question.type === "choice" ? Object.keys(question.criteria).length : 2;
  return n >= 2 && n <= (question.type === "score" ? 10 : 255);
}

export function encodeQuestion(name: string, question: Question): DecisionsQuestion {
  if (question.type === "noul") {
    return { type: "predicate", name, instructions: JSON.stringify({ ...question.instructions, criteria: question.criteria }) };
  }
  const instructions = JSON.stringify(question.instructions);
  if (question.type === "choice") {
    return { type: "choice", name, instructions, choices: Object.entries(question.criteria).map(([value, description]) => ({ value, description })) };
  }
  // Level labels are the indices Jev's score already uses, lowest first, so
  // the probability-weighted `score` OpenAI returns lands on the same scale.
  return { type: "score", name, instructions, levels: question.criteria.map((description, i) => ({ label: String(i), description: describe(description) })) };
}

export function encodeDecisions(model: string, state: unknown, questions: Record<string, Question>): DecisionsRequest {
  return { model, input: JSON.stringify(state), questions: Object.entries(questions).map(([name, q]) => encodeQuestion(name, q)) };
}

/** `[{ value, probability }]` to Jev's `{ value: probability }`; anything else passes through to be rejected by the reader. */
function distribution(value: unknown): unknown {
  if (!Array.isArray(value)) return value;
  return Object.fromEntries(value.map((entry) => record(entry) ? [String(entry.value), entry.probability] : [String(entry), Number.NaN]));
}

/**
 * One OpenAI answer in Jev's shape, or null for a refusal or an unknown type.
 * Values are not bounds-checked here: `readAnswer` and `readChoice` do that
 * for every provider, cache hits and replays included.
 */
export function decodeAnswer(answer: Record<string, unknown>): Record<string, unknown> | null {
  switch (answer.type) {
    case "predicate": return { type: "noul", noul: answer.probability };
    case "choice": return { type: "choice", choice: answer.choice, confidence: answer.confidence, probabilities: distribution(answer.probabilities) };
    case "score": return { type: "score", score: answer.score, confidence: answer.confidence ?? null, probabilities: distribution(answer.probabilities) };
    default: return null;
  }
}

/** Throws on a reply that does not answer exactly the questions asked; the caller turns that into no verdict for the batch. */
export function decodeDecisions(names: readonly string[], value: unknown): SystemOneResponse & { usage: Record<string, unknown> } {
  if (!record(value) || !Array.isArray(value.answers) || !record(value.usage)) throw new Error("OpenAI returned an invalid decision envelope");
  if (value.answers.length !== names.length) throw new Error(`OpenAI returned ${value.answers.length} answers for ${names.length} questions`);
  const answers: Record<string, unknown> = {};
  value.answers.forEach((answer, i) => {
    if (!record(answer) || answer.name !== names[i]) throw new Error("OpenAI answers do not match the questions asked");
    const decoded = decodeAnswer(answer);
    if (decoded) answers[names[i]!] = decoded;
  });
  return { ...(typeof value.model === "string" ? { model: value.model } : {}), answers, usage: value.usage };
}
