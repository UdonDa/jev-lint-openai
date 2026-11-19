/**
 * OpenAI's Decisions API (`POST /v1/decisions`) as a jev-lint client.
 *
 * Retries, pacing, budgets and accounting are `Jev`'s; this class supplies
 * what differs: the endpoint and credential, the wire format (translated by
 * `openai-decisions.ts`), the price, and the limits to refuse before sending.
 */
import { estimateTokens } from "./batch.ts";
import { Jev, JevError, Pacer, envNumber, fromEnv, type JevOptions } from "./jev.ts";
import { decodeDecisions, encodeDecisions, fitsDecisions } from "./openai-decisions.ts";
import type { Question, SystemOneResponse } from "./types.ts";

export const OPENAI_ENDPOINT = "https://api.openai.com/v1/decisions";
export const OPENAI_KEY_VARS = ["OPENAI_API_KEY"] as const;
export const OPENAI_KEY_SETUP = "Get an API key at https://platform.openai.com/api-keys, then set OPENAI_API_KEY";
export const OPENAI_DEFAULT_MODEL = "gpt-6-luna";

/**
 * From the gpt-6-luna model page and the Decisions guide, read 2026-11-20.
 * USD per million input tokens; `/v1/decisions` bills no output. `questions`
 * is the documented per-request maximum. `standardInputTokens` is where the
 * long-context rate starts: past it the WHOLE request is billed at twice
 * these prices, which no estimate here would show.
 */
export const OPENAI_MODELS = {
  [OPENAI_DEFAULT_MODEL]: { inputUsdPerMillion: 0.1, cachedUsdPerMillion: 0.01, cacheWriteUsdPerMillion: 0.125, questions: 400, standardInputTokens: 272_000 },
} as const;
export type OpenAIModel = keyof typeof OPENAI_MODELS;
type ModelPrice = (typeof OPENAI_MODELS)[OpenAIModel];

/**
 * The Build tier's 2,000,000 tokens per minute, the lowest tier listed for
 * gpt-6-luna, is 33,333 per second. `Jev`'s default bucket is TypeSafe's
 * measured 200,000 per second, six times this, and a Build key would spend a
 * run's first seconds collecting 429s. The pacer still probes upward on a
 * key with a higher tier. The burst is ten seconds of that rate: a choice,
 * not a measurement, so a run's first requests do not spend the minute at once.
 */
export const OPENAI_TOKENS_PER_SECOND = 2_000_000 / 60;
export const OPENAI_TOKEN_BURST = OPENAI_TOKENS_PER_SECOND * 10;

export interface OpenAIOptions extends Omit<JevOptions, "baseUrl" | "model"> { model?: OpenAIModel }

const count = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value);

/**
 * What one reply cost. Cached and cache-written tokens are already inside
 * `input_tokens`, so they are priced at their own rates and taken out of the
 * uncached remainder rather than added to it.
 */
export function decisionsUsd(price: ModelPrice, usage: unknown): number {
  if (!record(usage) || !count(usage.input_tokens)) throw new JevError("OpenAI returned invalid token usage");
  const details = usage.input_tokens_details ?? {};
  if (!record(details)) throw new JevError("OpenAI returned invalid token usage");
  const cached = details.cached_tokens ?? 0, written = details.cache_write_tokens ?? 0;
  if (!count(cached) || !count(written) || cached + written > usage.input_tokens) throw new JevError("OpenAI returned invalid token usage");
  const uncached = usage.input_tokens - cached - written;
  return (uncached * price.inputUsdPerMillion + cached * price.cachedUsdPerMillion + written * price.cacheWriteUsdPerMillion) / 1_000_000;
}

export class OpenAIDecisions extends Jev {
  readonly nativeModel: OpenAIModel;
  private billed = 0;

  constructor({ model = OPENAI_DEFAULT_MODEL, apiKey, pacer, ...options }: OpenAIOptions = {}) {
    if (!Object.hasOwn(OPENAI_MODELS, model)) throw new JevError("unknown OpenAI decisions model");
    // "" rather than undefined: `Jev` would otherwise fall back to the TypeSafe key and send it to OpenAI.
    super({ ...options, model, apiKey: apiKey ?? fromEnv(OPENAI_KEY_VARS) ?? "",
      pacer: pacer ?? new Pacer(envNumber("JEV_LINT_TOKENS_PER_SECOND") ?? OPENAI_TOKENS_PER_SECOND, envNumber("JEV_LINT_TOKEN_BURST") ?? OPENAI_TOKEN_BURST) });
    this.nativeModel = model;
  }

  private get price(): ModelPrice { return OPENAI_MODELS[this.nativeModel]; }
  override get usd(): number { return this.billed; }
  override get endpointIdentity(): string { return OPENAI_ENDPOINT; }
  protected override estimateUsd(tokens: number): number { return tokens / 1_000_000 * this.price.inputUsdPerMillion; }
  protected override measuredUsd(_tokens: number, usage: unknown): number { return decisionsUsd(this.price, usage); }

  protected override encode(state: unknown, questions: Record<string, Question>): { url: string; body: string } {
    return { url: OPENAI_ENDPOINT, body: JSON.stringify(encodeDecisions(this.model, state, questions)) };
  }

  protected override decode(parsed: unknown, questions: Record<string, Question>): SystemOneResponse {
    // Priced before the answers are checked: a reply that answers the wrong
    // questions is no verdict, but it was still billed.
    this.billed += decisionsUsd(this.price, record(parsed) ? parsed.usage : null);
    try { return decodeDecisions(Object.keys(questions), parsed); }
    catch (err: unknown) { throw new JevError(err instanceof Error ? err.message : String(err)); }
  }

  /** Questions OpenAI would refuse are left out and get no verdict; the rest of the batch is still asked. */
  override async ask(state: unknown, all: Record<string, Question>): Promise<SystemOneResponse> {
    const questions = Object.fromEntries(Object.entries(all).filter(([, q]) => fitsDecisions(q)));
    const n = Object.keys(questions).length;
    if (!n) return { answers: {}, usage: { input_tokens: 0, output_tokens: 0 } };
    if (!this.apiKey) throw new JevError(OPENAI_KEY_SETUP, { kind: "auth" });
    // "too_big" is what makes `askSplitting` halve the question set instead of dropping the batch.
    if (n > this.price.questions || estimateTokens(encodeDecisions(this.model, state, questions)) > this.price.standardInputTokens) {
      throw new JevError("max_tokens_exceeded: over the OpenAI decisions question or standard-rate input limit", { kind: "too_big" });
    }
    return super.ask(state, questions);
  }
}
