import { createHash } from "node:crypto";
import { extname } from "node:path";
import { Jev, API_KEY_VARS, API_KEY_SETUP, DEFAULT_MODEL, USD_PER_MTOK, fromEnv, type JevOptions, type AskClient } from "./jev.ts";
import { OpenAIDecisions, OPENAI_ENDPOINT, OPENAI_KEY_SETUP, OPENAI_KEY_VARS, OPENAI_MODELS, type OpenAIModel } from "./openai.ts";
import { OpenRouter, OPENROUTER_MODELS, OPENROUTER_JEV, OPENROUTER_ENDPOINT, OPENROUTER_KEY_SETUP, type OpenRouterModel } from "./openrouter.ts";

export type Provider = "jev" | "openrouter" | "openai";
export interface ModelSpec { provider: Provider; model: string; implicitOpenRouter: boolean; inputUsdPerMillion: number; keyAvailable: boolean; setup: string }

/** The providers selected by a `<prefix>:<id>` model name. Only OpenRouter is also inferred, from its key. */
const NATIVE = {
  openrouter: { endpoint: OPENROUTER_ENDPOINT, models: OPENROUTER_MODELS as Record<string, { inputUsdPerMillion: number }>, keyVars: ["OPENROUTER_API_KEY"], setup: OPENROUTER_KEY_SETUP, name: "OpenRouter" },
  openai: { endpoint: OPENAI_ENDPOINT, models: OPENAI_MODELS as Record<string, { inputUsdPerMillion: number }>, keyVars: OPENAI_KEY_VARS, setup: OPENAI_KEY_SETUP, name: "OpenAI" },
} as const;
type Native = keyof typeof NATIVE;
const prefixed = (model: string): Native | null =>
  (Object.keys(NATIVE) as Native[]).find((p) => model.startsWith(`${p}:`)) ?? null;

export function modelSpec(requested: string | null = null, baseUrl: string | null = null): ModelSpec {
  const model = requested ?? process.env.JEV_LINT_MODEL ?? DEFAULT_MODEL;
  const explicit = prefixed(model);
  const provider: Native | null = explicit ?? (model === DEFAULT_MODEL && fromEnv(NATIVE.openrouter.keyVars) ? "openrouter" : null);
  if (provider === null) return { provider: "jev", model, implicitOpenRouter: false, inputUsdPerMillion: USD_PER_MTOK, keyAvailable: !!fromEnv(API_KEY_VARS), setup: API_KEY_SETUP };
  const native = NATIVE[provider], id = explicit ? model.slice(provider.length + 1) : OPENROUTER_JEV;
  if (!Object.hasOwn(native.models, id)) throw new Error(`unknown ${native.name} decisions model: ${id}`);
  if (baseUrl && baseUrl.replace(/\/+$/, "") !== native.endpoint) throw new Error(`--base-url cannot override the ${native.name} endpoint`);
  return { provider, model: id, implicitOpenRouter: explicit === null, inputUsdPerMillion: native.models[id]!.inputUsdPerMillion,
    keyAvailable: !!fromEnv(native.keyVars), setup: native.setup };
}

export function createModelClient(options: JevOptions = {}): AskClient {
  const requested = options.model ?? process.env.JEV_LINT_MODEL ?? DEFAULT_MODEL;
  // An explicit direct-client credential must never follow an ambient native-provider selection.
  if (options.apiKey != null && !prefixed(options.model ?? "")) {
    return new Jev({ ...options, model: prefixed(requested) ? DEFAULT_MODEL : requested });
  }
  const spec = modelSpec(requested, options.baseUrl);
  if (spec.provider === "jev") return new Jev({ ...options, model: spec.model });
  const { model: _model, baseUrl: _baseUrl, ...rest } = options;
  if (spec.provider === "openai") return new OpenAIDecisions({ ...rest, model: spec.model as OpenAIModel });
  return new OpenRouter({ ...rest, model: spec.model as OpenRouterModel });
}

/** Keep all existing Jev cache locations; native backends use a separate model-scoped file. */
export function modelCachePath(path: string | null, model: string | null = null): string | null {
  const spec = modelSpec(model);
  if (!path || spec.provider === "jev") return path;
  const suffix = createHash("sha256").update(`native-v1\0${NATIVE[spec.provider].endpoint}\0${spec.model}`).digest("hex").slice(0, 12);
  const extension = extname(path);
  return `${extension ? path.slice(0, -extension.length) : path}.${spec.provider}-${suffix}${extension}`;
}
