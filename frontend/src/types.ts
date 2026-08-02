// Mirrors docs/API_CONTRACT.md. Every field here is produced by a measured run;
// there are no defaults or placeholder values anywhere in this app.

/** One provider's readiness. `configured` never implies anything about a key
 *  beyond its presence — /health reports booleans and never a value. */
export interface ProviderInfo {
  provider: string;
  model: string;
  configured: boolean;
  requires_key: string | null;
  reason: string | null;
  device?: string;
  host?: string;
}

/** A whole fallback chain: what it would try, and what it is using now. */
export interface ChainInfo {
  role: string;
  chain: string[];
  providers: ProviderInfo[];
  /** First entry that could serve right now, or null if the chain is exhausted. */
  active: string | null;
  last_used: string | null;
}

export interface ProviderStatus {
  offline_mode: boolean;
  /** Presence only. Never a key, never a fragment of one. */
  keys_configured: Record<string, boolean>;
  keys_missing: string[];
  embedding: ChainInfo;
  generation: ChainInfo;
  any_generation_available: boolean;
  any_embedding_available: boolean;
}

export type Mode = "local" | "cloud" | "auto";

/** One selectable model. `configured` is key presence only — never a key. */
export interface ProviderOption {
  provider: string;
  model: string;
  role: "embedding" | "generation";
  local: boolean;
  configured: boolean;
  reason: string | null;
  requires_key: string | null;
}

/** One entry in the model dropdown, already resolved by the backend into a
 *  coherent (embedding, generation) pair. */
export interface SelectionPreset {
  id: string;
  label: string;
  /** Which embedding provider this choice resolves to — disclosed, not hidden. */
  detail: string;
  mode: Mode;
  embedding_provider: string | null;
  generation_provider: string | null;
  local: boolean;
  available: boolean;
  reason: string | null;
}

export interface ProviderCatalogue {
  embedding: ProviderOption[];
  generation: ProviderOption[];
  selections: SelectionPreset[];
  default_chains: { embedding: string[]; generation: string[] };
  modes: { available: Mode[]; default: Mode };
  note: string;
}

/** Whether one execution mode can actually run, answered independently.
 *  Ollama being down must not make cloud look broken, and vice versa. */
export interface ModeReadiness {
  mode: string;
  ready: boolean;
  reason: string | null;
  embedding: {
    ready: boolean;
    reason?: string | null;
    providers: string[];
    active: string | null;
  };
  generation: {
    ready: boolean;
    reason?: string | null;
    providers: string[];
    active: string | null;
  };
}

export interface Health {
  status: string;
  version: string;
  warm: boolean;
  force_fast_mode: boolean;
  modes: {
    available: Mode[];
    default: Mode;
    local: ModeReadiness;
    cloud: ModeReadiness;
  };
  tokenizer: { backend: string; exact: boolean };
  embeddings: {
    model: string;
    device: string;
    available: boolean;
    provider: string | null;
    chain: string[];
  };
  providers: ProviderStatus;
  generation: {
    available: boolean;
    model: string;
    chain: string[];
    error: string | null;
  };
  capabilities: Record<string, boolean>;
}

/** One stage of the pipeline. Always six, in fixed order. */
export interface Stage {
  name: string;
  status: "ok" | "skipped" | "failed";
  duration_ms: number;
  chunks_in: number;
  chunks_out: number;
  tokens_in: number;
  tokens_out: number;
  tokens_removed: number;
  reduction_pct: number;
  /** Present whenever status is "skipped" - guaranteed by a backend test. */
  note: string | null;
  /** Which chain entry actually served this stage; null for stages that call
   *  no model. Names the provider that answered, not the one tried first. */
  provider_used: string | null;
  details: Record<string, unknown>;
}

/** A character range over the ORIGINAL text the client submitted. */
export interface Span {
  start: number;
  end: number;
  kept: boolean;
  count: number;
  tokens: number;
  kind: string;
  symbol: string | null;
  density: number | null;
  reason: string | null;
  duplicate_count: number;
  source_file?: string;
}

export interface CompressSummary {
  source_name: string;
  detected_kind: string;
  /** Mode this run actually executed under. */
  mode: Mode;
  /** stage name -> provider that served it, e.g. {redundancy: "gemini"}. */
  providers_used: Record<string, string>;
  original_tokens: number;
  compressed_tokens: number;
  tokens_saved: number;
  compression_ratio: number;
  compression_pct: number;
  budget_tokens: number;
  budget_ratio: number;
  chunks_total: number;
  chunks_kept: number;
  chunks_dropped: number;
  total_ms: number;
  tokenizer_exact: boolean;
}

/** Per-file outcome of a multi-file upload. A skip is normal, not an error. */
export interface FileStatus {
  name: string;
  status: "done" | "skipped";
  reason: string | null;
  characters: number;
  /** Present for PDFs only. */
  pages?: number;
}

/** Measured trustworthiness of one compression. Deterministic — no model. */
export interface ConfidenceReport {
  score: number;
  band: "high" | "moderate" | "low" | "unknown";
  components: Record<string, number>;
  /** What is dragging the score down, naming specifics. */
  reasons: string[];
  evidence: Record<string, unknown>;
  method: string;
}

/** One omission in the compressed text, addressable via /expand. */
export interface MarkerRef {
  id: string;
  kind: "dropped" | "collapsed";
  sections: number;
  tokens: number;
  start_line?: number;
  end_line?: number;
  start_char?: number;
  end_char?: number;
}

/** What an omission actually contained. */
export interface Expansion {
  compression_id: string;
  marker_id: string;
  kind: string;
  sections: number;
  tokens: number;
  text: string;
  chunks: Array<{
    id: string;
    kind: string;
    symbol: string | null;
    tokens: number;
    start_line: number;
    end_line: number;
    text: string;
  }>;
}

export interface CompressResponse {
  summary: CompressSummary;
  confidence: ConfidenceReport | null;
  compression_id: string;
  markers: MarkerRef[];
  stages: Stage[];
  audit_trail: Array<Record<string, unknown>>;
  compressed_text: string;
  spans: Span[];
  files?: FileStatus[];
}

export interface EvaluateAggregate {
  items: number;
  compression_ratio: number;
  tokens_before: number;
  tokens_after: number;
  cost_before_usd: number;
  cost_after_usd: number;
  cost_reduction: number;
  latency_before_ms: number;
  latency_after_ms: number;
  latency_speedup: number;
  accuracy_before: number;
  accuracy_after: number;
  accuracy_retention: number;
  /** The compressor's own ceiling - measured with no model involved. */
  fact_survival_rate: number;
}

export interface EvaluateResponse {
  report_id: string;
  cached: boolean;
  config: {
    downstream_model: string;
    accuracy_metric: string;
    budget_ratio: number;
    pricing_model: string;
  };
  aggregate: EvaluateAggregate;
  fact_survival: {
    overall: number;
    facts_surviving: number;
    facts_total: number;
    by_context: Record<string, { surviving: number; total: number; rate: number }>;
  };
  diagnostic?: { label: string; available: boolean; reason?: string; verdict?: string };
}

export interface ApiError {
  error: { code: string; message: string; detail?: Record<string, unknown> };
}

/** One arm of the /answer race: the same question against one context. */
export interface AnswerArm {
  answer: string;
  error: string | null;
  tokens_in: number;
  tokens_out: number;
  latency_ms: number;
  cost_usd: number;
  /** The prompt was refused for size, not a transient provider failure. */
  too_large: boolean;
}

export interface AnswerResponse {
  /** The full prompt was rejected as too large while the compressed one
   *  answered — compression was the difference between an answer and none. */
  unlocked: boolean;
  question: string;
  model: string;
  mode: Mode;
  providers_used: Record<string, string>;
  full: AnswerArm;
  compressed: AnswerArm;
  delta: {
    tokens_saved: number;
    compression_pct: number;
    cost_saved_usd: number;
    cost_reduction_pct: number;
    speedup: number;
    pricing_model: string;
    full_context_rejected: boolean;
  };
  confidence: ConfidenceReport | null;
  compressed_text: string;
  note: string;
}
