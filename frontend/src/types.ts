// Mirrors docs/API_CONTRACT.md. Every field here is produced by a measured run;
// there are no defaults or placeholder values anywhere in this app.

export interface Health {
  status: string;
  version: string;
  warm: boolean;
  force_fast_mode: boolean;
  tokenizer: { backend: string; exact: boolean };
  embeddings: { model: string; device: string; available: boolean };
  ollama: { available: boolean; model: string; error: string | null };
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
}

export interface CompressSummary {
  source_name: string;
  detected_kind: string;
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

export interface CompressResponse {
  summary: CompressSummary;
  stages: Stage[];
  audit_trail: Array<Record<string, unknown>>;
  compressed_text: string;
  spans: Span[];
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
