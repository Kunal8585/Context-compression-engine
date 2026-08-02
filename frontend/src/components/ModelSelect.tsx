import type { ProviderCatalogue, SelectionPreset } from "../types";

/**
 * One dropdown: which model runs this compression.
 *
 * A compression actually uses two models — one to embed for clustering, one to
 * rewrite — and no single vendor supplies both for every choice (Groq has no
 * embeddings API, Cohere has no chat API). Rather than make that the user's
 * problem with two controls, each entry here names the model being chosen and
 * the backend resolves the pairing. The line under the dropdown states which
 * embedding provider came with it, so the pairing is disclosed rather than
 * hidden.
 *
 * Options come from `GET /providers`, so a model added to config.yaml appears
 * here with no frontend change, and one that cannot run is disabled carrying
 * the backend's own reason.
 */
export function ModelSelect({
  catalogue,
  value,
  onChange,
  disabled,
  compact = false,
}: {
  catalogue: ProviderCatalogue | undefined;
  value: string;
  onChange: (id: string) => void;
  disabled?: boolean;
  /** Inline in the workspace toolbar: no heading, no helper line. The
   *  resolved-provider detail still reaches the user, on the result banner
   *  after the run, where it describes what actually happened rather than
   *  what was requested. */
  compact?: boolean;
}) {
  const presets = catalogue?.selections ?? [];
  const selected = presets.find((p) => p.id === value);

  return (
    <label className="block">
      {!compact && (
        <span className="mb-1 flex items-baseline gap-2">
          <span className="text-sm font-medium text-neutral-300">Model</span>
          <span className="text-[10px] uppercase tracking-wide text-neutral-600">
            embeddings + rewriting
          </span>
        </span>
      )}

      <select
        aria-label="Model"
        disabled={disabled || presets.length === 0}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`w-full rounded border border-neutral-800 bg-neutral-950 px-3 text-sm text-neutral-100 outline-none focus:border-neutral-600 disabled:opacity-50 ${
          compact ? "py-1.5" : "py-2 sm:max-w-md"
        }`}
      >
        {presets.length === 0 && <option value="auto">Loading models…</option>}
        {presets.map((preset) => (
          <option
            key={preset.id}
            value={preset.id}
            disabled={!preset.available}
          >
            {preset.label}
            {preset.available ? "" : ` — ${preset.reason ?? "unavailable"}`}
          </option>
        ))}
      </select>

      {!compact && (
      <span className="mt-1 block text-xs text-neutral-600">
        {selected ? (
          selected.available ? (
            <>
              {selected.detail}
              {selected.id !== "auto" && (
                <span className="text-neutral-700">
                  {" "}
                  · pinned, no fallback if it fails
                </span>
              )}
            </>
          ) : (
            <span className="text-amber-500/80">
              {selected.reason ?? "not configured"}
            </span>
          )
        ) : (
          <>choose which model compresses this input</>
        )}
      </span>
      )}
    </label>
  );
}

/** Human label for a provider name reported in `providers_used`.
 *
 *  Undefined is a real, explainable state rather than a mystery: the stage
 *  either had nothing to do (a single-chunk input skips the embedding pass) or
 *  its pinned provider failed and, being pinned, had no fallback. Saying
 *  "unknown" would invite the reader to assume a bug. */
export function providerLabel(provider: string | undefined): string {
  if (!provider) return "none";
  const names: Record<string, string> = {
    local: "Llama 3.2 3B (local)",
    groq: "Groq (cloud)",
    gemini: "Gemini (cloud)",
    openai: "OpenAI (cloud)",
    cohere: "Cohere (cloud)",
    openrouter: "OpenRouter (cloud)",
  };
  return names[provider] ?? provider;
}

/** The request fields a preset resolves to. */
export function presetToPins(preset: SelectionPreset | undefined) {
  return {
    mode: preset?.mode ?? "auto",
    embedding_provider: preset?.embedding_provider ?? null,
    generation_provider: preset?.generation_provider ?? null,
  };
}
