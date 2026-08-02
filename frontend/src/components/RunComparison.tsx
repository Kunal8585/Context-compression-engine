import type { CompressResponse, Mode } from "../types";
import { providerLabel } from "./ModelSelect";

export interface RunRecord {
  mode: Mode;
  /** Stage 3. Runs on every compression. */
  embeddingProvider: string | undefined;
  /** Stage 6. Undefined whenever the stage was skipped, which is common. */
  generationProvider: string | undefined;
  /** Why stage 6 did not run, when it did not. */
  generationSkipped: string | null;
  wallMs: number;
  originalTokens: number;
  compressedTokens: number;
  compressionPct: number;
  confidence: number | null;
}

export function toRunRecord(
  result: CompressResponse,
  wallMs: number,
): RunRecord {
  const used = result.summary.providers_used ?? {};
  const stage6 = result.stages.find((s) => s.name === "abstractive");
  return {
    mode: result.summary.mode,
    embeddingProvider: used.redundancy,
    // Reported separately from embeddings rather than collapsed into one
    // "provider". They are routinely different providers, and stage 6 is
    // frequently skipped entirely - showing only the embedding provider made
    // a working Groq look like it was never configured.
    generationProvider: used.abstractive,
    generationSkipped:
      !used.abstractive && stage6?.status === "skipped"
        ? (stage6.note ?? "stage 6 did not run")
        : null,
    wallMs,
    originalTokens: result.summary.original_tokens,
    compressedTokens: result.summary.compressed_tokens,
    compressionPct: result.summary.compression_pct,
    confidence: result.confidence?.score ?? null,
  };
}

/**
 * Side-by-side of the two most recent runs.
 *
 * The point of the mode toggle is the comparison, and a comparison you have to
 * hold in your head is not one you can show an audience. This keeps the
 * previous run on screen so switching Local → Cloud and re-running produces a
 * visible delta rather than a replaced set of numbers.
 *
 * It renders only when the two runs actually used *different providers*. The
 * gate is on providers rather than mode because most model choices resolve to
 * mode `auto` — comparing Groq against Gemini is the interesting case and both
 * are `auto`, so gating on mode would have hidden exactly the comparison this
 * panel exists for.
 */
export function RunComparison({
  current,
  previous,
}: {
  current: RunRecord;
  previous: RunRecord | null;
}) {
  const samePair =
    previous &&
    previous.embeddingProvider === current.embeddingProvider &&
    previous.generationProvider === current.generationProvider;
  if (!previous || samePair) return null;

  const faster = current.wallMs < previous.wallMs;
  const deltaMs = Math.abs(current.wallMs - previous.wallMs);
  const speedup =
    Math.max(current.wallMs, previous.wallMs) /
    Math.max(1, Math.min(current.wallMs, previous.wallMs));

  return (
    <section className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4">
      <h2 className="mb-3 text-sm font-medium text-neutral-300">
        This run vs previous
        <span className="ml-2 font-normal text-neutral-500">
          {previous.generationProvider ?? previous.embeddingProvider ?? "?"} →{" "}
          {current.generationProvider ?? current.embeddingProvider ?? "?"}
        </span>
      </h2>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <RunCard record={previous} label="Previous" muted />
        <RunCard record={current} label="This run" />
      </div>

      <p className="mt-3 text-xs text-neutral-500">
        <span className={faster ? "text-emerald-400" : "text-amber-400"}>
          {faster ? "Faster" : "Slower"} by {(deltaMs / 1000).toFixed(1)}s
        </span>{" "}
        ({speedup.toFixed(1)}× {faster ? "speedup" : "slowdown"}) · compression{" "}
        {previous.compressionPct.toFixed(1)}% → {current.compressionPct.toFixed(1)}%
        {current.compressedTokens !== previous.compressedTokens && (
          <>
            {" "}
            · different embedding models cluster differently, so token counts
            need not match
          </>
        )}
      </p>
    </section>
  );
}

function RunCard({
  record,
  label,
  muted,
}: {
  record: RunRecord;
  label: string;
  muted?: boolean;
}) {
  return (
    <div
      className={`rounded border p-3 ${
        muted
          ? "border-neutral-800 bg-neutral-950/40 opacity-70"
          : "border-neutral-700 bg-neutral-950"
      }`}
    >
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium text-neutral-400">{label}</span>
        <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-neutral-400">
          {record.embeddingProvider === "local" ? "local" : "cloud"}
        </span>
      </div>
      <p className="mt-1 font-mono text-[11px] leading-relaxed text-neutral-500">
        embed {providerLabel(record.embeddingProvider)}
        <br />
        gen{" "}
        {record.generationProvider
          ? providerLabel(record.generationProvider)
          : "not used"}
      </p>
      <dl className="mt-2 grid grid-cols-3 gap-2 text-center">
        <Metric label="wall" value={`${(record.wallMs / 1000).toFixed(1)}s`} />
        <Metric label="smaller" value={`${record.compressionPct.toFixed(0)}%`} />
        <Metric
          label="confidence"
          value={
            record.confidence === null
              ? "—"
              : `${Math.round(record.confidence * 100)}%`
          }
        />
      </dl>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dd className="font-mono text-sm text-neutral-200">{value}</dd>
      <dt className="text-[10px] uppercase tracking-wide text-neutral-600">
        {label}
      </dt>
    </div>
  );
}
