import { useEffect, useState } from "react";
import type { AnswerResponse } from "../types";

/**
 * The payoff, side by side.
 *
 * A compression ratio is an abstraction; "the small prompt answered the same
 * question, 37× faster, for 69% less" is not. This races the *same question*
 * against the *same model* twice — full context vs compressed — and shows both
 * answers next to each other so the trade-off is visible rather than asserted.
 *
 * The timers run live while the request is in flight. That matters for a demo:
 * the compressed pane visibly finishes while the full pane is still going,
 * which is the entire argument for the project made in one second of watching.
 *
 * Where the two answers differ, that is shown too. A tool that hid the cases
 * where compression cost an answer would not be worth trusting on the cases
 * where it did not.
 */
export function AnswerRace({
  data,
  pending,
}: {
  data: AnswerResponse | undefined;
  pending: boolean;
}) {
  const elapsed = useElapsed(pending);
  if (!pending && !data) return null;

  return (
    <section className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-sm font-medium text-neutral-300">
          Same question, same model, both contexts
        </h2>
        {data && (
          <span className="font-mono text-xs text-neutral-500">
            {data.model}
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <Pane
          label="Full context"
          tone="neutral"
          pending={pending}
          elapsed={elapsed}
          arm={data?.full}
        />
        <Pane
          label="Compressed"
          tone="emerald"
          pending={pending}
          elapsed={elapsed}
          arm={data?.compressed}
        />
      </div>

      {data?.unlocked && (
        <p className="mt-3 rounded border border-emerald-800 bg-emerald-950/30 p-3 text-sm leading-relaxed text-emerald-200">
          <strong>The full prompt was rejected as too large.</strong> At{" "}
          {data.full.tokens_in.toLocaleString()} tokens it exceeds what the
          model accepts, so the uncompressed context has no answer at any
          price. The compressed prompt fit and answered — here compression is
          not an optimisation, it is the difference between a result and none.
        </p>
      )}

      {data && <Verdict data={data} />}
    </section>
  );
}

/** Ticks while a request is in flight so the race is watchable, not inferred. */
function useElapsed(running: boolean) {
  const [ms, setMs] = useState(0);
  useEffect(() => {
    if (!running) return;
    setMs(0);
    const started = performance.now();
    const id = setInterval(() => setMs(performance.now() - started), 50);
    return () => clearInterval(id);
  }, [running]);
  return ms;
}

function Pane({
  label,
  tone,
  pending,
  elapsed,
  arm,
}: {
  label: string;
  tone: "neutral" | "emerald";
  pending: boolean;
  elapsed: number;
  arm: AnswerResponse["full"] | undefined;
}) {
  const done = Boolean(arm);
  const accent = tone === "emerald" ? "text-emerald-400" : "text-neutral-300";
  const border =
    tone === "emerald" ? "border-emerald-900/60" : "border-neutral-800";

  return (
    <div className={`rounded border ${border} bg-neutral-950 p-3`}>
      <div className="flex items-baseline justify-between gap-2">
        <span className={`text-xs font-medium ${accent}`}>{label}</span>
        <span className="font-mono text-xs tabular-nums text-neutral-500">
          {done
            ? `${(arm!.latency_ms / 1000).toFixed(2)}s`
            : pending
              ? `${(elapsed / 1000).toFixed(2)}s…`
              : "—"}
        </span>
      </div>

      <div className="mt-1 font-mono text-[11px] text-neutral-600">
        {done
          ? `${arm!.tokens_in.toLocaleString()} tokens in · $${arm!.cost_usd.toFixed(6)}`
          : "waiting…"}
      </div>

      <div className="mt-2 min-h-[5rem] whitespace-pre-wrap text-sm leading-relaxed text-neutral-200">
        {done ? (
          arm!.error ? (
            // "Too large" is a different animal from "the provider fell over":
            // it is the expected, informative outcome on a big input, so it
            // reads as a finding rather than a crash.
            <span className={arm!.too_large ? "text-amber-400" : "text-red-400"}>
              {arm!.too_large
                ? "Rejected — this prompt exceeds the model's limit."
                : arm!.error}
            </span>
          ) : (
            arm!.answer
          )
        ) : (
          <span className="inline-block h-3 w-24 animate-pulse rounded bg-neutral-800" />
        )}
      </div>
    </div>
  );
}

function Verdict({ data }: { data: AnswerResponse }) {
  const d = data.delta;
  // Normalised comparison: the same answer phrased differently is still the
  // same answer, so this is a signal to read, not a score.
  const normalise = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  const bothAnswered = !data.full.error && !data.compressed.error;
  const same =
    bothAnswered &&
    normalise(data.full.answer) === normalise(data.compressed.answer);

  return (
    <>
      <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="faster" value={`${d.speedup.toFixed(1)}×`} good />
        <Stat label="cheaper" value={`${d.cost_reduction_pct.toFixed(0)}%`} good />
        <Stat
          label="tokens saved"
          value={d.tokens_saved.toLocaleString()}
          good
        />
        <Stat
          label="answers"
          value={!bothAnswered ? "only compressed" : same ? "identical" : "differ"}
          good={!bothAnswered || same}
        />
      </div>

      <p className="mt-3 text-xs leading-relaxed text-neutral-600">
        {!bothAnswered
          ? "Only one arm returned an answer, so there is nothing to compare on wording — the comparison above is about whether an answer was possible at all."
          : same
            ? "Both prompts produced the same answer — the removed tokens carried no information this question needed."
            : "The answers differ. Compare them above: compression may have dropped something this question needed, or the model simply phrased it differently."}{" "}
        Cost uses published {d.pricing_model} pricing on measured token counts.{" "}
        {data.note}
      </p>
    </>
  );
}

function Stat({
  label,
  value,
  good,
}: {
  label: string;
  value: string;
  good?: boolean;
}) {
  return (
    <div className="rounded border border-neutral-800 bg-neutral-950 p-2 text-center">
      <div
        className={`font-mono text-lg tabular-nums ${
          good ? "text-emerald-400" : "text-amber-400"
        }`}
      >
        {value}
      </div>
      <div className="text-[10px] uppercase tracking-wide text-neutral-600">
        {label}
      </div>
    </div>
  );
}
