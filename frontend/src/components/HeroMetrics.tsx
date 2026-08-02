import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { EvaluateResponse } from "../types";

/** Count-up tween. Plain rAF - a spring library buys nothing here. */
function useCountUp(target: number, durationMs = 900) {
  const [value, setValue] = useState(0);
  const frame = useRef<number>(0);

  useEffect(() => {
    const started = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - started) / durationMs);
      // easeOutCubic - fast start, soft landing
      setValue(target * (1 - Math.pow(1 - t, 3)));
      if (t < 1) frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame.current);
  }, [target, durationMs]);

  return value;
}

function Card({
  label,
  value,
  format,
  detail,
  onClick,
  expanded,
}: {
  label: string;
  value: number;
  format: (n: number) => string;
  detail: string;
  onClick?: () => void;
  expanded?: boolean;
}) {
  const animated = useCountUp(value);
  const interactive = Boolean(onClick);

  return (
    <div className="flex-1 min-w-[220px]">
      <div
        onClick={onClick}
        role={interactive ? "button" : undefined}
        tabIndex={interactive ? 0 : undefined}
        onKeyDown={(e) => {
          if (interactive && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            onClick?.();
          }
        }}
        className={`rounded-lg border border-neutral-800 bg-neutral-900/60 p-5 transition-colors ${
          interactive
            ? "cursor-pointer hover:border-neutral-600 focus:outline-none focus:border-emerald-600"
            : ""
        }`}
      >
        <div className="flex items-start justify-between gap-2">
          <span className="text-[11px] uppercase tracking-wider text-neutral-500">
            {label}
          </span>
          {interactive && (
            <span className="text-[11px] text-emerald-500 shrink-0">
              {expanded ? "hide" : "why?"}
            </span>
          )}
        </div>
        <div className="mt-2 font-mono text-4xl text-neutral-50 tabular-nums">
          {format(animated)}
        </div>
        <div className="mt-1 text-xs text-neutral-500">{detail}</div>
      </div>
    </div>
  );
}

export function HeroMetrics({ report }: { report: EvaluateResponse }) {
  const [open, setOpen] = useState(false);
  const a = report.aggregate;

  const pct = (n: number) => `${(n * 100).toFixed(1)}%`;
  const usd = (n: number) => `$${n.toFixed(5)}`;

  return (
    <section>
      {/* items-start: the expanding card grows downward on its own instead of
          stretching its three siblings to match. */}
      <div className="flex flex-wrap items-start gap-4">
        <Card
          label="Compression ratio"
          value={a.compression_ratio}
          format={pct}
          detail={`${a.tokens_before.toLocaleString()} → ${a.tokens_after.toLocaleString()} tokens`}
        />
        <Card
          label="Cost reduction"
          value={a.cost_reduction}
          format={pct}
          detail={`${usd(a.cost_before_usd)} → ${usd(a.cost_after_usd)} · ${report.config.pricing_model}`}
        />
        <Card
          label="Latency speedup"
          value={a.latency_speedup}
          format={(n) => `${n.toFixed(2)}×`}
          detail={`${(a.latency_before_ms / 1000).toFixed(0)}s → ${(a.latency_after_ms / 1000).toFixed(0)}s`}
        />
        {/* Fact survival, not accuracy retention.

            Retention is a joint verdict on the compressor AND the answering
            model — a small model that cannot find a fact sitting in its own
            context drags it down while saying nothing about what compression
            removed. Fact survival is the compressor's own number: measured
            deterministically, no model involved. It is also the ceiling
            retention is bounded by, so it is the more informative headline.
            Retention is still shown, in the panel below, where it can be
            explained rather than mistaken for a compression metric. */}
        <Card
          label="Key facts preserved"
          value={report.fact_survival.overall}
          format={pct}
          detail={`${report.fact_survival.facts_surviving}/${report.fact_survival.facts_total} facts · no model involved`}
          onClick={() => setOpen((v) => !v)}
          expanded={open}
        />
      </div>

      {/* Inline, full-width, below the cards - not a modal or a tooltip. Given
          judges may read this on a projector at 1280px, a single narrow column
          of wrapped body text was the wrong shape for the most important
          explanation on the page. */}
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-4 rounded-lg border border-emerald-900/60 bg-emerald-950/20 p-5">
              <p className="text-base leading-relaxed text-neutral-200">
                <span className="font-mono text-emerald-400">
                  {pct(report.fact_survival.overall)}
                </span>{" "}
                of key facts survive compression (
                {report.fact_survival.facts_surviving}/
                {report.fact_survival.facts_total}), measured by checking the
                compressed text directly — <strong>no model is asked
                anything</strong>, so this number is the compressor&apos;s own
                and is reproducible exactly.
              </p>
              <p className="mt-3 text-base leading-relaxed text-neutral-300">
                Downstream <strong>accuracy retention</strong> is{" "}
                <span className="font-mono text-amber-400">
                  {pct(a.accuracy_retention)}
                </span>{" "}
                ({pct(a.accuracy_before)} → {pct(a.accuracy_after)} key-fact
                recall against {report.config.downstream_model}). It is bounded
                by the number above: the compressor cannot preserve more than
                it preserved, and the answering model cannot retrieve more than
                the compressor kept. Where the two differ, the gap is
                retrieval, not information loss.
              </p>
              <p className="mt-3 text-sm text-neutral-500">
                Both are reported because they answer different questions.
                Accuracy metric: {report.config.accuracy_metric}.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <p className="mt-3 text-xs text-neutral-600">
        Measured over a {a.items}-item benchmark against{" "}
        {report.config.downstream_model}
        {report.diagnostic && !report.diagnostic.available && (
          <> · {report.diagnostic.label}: {report.diagnostic.reason}</>
        )}
      </p>
    </section>
  );
}
