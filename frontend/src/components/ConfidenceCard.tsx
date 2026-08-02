import { useState } from "react";
import type { ConfidenceReport } from "../types";

/**
 * How much to trust this particular compression.
 *
 * Shown with its reasons rather than as a bare number, on purpose. "0.44" tells
 * a user nothing they can act on; "37 of 214 numbers that survived dedup were
 * dropped by the budget, e.g. 240, 8.4" tells them whether the thing they care
 * about is still in there — and if it isn't, that raising the budget is the fix.
 *
 * The score is deterministic: weighted ratios over counts measured during the
 * run, no model involved. It predicts fact survival and is checked against the
 * four sample contexts where fact survival was measured directly — so it is
 * labelled a prediction here, never a guarantee.
 */
const BANDS = {
  high: {
    dot: "bg-emerald-500",
    text: "text-emerald-400",
    bar: "bg-emerald-500",
    label: "High confidence",
    blurb: "Little unique content was removed — mostly duplicates collapsed.",
  },
  moderate: {
    dot: "bg-amber-500",
    text: "text-amber-400",
    bar: "bg-amber-500",
    label: "Moderate confidence",
    blurb: "Some scoring content was evicted to meet the budget.",
  },
  low: {
    dot: "bg-red-500",
    text: "text-red-400",
    bar: "bg-red-500",
    label: "Low confidence",
    blurb:
      "This input has little redundancy, so the budget removed unique facts. Raise the budget if answers look incomplete.",
  },
  unknown: {
    dot: "bg-neutral-600",
    text: "text-neutral-400",
    bar: "bg-neutral-600",
    label: "Not scored",
    blurb: "Nothing to judge.",
  },
} as const;

const COMPONENT_LABELS: Record<string, string> = {
  number_retention: "Numbers kept",
  identifier_retention: "Identifiers kept",
  density_retention: "Density mass kept",
  lossless_share: "Removed as duplicates",
  dependency_integrity: "Code deps intact",
};

export function ConfidenceCard({ confidence }: { confidence: ConfidenceReport }) {
  const [open, setOpen] = useState(false);
  const band = BANDS[confidence.band] ?? BANDS.unknown;
  const pct = Math.round(confidence.score * 100);

  return (
    <section className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${band.dot}`} />
          <h2 className="text-sm font-medium text-neutral-300">
            Compression confidence
          </h2>
          <span className={`text-sm font-medium ${band.text}`}>
            {band.label} · {pct}%
          </span>
        </div>
        <button
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="text-xs text-neutral-600 hover:text-neutral-400"
        >
          {open ? "hide breakdown" : "breakdown"}
        </button>
      </div>

      <div className="mt-2 h-1.5 w-full overflow-hidden rounded bg-neutral-800">
        <div
          className={`h-full rounded ${band.bar} transition-all`}
          style={{ width: `${pct}%` }}
        />
      </div>

      <p className="mt-2 text-xs text-neutral-500">{band.blurb}</p>

      <ul className="mt-2 space-y-1">
        {confidence.reasons.map((reason) => (
          <li key={reason} className="text-xs leading-relaxed text-neutral-400">
            · {reason}
          </li>
        ))}
      </ul>

      {open && (
        <div className="mt-3 border-t border-neutral-800 pt-3">
          <ul className="space-y-1.5">
            {Object.entries(confidence.components).map(([key, value]) => (
              <li key={key} className="flex items-center gap-3 text-xs">
                <span className="w-44 shrink-0 text-neutral-500">
                  {COMPONENT_LABELS[key] ?? key}
                </span>
                <span className="h-1 flex-1 overflow-hidden rounded bg-neutral-800">
                  <span
                    className="block h-full rounded bg-neutral-600"
                    style={{ width: `${Math.round(value * 100)}%` }}
                  />
                </span>
                <span className="w-10 shrink-0 text-right font-mono text-neutral-400">
                  {Math.round(value * 100)}%
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-[11px] leading-relaxed text-neutral-600">
            {confidence.method} Retention is measured against what survived
            deduplication, not the raw input — collapsing 400 identical log lines
            keeps a representative and a count, so it is not a loss.
          </p>
        </div>
      )}
    </section>
  );
}
