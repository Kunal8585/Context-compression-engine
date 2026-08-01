import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { Stage } from "../types";

/** One-line summary per stage, derived from whatever that stage measures. */
function summarise(stage: Stage): string {
  const d = stage.details ?? {};
  const n = (key: string) => Number(d[key] ?? 0);

  switch (stage.name) {
    case "chunking":
      return `${stage.chunks_out} chunks from ${stage.tokens_in.toLocaleString()} tokens`;
    case "redundancy": {
      const parts = [
        n("exact_collapsed") && `${n("exact_collapsed")} exact`,
        n("structural_collapsed") && `${n("structural_collapsed")} structural`,
        n("embedding_collapsed") && `${n("embedding_collapsed")} semantic`,
      ].filter(Boolean);
      return `${stage.chunks_in} → ${stage.chunks_out} chunks${
        parts.length ? ` (${parts.join(", ")})` : " (no duplicates found)"
      }`;
    }
    case "density":
      return `${stage.chunks_in} chunks scored · ${String(d.entity_backend ?? "")}`;
    case "selection":
      return `${stage.chunks_in} → ${stage.chunks_out} chunks kept · ${n(
        "dropped_tokens",
      ).toLocaleString()} tokens dropped`;
    case "abstractive":
      return stage.status === "skipped"
        ? "not run"
        : `${n("accepted")} accepted, ${n("rejected")} rejected`;
    case "reconstruction":
      return `${stage.tokens_out.toLocaleString()} token prompt · ${n(
        "marker_count",
      )} audit markers`;
    default:
      return `${stage.tokens_in} → ${stage.tokens_out} tokens`;
  }
}

const STATUS_STYLES: Record<string, string> = {
  ok: "text-emerald-500 border-emerald-900/60 bg-emerald-950/30",
  skipped: "text-amber-500 border-amber-900/60 bg-amber-950/30",
  failed: "text-red-400 border-red-900/60 bg-red-950/30",
};

export function StageAccordion({ stages }: { stages: Stage[] }) {
  // Stage 6 is open by default when skipped: its note is the proof that the
  // skip was measured and deliberate, not a silent failure.
  const [open, setOpen] = useState<string[]>(
    stages.filter((s) => s.status === "skipped").map((s) => s.name),
  );

  const toggle = (name: string) =>
    setOpen((current) =>
      current.includes(name)
        ? current.filter((n) => n !== name)
        : [...current, name],
    );

  return (
    <section>
      <h2 className="mb-3 text-sm font-medium text-neutral-300">
        Pipeline stages
        <span className="ml-2 text-xs font-normal text-neutral-600">
          every run reports all six
        </span>
      </h2>

      <div className="divide-y divide-neutral-800 overflow-hidden rounded-lg border border-neutral-800 bg-neutral-900/40">
        {stages.map((stage, index) => {
          const isOpen = open.includes(stage.name);
          return (
            <div key={stage.name}>
              <button
                onClick={() => toggle(stage.name)}
                className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-neutral-900"
              >
                <span className="w-4 shrink-0 font-mono text-xs text-neutral-600">
                  {index + 1}
                </span>
                <span className="w-32 shrink-0 text-sm text-neutral-200">
                  {stage.name}
                </span>
                <span
                  className={`shrink-0 rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${
                    STATUS_STYLES[stage.status] ?? STATUS_STYLES.ok
                  }`}
                >
                  {stage.status}
                </span>
                <span className="flex-1 truncate font-mono text-xs text-neutral-500">
                  {summarise(stage)}
                </span>
                <span className="shrink-0 font-mono text-[11px] text-neutral-600">
                  {stage.duration_ms.toFixed(0)}ms
                </span>
              </button>

              <AnimatePresence initial={false}>
                {isOpen && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.18 }}
                    className="overflow-hidden"
                  >
                    <div className="border-t border-neutral-800/60 bg-neutral-950 px-4 py-3 pl-11">
                      {/* Verbatim note. When a stage is skipped this sentence
                          is the whole point - it says we checked and why. */}
                      {stage.note && (
                        <p
                          className={`text-xs leading-relaxed ${
                            stage.status === "skipped"
                              ? "text-amber-300/90"
                              : "text-neutral-400"
                          }`}
                        >
                          {stage.status === "skipped" && (
                            <span className="font-medium">Skipped — </span>
                          )}
                          {stage.note}
                        </p>
                      )}
                      {!stage.note && stage.status === "ok" && (
                        <p className="text-xs text-neutral-500">
                          Ran normally. {stage.tokens_in.toLocaleString()} →{" "}
                          {stage.tokens_out.toLocaleString()} tokens (
                          {stage.reduction_pct.toFixed(1)}% removed).
                        </p>
                      )}
                      <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-[11px] text-neutral-500 sm:grid-cols-3">
                        {Object.entries(stage.details ?? {})
                          .filter(
                            ([, value]) =>
                              typeof value === "number" ||
                              typeof value === "string" ||
                              typeof value === "boolean",
                          )
                          .slice(0, 9)
                          .map(([key, value]) => (
                            <div key={key} className="flex justify-between gap-2">
                              <dt className="truncate text-neutral-600">{key}</dt>
                              <dd className="text-neutral-400">{String(value)}</dd>
                            </div>
                          ))}
                      </dl>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>
    </section>
  );
}
