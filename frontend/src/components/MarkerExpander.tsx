import { useState } from "react";
import { expandMarker } from "../api";
import type { Expansion, MarkerRef } from "../types";

/**
 * Recover what a marker hides, one marker at a time.
 *
 * Every other compressor in this space is one-way: it decides what to drop and
 * the caller lives with it. Because stage 7 already annotates each omission,
 * and each annotation carries an id, an omission has an *address* — so a
 * consumer that reads `[... omitted #d3 4 section(s) ...]` and decides it
 * actually needs that material can ask for exactly it, instead of re-running
 * the whole compression at a looser budget and hoping.
 *
 * That is the difference between compression as destruction and compression as
 * a lossy view over retained content.
 */
export function MarkerExpander({
  compressionId,
  markers,
}: {
  compressionId: string;
  markers: MarkerRef[];
}) {
  const [open, setOpen] = useState<string | null>(null);
  const [cache, setCache] = useState<Record<string, Expansion>>({});
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!markers.length) return null;

  const droppedTokens = markers.reduce((total, m) => total + m.tokens, 0);

  async function toggle(id: string) {
    if (open === id) {
      setOpen(null);
      return;
    }
    setOpen(id);
    setError(null);
    if (cache[id]) return;
    setLoading(id);
    try {
      const expansion = await expandMarker(compressionId, id);
      setCache((current) => ({ ...current, [id]: expansion }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(null);
    }
  }

  return (
    <section className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4">
      <h2 className="text-sm font-medium text-neutral-300">
        Recover omitted content
        <span className="ml-2 font-normal text-neutral-500">
          {markers.length} marker{markers.length === 1 ? "" : "s"} ·{" "}
          {droppedTokens.toLocaleString()} tokens addressable
        </span>
      </h2>
      <p className="mt-1 text-xs text-neutral-600">
        Each omission in the compressed prompt carries an id. Click one to pull
        that content back without re-compressing — the compression is a view,
        not a deletion.
      </p>

      <ul className="mt-3 space-y-1.5">
        {markers.map((marker) => {
          const isOpen = open === marker.id;
          const expansion = cache[marker.id];
          return (
            <li key={marker.id}>
              <button
                onClick={() => toggle(marker.id)}
                aria-expanded={isOpen}
                className="flex w-full items-center gap-3 rounded border border-neutral-800 bg-neutral-950 px-3 py-2 text-left text-xs hover:border-neutral-600"
              >
                <span className="font-mono text-emerald-500">#{marker.id}</span>
                <span className="text-neutral-400">
                  {marker.kind === "collapsed"
                    ? `${marker.sections} near-identical occurrence${marker.sections === 1 ? "" : "s"} collapsed`
                    : `${marker.sections} section${marker.sections === 1 ? "" : "s"} dropped`}
                  {marker.start_line !== undefined &&
                    ` · lines ${marker.start_line}-${marker.end_line}`}
                </span>
                <span className="ml-auto shrink-0 font-mono text-neutral-600">
                  {marker.tokens.toLocaleString()} tok
                </span>
                <span className="shrink-0 text-neutral-500">
                  {loading === marker.id ? "…" : isOpen ? "hide" : "recover"}
                </span>
              </button>

              {isOpen && expansion && (
                <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded border border-emerald-900/50 bg-emerald-950/10 p-3 font-mono text-[11px] leading-relaxed text-neutral-300">
                  {expansion.text || "(no text recorded for this marker)"}
                </pre>
              )}
              {isOpen && error && !expansion && (
                <p className="mt-1 px-3 text-xs text-amber-500/80">{error}</p>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
