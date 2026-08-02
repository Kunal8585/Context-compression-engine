import { useMemo } from "react";
import type { CompressResponse } from "../types";

/**
 * Side-by-side original vs compressed.
 *
 * The backend never echoes the input back - it returns character spans over the
 * text we already hold, which is what keeps the response ~70 KB instead of
 * 2.17 MB on a large log. So the left pane is rendered by slicing our own copy
 * with those spans, dimming everything the selector dropped.
 */
export function DiffView({
  original,
  result,
}: {
  original: string;
  result: CompressResponse;
}) {
  const segments = useMemo(() => {
    const out: Array<{
      text: string;
      kept: boolean;
      key: string;
      source?: string;
      /** Only the first span of each file carries a badge — see below. */
      showBadge?: boolean;
    }> = [];
    let cursor = 0;
    // A badge on every span would put one in front of every paragraph of a
    // single-file upload. What a reader actually needs is the point where
    // provenance *changes*, so the badge marks transitions between files.
    let lastSource: string | undefined;
    result.spans.forEach((span, index) => {
      if (span.start > cursor) {
        // Whitespace between chunks - keep it so line numbers stay honest.
        out.push({
          text: original.slice(cursor, span.start),
          kept: true,
          key: `gap-${index}`,
        });
      }
      const source = span.source_file;
      out.push({
        text: original.slice(span.start, span.end),
        kept: span.kept,
        key: `span-${index}`,
        source,
        showBadge: Boolean(source) && source !== lastSource,
      });
      if (source) lastSource = source;
      cursor = Math.max(cursor, span.end);
    });
    if (cursor < original.length) {
      out.push({ text: original.slice(cursor), kept: true, key: "tail" });
    }
    return out;
  }, [original, result.spans]);

  const { summary } = result;
  const droppedTokens = result.spans
    .filter((s) => !s.kept)
    .reduce((total, s) => total + s.tokens, 0);
  const sourceFiles = useMemo(
    () => [...new Set(result.spans.map((s) => s.source_file).filter(Boolean))],
    [result.spans],
  );

  return (
    <section>
      <div className="mb-3 flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h2 className="text-sm font-medium text-neutral-300">Before / after</h2>
        <span className="font-mono text-xs text-neutral-500">
          {summary.original_tokens.toLocaleString()} →{" "}
          {summary.compressed_tokens.toLocaleString()} tokens ·{" "}
          <span className="text-emerald-400">
            {summary.compression_pct.toFixed(1)}% smaller
          </span>{" "}
          · budget {summary.budget_tokens.toLocaleString()} ·{" "}
          {summary.total_ms.toFixed(0)}ms
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Pane
          title="Original"
          meta={
            sourceFiles.length > 1
              ? `${sourceFiles.length} files · ${summary.chunks_total} chunks · ${summary.original_tokens.toLocaleString()} tokens`
              : `${summary.chunks_total} chunks · ${summary.original_tokens.toLocaleString()} tokens`
          }
        >
          {segments.map((segment) => (
            <span key={segment.key} title={segment.source}>
              {segment.showBadge && <SourceBadge name={segment.source!} />}
              {segment.kept ? (
                segment.text
              ) : (
                <span
                  title="dropped by the budget selector"
                  className="bg-red-950/30 text-neutral-600 line-through decoration-neutral-700"
                >
                  {segment.text}
                </span>
              )}
            </span>
          ))}
        </Pane>

        <Pane
          title="Compressed"
          meta={`${summary.chunks_kept} of ${summary.chunks_total} chunks kept · ${summary.compressed_tokens.toLocaleString()} tokens`}
        >
          {/* Audit markers the reconstructor inserted are highlighted so it is
              obvious the prompt declares its own omissions. */}
          {result.compressed_text.split(/(\[[^\]\n]*\])/g).map((part, index) =>
            part.startsWith("[") && part.endsWith("]") ? (
              <span
                key={index}
                className="text-amber-500/80 bg-amber-950/20 rounded px-0.5"
              >
                {part}
              </span>
            ) : (
              <span key={index}>{part}</span>
            ),
          )}
        </Pane>
      </div>

      <p className="mt-2 text-xs text-neutral-600">
        <span className="inline-block h-2 w-3 bg-red-950/60 align-middle" />{" "}
        dropped ({droppedTokens.toLocaleString()} tokens) ·{" "}
        <span className="text-amber-500/80">[markers]</span> state what was
        removed, so the prompt is auditable
        {sourceFiles.length > 1 && (
          <>
            {" · "}
            <span className="rounded bg-neutral-800 px-1 text-[9px] text-neutral-400">
              file.ext
            </span>{" "}
            badges mark where each uploaded file begins
          </>
        )}
      </p>
    </section>
  );
}

/** Provenance tag shown where the source file changes in a multi-file upload. */
function SourceBadge({ name }: { name: string }) {
  return (
    <span className="mr-1 rounded bg-neutral-800 px-1 align-middle text-[9px] text-neutral-400">
      {name}
    </span>
  );
}

function Pane({
  title,
  meta,
  children,
}: {
  title: string;
  meta: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950">
      <div className="flex items-baseline justify-between border-b border-neutral-800 px-3 py-2">
        <span className="text-xs font-medium text-neutral-300">{title}</span>
        <span className="font-mono text-[11px] text-neutral-600">{meta}</span>
      </div>
      <pre className="max-h-[26rem] overflow-auto whitespace-pre-wrap break-words p-3 font-mono text-[11px] leading-relaxed text-neutral-400">
        {children}
      </pre>
    </div>
  );
}
