import { Component, type ReactNode, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { compress, getEvaluation, getHealth } from "./api";
import { HeroMetrics } from "./components/HeroMetrics";
import { DiffView } from "./components/DiffView";
import { StageAccordion } from "./components/StageAccordion";

/**
 * Give the paste a plausible filename.
 *
 * The backend picks its chunker from the file extension: `.py` routes to
 * tree-sitter (function/class chunks, structural dedup), `.log` to record
 * chunking. Sending everything as `.txt` silently downgrades pasted source to
 * paragraph chunking — 34 prose blocks and "no duplicates found" instead of 24
 * functions with the near-identical validators collapsed.
 */
function inferFilename(text: string): string {
  const head = text.slice(0, 4000);
  const lines = head.split("\n").filter((l) => l.trim()).slice(0, 40);
  const logLike = lines.filter((l) =>
    /^\s*\[?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}|\d{2}:\d{2}:\d{2})|^\s*\[?(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b/.test(l),
  ).length;
  if (lines.length >= 3 && logLike / lines.length >= 0.4) return "pasted.log";
  if (/^\s*(def |class |from \s*\w+ import|import \w+)/m.test(head)) return "pasted.py";
  if (/^\s*(function |const |let |export |import .* from )/m.test(head)) return "pasted.js";
  if (/^#{1,6}\s+\S/m.test(head)) return "pasted.md";
  return "pasted.txt";
}

/** A crash must never produce a blank white screen during judging. */
class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div className="mx-auto max-w-2xl p-8">
          <Notice tone="error" title="Something broke in the interface">
            {this.state.error.message} — reload the page to try again.
          </Notice>
        </div>
      );
    }
    return this.props.children;
  }
}

function Notice({
  tone,
  title,
  children,
}: {
  tone: "error" | "warn";
  title: string;
  children: ReactNode;
}) {
  const styles =
    tone === "error"
      ? "border-red-900/60 bg-red-950/30 text-red-200"
      : "border-amber-900/60 bg-amber-950/30 text-amber-200";
  return (
    <div className={`rounded-lg border p-4 text-sm ${styles}`}>
      <p className="font-medium">{title}</p>
      <p className="mt-1 opacity-90">{children}</p>
    </div>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [budget, setBudget] = useState(30);
  const [submitted, setSubmitted] = useState("");

  const health = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    retry: 1,
    refetchInterval: (q) => (q.state.data?.warm ? false : 3000),
  });

  const evaluation = useQuery({
    queryKey: ["evaluate"],
    queryFn: getEvaluation,
    retry: 1,
    enabled: health.isSuccess,
  });

  const run = useMutation({
    mutationFn: () => {
      setSubmitted(text);
      return compress(text, budget / 100, inferFilename(text));
    },
  });

  const backendDown = health.isError;
  const warming = health.data && !health.data.warm;

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-200">
      <div className="mx-auto max-w-6xl space-y-8 px-6 py-10">
        <header className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold text-neutral-50">
              Context Compression Engine
            </h1>
            <p className="mt-1 text-sm text-neutral-500">
              Algorithmic prompt compression — chunk, deduplicate, score,
              select, reconstruct. Every number below is measured.
            </p>
          </div>
          <StatusPill
            down={backendDown}
            warming={Boolean(warming)}
            label={
              backendDown
                ? "Backend unavailable"
                : warming
                  ? "Loading models…"
                  : `Ready · ${health.data?.tokenizer.backend ?? ""}`
            }
          />
        </header>

        {backendDown && (
          <Notice tone="error" title="Backend unavailable">
            Cannot reach the API. Start it with{" "}
            <code className="font-mono">./run.sh --backend</code> and reload.
          </Notice>
        )}

        {health.data?.tokenizer.exact === false && (
          <Notice tone="warn" title="Token counts are estimates">
            tiktoken could not load, so every ratio on this page is approximate
            rather than measured.
          </Notice>
        )}

        {/* --- 1. hero metrics --- */}
        {evaluation.data && <HeroMetrics report={evaluation.data} />}
        {evaluation.isError && !backendDown && (
          <Notice tone="warn" title="No benchmark report available">
            Run <code className="font-mono">python -m eval.harness</code> to
            generate one. Compression below still works.
          </Notice>
        )}

        {/* --- input --- */}
        <section className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-5">
          <label className="mb-2 block text-sm font-medium text-neutral-300">
            Paste a context — source file, log dump, or document
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste anything here…"
            spellCheck={false}
            className="h-40 w-full resize-y rounded border border-neutral-800 bg-neutral-950 p-3 font-mono text-xs text-neutral-300 outline-none focus:border-neutral-600"
          />
          <div className="mt-4 flex flex-wrap items-center gap-4">
            <label className="flex items-center gap-3 text-sm text-neutral-400">
              Token budget
              <input
                type="range"
                min={5}
                max={70}
                step={5}
                value={budget}
                onChange={(e) => setBudget(Number(e.target.value))}
                className="w-40 accent-emerald-600"
              />
              <span className="w-24 font-mono text-xs text-neutral-300">
                {budget}% of input
              </span>
            </label>
            <button
              onClick={() => run.mutate()}
              disabled={!text.trim() || run.isPending || backendDown}
              className="rounded bg-emerald-700 px-5 py-2 text-sm font-medium text-white transition-colors hover:bg-emerald-600 disabled:cursor-not-allowed disabled:bg-neutral-800 disabled:text-neutral-500"
            >
              {run.isPending ? "Processing…" : "Compress"}
            </button>
            {run.isPending && (
              <span className="text-xs text-neutral-500">
                running six stages locally…
              </span>
            )}
          </div>

          {run.isError && (
            <div className="mt-4">
              <Notice tone="error" title="Compression failed">
                {(run.error as Error).message}
              </Notice>
            </div>
          )}
        </section>

        {/* --- 2 + 3. diff view and stage accordion --- */}
        {run.data && (
          <>
            <DiffView original={submitted} result={run.data} />
            <StageAccordion stages={run.data.stages} />
          </>
        )}

        <footer className="border-t border-neutral-900 pt-4 text-xs text-neutral-700">
          Runs entirely locally — no API key, no cloud inference.
          {health.data && ` Embeddings on ${health.data.embeddings.device}.`}
        </footer>
      </div>
    </div>
  );
}

function StatusPill({
  down,
  warming,
  label,
}: {
  down: boolean;
  warming: boolean;
  label: string;
}) {
  const colour = down
    ? "bg-red-500"
    : warming
      ? "bg-amber-500 animate-pulse"
      : "bg-emerald-500";
  return (
    <span className="flex items-center gap-2 rounded-full border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs text-neutral-400">
      <span className={`h-2 w-2 rounded-full ${colour}`} />
      {label}
    </span>
  );
}

export { ErrorBoundary };
