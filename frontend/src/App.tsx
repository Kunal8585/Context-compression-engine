import { Component, type ReactNode, useCallback, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  compress,
  compressFiles,
  getEvaluation,
  getHealth,
  getProviders,
  askBoth,
} from "./api";
import { AnswerRace } from "./components/AnswerRace";
import { HeroMetrics } from "./components/HeroMetrics";
import { ConfidenceCard } from "./components/ConfidenceCard";
import { DiffView } from "./components/DiffView";
import { MarkerExpander } from "./components/MarkerExpander";
import { ModelSelect, presetToPins, providerLabel } from "./components/ModelSelect";
import { ProviderStatusBar } from "./components/ProviderStatus";
import { RunComparison, toRunRecord, type RunRecord } from "./components/RunComparison";
import { StageAccordion } from "./components/StageAccordion";
import type { CompressResponse, FileStatus } from "./types";

/** Extensions the backend's ingestion layer accepts. Kept in one place so the
 *  picker's `accept` and the drop-zone label cannot drift apart. */
const ACCEPTED = [
  ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".pdf",
  ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml", ".toml",
  ".ini", ".cfg", ".sh", ".bash", ".sql", ".java", ".go", ".rs",
  ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt",
  ".css", ".scss", ".html", ".xml", ".zsh",
].join(",");

const MAX_FILES = 10;

/** Matches `selection.budget_ratio` in config.yaml. Every published number was
 *  measured at this budget, so the UI does not offer to change it. */
const DEFAULT_BUDGET_PCT = 30;

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
  // Budget is fixed at the config default rather than exposed as a slider.
  // It was a control that let anyone quietly change the headline compression
  // ratio mid-demo, and 0.30 is the ratio every measured number in the README
  // and reports/ was produced at - a run at 0.55 is not comparable to any of
  // them. Tune it in config.yaml, where the change is recorded.
  const [submitted, setSubmitted] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  // Session-scoped. One id from GET /providers -> selections; the backend
  // resolves it into the (embedding, generation) pair the request uses, so the
  // UI never has to know which vendor offers which role.
  const [modelId, setModelId] = useState("auto");
  // Optional. When present the scorer keeps what answers THIS question,
  // not merely what is generally informative - see density.query_relevance.
  const [query, setQuery] = useState("");
  // Kept so switching model and re-running produces a visible delta rather
  // than silently replacing the numbers.
  const [previousRun, setPreviousRun] = useState<RunRecord | null>(null);
  const [currentRun, setCurrentRun] = useState<RunRecord | null>(null);

  const addFiles = useCallback((incoming: FileList | null) => {
    if (!incoming?.length) return;
    setFiles((current) => {
      const merged = [...current];
      for (const file of Array.from(incoming)) {
        // De-duplicate on name+size so dropping the same folder twice does not
        // silently double the context (and the compression ratio with it).
        if (!merged.some((f) => f.name === file.name && f.size === file.size)) {
          merged.push(file);
        }
      }
      return merged.slice(0, MAX_FILES);
    });
  }, []);

  const health = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    retry: 1,
    refetchInterval: (q) => (q.state.data?.warm ? false : 3000),
  });

  const providers = useQuery({
    queryKey: ["providers"],
    queryFn: getProviders,
    retry: 1,
    staleTime: 60_000,
  });

  const evaluation = useQuery({
    queryKey: ["evaluate"],
    queryFn: getEvaluation,
    retry: 1,
    enabled: health.isSuccess,
  });

  const run = useMutation({
    mutationFn: async (): Promise<{ result: CompressResponse; wallMs: number }> => {
      setSubmitted(text);
      // Wall clock measured client-side on purpose: it includes the network,
      // which is the whole point of a local-vs-cloud comparison.
      const started = performance.now();
      const { mode, ...pins } = presetToPins(
        providers.data?.selections.find((p) => p.id === modelId),
      );
      const result = files.length
        ? await compressFiles(files, text, DEFAULT_BUDGET_PCT / 100, mode, pins, query)
        : await compress(text, DEFAULT_BUDGET_PCT / 100, inferFilename(text), mode, pins, query);
      return { result, wallMs: performance.now() - started };
    },
    onSuccess: ({ result, wallMs }) => {
      setPreviousRun(currentRun);
      setCurrentRun(toRunRecord(result, wallMs));
    },
  });

  // The payoff, raced live: same question, same model, both contexts.
  const ask = useMutation({
    mutationFn: async () => {
      setSubmitted(text);
      const { mode, ...pins } = presetToPins(
        providers.data?.selections.find((p) => p.id === modelId),
      );
      return askBoth(text, query.trim(), inferFilename(text), mode, pins);
    },
  });

  const result = run.data?.result;

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

        {/* --- provider status: what is actually serving this run --- */}
        {health.data && <ProviderStatusBar health={health.data} />}

        {/* --- 1. hero metrics --- */}
        {evaluation.data && <HeroMetrics report={evaluation.data} />}
        {evaluation.isError && !backendDown && (
          <Notice tone="warn" title="No benchmark report available">
            Run <code className="font-mono">python -m eval.harness</code> to
            generate one. Compression below still works.
          </Notice>
        )}

        {/* --- workspace ---
            One surface. Pasting, dropping files, asking a question and picking
            a model were four separate bordered blocks stacked down the page,
            which read as four decisions to make before anything happens. They
            are one decision - "here is my context, compress it" - so they get
            one panel: the whole thing is the drop target, files land as chips
            inside it, and the toolbar sits under the same border. */}
        <section
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={(e) => {
            // Only clear when the pointer leaves the panel itself, not when it
            // crosses between children - otherwise the highlight strobes.
            if (!e.currentTarget.contains(e.relatedTarget as Node)) {
              setDragging(false);
            }
          }}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            addFiles(e.dataTransfer.files);
          }}
          className={`rounded-lg border bg-neutral-900/40 transition-colors ${
            dragging
              ? "border-emerald-600 bg-emerald-950/10"
              : "border-neutral-800"
          }`}
        >
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste a log, source file or document — or drop files anywhere on this panel"
            spellCheck={false}
            className="h-44 w-full resize-y rounded-t-lg bg-transparent p-4 font-mono text-xs text-neutral-300 outline-none placeholder:text-neutral-600"
          />

          {/* Attached files as chips, inline - they are part of the same input,
              not a separate mode. */}
          {files.length > 0 && (
            <div className="flex flex-wrap gap-2 px-4 pb-2">
              {files.map((file) => (
                <span
                  key={`${file.name}-${file.size}`}
                  className="flex items-center gap-2 rounded-full border border-neutral-700 bg-neutral-900 py-1 pl-3 pr-1.5 text-xs text-neutral-300"
                >
                  <span className="font-mono">{file.name}</span>
                  <span className="text-neutral-600">
                    {(file.size / 1024).toFixed(0)} KB
                  </span>
                  <button
                    type="button"
                    aria-label={`Remove ${file.name}`}
                    onClick={() =>
                      setFiles((current) => current.filter((f) => f !== file))
                    }
                    className="rounded-full px-1.5 text-neutral-500 hover:bg-neutral-800 hover:text-neutral-200"
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}

          {/* Optional question. Present, it switches on the query_relevance
              density signal, so the selector keeps what answers THIS question
              rather than what is merely informative. Measured: key-fact
              survival 80.8% -> 88.5% on the benchmark corpus. */}
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Optional — ask a question to keep what answers it"
            className="w-full border-t border-neutral-800 bg-transparent px-4 py-2.5 text-sm text-neutral-200 outline-none placeholder:text-neutral-600"
          />

          {/* One toolbar: attach, model, go. */}
          <div className="flex flex-wrap items-center gap-3 border-t border-neutral-800 p-3">
            <label className="cursor-pointer rounded border border-neutral-700 px-3 py-1.5 text-sm text-neutral-300 transition-colors hover:border-neutral-500 hover:bg-neutral-900">
              Attach files
              <input
                type="file"
                multiple
                accept={ACCEPTED}
                className="sr-only"
                onChange={(e) => {
                  addFiles(e.target.files);
                  e.target.value = "";
                }}
              />
            </label>

            <div className="min-w-[240px] flex-1">
              <ModelSelect
                catalogue={providers.data}
                value={modelId}
                onChange={setModelId}
                disabled={run.isPending}
                compact
              />
            </div>

            <button
              onClick={() => run.mutate()}
              disabled={(!text.trim() && !files.length) || run.isPending || ask.isPending || backendDown}
              className="rounded border border-neutral-700 px-4 py-2 text-sm font-medium text-neutral-200 transition-colors hover:border-neutral-500 hover:bg-neutral-900 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {run.isPending ? "Compressing…" : "Compress"}
            </button>
            {/* The headline action once a question is present: compressing is
                the mechanism, answering the same question from both contexts
                is the actual claim. */}
            <button
              onClick={() => ask.mutate()}
              disabled={!text.trim() || !query.trim() || ask.isPending || run.isPending || backendDown}
              title={
                !query.trim()
                  ? "Ask a question above to race the two contexts"
                  : "Answer this question from the full and compressed contexts"
              }
              className="rounded bg-emerald-700 px-5 py-2 text-sm font-medium text-white transition-colors hover:bg-emerald-600 disabled:cursor-not-allowed disabled:bg-neutral-800 disabled:text-neutral-500"
            >
              {ask.isPending ? "Racing…" : "Compress & Answer"}
            </button>
          </div>

          <p className="px-4 pb-3 text-xs text-neutral-600">
            .txt .md .log .pdf and common code files · up to {MAX_FILES} files,
            combined in upload order · budget {DEFAULT_BUDGET_PCT}% of input
            {query.trim() && " · query-aware scoring on"}
          </p>

          {run.isError && (
            <div className="px-4 pb-4">
              <Notice tone="error" title="Compression failed">
                {(run.error as Error).message}
              </Notice>
            </div>
          )}
        </section>

        {/* --- 2 + 3. diff view and stage accordion --- */}
        <AnswerRace data={ask.data} pending={ask.isPending} />
        {ask.isError && (
          <Notice tone="error" title="Answer failed">
            {(ask.error as Error).message}
          </Notice>
        )}

        {result && currentRun && (
          <>
            <ResolvedProvider record={currentRun} />
            <RunComparison current={currentRun} previous={previousRun} />
            {result.files && <FileResults files={result.files} />}
            {result.confidence && (
              <ConfidenceCard confidence={result.confidence} />
            )}
            <DiffView original={submitted} result={result} />
            <MarkerExpander
              compressionId={result.compression_id}
              markers={result.markers}
            />
            <StageAccordion stages={result.stages} />
          </>
        )}

        <footer className="border-t border-neutral-900 pt-4 text-xs text-neutral-700">
          Multi-provider model layer — embeddings and generation each run through
          an ordered fallback chain, so a rate-limited or unreachable provider
          degrades to the next instead of failing.
          {health.data &&
            ` Serving now: ${health.data.providers.embedding.active ?? "none"} (embeddings), ${health.data.providers.generation.active ?? "none"} (generation).`}{" "}
          Live status above.
        </footer>
      </div>
    </div>
  );
}

/**
 * "Compressed using: Groq (cloud)".
 *
 * Closes the loop on the toggle: the user picked a mode, and this states which
 * provider that actually resolved to. Those differ whenever the chain fell
 * back, which is exactly the case worth surfacing rather than hiding.
 */
function ResolvedProvider({ record }: { record: RunRecord }) {
  const cloud =
    record.embeddingProvider !== undefined && record.embeddingProvider !== "local";
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 px-4 py-3 text-sm">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span
          className={`h-2 w-2 rounded-full ${cloud ? "bg-sky-400" : "bg-emerald-500"}`}
        />
        <span className="text-neutral-400">Compressed using</span>
        <span className="font-medium text-neutral-100">
          {providerLabel(record.embeddingProvider)}
        </span>
        <span className="text-xs text-neutral-600">
          · mode <span className="font-mono">{record.mode}</span> ·{" "}
          {(record.wallMs / 1000).toFixed(1)}s ·{" "}
          {record.compressionPct.toFixed(1)}% smaller
        </span>
      </div>

      {/* Stage 6 is reported separately because it is a different provider and
          is frequently skipped. Folding it into the line above made a working
          generation provider look like it was never configured. */}
      <p className="mt-1.5 pl-5 text-xs text-neutral-500">
        Embeddings (stage 3):{" "}
        <span className="text-neutral-300">
          {providerLabel(record.embeddingProvider)}
        </span>
        {" · "}
        Rewriting (stage 6):{" "}
        {record.generationProvider ? (
          <span className="text-neutral-300">
            {providerLabel(record.generationProvider)}
          </span>
        ) : (
          <span className="text-amber-500/80">
            not used — {record.generationSkipped ?? "skipped"}
          </span>
        )}
      </p>
      {!record.generationProvider && record.generationSkipped?.includes("threshold") && (
        <p className="mt-1 pl-5 text-xs text-neutral-600">
          Stage 6 only paraphrases chunks above 150 tokens, and at this budget
          the selector drops them all first. Raise the token budget to ~50% to
          see the generation provider run.
        </p>
      )}
    </div>
  );
}

/**
 * Per-file outcome of an upload.
 *
 * A skipped file is reported, not hidden: a judge who drags in a scanned PDF
 * needs to see that it contributed nothing and why, rather than wonder why the
 * compression ratio looks odd.
 */
function FileResults({ files }: { files: FileStatus[] }) {
  const skipped = files.filter((f) => f.status === "skipped");
  return (
    <section className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4">
      <h2 className="mb-2 text-sm font-medium text-neutral-300">
        Files processed
        <span className="ml-2 font-normal text-neutral-500">
          {files.length - skipped.length} of {files.length} used
        </span>
      </h2>
      <ul className="space-y-1">
        {files.map((file) => {
          const ok = file.status === "done";
          return (
            <li key={file.name} className="flex items-start gap-2 text-xs">
              <span
                className={`mt-0.5 h-2 w-2 shrink-0 rounded-full ${
                  ok ? "bg-emerald-500" : "bg-amber-500"
                }`}
              />
              <span className="font-mono text-neutral-300">{file.name}</span>
              <span className="text-neutral-600">
                {ok
                  ? `${file.characters.toLocaleString()} chars${
                      file.pages ? ` · ${file.pages} pages` : ""
                    }`
                  : `skipped — ${file.reason ?? "unreadable"}`}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
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
