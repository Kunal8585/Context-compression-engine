import { useState } from "react";
import type { ChainInfo, Health, ProviderInfo } from "../types";

/**
 * Which model providers are actually serving this run.
 *
 * The point of showing this is that the fallback chain is the architecture's
 * main claim, and a claim you can watch resolve is worth more than one in a
 * README. Each chain renders in configured order with the live entry marked,
 * so "if one API is rate-limited the system falls back to the next" is
 * something a judge can see rather than take on trust.
 *
 * Nothing here can leak a secret: `/health` reports key presence as a boolean
 * and never returns a key, so there is no value available to render even by
 * accident.
 */
export function ProviderStatusBar({ health }: { health: Health }) {
  const [open, setOpen] = useState(false);
  const { providers } = health;
  const degraded =
    !providers.any_generation_available || !providers.any_embedding_available;

  return (
    <section className="rounded-lg border border-neutral-800 bg-neutral-900/40">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-4 px-4 py-3 text-left"
      >
        <span className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <span className="text-sm font-medium text-neutral-300">
            Model providers
          </span>
          <ActiveTag label="embeddings" chain={providers.embedding} />
          <ActiveTag label="generation" chain={providers.generation} />
          {providers.offline_mode && (
            <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[10px] text-neutral-400">
              offline mode — local only
            </span>
          )}
          {degraded && (
            <span className="rounded bg-amber-950/40 px-1.5 py-0.5 text-[10px] text-amber-400">
              chain exhausted
            </span>
          )}
        </span>
        <span className="shrink-0 text-xs text-neutral-600">
          {open ? "hide" : "details"}
        </span>
      </button>

      {open && (
        <div className="grid grid-cols-1 gap-4 border-t border-neutral-800 p-4 md:grid-cols-2">
          <Chain title="Embeddings (stage 3)" chain={providers.embedding} />
          <Chain
            title="Generation (stage 6 + eval)"
            chain={providers.generation}
          />
          <p className="text-xs leading-relaxed text-neutral-600 md:col-span-2">
            Chains are tried top to bottom. A provider with no key configured is
            skipped, not an error; one that fails or times out is passed over and
            retried after a cooldown. Keys are read from the environment and are
            never returned by this endpoint — only whether each one is present.
          </p>
        </div>
      )}
    </section>
  );
}

function ActiveTag({ label, chain }: { label: string; chain: ChainInfo }) {
  const active = chain.active;
  return (
    <span className="flex items-center gap-1.5 text-xs">
      <span
        className={`h-2 w-2 rounded-full ${
          active ? "bg-emerald-500" : "bg-amber-500"
        }`}
      />
      <span className="text-neutral-500">{label}</span>
      <span className="font-mono text-neutral-300">{active ?? "none"}</span>
    </span>
  );
}

function Chain({ title, chain }: { title: string; chain: ChainInfo }) {
  return (
    <div>
      <h3 className="mb-2 text-xs font-medium text-neutral-400">{title}</h3>
      <ol className="space-y-1">
        {chain.providers.map((provider, index) => (
          <ProviderRow
            key={provider.provider}
            provider={provider}
            position={index + 1}
            active={provider.provider === chain.active}
          />
        ))}
      </ol>
    </div>
  );
}

function ProviderRow({
  provider,
  position,
  active,
}: {
  provider: ProviderInfo;
  position: number;
  active: boolean;
}) {
  return (
    <li className="flex items-baseline gap-2 text-xs">
      <span className="w-4 shrink-0 text-right font-mono text-neutral-700">
        {position}
      </span>
      <span
        className={`font-mono ${active ? "text-emerald-400" : "text-neutral-400"}`}
      >
        {provider.provider}
      </span>
      <span className="truncate font-mono text-[10px] text-neutral-600">
        {provider.model}
      </span>
      <span className="ml-auto shrink-0 text-[10px]">
        {active ? (
          <span className="text-emerald-500">serving</span>
        ) : provider.configured ? (
          <span className="text-neutral-600">standby</span>
        ) : (
          <span className="text-neutral-700" title={provider.reason ?? undefined}>
            {provider.requires_key ? "no key" : "unavailable"}
          </span>
        )}
      </span>
    </li>
  );
}
