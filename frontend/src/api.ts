import type {
  CompressResponse,
  EvaluateResponse,
  Health,
  Mode,
  Expansion,
  ProviderCatalogue,
  AnswerResponse,
} from "./types";

/** Which provider each role is pinned to; null means "follow the mode". */
export interface ProviderPins {
  embedding_provider: string | null;
  generation_provider: string | null;
}

// Dev goes through Vite's proxy; a deployed build points at VITE_API_URL.
const BASE = import.meta.env.VITE_API_URL ?? "/api";

/** Unwraps the backend's `{error: {code, message}}` envelope into a real Error. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      ...init,
      headers: init?.body instanceof FormData
        ? init?.headers
        : { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new Error(
      "Cannot reach the backend. Is it running on port 8000?",
    );
  }

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      body?.error?.message ??
      body?.detail?.[0]?.msg ??
      `Request failed (${response.status})`;
    throw new Error(message);
  }
  return body as T;
}

export const getHealth = () => request<Health>("/health");

/** Every selectable model per role. What the model pickers are built from. */
export const getProviders = () => request<ProviderCatalogue>("/providers");

/** Recover what one marker in a compressed prompt hides. The escape hatch. */
export const expandMarker = (compressionId: string, markerId: string) =>
  request<Expansion>(`/expand/${compressionId}/${markerId}`);

export const getEvaluation = () =>
  request<EvaluateResponse>("/evaluate", {
    method: "POST",
    // Serves the last measured report instantly. A live run is minutes of local
    // inference and must never block the UI.
    body: JSON.stringify({ run: false }),
  });

export const compress = (
  text: string,
  budgetRatio: number,
  name: string,
  mode: Mode = "auto",
  pins: ProviderPins = { embedding_provider: null, generation_provider: null },
  query = "",
) =>
  request<CompressResponse>("/compress", {
    method: "POST",
    body: JSON.stringify({
      text,
      name,
      budget_ratio: budgetRatio,
      // Let the backend run stage 6 when a provider is available. It retains
      // its own safe skip/fallback for unavailable models.
      fast_mode: false,
      mode,
      ...(pins.embedding_provider
        ? { embedding_provider: pins.embedding_provider }
        : {}),
      ...(pins.generation_provider
        ? { generation_provider: pins.generation_provider }
        : {}),
      // Turns on the query_relevance density signal: the scorer keeps what
      // answers this question, not merely what is generally informative.
      ...(query.trim() ? { query: query.trim() } : {}),
    }),
  });

export const compressFiles = (
  files: File[],
  text: string,
  budgetRatio: number,
  mode: Mode = "auto",
  pins: ProviderPins = { embedding_provider: null, generation_provider: null },
  query = "",
) => {
  const body = new FormData();
  files.forEach((file) => body.append("files", file));
  if (text.trim()) body.append("text", text);
  body.append("budget_ratio", String(budgetRatio));
  body.append("fast_mode", "false");
  body.append("mode", mode);
  if (pins.embedding_provider)
    body.append("embedding_provider", pins.embedding_provider);
  if (pins.generation_provider)
    body.append("generation_provider", pins.generation_provider);
  if (query.trim()) body.append("query", query.trim());
  return request<CompressResponse>("/compress", { method: "POST", body });
};

/** Race the same question against the full and compressed contexts. */
export const askBoth = (
  text: string,
  question: string,
  name: string,
  mode: Mode = "auto",
  pins: ProviderPins = { embedding_provider: null, generation_provider: null },
) =>
  request<AnswerResponse>("/answer", {
    method: "POST",
    body: JSON.stringify({
      text,
      question,
      name,
      mode,
      ...(pins.embedding_provider
        ? { embedding_provider: pins.embedding_provider }
        : {}),
      ...(pins.generation_provider
        ? { generation_provider: pins.generation_provider }
        : {}),
    }),
  });
