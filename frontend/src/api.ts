import type {
  CompressResponse,
  EvaluateResponse,
  Health,
} from "./types";

// Dev goes through Vite's proxy; a deployed build points at VITE_API_URL.
const BASE = import.meta.env.VITE_API_URL ?? "/api";

/** Unwraps the backend's `{error: {code, message}}` envelope into a real Error. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
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

export const getEvaluation = () =>
  request<EvaluateResponse>("/evaluate", {
    method: "POST",
    // Serves the last measured report instantly. A live run is minutes of local
    // inference and must never block the UI.
    body: JSON.stringify({ run: false }),
  });

export const compress = (text: string, budgetRatio: number, name: string) =>
  request<CompressResponse>("/compress", {
    method: "POST",
    body: JSON.stringify({
      text,
      name,
      budget_ratio: budgetRatio,
      fast_mode: true,
    }),
  });
