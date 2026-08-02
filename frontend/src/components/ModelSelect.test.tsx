import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ModelSelect, presetToPins, providerLabel } from "./ModelSelect";
import type { ProviderCatalogue, SelectionPreset } from "../types";

/**
 * One dropdown, every model. The parts worth pinning down:
 * a model that cannot run is never selectable, the embedding provider the
 * choice resolves to is disclosed rather than hidden, and the id selected maps
 * to the exact request fields the backend expects.
 */

function preset(
  id: string,
  label: string,
  overrides: Partial<SelectionPreset> = {},
): SelectionPreset {
  return {
    id,
    label,
    detail: `embeddings for ${id}`,
    mode: "auto",
    embedding_provider: id === "local" ? "local" : "cohere",
    generation_provider: id === "auto" ? null : id,
    local: id === "local",
    available: true,
    reason: null,
    ...overrides,
  };
}

const CATALOGUE: ProviderCatalogue = {
  embedding: [],
  generation: [],
  selections: [
    preset("auto", "Auto — best available", {
      embedding_provider: null,
      generation_provider: null,
    }),
    preset("local", "Local — llama3.2:3b", { mode: "local" }),
    preset("groq", "groq — llama-3.3-70b-versatile"),
    preset("gemini", "gemini — gemini-2.0-flash", {
      available: false,
      reason: "GOOGLE_API_KEY is not configured",
    }),
    preset("openai", "openai — gpt-4o-mini"),
  ],
  default_chains: { embedding: ["cohere", "local"], generation: ["groq", "local"] },
  modes: { available: ["local", "cloud", "auto"], default: "auto" },
  note: "",
};

function renderSelect(
  catalogue: ProviderCatalogue | undefined = CATALOGUE,
  value = "auto",
) {
  const onChange = vi.fn();
  render(
    <ModelSelect catalogue={catalogue} value={value} onChange={onChange} />,
  );
  return { onChange, select: screen.getByLabelText("Model") as HTMLSelectElement };
}

describe("ModelSelect", () => {
  it("renders exactly one control for the whole choice", () => {
    renderSelect();
    expect(screen.getAllByRole("combobox")).toHaveLength(1);
  });

  it("lists every model, local and cloud, in one list", () => {
    const { select } = renderSelect();
    const values = Array.from(select.querySelectorAll("option")).map(
      (o) => (o as HTMLOptionElement).value,
    );
    expect(values).toEqual(["auto", "local", "groq", "gemini", "openai"]);
  });

  it("shows the concrete model id in each label", () => {
    const { select } = renderSelect();
    const labels = Array.from(select.querySelectorAll("option")).map(
      (o) => o.textContent ?? "",
    );
    expect(labels.some((l) => l.includes("llama-3.3-70b-versatile"))).toBe(true);
    expect(labels.some((l) => l.includes("llama3.2:3b"))).toBe(true);
  });

  it("disables a model that cannot run and says why", () => {
    const { select } = renderSelect();
    const gemini = Array.from(select.querySelectorAll("option")).find(
      (o) => (o as HTMLOptionElement).value === "gemini",
    ) as HTMLOptionElement;
    expect(gemini.disabled).toBe(true);
    expect(gemini.textContent).toContain("GOOGLE_API_KEY is not configured");
  });

  it("discloses which embedding provider the choice resolves to", () => {
    // A compression uses two models; hiding the second would be a lie of
    // omission about what produced the numbers.
    renderSelect(CATALOGUE, "groq");
    expect(screen.getByText(/embeddings for groq/)).toBeInTheDocument();
  });

  it("warns that a specific model has no fallback", () => {
    renderSelect(CATALOGUE, "groq");
    expect(screen.getByText(/pinned, no fallback if it fails/)).toBeInTheDocument();
  });

  it("does not warn about fallback on Auto, which is the fallback", () => {
    renderSelect(CATALOGUE, "auto");
    expect(screen.queryByText(/no fallback/)).toBeNull();
  });

  it("reports the chosen id back", () => {
    const { onChange, select } = renderSelect();
    fireEvent.change(select, { target: { value: "groq" } });
    expect(onChange).toHaveBeenCalledWith("groq");
  });

  it("is inert before the catalogue loads, rather than offering nothing", () => {
    // Rendered directly: passing `undefined` to renderSelect would hit its
    // default parameter and silently test the loaded case instead.
    render(<ModelSelect catalogue={undefined} value="auto" onChange={vi.fn()} />);
    const select = screen.getByLabelText("Model") as HTMLSelectElement;
    expect(select).toBeDisabled();
    expect(select.textContent).toContain("Loading models");
  });
});

describe("presetToPins", () => {
  it("maps a specific model to explicit request fields", () => {
    const pins = presetToPins(CATALOGUE.selections.find((p) => p.id === "groq"));
    expect(pins).toEqual({
      mode: "auto",
      embedding_provider: "cohere",
      generation_provider: "groq",
    });
  });

  it("maps Auto to no pins at all, so the chain is used", () => {
    const pins = presetToPins(CATALOGUE.selections.find((p) => p.id === "auto"));
    expect(pins).toEqual({
      mode: "auto",
      embedding_provider: null,
      generation_provider: null,
    });
  });

  it("maps Local to local mode on both roles", () => {
    const pins = presetToPins(CATALOGUE.selections.find((p) => p.id === "local"));
    expect(pins).toEqual({
      mode: "local",
      embedding_provider: "local",
      generation_provider: "local",
    });
  });

  it("falls back to auto for an unknown id", () => {
    expect(presetToPins(undefined).mode).toBe("auto");
  });
});

describe("providerLabel", () => {
  it("distinguishes local from cloud", () => {
    expect(providerLabel("local")).toBe("Llama 3.2 3B (local)");
    expect(providerLabel("groq")).toBe("Groq (cloud)");
  });

  it("falls back to the raw name for an unknown provider", () => {
    expect(providerLabel("new-vendor")).toBe("new-vendor");
  });

  it("calls an absent provider 'none', not 'unknown'", () => {
    // A stage with no provider is explainable (nothing to embed, or a pinned
    // provider failed with no fallback), not a mystery.
    expect(providerLabel(undefined)).toBe("none");
  });
});
