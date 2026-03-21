import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";

import { App } from "./App";

const mockBootstrapSession = vi.fn();
const mockRestartSession = vi.fn();
const mockBackendStatus = vi.fn();

vi.mock("./lib/tauri", () => ({
  bootstrapSession: () => mockBootstrapSession(),
  restartSession: () => mockRestartSession(),
  backendStatus: () => mockBackendStatus(),
  pickFolder: vi.fn(),
}));

function renderApp() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>,
  );
}

test("App shows restart action after bootstrap failure", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        version: "0.1.0",
        runtime_ready: true,
        paths: {},
        models: {
          embedding: "qwen3-embedding:4b",
          generator: "qwen3:8b",
          reranker: "Qwen/Qwen3-Reranker-4B",
        },
        ollama: {
          base_url: "http://127.0.0.1:11434",
          is_local_endpoint: true,
          reachable: true,
          error: null,
          available_models: ["qwen3:8b"],
          embedding_model_available: true,
          generator_model_available: true,
        },
        reranker_cache: {
          available: true,
          dependencies_available: true,
          dependency_error: null,
          missing_files: [],
        },
        sqlite: {
          all_required_tables_present: true,
          required_tables: {},
        },
      }),
      text: async () => "",
    }),
  );
  mockBootstrapSession.mockRejectedValueOnce(new Error("boom"));
  mockRestartSession.mockResolvedValueOnce({
    apiBaseUrl: "http://127.0.0.1:8787",
    sessionToken: "token",
    version: "0.1.0",
    gptRagHome: "/tmp/gpt-rag-gui-test",
    runtimeMode: "dev",
    runtimeSource: "repo-venv",
  });
  mockBackendStatus.mockResolvedValue({
    sessionPresent: true,
    apiAlive: true,
    workerAlive: true,
    apiBaseUrl: "http://127.0.0.1:8787",
    gptRagHome: "/tmp/gpt-rag-gui-test",
    runtimeMode: "dev",
    runtimeSource: "repo-venv",
  });

  renderApp();

  expect(await screen.findByText("Desktop bootstrap failed")).toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "Restart local services" }));

  await waitFor(() => {
    expect(screen.getByText("GPT_RAG")).toBeInTheDocument();
  });
  expect(mockRestartSession).toHaveBeenCalledTimes(1);
  vi.unstubAllGlobals();
});
