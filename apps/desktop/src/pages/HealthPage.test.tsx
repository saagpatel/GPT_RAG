import { screen } from "@testing-library/react";

import { HealthPage } from "./HealthPage";
import { createMockApi, renderWithApp } from "../test/renderWithApp";

test("HealthPage surfaces blocked runtime setup and runtime source details", async () => {
  const api = createMockApi({
    getHealth: async () => ({
      version: "0.1.0",
      runtime_ready: false,
      paths: {},
      models: {
        embedding: "qwen3-embedding:4b",
        generator: "qwen3:8b",
        reranker: "Qwen/Qwen3-Reranker-4B",
      },
      ollama: {
        base_url: "http://127.0.0.1:11434",
        is_local_endpoint: true,
        reachable: false,
        error: "Ollama is unreachable",
        available_models: [],
        embedding_model_available: false,
        generator_model_available: false,
      },
      reranker_cache: {
        available: false,
        dependencies_available: false,
        dependency_error: "sentence-transformers missing",
        missing_files: ["config.json"],
      },
      sqlite: {
        all_required_tables_present: false,
        required_tables: {},
      },
    }),
  });

  renderWithApp(<HealthPage />, {
    api,
    getBackendStatus: async () => ({
      sessionPresent: true,
      apiAlive: true,
      workerAlive: true,
      apiBaseUrl: "http://127.0.0.1:8787",
      gptRagHome: "/tmp/gpt-rag-gui-test",
      runtimeMode: "packaged",
      runtimeSource: "bundled-sidecar",
    }),
  });

  expect(await screen.findByText("Action needed")).toBeInTheDocument();
  expect(screen.getByText("Ollama")).toBeInTheDocument();
  expect(screen.getAllByText("Copy pull command")).toHaveLength(2);
  expect(screen.getByText("bundled-sidecar")).toBeInTheDocument();
  expect(screen.getAllByText("alive")).toHaveLength(2);
  expect(screen.getByText("http://127.0.0.1:8787")).toBeInTheDocument();
  expect(screen.getByText("packaged")).toBeInTheDocument();
});
