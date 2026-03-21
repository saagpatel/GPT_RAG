import { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";

import type { GuiApiLike } from "../lib/api";
import { SessionProvider } from "../lib/session";
import type { DesktopBackendStatus, SessionBootstrap } from "../types";

const defaultBootstrap: SessionBootstrap = {
  apiBaseUrl: "http://127.0.0.1:8787",
  sessionToken: "test-token",
  version: "0.1.0",
  gptRagHome: "/tmp/gpt-rag-gui-test",
  runtimeMode: "dev",
  runtimeSource: "repo-venv",
};

const defaultBackendStatus: DesktopBackendStatus = {
  sessionPresent: true,
  apiAlive: true,
  workerAlive: true,
  apiBaseUrl: defaultBootstrap.apiBaseUrl,
  gptRagHome: defaultBootstrap.gptRagHome,
  runtimeMode: defaultBootstrap.runtimeMode,
  runtimeSource: defaultBootstrap.runtimeSource,
};

function ensureTestLocalStorage() {
  const existing = globalThis.localStorage as Storage | undefined;
  if (
    existing &&
    typeof existing.getItem === "function" &&
    typeof existing.setItem === "function"
  ) {
    return existing;
  }

  let store: Record<string, string> = {};
  const nextStorage = {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => {
      store[key] = value;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
    key: (index: number) => Object.keys(store)[index] ?? null,
    get length() {
      return Object.keys(store).length;
    },
  } satisfies Storage;

  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: nextStorage,
  });
  return nextStorage;
}

export function setTestLocalStorageItem(key: string, value: string) {
  ensureTestLocalStorage().setItem(key, value);
}

export function createMockApi(
  overrides: Partial<GuiApiLike> = {},
): GuiApiLike {
  return {
    getHealth: async () => {
      throw new Error("getHealth not mocked");
    },
    initialize: async () => ({ status: "initialized" }),
    getReindexStatus: async () => ({
      status: "status",
      sqlite_path: "/tmp/rag.db",
      lancedb_path: "/tmp/vectors",
      embedding_model: "embed-model",
      chunk_count: 0,
      vector_count: 0,
      remaining_count: 0,
      completion_percentage: 0,
    }),
    search: async () => ({ query: "", mode: "lexical", results: [] }),
    createJob: async () => ({
      job: {
        id: 1,
        kind: "inspect",
        status: "pending",
        request_json: {},
        result_json: null,
        error_json: null,
        created_at: "",
        started_at: null,
        finished_at: null,
        heartbeat_at: null,
        cancel_requested: false,
        worker_id: null,
      },
    }),
    listJobs: async () => ({ jobs: [] }),
    getJob: async () => {
      throw new Error("getJob not mocked");
    },
    cancelJob: async () => ({
      job: {
        id: 1,
        kind: "inspect",
        status: "cancelled",
        request_json: {},
        result_json: null,
        error_json: null,
        created_at: "",
        started_at: null,
        finished_at: null,
        heartbeat_at: null,
        cancel_requested: true,
        worker_id: null,
      },
    }),
    listTraces: async () => ({ trace_path: "/tmp/traces", count: 0, traces: [] }),
    getTrace: async () => {
      throw new Error("getTrace not mocked");
    },
    jobsWebSocketUrl: () => "ws://127.0.0.1:8787/ws/jobs?token=test-token",
    ...overrides,
  };
}

export function renderWithApp(
  ui: ReactNode,
  {
    api = createMockApi(),
    bootstrap = defaultBootstrap,
    restartSession = async () => {},
    getBackendStatus = async () => defaultBackendStatus,
  }: {
    api?: GuiApiLike;
    bootstrap?: SessionBootstrap;
    restartSession?: () => Promise<void>;
    getBackendStatus?: () => Promise<DesktopBackendStatus>;
  } = {},
) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  ensureTestLocalStorage();

  return render(
    <QueryClientProvider client={queryClient}>
      <SessionProvider value={{ api, bootstrap, restartSession, getBackendStatus }}>
        <MemoryRouter>{ui}</MemoryRouter>
      </SessionProvider>
    </QueryClientProvider>,
  );
}
