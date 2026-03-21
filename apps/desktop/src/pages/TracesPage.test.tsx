import userEvent from "@testing-library/user-event";
import { screen } from "@testing-library/react";
import { vi } from "vitest";

import { TracesPage } from "./TracesPage";
import { createMockApi, renderWithApp } from "../test/renderWithApp";

test("TracesPage renders summary cards and selected trace metadata", async () => {
  vi.stubGlobal("navigator", {
    clipboard: {
      writeText: vi.fn().mockResolvedValue(undefined),
    },
  });
  const user = userEvent.setup();
  const api = createMockApi({
    listTraces: async () => ({
      trace_path: "/tmp/traces",
      count: 2,
      traces: [
        {
          path: "/tmp/traces/inspect.json",
          name: "inspect.json",
          type: "inspect",
          timestamp: "2026-03-15T10:00:00Z",
          query: "socket timeout",
          size_bytes: 128,
        },
        {
          path: "/tmp/traces/ask.json",
          name: "ask.json",
          type: "ask",
          timestamp: "2026-03-15T10:05:00Z",
          query: "What does the corpus say about socket timeouts?",
          size_bytes: 256,
        },
      ],
    }),
    getTrace: async () => ({
      metadata: {
        path: "/tmp/traces/ask.json",
        name: "ask.json",
        type: "ask",
        timestamp: "2026-03-15T10:05:00Z",
        query: "What does the corpus say about socket timeouts?",
        size_bytes: 256,
      },
      payload: {
        query: "What does the corpus say about socket timeouts?",
        generated_answer: {
          answer: "Increase the timeout. [C1]",
        },
      },
    }),
  });

  renderWithApp(<TracesPage />, { api });

  expect(await screen.findByText("/tmp/traces")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /ask/i }));
  await user.click(screen.getByRole("button", { name: "Copy path" }));

  expect(await screen.findByText("ask.json")).toBeInTheDocument();
  expect(screen.getByText("256 bytes")).toBeInTheDocument();
  expect(screen.getByText("Trace path copied.")).toBeInTheDocument();
  vi.unstubAllGlobals();
});
