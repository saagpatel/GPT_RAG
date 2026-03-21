import userEvent from "@testing-library/user-event";
import { screen } from "@testing-library/react";

import { AskPage } from "./AskPage";
import { createMockApi, renderWithApp } from "../test/renderWithApp";

test("AskPage renders answer text, citations, and insufficient-evidence state", async () => {
  const user = userEvent.setup();
  const api = createMockApi({
    createJob: async () => ({
      job: {
        id: 11,
        kind: "ask",
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
    getJob: async () => ({
      job: {
        id: 11,
        kind: "ask",
        status: "completed",
        request_json: {},
        result_json: {
          generated_answer: {
            answer: "The corpus says to increase the socket timeout. [C1]",
            citations: [
              {
                label: "C1",
                display: "Socket Timeout Guide - socket_timeout_guide.md",
                quote: "Increase the timeout value when network calls stall.",
              },
            ],
            warnings: ["Limited evidence."],
            retrieval_summary: {
              retrieved_count: 2,
              used_chunk_count: 0,
              cited_chunk_count: 1,
              generator_called: false,
            },
            used_chunks: [
              {
                label: "C1",
                document_title: "Socket Timeout Guide",
                source_path: "/tmp/socket_timeout_guide.md",
                chunk_text_excerpt: "Increase the timeout value when network calls stall.",
              },
            ],
          },
          retrieval_snapshot: {
            snapshot_id: "snapshot-1",
            result_count: 2,
            trace_path: "/tmp/ask-trace.json",
          },
          answer_context_diversity: {
            used_chunk_count: 1,
            unique_document_count: 1,
          },
        },
        error_json: null,
        created_at: "",
        started_at: "",
        finished_at: "",
        heartbeat_at: "",
        cancel_requested: false,
        worker_id: "worker-2",
      },
      events: [],
    }),
  });

  renderWithApp(<AskPage />, { api });

  await user.type(
    screen.getByLabelText("Ask query"),
    "What does the local corpus say about socket timeouts?",
  );
  await user.click(screen.getByRole("button", { name: "Queue ask job" }));

  expect(
    await screen.findByText("The corpus says to increase the socket timeout. [C1]"),
  ).toBeInTheDocument();
  expect(screen.getByText("Insufficient evidence")).toBeInTheDocument();
  expect(screen.getByText("Limited evidence.")).toBeInTheDocument();
  expect(screen.getByText("Socket Timeout Guide - socket_timeout_guide.md")).toBeInTheDocument();
  expect(screen.getByText("/tmp/ask-trace.json")).toBeInTheDocument();
});
