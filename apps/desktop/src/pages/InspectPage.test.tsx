import userEvent from "@testing-library/user-event";
import { screen } from "@testing-library/react";

import { InspectPage } from "./InspectPage";
import { createMockApi, renderWithApp, setTestLocalStorageItem } from "../test/renderWithApp";

test("InspectPage renders inspect result rows", async () => {
  setTestLocalStorageItem(
    "gpt_rag_recent_inspect_queries",
    JSON.stringify(["socket timeout"]),
  );
  const user = userEvent.setup();
  const api = createMockApi({
    createJob: async () => ({
      job: {
        id: 7,
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
    getJob: async () => ({
      job: {
        id: 7,
        kind: "inspect",
        status: "completed",
        request_json: {},
        result_json: {
          query: "socket timeout",
          trace_path: "/tmp/inspect.json",
          diversity: {
            unique_document_count: 1,
          },
          results: [
            {
              chunk_id: 1,
              stable_id: "chunk-stable-1",
              final_rank: 1,
              title: "Socket Timeout Guide",
              fusion_score: 0.9,
              reranker_score: 0.8,
              lexical_score: 1,
              semantic_score: 0.75,
              section_title: "Networking",
              source_path: "/tmp/socket.md",
              text: "Increase the timeout threshold when sockets are unstable.",
            },
          ],
        },
        error_json: null,
        created_at: "",
        started_at: "",
        finished_at: "",
        heartbeat_at: "",
        cancel_requested: false,
        worker_id: "worker-1",
      },
      events: [],
    }),
  });

  renderWithApp(<InspectPage />, { api });

  expect(screen.getByRole("button", { name: "socket timeout" })).toBeInTheDocument();
  await user.type(screen.getByLabelText("Inspect query"), "socket timeout");
  await user.click(screen.getByRole("button", { name: "Queue inspect job" }));

  expect((await screen.findAllByText("Socket Timeout Guide")).length).toBeGreaterThanOrEqual(1);
  expect(screen.getByText("/tmp/inspect.json")).toBeInTheDocument();
  expect(
    screen.getAllByText("Increase the timeout threshold when sockets are unstable.").length,
  ).toBeGreaterThanOrEqual(2);
  expect(screen.getByText("Score breakdown")).toBeInTheDocument();
});
