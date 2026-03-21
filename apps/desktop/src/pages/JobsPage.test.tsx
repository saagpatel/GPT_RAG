import userEvent from "@testing-library/user-event";
import { screen } from "@testing-library/react";

import { JobsPage } from "./JobsPage";
import { createMockApi, renderWithApp } from "../test/renderWithApp";

test("JobsPage shows running jobs and progress events", async () => {
  const user = userEvent.setup();
  const api = createMockApi({
    listJobs: async () => ({
      jobs: [
        {
          id: 9,
          kind: "reindex_vectors",
          status: "running",
          request_json: {},
          result_json: null,
          error_json: null,
          created_at: "",
          started_at: "",
          finished_at: null,
          heartbeat_at: "",
          cancel_requested: false,
          worker_id: "worker-1",
        },
      ],
    }),
    getJob: async () => ({
      job: {
        id: 9,
        kind: "reindex_vectors",
        status: "running",
        request_json: {},
        result_json: null,
        error_json: null,
        created_at: "",
        started_at: "",
        finished_at: null,
        heartbeat_at: "",
        cancel_requested: false,
        worker_id: "worker-1",
      },
      events: [
        {
          id: 1,
          job_id: 9,
          sequence: 1,
          created_at: "",
          event_type: "reindex_batch",
          payload_json: {
            stage: "reindex_batch",
            message: "Vector indexing batch completed.",
            counts: {
              vectors_done: 4,
              vectors_total: 10,
            },
          },
        },
      ],
    }),
  });

  renderWithApp(<JobsPage />, { api });

  expect(await screen.findByText("reindex_vectors")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /reindex_vectors/i }));
  expect(await screen.findByText("Vector indexing batch completed.")).toBeInTheDocument();
  expect(screen.getByLabelText("Vectors progress")).toBeInTheDocument();
});
