import userEvent from "@testing-library/user-event";
import { screen } from "@testing-library/react";

import { LibraryPage } from "./LibraryPage";
import { createMockApi, renderWithApp } from "../test/renderWithApp";

test("LibraryPage queues ingest preview jobs", async () => {
  const user = userEvent.setup();
  const api = createMockApi({
    createJob: async (input) => ({
      job: {
        id: 42,
        kind: String(input.kind),
        status: "pending",
        request_json: input,
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
  });

  renderWithApp(<LibraryPage />, { api });

  await user.type(screen.getByLabelText("Manual library path"), "/Users/d/Knowledge");
  await user.click(screen.getByRole("button", { name: "Add path" }));
  await user.click(screen.getByRole("button", { name: "Preview ingest" }));

  expect(await screen.findByText("Queued ingest preview job #42.")).toBeInTheDocument();
});
