import userEvent from "@testing-library/user-event";
import { screen } from "@testing-library/react";

import { SearchPage } from "./SearchPage";
import { createMockApi, renderWithApp, setTestLocalStorageItem } from "../test/renderWithApp";

test("SearchPage renders recent queries and result cards", async () => {
  setTestLocalStorageItem(
    "gpt_rag_recent_search_queries",
    JSON.stringify(["pgvector", "socket timeout"]),
  );
  const user = userEvent.setup();
  const api = createMockApi({
    search: async () => ({
      query: "pgvector",
      mode: "lexical",
      results: [
        {
          chunk_id: 1,
          title: "pgvector Usage Guide",
          section_title: "Extensions",
          source_path: "/tmp/pgvector.md",
          text: "Install the pgvector extension before creating embeddings columns.",
          lexical_score: 1,
          semantic_score: 0.2,
          fusion_score: 0.8,
          reranker_score: 0.7,
          final_rank: 1,
        },
      ],
    }),
  });

  renderWithApp(<SearchPage />, { api });

  await user.click(screen.getByRole("button", { name: "pgvector" }));
  await user.click(screen.getByRole("button", { name: "Search" }));

  expect(await screen.findByText("pgvector Usage Guide")).toBeInTheDocument();
  expect(
    screen.getByText("Install the pgvector extension before creating embeddings columns."),
  ).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Open in Inspect" })).toBeInTheDocument();
});
