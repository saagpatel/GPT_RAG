import { FormEvent, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { useMutation } from "@tanstack/react-query";

import { Card } from "../components/Card";
import { rememberRecentValue, loadRecentValues } from "../lib/localUiState";
import { useSession } from "../lib/session";
import type { SearchMode } from "../types";

const RECENT_SEARCH_QUERIES_KEY = "gpt_rag_recent_search_queries";

export function SearchPage() {
  const { api } = useSession();
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [limit, setLimit] = useState("8");
  const [recentQueries, setRecentQueries] = useState<string[]>([]);
  const mutation = useMutation({
    mutationFn: () =>
      api.search({
        query,
        mode,
        limit: Number(limit) || 8,
      }),
  });

  useEffect(() => {
    setRecentQueries(loadRecentValues(RECENT_SEARCH_QUERIES_KEY));
  }, []);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) {
      return;
    }
    setRecentQueries((current) =>
      rememberRecentValue(RECENT_SEARCH_QUERIES_KEY, current, query.trim()),
    );
    mutation.mutate();
  }

  return (
    <div className="page-grid">
      <Card title="Search" subtitle="Use quick synchronous retrieval to scout your library before a deeper inspect or ask run.">
        <form className="stack" onSubmit={handleSubmit}>
          <input
            aria-label="Search query"
            placeholder="socket timeout"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <div className="inline-fields">
            <label>
              Mode
              <select value={mode} onChange={(event) => setMode(event.target.value as SearchMode)}>
                <option value="lexical">lexical</option>
                <option value="semantic">semantic</option>
                <option value="hybrid">hybrid</option>
              </select>
            </label>
            <label>
              Limit
              <input value={limit} onChange={(event) => setLimit(event.target.value)} />
            </label>
          </div>
          {recentQueries.length ? (
            <div className="stack">
              <span className="metric-label">Recent queries</span>
              <div className="chip-list">
                {recentQueries.map((recentQuery) => (
                  <button
                    key={recentQuery}
                    className="chip"
                    onClick={() => setQuery(recentQuery)}
                    type="button"
                  >
                    {recentQuery}
                  </button>
                ))}
              </div>
            </div>
          ) : null}
          <button type="submit">Search</button>
        </form>
      </Card>

      <Card title="Results" subtitle="File name, section, and excerpt first. Score details stay visible but secondary.">
        {mutation.isPending ? <p>Running retrieval…</p> : null}
        {mutation.data?.results.length ? (
          <div className="stack">
            {mutation.data.results.map((result) => (
              <div className="result-card" key={result.chunk_id}>
                <div className="result-card-header">
                  <div>
                    <h3>{result.title ?? "Untitled"}</h3>
                    <p className="table-subtext">
                      {result.section_title ?? "No section"} · rank {result.final_rank ?? "-"}
                    </p>
                  </div>
                  <button
                    onClick={() =>
                      navigate(`/inspect?query=${encodeURIComponent(mutation.data.query)}`)
                    }
                    type="button"
                  >
                    Open in Inspect
                  </button>
                </div>
                <p className="mono-text">{result.source_path}</p>
                <p>{result.text ?? "No excerpt available yet."}</p>
                <div className="result-score-row">
                  <span>Lexical {result.lexical_score ?? "-"}</span>
                  <span>Semantic {result.semantic_score ?? "-"}</span>
                  <span>Fusion {result.fusion_score ?? "-"}</span>
                  <span>Reranker {result.reranker_score ?? "-"}</span>
                </div>
              </div>
            ))}
          </div>
        ) : mutation.isSuccess ? (
          <div className="empty-state">
            <h3>No matching chunks yet</h3>
            <p>Try a broader query, switch to lexical mode, or ingest more notes first.</p>
          </div>
        ) : (
          <div className="empty-state">
            <h3>No search results yet</h3>
            <p>Run a search to quickly scan what the local library can already retrieve.</p>
          </div>
        )}
      </Card>
    </div>
  );
}
