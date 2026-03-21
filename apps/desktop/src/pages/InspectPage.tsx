import { FormEvent, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { useMutation, useQuery } from "@tanstack/react-query";

import { Card } from "../components/Card";
import { StatusPill } from "../components/StatusPill";
import { loadRecentValues, rememberRecentValue } from "../lib/localUiState";
import { useSession } from "../lib/session";
import type { SearchResult } from "../types";

const RECENT_INSPECT_QUERIES_KEY = "gpt_rag_recent_inspect_queries";

function formatScore(value: unknown): string {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

export function InspectPage() {
  const { api } = useSession();
  const [searchParams] = useSearchParams();
  const [query, setQuery] = useState(searchParams.get("query") ?? "");
  const [recentQueries, setRecentQueries] = useState<string[]>([]);
  const [jobId, setJobId] = useState<number | null>(null);
  const [selectedChunkId, setSelectedChunkId] = useState<number | null>(null);

  const createMutation = useMutation({
    mutationFn: () =>
      api.createJob({
        kind: "inspect",
        query,
        limit: 6,
        save_trace: true,
      }),
    onSuccess: (response) => setJobId(response.job.id),
  });
  const jobQuery = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.getJob(jobId as number),
    enabled: jobId !== null,
    refetchInterval: 1500,
  });

  useEffect(() => {
    setRecentQueries(loadRecentValues(RECENT_INSPECT_QUERIES_KEY));
  }, []);

  useEffect(() => {
    const fromRoute = searchParams.get("query");
    if (fromRoute) {
      setQuery(fromRoute);
    }
  }, [searchParams]);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) {
      return;
    }
    setRecentQueries((current) =>
      rememberRecentValue(RECENT_INSPECT_QUERIES_KEY, current, query.trim()),
    );
    createMutation.mutate();
  }

  const resultPayload = jobQuery.data?.job.result_json;
  const results = useMemo(
    () => (Array.isArray(resultPayload?.results) ? (resultPayload.results as SearchResult[]) : []),
    [resultPayload],
  );
  const selectedResult =
    results.find((result) => result.chunk_id === selectedChunkId) ?? results[0] ?? null;

  useEffect(() => {
    if (!results.length) {
      setSelectedChunkId(null);
      return;
    }
    if (!results.some((result) => result.chunk_id === selectedChunkId)) {
      setSelectedChunkId(results[0].chunk_id);
    }
  }, [results, selectedChunkId]);

  return (
    <div className="page-grid">
      <Card title="Inspect retrieval" subtitle="Run the full inspect workflow as a tracked job.">
        <form className="stack" onSubmit={handleSubmit}>
          <input
            aria-label="Inspect query"
            placeholder="socket timeout"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          {recentQueries.length ? (
            <div className="stack">
              <span className="metric-label">Recent inspect queries</span>
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
          <button type="submit">Queue inspect job</button>
        </form>
        {jobQuery.data ? (
          <p className="callout">
            Job #{jobQuery.data.job.id} <StatusPill status={jobQuery.data.job.status} />
          </p>
        ) : null}
      </Card>

      <Card title="Inspect summary" subtitle="Query details, trace output, and source spread at a glance.">
        {resultPayload ? (
          <div className="metric-grid">
            <div>
              <span className="metric-label">Query</span>
              <strong>{String(resultPayload.query ?? query)}</strong>
            </div>
            <div>
              <span className="metric-label">Result count</span>
              <strong>{results.length}</strong>
            </div>
            <div>
              <span className="metric-label">Trace path</span>
              <strong className="mono-text">{String(resultPayload.trace_path ?? "-")}</strong>
            </div>
            <div>
              <span className="metric-label">Unique documents</span>
              <strong>
                {String(
                  (resultPayload.diversity as Record<string, unknown> | undefined)
                    ?.unique_document_count ?? "-",
                )}
              </strong>
            </div>
          </div>
        ) : (
          <p>No inspect results yet.</p>
        )}
      </Card>

      <Card title="Ranked chunks" subtitle="Source path and excerpt stay visually primary, with score details still inspectable.">
        {results.length ? (
          <div className="inspect-layout">
            <div className="inspect-results-list">
              {results.map((result) => (
                <button
                  className={`inspect-result-card ${
                    selectedResult?.chunk_id === result.chunk_id ? "inspect-result-card-selected" : ""
                  }`}
                  key={result.chunk_id}
                  onClick={() => setSelectedChunkId(result.chunk_id)}
                  type="button"
                >
                  <div className="result-card-header">
                    <div>
                      <h3>{result.title ?? "Untitled"}</h3>
                      <p className="table-subtext">
                        {result.section_title ?? "No section"} · rank {result.final_rank ?? "-"}
                      </p>
                    </div>
                    <StatusPill status={`#${result.final_rank ?? "-"}`} />
                  </div>
                  <p className="mono-text">{result.source_path}</p>
                  <p className="result-excerpt">{result.text ?? "No chunk text available."}</p>
                  <div className="chip-list">
                    {result.exact_title_match ? <span className="chip">Exact title</span> : null}
                    {result.exact_source_name_match ? <span className="chip">Exact file</span> : null}
                    {result.phrase_match ? <span className="chip">Phrase</span> : null}
                  </div>
                </button>
              ))}
            </div>

            {selectedResult ? (
              <div className="detail-panel">
                <div className="detail-panel-header">
                  <div>
                    <p className="eyebrow">Selected chunk</p>
                    <h3>{selectedResult.title ?? "Untitled"}</h3>
                  </div>
                  <StatusPill status={jobQuery.data?.job.status ?? "pending"} />
                </div>
                <div className="metric-grid">
                  <div>
                    <span className="metric-label">Source path</span>
                    <strong className="mono-text">{selectedResult.source_path}</strong>
                  </div>
                  <div>
                    <span className="metric-label">Section</span>
                    <strong>{selectedResult.section_title ?? "-"}</strong>
                  </div>
                  <div>
                    <span className="metric-label">Chunk id</span>
                    <strong>{selectedResult.chunk_id}</strong>
                  </div>
                  <div>
                    <span className="metric-label">Stable id</span>
                    <strong className="mono-text">
                      {String(selectedResult.stable_id ?? "-")}
                    </strong>
                  </div>
                </div>
                <div className="excerpt-panel">
                  <p>{selectedResult.text ?? "No chunk text available."}</p>
                </div>
                <details className="details-panel" open>
                  <summary>Score breakdown</summary>
                  <div className="metric-grid">
                    <div>
                      <span className="metric-label">Lexical</span>
                      <strong>{formatScore(selectedResult.lexical_score)}</strong>
                    </div>
                    <div>
                      <span className="metric-label">Semantic</span>
                      <strong>{formatScore(selectedResult.semantic_score)}</strong>
                    </div>
                    <div>
                      <span className="metric-label">Fusion</span>
                      <strong>{formatScore(selectedResult.fusion_score)}</strong>
                    </div>
                    <div>
                      <span className="metric-label">Reranker</span>
                      <strong>{formatScore(selectedResult.reranker_score)}</strong>
                    </div>
                  </div>
                </details>
              </div>
            ) : null}
          </div>
        ) : (
          <p>No inspect results yet.</p>
        )}
      </Card>
    </div>
  );
}
