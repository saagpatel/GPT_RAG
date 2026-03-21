import { FormEvent, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { useMutation, useQuery } from "@tanstack/react-query";

import { Card } from "../components/Card";
import { StatusPill } from "../components/StatusPill";
import { loadRecentValues, rememberRecentValue } from "../lib/localUiState";
import { useSession } from "../lib/session";

const RECENT_ASK_QUERIES_KEY = "gpt_rag_recent_ask_queries";

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : null;
}

export function AskPage() {
  const { api } = useSession();
  const [searchParams] = useSearchParams();
  const [query, setQuery] = useState(searchParams.get("query") ?? "");
  const [recentQueries, setRecentQueries] = useState<string[]>([]);
  const [jobId, setJobId] = useState<number | null>(null);

  const createMutation = useMutation({
    mutationFn: () =>
      api.createJob({
        kind: "ask",
        query,
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
    setRecentQueries(loadRecentValues(RECENT_ASK_QUERIES_KEY));
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
      rememberRecentValue(RECENT_ASK_QUERIES_KEY, current, query.trim()),
    );
    createMutation.mutate();
  }

  const resultPayload = jobQuery.data?.job.result_json ?? null;
  const generatedAnswer = asRecord(asRecord(resultPayload)?.generated_answer);
  const retrievalSnapshot = asRecord(asRecord(resultPayload)?.retrieval_snapshot);
  const answerContextDiversity = asRecord(asRecord(resultPayload)?.answer_context_diversity);
  const citations = Array.isArray(generatedAnswer?.citations)
    ? (generatedAnswer.citations as Array<Record<string, unknown> | string>)
    : [];
  const warnings = Array.isArray(generatedAnswer?.warnings)
    ? (generatedAnswer.warnings as string[])
    : [];
  const usedChunks = Array.isArray(generatedAnswer?.used_chunks)
    ? (generatedAnswer.used_chunks as Array<Record<string, unknown>>)
    : [];
  const retrievalSummary = asRecord(generatedAnswer?.retrieval_summary);
  const renderedAnswer = useMemo(
    () => (generatedAnswer ? String(generatedAnswer.answer ?? "") : ""),
    [generatedAnswer],
  );
  const insufficientEvidence =
    retrievalSummary?.generator_called === false ||
    (typeof retrievalSummary?.used_chunk_count === "number" &&
      Number(retrievalSummary.used_chunk_count) === 0);

  return (
    <div className="page-grid">
      <Card title="Ask grounded question" subtitle="Run retrieval plus grounded answer generation.">
        <form className="stack" onSubmit={handleSubmit}>
          <input
            aria-label="Ask query"
            placeholder="What does the local corpus say about socket timeouts?"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          {recentQueries.length ? (
            <div className="stack">
              <span className="metric-label">Recent ask queries</span>
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
          <button type="submit">Queue ask job</button>
        </form>
        {jobQuery.data ? (
          <p className="callout">
            Job #{jobQuery.data.job.id} <StatusPill status={jobQuery.data.job.status} />
          </p>
        ) : null}
      </Card>

      <Card title="Grounded answer" subtitle="Answer first, with warnings, citations, and retrieval detail below.">
        {generatedAnswer ? (
          <div className="stack">
            <div className={`answer-panel ${insufficientEvidence ? "answer-panel-cautious" : ""}`}>
              {insufficientEvidence ? (
                <div className="insufficient-banner">
                  <p className="eyebrow">Insufficient evidence</p>
                  <h3>The app did not find strong enough support to answer confidently.</h3>
                </div>
              ) : null}
              <p className="answer-body">{renderedAnswer}</p>
            </div>
            {warnings.length ? (
              <div className="callout negative">
                <h3 className="panel-title">Warnings</h3>
                <ul className="flat-list">
                  {warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </div>
            ) : null}

            <div className="stack">
              <h3 className="panel-title">Citations</h3>
              {citations.length ? (
                <div className="citation-list">
                  {citations.map((citation) => {
                    if (typeof citation === "string") {
                      return (
                        <div className="citation-card" key={citation}>
                          <strong>{citation}</strong>
                        </div>
                      );
                    }
                    const label = String(citation.label ?? citation.display ?? "Citation");
                    return (
                      <div className="citation-card" key={label}>
                        <strong>{label}</strong>
                        <div className="table-subtext">
                          {String(citation.display ?? citation.source_path ?? "-")}
                        </div>
                        {"quote" in citation ? (
                          <p className="citation-quote">{String(citation.quote ?? "")}</p>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p>No citations were attached to this answer.</p>
              )}
            </div>

            <details className="details-panel" open>
              <summary>Used chunks and retrieval summary</summary>
              <div className="stack">
                {retrievalSummary ? (
                  <div className="metric-grid">
                    <div>
                      <span className="metric-label">Retrieved chunks</span>
                      <strong>{String(retrievalSummary.retrieved_count ?? "-")}</strong>
                    </div>
                    <div>
                      <span className="metric-label">Used chunks</span>
                      <strong>{String(retrievalSummary.used_chunk_count ?? "-")}</strong>
                    </div>
                    <div>
                      <span className="metric-label">Cited chunks</span>
                      <strong>{String(retrievalSummary.cited_chunk_count ?? "-")}</strong>
                    </div>
                    <div>
                      <span className="metric-label">Generator called</span>
                      <strong>{String(retrievalSummary.generator_called ?? "-")}</strong>
                    </div>
                  </div>
                ) : null}
                {usedChunks.length ? (
                  <div className="used-chunk-list">
                    {usedChunks.map((chunk) => (
                      <div className="detail-panel" key={String(chunk.label ?? chunk.chunk_id)}>
                        <strong>{String(chunk.label ?? "Chunk")}</strong>
                        <div className="table-subtext">
                          {String(chunk.document_title ?? chunk.source_path ?? "-")}
                        </div>
                        <div className="table-subtext mono-text">
                          {String(chunk.source_path ?? "-")}
                        </div>
                        <p>{String(chunk.chunk_text_excerpt ?? chunk.text ?? "")}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="table-subtext">
                    No used chunks were recorded for this answer run.
                  </p>
                )}
              </div>
            </details>

            <details className="details-panel">
              <summary>Retrieval snapshot and answer context diversity</summary>
              <div className="stack">
                {retrievalSnapshot ? (
                  <div className="metric-grid">
                    <div>
                      <span className="metric-label">Snapshot id</span>
                      <strong>{String(retrievalSnapshot.snapshot_id ?? "-")}</strong>
                    </div>
                    <div>
                      <span className="metric-label">Result count</span>
                      <strong>{String(retrievalSnapshot.result_count ?? "-")}</strong>
                    </div>
                    <div>
                      <span className="metric-label">Trace path</span>
                      <strong className="mono-text">
                        {String(retrievalSnapshot.trace_path ?? resultPayload?.trace_path ?? "-")}
                      </strong>
                    </div>
                  </div>
                ) : null}
                {answerContextDiversity ? (
                  <div className="metric-grid">
                    <div>
                      <span className="metric-label">Used chunk count</span>
                      <strong>{String(answerContextDiversity.used_chunk_count ?? "-")}</strong>
                    </div>
                    <div>
                      <span className="metric-label">Unique documents</span>
                      <strong>{String(answerContextDiversity.unique_document_count ?? "-")}</strong>
                    </div>
                  </div>
                ) : null}
              </div>
            </details>
          </div>
        ) : (
          <p>No answer job has completed yet.</p>
        )}
      </Card>
    </div>
  );
}
