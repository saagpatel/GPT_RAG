import { FormEvent, useEffect, useMemo, useState } from "react";

import { useMutation, useQuery } from "@tanstack/react-query";

import { Card } from "../components/Card";
import { StatusPill } from "../components/StatusPill";
import { loadRecentValues, persistRecentValues, rememberRecentValue } from "../lib/localUiState";
import { useSession } from "../lib/session";
import { pickFolder } from "../lib/tauri";

const RECENT_PATHS_KEY = "gpt_rag_recent_library_paths";

function latestCompletedReindex(jobs: Array<{ kind: string; status: string; result_json: Record<string, unknown> | null }>) {
  return jobs.find((job) => job.kind === "reindex_vectors" && job.status === "completed");
}

export function LibraryPage() {
  const { api } = useSession();
  const [selectedPaths, setSelectedPaths] = useState<string[]>([]);
  const [manualPath, setManualPath] = useState("");
  const [recentPaths, setRecentPaths] = useState<string[]>([]);
  const [pathError, setPathError] = useState<string | null>(null);
  const [skipEmbeddings, setSkipEmbeddings] = useState(true);
  const [batchSize, setBatchSize] = useState("64");
  const [untilSeconds, setUntilSeconds] = useState("60");

  useEffect(() => {
    setRecentPaths(loadRecentValues(RECENT_PATHS_KEY));
  }, []);

  useEffect(() => {
    persistRecentValues(RECENT_PATHS_KEY, recentPaths);
  }, [recentPaths]);

  const vectorStatusQuery = useQuery({
    queryKey: ["vector-status"],
    queryFn: () => api.getReindexStatus(),
    refetchInterval: 3000,
  });
  const jobsQuery = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.listJobs(),
    refetchInterval: 3000,
  });
  const previewMutation = useMutation({
    mutationFn: (paths: string[]) =>
      api.createJob({
        kind: "ingest_preview",
        paths,
        skip_embeddings: true,
      }),
  });
  const ingestMutation = useMutation({
    mutationFn: (paths: string[]) =>
      api.createJob({
        kind: "ingest_run",
        paths,
        skip_embeddings: skipEmbeddings,
        batch_size: Number(batchSize) || undefined,
      }),
  });
  const reindexMutation = useMutation({
    mutationFn: () =>
      api.createJob({
        kind: "reindex_vectors",
        resume: true,
        until_seconds: Number(untilSeconds) || undefined,
        batch_size: Number(batchSize) || undefined,
      }),
  });

  const lastReindexJob = useMemo(
    () => latestCompletedReindex(jobsQuery.data?.jobs ?? []),
    [jobsQuery.data?.jobs],
  );
  const lastReindexResult =
    lastReindexJob && lastReindexJob.result_json ? lastReindexJob.result_json : null;

  function validatePaths(): string[] | null {
    if (!selectedPaths.length) {
      setPathError("Add at least one local folder before queueing ingest work.");
      return null;
    }
    setPathError(null);
    return selectedPaths;
  }

  function addPath(nextPath: string) {
    const normalized = nextPath.trim();
    if (!normalized) {
      return;
    }
    setSelectedPaths((current) =>
      current.includes(normalized) ? current : [...current, normalized],
    );
    setRecentPaths((current) => rememberRecentValue(RECENT_PATHS_KEY, current, normalized));
    setManualPath("");
    setPathError(null);
  }

  async function handlePickFolder() {
    const selected = await pickFolder();
    if (selected) {
      addPath(selected);
    }
  }

  function handleAddManualPath(event: FormEvent) {
    event.preventDefault();
    addPath(manualPath);
  }

  function handlePreview(event: FormEvent) {
    event.preventDefault();
    const validPaths = validatePaths();
    if (validPaths) {
      previewMutation.mutate(validPaths);
    }
  }

  function handleIngest() {
    const validPaths = validatePaths();
    if (validPaths) {
      ingestMutation.mutate(validPaths);
    }
  }

  return (
    <div className="page-grid">
      <Card
        title="Vector completion"
        subtitle="Large-corpus work stays operationally clear here: ingest quickly first, then resume vectors on your own budget."
        actions={
          vectorStatusQuery.data ? (
            <StatusPill status={`${vectorStatusQuery.data.completion_percentage}%`} />
          ) : null
        }
      >
        {vectorStatusQuery.data ? (
          <div className="stack">
            <div className="metric-grid">
              <div>
                <span className="metric-label">Chunks in SQLite</span>
                <strong>{vectorStatusQuery.data.chunk_count}</strong>
              </div>
              <div>
                <span className="metric-label">Vectors present</span>
                <strong>{vectorStatusQuery.data.vector_count}</strong>
              </div>
              <div>
                <span className="metric-label">Remaining chunks</span>
                <strong>{vectorStatusQuery.data.remaining_count}</strong>
              </div>
              <div>
                <span className="metric-label">Embedding model</span>
                <strong>{vectorStatusQuery.data.embedding_model}</strong>
              </div>
            </div>
            <div className="progress-row">
              <header>
                <span>Vector completion</span>
                <span>{vectorStatusQuery.data.completion_percentage}%</span>
              </header>
              <div aria-label="Vector completion progress" className="progress-bar">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${vectorStatusQuery.data.completion_percentage}%` }}
                />
              </div>
            </div>
          </div>
        ) : (
          <p>Loading vector completion…</p>
        )}
      </Card>

      <Card title="Choose local folders" subtitle="Use the native picker or add paths manually.">
        <div className="stack">
          <div className="button-row">
            <button onClick={() => void handlePickFolder()} type="button">
              Pick folder
            </button>
          </div>

          <form className="inline-fields" onSubmit={handleAddManualPath}>
            <label>
              Add path manually
              <input
                aria-label="Manual library path"
                placeholder="/Users/d/Knowledge"
                value={manualPath}
                onChange={(event) => setManualPath(event.target.value)}
              />
            </label>
            <div className="inline-action">
              <span className="metric-label">Manual path action</span>
              <button type="submit">Add path</button>
            </div>
          </form>

          {selectedPaths.length ? (
            <div className="stack">
              <span className="metric-label">Selected folders</span>
              <div className="chip-list">
                {selectedPaths.map((path) => (
                  <button
                    key={path}
                    className="chip removable-chip"
                    onClick={() =>
                      setSelectedPaths((current) => current.filter((item) => item !== path))
                    }
                    type="button"
                  >
                    {path}
                    <span aria-hidden="true">×</span>
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          {recentPaths.length ? (
            <div className="stack">
              <span className="metric-label">Recent folders</span>
              <div className="chip-list">
                {recentPaths.map((path) => (
                  <button key={path} className="chip" onClick={() => addPath(path)} type="button">
                    {path}
                  </button>
                ))}
              </div>
            </div>
          ) : null}
          {pathError ? <p className="callout negative">{pathError}</p> : null}
        </div>
      </Card>

      <Card title="Fast ingest now" subtitle="Get SQLite + lexical search ready first, then come back for vectors.">
        <div className="stack">
          <p className="table-subtext">
            This is the recommended large-corpus path for your notes: preview first, then ingest
            with embeddings skipped so the library becomes usable quickly.
          </p>
          <form className="stack" onSubmit={handlePreview}>
            <label className="checkbox-row">
              <input
                checked={skipEmbeddings}
                onChange={(event) => setSkipEmbeddings(event.target.checked)}
                type="checkbox"
              />
              Skip embeddings during ingest
            </label>
            <label>
              Batch size for any embedding work that does happen
              <input
                aria-label="Batch size"
                value={batchSize}
                onChange={(event) => setBatchSize(event.target.value)}
              />
            </label>
            <div className="button-row">
              <button type="submit">Preview ingest</button>
              <button onClick={handleIngest} type="button">
                Run ingest
              </button>
            </div>
          </form>
          {previewMutation.data ? (
            <p className="callout">Queued ingest preview job #{previewMutation.data.job.id}.</p>
          ) : null}
          {ingestMutation.data ? (
            <p className="callout">Queued ingest run job #{ingestMutation.data.job.id}.</p>
          ) : null}
        </div>
      </Card>

      <Card title="Continue vector indexing" subtitle="Resume vectors on a time budget so long runs stay easy to control.">
        <div className="stack">
          <div className="inline-fields">
            <label>
              Batch size
              <input
                aria-label="Vector batch size"
                value={batchSize}
                onChange={(event) => setBatchSize(event.target.value)}
              />
            </label>
            <label>
              Time budget (seconds)
              <input
                aria-label="Reindex time budget (seconds)"
                value={untilSeconds}
                onChange={(event) => setUntilSeconds(event.target.value)}
              />
            </label>
          </div>
          <div className="button-row">
            <button onClick={() => reindexMutation.mutate()} type="button">
              Continue vector indexing
            </button>
          </div>
          {reindexMutation.data ? (
            <p className="callout">Queued vector job #{reindexMutation.data.job.id}.</p>
          ) : null}

          {lastReindexResult ? (
            <div className="detail-panel">
              <div className="detail-panel-header">
                <div>
                  <p className="eyebrow">Last completed vector run</p>
                  <h3>Most recent reindex summary</h3>
                </div>
              </div>
              <div className="metric-grid">
                <div>
                  <span className="metric-label">Indexed chunks</span>
                  <strong>{String(lastReindexResult.indexed_count ?? "-")}</strong>
                </div>
                <div>
                  <span className="metric-label">Remaining after run</span>
                  <strong>{String(lastReindexResult.remaining_count ?? "-")}</strong>
                </div>
                <div>
                  <span className="metric-label">Elapsed seconds</span>
                  <strong>{String(lastReindexResult.elapsed_seconds ?? "-")}</strong>
                </div>
                <div>
                  <span className="metric-label">Throughput</span>
                  <strong>{String(lastReindexResult.throughput_chunks_per_second ?? "-")}</strong>
                </div>
              </div>
            </div>
          ) : (
            <p className="table-subtext">
              No completed vector run has been recorded in this app home yet.
            </p>
          )}
        </div>
      </Card>
    </div>
  );
}
