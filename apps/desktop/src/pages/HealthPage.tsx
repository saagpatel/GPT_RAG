import { useState } from "react";

import { useMutation, useQuery } from "@tanstack/react-query";

import { Card } from "../components/Card";
import { StatusPill } from "../components/StatusPill";
import { copyText } from "../lib/localUiState";
import { useSession } from "../lib/session";
import type { DoctorReport } from "../types";

interface RuntimeIssue {
  label: string;
  detail: string;
  severity: "blocked" | "warning";
}

function buildRuntimeIssues(report: DoctorReport): RuntimeIssue[] {
  const issues: RuntimeIssue[] = [];

  if (!report.ollama.reachable) {
    issues.push({
      label: "Ollama",
      detail: report.ollama.error ?? "The local Ollama runtime is not reachable.",
      severity: "blocked",
    });
  }
  if (!report.ollama.embedding_model_available) {
    issues.push({
      label: "Embedding model",
      detail: `Pull ${report.models.embedding} locally before semantic indexing can run.`,
      severity: "blocked",
    });
  }
  if (!report.ollama.generator_model_available) {
    issues.push({
      label: "Generator model",
      detail: `Pull ${report.models.generator} locally before grounded answering can run.`,
      severity: "blocked",
    });
  }
  if (!report.reranker_cache.dependencies_available) {
    issues.push({
      label: "Reranker dependencies",
      detail:
        report.reranker_cache.dependency_error ??
        "The reranker Python dependencies are not installed in the desktop runtime.",
      severity: "blocked",
    });
  }
  if (!report.reranker_cache.available) {
    issues.push({
      label: "Reranker cache",
      detail: report.reranker_cache.missing_files.length
        ? `Missing files: ${report.reranker_cache.missing_files.join(", ")}`
        : "The local reranker cache is not ready yet.",
      severity: "warning",
    });
  }
  if (!report.sqlite.all_required_tables_present) {
    issues.push({
      label: "App state",
      detail: "SQLite has not been initialized yet. Use Initialize state once.",
      severity: "warning",
    });
  }

  return issues;
}

export function HealthPage() {
  const { api, getBackendStatus, restartSession } = useSession();
  const [copyMessage, setCopyMessage] = useState<string | null>(null);
  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: () => api.getHealth(),
    refetchInterval: 15_000,
  });
  const shellStatusQuery = useQuery({
    queryKey: ["desktop-backend-status"],
    queryFn: getBackendStatus,
    refetchInterval: 1500,
  });
  const initMutation = useMutation({
    mutationFn: () => api.initialize(),
  });
  const runtimeCheckMutation = useMutation({
    mutationFn: () => api.createJob({ kind: "runtime_check" }),
  });
  const restartMutation = useMutation({
    mutationFn: restartSession,
  });

  const doctorReport = healthQuery.data ?? null;
  const issues = doctorReport ? buildRuntimeIssues(doctorReport) : [];
  const blockedIssues = issues.filter((issue) => issue.severity === "blocked");
  const warningIssues = issues.filter((issue) => issue.severity === "warning");

  async function handleCopy(command: string) {
    const copied = await copyText(command);
    setCopyMessage(copied ? `Copied: ${command}` : "Copy failed in this environment.");
  }

  return (
    <div className="page-grid">
      <Card
        title="Runtime readiness"
        subtitle="A clearer first-run view of what is ready, what is blocked, and how to fix it."
        actions={<StatusPill status={doctorReport?.runtime_ready ?? false} />}
      >
        {healthQuery.isLoading ? <p>Loading local runtime status…</p> : null}
        {doctorReport ? (
          <div className="stack">
            <div className={`hero-panel ${doctorReport.runtime_ready ? "hero-ready" : "hero-blocked"}`}>
              <p className="eyebrow">
                {doctorReport.runtime_ready ? "Ready to use" : "Action needed"}
              </p>
              <h3>
                {doctorReport.runtime_ready
                  ? "The local runtime is healthy."
                  : "The app launched, but the local runtime still needs setup."}
              </h3>
              <p>
                {doctorReport.runtime_ready
                  ? "You can ingest, inspect, ask, and resume vector work from the desktop app."
                  : "Use the blocked items below to fix software, models, or reranker setup before treating the results as fully ready."}
              </p>
            </div>

            {blockedIssues.length ? (
              <div className="stack">
                <h3 className="panel-title">Blocked items</h3>
                {blockedIssues.map((issue) => (
                  <div className="callout negative" key={issue.label}>
                    <strong>{issue.label}</strong>
                    <p>{issue.detail}</p>
                  </div>
                ))}
              </div>
            ) : null}

            {warningIssues.length ? (
              <div className="stack">
                <h3 className="panel-title">Still worth checking</h3>
                {warningIssues.map((issue) => (
                  <div className="callout warning" key={issue.label}>
                    <strong>{issue.label}</strong>
                    <p>{issue.detail}</p>
                  </div>
                ))}
              </div>
            ) : null}

            <div className="command-grid">
              <div className="command-card">
                <p className="eyebrow">Embedding model</p>
                <strong className="mono-text">{doctorReport.models.embedding}</strong>
                <div className="button-row">
                  <button
                    onClick={() => void handleCopy(`ollama pull ${doctorReport.models.embedding}`)}
                    type="button"
                  >
                    Copy pull command
                  </button>
                </div>
              </div>
              <div className="command-card">
                <p className="eyebrow">Generator model</p>
                <strong className="mono-text">{doctorReport.models.generator}</strong>
                <div className="button-row">
                  <button
                    onClick={() => void handleCopy(`ollama pull ${doctorReport.models.generator}`)}
                    type="button"
                  >
                    Copy pull command
                  </button>
                </div>
              </div>
            </div>

            <div className="metric-grid">
              <div>
                <span className="metric-label">Ollama endpoint</span>
                <strong className="mono-text">{doctorReport.ollama.base_url}</strong>
              </div>
              <div>
                <span className="metric-label">Ollama reachable</span>
                <strong>{doctorReport.ollama.reachable ? "yes" : "no"}</strong>
              </div>
              <div>
                <span className="metric-label">Embedding model ready</span>
                <strong>{String(doctorReport.ollama.embedding_model_available)}</strong>
              </div>
              <div>
                <span className="metric-label">Generator model ready</span>
                <strong>{String(doctorReport.ollama.generator_model_available)}</strong>
              </div>
              <div>
                <span className="metric-label">Reranker cache</span>
                <strong>{doctorReport.reranker_cache.available ? "ready" : "missing files"}</strong>
              </div>
              <div>
                <span className="metric-label">Reranker dependencies</span>
                <strong>
                  {doctorReport.reranker_cache.dependencies_available ? "installed" : "missing"}
                </strong>
              </div>
            </div>
            {copyMessage ? <p className="callout">{copyMessage}</p> : null}
          </div>
        ) : null}
      </Card>

      <Card
        title="Desktop services"
        subtitle="Local API and worker liveness for the desktop shell."
        actions={
          shellStatusQuery.data ? (
            <StatusPill
              status={shellStatusQuery.data.apiAlive && shellStatusQuery.data.workerAlive}
            />
          ) : null
        }
      >
        {shellStatusQuery.data ? (
          <div className="metric-grid">
            <div>
              <span className="metric-label">API sidecar</span>
              <strong>{shellStatusQuery.data.apiAlive ? "alive" : "down"}</strong>
            </div>
            <div>
              <span className="metric-label">Worker sidecar</span>
              <strong>{shellStatusQuery.data.workerAlive ? "alive" : "down"}</strong>
            </div>
            <div>
              <span className="metric-label">Runtime mode</span>
              <strong>{shellStatusQuery.data.runtimeMode}</strong>
            </div>
            <div>
              <span className="metric-label">Runtime source</span>
              <strong className="mono-text">{shellStatusQuery.data.runtimeSource}</strong>
            </div>
            <div>
              <span className="metric-label">Loopback API</span>
              <strong className="mono-text">{shellStatusQuery.data.apiBaseUrl ?? "-"}</strong>
            </div>
            <div>
              <span className="metric-label">App home</span>
              <strong className="mono-text">{shellStatusQuery.data.gptRagHome || "-"}</strong>
            </div>
          </div>
        ) : (
          <p>Checking desktop sidecars…</p>
        )}
        <div className="button-row">
          <button onClick={() => restartMutation.mutate()} type="button">
            Restart local services
          </button>
          <button onClick={() => runtimeCheckMutation.mutate()} type="button">
            Rerun runtime check
          </button>
        </div>
        {restartMutation.isSuccess ? (
          <p className="callout">Desktop sidecars restarted successfully.</p>
        ) : null}
        {runtimeCheckMutation.data ? (
          <p className="callout">Queued runtime check job #{runtimeCheckMutation.data.job.id}.</p>
        ) : null}
      </Card>

      <Card title="Actions" subtitle="Initialize local state if this is a fresh app home.">
        <div className="button-row">
          <button onClick={() => initMutation.mutate()} type="button">
            Initialize state
          </button>
        </div>
        {initMutation.data ? <p className="callout">Initialized local state.</p> : null}
      </Card>
    </div>
  );
}
