import { useEffect, useMemo, useState } from "react";

import { useMutation, useQuery } from "@tanstack/react-query";

import { Card } from "../components/Card";
import { JsonView } from "../components/JsonView";
import { StatusPill } from "../components/StatusPill";
import { useSession } from "../lib/session";
import type { GuiJob, GuiJobEventRow, JobProgressMetric } from "../types";

type JobFilter = "running" | "failed" | "completed" | "all";

function jobPriority(job: GuiJob): number {
  if (job.status === "running") {
    return 0;
  }
  if (job.status === "pending") {
    return 1;
  }
  if (job.status === "interrupted") {
    return 2;
  }
  return 3;
}

function deriveProgress(event: GuiJobEventRow): JobProgressMetric[] {
  const counts = event.payload_json.counts ?? {};
  const progressPairs: Array<[string, string, string]> = [
    ["Documents", "documents_done", "documents_seen"],
    ["Chunks", "chunks_done", "chunks_total"],
    ["Vectors", "vectors_done", "vectors_total"],
  ];

  return progressPairs.flatMap(([label, doneKey, totalKey]) => {
    const done = Number(counts[doneKey] ?? 0);
    const total = Number(counts[totalKey] ?? 0);
    if (!total || Number.isNaN(total) || done > total) {
      return [];
    }
    return [
      {
        label,
        complete: done,
        total,
        percent: Math.round((done / total) * 100),
      },
    ];
  });
}

function filterJobs(jobs: GuiJob[], filter: JobFilter): GuiJob[] {
  switch (filter) {
    case "running":
      return jobs.filter((job) => job.status === "running" || job.status === "pending");
    case "failed":
      return jobs.filter((job) =>
        ["failed", "cancelled", "interrupted"].includes(job.status),
      );
    case "completed":
      return jobs.filter((job) => job.status === "completed");
    default:
      return jobs;
  }
}

export function JobsPage() {
  const { api } = useSession();
  const [selectedJobId, setSelectedJobId] = useState<number | null>(null);
  const [filter, setFilter] = useState<JobFilter>("running");
  const jobsQuery = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.listJobs(),
    refetchInterval: 2000,
  });
  const sortedJobs = useMemo(
    () =>
      [...(jobsQuery.data?.jobs ?? [])].sort((left, right) => {
        return jobPriority(left) - jobPriority(right) || right.id - left.id;
      }),
    [jobsQuery.data?.jobs],
  );
  const jobs = filterJobs(sortedJobs, filter);

  useEffect(() => {
    if (selectedJobId !== null && jobs.some((job) => job.id === selectedJobId)) {
      return;
    }
    setSelectedJobId(jobs[0]?.id ?? null);
  }, [jobs, selectedJobId]);

  const detailQuery = useQuery({
    queryKey: ["job", selectedJobId],
    queryFn: () => api.getJob(selectedJobId as number),
    enabled: selectedJobId !== null,
    refetchInterval: 1500,
  });
  const cancelMutation = useMutation({
    mutationFn: (jobId: number) => api.cancelJob(jobId),
  });

  return (
    <div className="page-grid jobs-layout">
      <Card title="Jobs" subtitle="Running work stays pinned first, and interrupted jobs stay visible after a restart.">
        <div className="button-row filter-row">
          {(["running", "failed", "completed", "all"] as JobFilter[]).map((value) => (
            <button
              className={filter === value ? "filter-button active" : "filter-button"}
              key={value}
              onClick={() => setFilter(value)}
              type="button"
            >
              {value}
            </button>
          ))}
        </div>

        {jobs.length ? (
          <ul className="job-list">
            {jobs.map((job) => (
              <li key={job.id}>
                <button className="job-row" onClick={() => setSelectedJobId(job.id)} type="button">
                  <span>
                    <strong>{job.kind}</strong>
                    <span className="job-meta">#{job.id}</span>
                    {job.status === "interrupted" ? (
                      <span className="table-subtext">Resume from Library or rerun the job.</span>
                    ) : null}
                  </span>
                  <StatusPill status={job.status} />
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p>No jobs match this filter yet.</p>
        )}
      </Card>

      <Card
        title="Job detail"
        subtitle="Progress grouped by stage, with clear failure and recovery context."
        actions={
          selectedJobId !== null ? (
            <button onClick={() => cancelMutation.mutate(selectedJobId)} type="button">
              Cancel job
            </button>
          ) : null
        }
      >
        {detailQuery.data ? (
          <div className="stack">
            <div className="metric-grid">
              <div>
                <span className="metric-label">Kind</span>
                <strong>{detailQuery.data.job.kind}</strong>
              </div>
              <div>
                <span className="metric-label">Status</span>
                <strong>
                  <StatusPill status={detailQuery.data.job.status} />
                </strong>
              </div>
              <div>
                <span className="metric-label">Created</span>
                <strong>{detailQuery.data.job.created_at || "-"}</strong>
              </div>
              <div>
                <span className="metric-label">Worker</span>
                <strong>{detailQuery.data.job.worker_id || "-"}</strong>
              </div>
            </div>

            {detailQuery.data.job.status === "interrupted" ? (
              <div className="callout warning">
                This job was interrupted by an app or worker restart. If it was vector work, use
                Library to continue from the saved progress point.
              </div>
            ) : null}

            {detailQuery.data.events.map((event) => {
              const progress = deriveProgress(event);
              return (
                <div className="event-card" key={event.id}>
                  <div className="event-header">
                    <div>
                      <p className="eyebrow">{event.payload_json.stage ?? event.event_type}</p>
                      <strong>{String(event.payload_json.message ?? "")}</strong>
                    </div>
                    <StatusPill status={event.payload_json.status ?? "running"} />
                  </div>
                  {progress.map((metric) => (
                    <div className="progress-row" key={metric.label}>
                      <header>
                        <span>{metric.label}</span>
                        <span>
                          {metric.complete} / {metric.total}
                        </span>
                      </header>
                      <div aria-label={`${metric.label} progress`} className="progress-bar">
                        <div
                          className="progress-bar-fill"
                          style={{ width: `${metric.percent}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              );
            })}

            {detailQuery.data.job.error_json ? (
              <p className="callout negative">
                {String(detailQuery.data.job.error_json.message ?? "Job failed.")}
              </p>
            ) : null}
            {detailQuery.data.job.result_json ? (
              <details className="details-panel" open>
                <summary>Result payload</summary>
                <JsonView value={detailQuery.data.job.result_json} />
              </details>
            ) : null}
            {detailQuery.data.job.error_json ? (
              <details className="details-panel" open>
                <summary>Error payload</summary>
                <JsonView value={detailQuery.data.job.error_json} />
              </details>
            ) : null}
          </div>
        ) : (
          <p>Select a job to inspect its progress and result.</p>
        )}
      </Card>
    </div>
  );
}
