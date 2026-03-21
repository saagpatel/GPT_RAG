import { useState } from "react";

import { useQuery } from "@tanstack/react-query";

import { Card } from "../components/Card";
import { JsonView } from "../components/JsonView";
import { copyText } from "../lib/localUiState";
import { useSession } from "../lib/session";

export function TracesPage() {
  const { api } = useSession();
  const [selected, setSelected] = useState<{ type: string; name: string } | null>(null);
  const [copyMessage, setCopyMessage] = useState<string | null>(null);
  const tracesQuery = useQuery({
    queryKey: ["traces"],
    queryFn: () => api.listTraces(),
  });
  const traceQuery = useQuery({
    queryKey: ["trace", selected?.type, selected?.name],
    queryFn: () => api.getTrace(selected?.type ?? "", selected?.name ?? ""),
    enabled: selected !== null,
  });

  async function handleCopy(text: string, label: string) {
    const copied = await copyText(text);
    setCopyMessage(copied ? `${label} copied.` : "Copy failed in this environment.");
  }

  return (
    <div className="page-grid jobs-layout">
      <Card title="Managed traces" subtitle="Saved inspect and ask artifacts, kept read-only in the local release candidate.">
        {tracesQuery.data ? (
          <div className="trace-summary-grid">
            <div className="summary-card">
              <p className="eyebrow">Inspect traces</p>
              <strong>
                {tracesQuery.data.traces.filter((trace) => trace.type === "inspect").length}
              </strong>
            </div>
            <div className="summary-card">
              <p className="eyebrow">Ask traces</p>
              <strong>
                {tracesQuery.data.traces.filter((trace) => trace.type === "ask").length}
              </strong>
            </div>
            <div className="summary-card">
              <p className="eyebrow">Trace directory</p>
              <strong className="mono-text">{tracesQuery.data.trace_path}</strong>
            </div>
          </div>
        ) : null}
        {tracesQuery.data?.traces.length ? (
          <ul className="job-list">
            {tracesQuery.data.traces.map((trace) => (
              <li key={trace.path}>
                <button
                  className={`job-row ${selected?.name === trace.name ? "job-row-selected" : ""}`}
                  onClick={() => setSelected({ type: String(trace.type), name: trace.name })}
                  type="button"
                >
                  <span>
                    <strong>{trace.type}</strong>
                    <span className="job-meta">{trace.query ?? trace.name}</span>
                    <span className="table-subtext mono-text">{trace.path}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p>No inspect or ask traces are available yet.</p>
        )}
      </Card>

      <Card title="Trace detail" subtitle="Metadata summary first, raw JSON second.">
        {traceQuery.data ? (
          <div className="stack">
            <div className="metric-grid">
              <div>
                <span className="metric-label">Type</span>
                <strong>{traceQuery.data.metadata.type}</strong>
              </div>
              <div>
                <span className="metric-label">Timestamp</span>
                <strong>{traceQuery.data.metadata.timestamp ?? "-"}</strong>
              </div>
              <div>
                <span className="metric-label">Path</span>
                <strong className="mono-text">{traceQuery.data.metadata.path}</strong>
              </div>
              <div>
                <span className="metric-label">Query</span>
                <strong>{traceQuery.data.metadata.query ?? "-"}</strong>
              </div>
              <div>
                <span className="metric-label">Size</span>
                <strong>{traceQuery.data.metadata.size_bytes} bytes</strong>
              </div>
            </div>
            <div className="detail-panel">
              <div className="detail-panel-header">
                <div>
                  <p className="eyebrow">Selected trace</p>
                  <h3>{traceQuery.data.metadata.name}</h3>
                </div>
              </div>
              <p className="table-subtext">
                This screen stays read-only in the desktop release candidate. Diffing, pruning,
                and destructive trace maintenance remain in the CLI.
              </p>
              <div className="button-row">
                <button
                  onClick={() => void handleCopy(traceQuery.data.metadata.path, "Trace path")}
                  type="button"
                >
                  Copy path
                </button>
                <button
                  onClick={() =>
                    void handleCopy(JSON.stringify(traceQuery.data.payload, null, 2), "Trace JSON")
                  }
                  type="button"
                >
                  Copy JSON
                </button>
              </div>
            </div>
            {copyMessage ? <p className="callout">{copyMessage}</p> : null}
            <JsonView value={traceQuery.data.payload} />
          </div>
        ) : (
          <p>Select a trace to inspect its payload.</p>
        )}
      </Card>
    </div>
  );
}
