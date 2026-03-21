import { NavLink, Outlet } from "react-router-dom";

import { useQuery } from "@tanstack/react-query";

import { useJobEvents } from "../hooks/useJobEvents";
import { useSession } from "../lib/session";

const NAV_ITEMS = [
  { to: "/", label: "Health" },
  { to: "/library", label: "Library" },
  { to: "/search", label: "Search" },
  { to: "/inspect", label: "Inspect" },
  { to: "/ask", label: "Ask" },
  { to: "/jobs", label: "Jobs" },
  { to: "/traces", label: "Traces" },
];

export function AppShell() {
  useJobEvents();
  const { api, bootstrap } = useSession();
  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: () => api.getHealth(),
    refetchInterval: 15_000,
  });
  const jobsQuery = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.listJobs(),
    refetchInterval: 5_000,
  });
  const interruptedJobs = (jobsQuery.data?.jobs ?? []).filter((job) => job.status === "interrupted");
  const runtimeBlocked = healthQuery.data ? !healthQuery.data.runtime_ready : false;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <p className="eyebrow">Local desktop shell</p>
          <h1>GPT_RAG</h1>
          <p className="brand-detail">Version {bootstrap.version}</p>
        </div>
        <nav className="nav-list" aria-label="Primary">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
              to={item.to}
              end={item.to === "/"}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <p className="eyebrow">App home</p>
          <p className="mono-text">{bootstrap.gptRagHome || "System default"}</p>
          <p className="eyebrow">Runtime</p>
          <p>{bootstrap.runtimeMode === "packaged" ? "Bundled sidecars" : "Development shell"}</p>
          <p className="mono-text">{bootstrap.runtimeSource}</p>
        </div>
      </aside>
      <main className="main-panel">
        {runtimeBlocked ? (
          <div className="app-banner negative">
            <strong>Runtime setup still needs attention.</strong>
            <span>
              The app launched, but one or more local prerequisites are missing. Open Health for
              exact fix steps.
            </span>
          </div>
        ) : null}
        {interruptedJobs.length ? (
          <div className="app-banner warning">
            <strong>{interruptedJobs.length} interrupted job{interruptedJobs.length === 1 ? "" : "s"} detected.</strong>
            <span>
              Review Jobs or return to Library to resume vector indexing after a restart.
            </span>
          </div>
        ) : null}
        <Outlet />
      </main>
    </div>
  );
}
