import { useEffect, useState } from "react";
import { HashRouter, Route, Routes } from "react-router-dom";

import { AppShell } from "./components/AppShell";
import { GuiApiClient } from "./lib/api";
import { SessionProvider } from "./lib/session";
import { backendStatus, bootstrapSession, restartSession } from "./lib/tauri";
import { AskPage } from "./pages/AskPage";
import { HealthPage } from "./pages/HealthPage";
import { InspectPage } from "./pages/InspectPage";
import { JobsPage } from "./pages/JobsPage";
import { LibraryPage } from "./pages/LibraryPage";
import { SearchPage } from "./pages/SearchPage";
import { TracesPage } from "./pages/TracesPage";
import type { SessionBootstrap } from "./types";

export function App() {
  const [bootstrap, setBootstrap] = useState<SessionBootstrap | null>(null);
  const [launchState, setLaunchState] = useState<"launching" | "ready" | "failed">("launching");
  const [launchError, setLaunchError] = useState<string | null>(null);

  async function launchSession({ restart = false }: { restart?: boolean } = {}) {
    setLaunchState("launching");
    setLaunchError(null);
    try {
      const nextBootstrap = restart ? await restartSession() : await bootstrapSession();
      setBootstrap(nextBootstrap);
      setLaunchState("ready");
    } catch (error) {
      setBootstrap(null);
      setLaunchState("failed");
      setLaunchError(error instanceof Error ? error.message : "Desktop bootstrap failed.");
    }
  }

  useEffect(() => {
    void launchSession();
  }, []);

  if (launchState === "launching") {
    return <div className="launch-screen">Starting local desktop services…</div>;
  }

  if (launchState === "failed" || !bootstrap) {
    return (
      <div className="launch-screen error-screen">
        <h1>Desktop bootstrap failed</h1>
        <p>{launchError ?? "The local API and worker could not be started."}</p>
        <div className="button-row">
          <button onClick={() => void launchSession({ restart: true })} type="button">
            Restart local services
          </button>
          <button onClick={() => void launchSession()} type="button">
            Retry launch
          </button>
        </div>
      </div>
    );
  }

  const sessionValue = {
    bootstrap,
    api: new GuiApiClient(bootstrap),
    restartSession: async () => {
      await launchSession({ restart: true });
    },
    getBackendStatus: backendStatus,
  };

  return (
    <SessionProvider value={sessionValue}>
      <HashRouter>
        <Routes>
          <Route element={<AppShell />} path="/">
            <Route element={<HealthPage />} index />
            <Route element={<LibraryPage />} path="library" />
            <Route element={<SearchPage />} path="search" />
            <Route element={<InspectPage />} path="inspect" />
            <Route element={<AskPage />} path="ask" />
            <Route element={<JobsPage />} path="jobs" />
            <Route element={<TracesPage />} path="traces" />
          </Route>
        </Routes>
      </HashRouter>
    </SessionProvider>
  );
}
