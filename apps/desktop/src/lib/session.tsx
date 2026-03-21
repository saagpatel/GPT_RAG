import { createContext, useContext } from "react";

import type { GuiApiLike } from "./api";
import type { DesktopBackendStatus, SessionBootstrap } from "../types";

export interface SessionContextValue {
  bootstrap: SessionBootstrap;
  api: GuiApiLike;
  restartSession: () => Promise<void>;
  getBackendStatus: () => Promise<DesktopBackendStatus>;
}

const SessionContext = createContext<SessionContextValue | null>(null);

export function SessionProvider({
  value,
  children,
}: {
  value: SessionContextValue;
  children: React.ReactNode;
}) {
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionContextValue {
  const value = useContext(SessionContext);
  if (!value) {
    throw new Error("useSession must be used inside SessionProvider.");
  }
  return value;
}
