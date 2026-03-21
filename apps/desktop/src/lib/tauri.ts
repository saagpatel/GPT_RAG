import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

import type { DesktopBackendStatus, SessionBootstrap } from "../types";

export async function bootstrapSession(): Promise<SessionBootstrap> {
  return invoke<SessionBootstrap>("bootstrap_session");
}

export async function restartSession(): Promise<SessionBootstrap> {
  return invoke<SessionBootstrap>("restart_session");
}

export async function backendStatus(): Promise<DesktopBackendStatus> {
  return invoke<DesktopBackendStatus>("backend_status");
}

export async function pickFolder(): Promise<string | null> {
  const selected = await open({
    directory: true,
    multiple: false,
  });
  return typeof selected === "string" ? selected : null;
}
