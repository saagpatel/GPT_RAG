import { useEffect } from "react";

import { useQueryClient } from "@tanstack/react-query";

import { isJobEventMessage } from "../lib/api";
import { useSession } from "../lib/session";

export function useJobEvents() {
  const queryClient = useQueryClient();
  const { api } = useSession();

  useEffect(() => {
    const socket = new WebSocket(api.jobsWebSocketUrl());

    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data) as unknown;
      if (!isJobEventMessage(payload)) {
        return;
      }
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      queryClient.invalidateQueries({ queryKey: ["job", payload.event.job_id] });
    };

    return () => {
      socket.close();
    };
  }, [api, queryClient]);
}
