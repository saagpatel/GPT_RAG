import type {
  CreateJobResponse,
  DoctorReport,
  GuiJobDetailResponse,
  GuiJobsResponse,
  JobEventMessage,
  SearchMode,
  SearchResponse,
  SessionBootstrap,
  TraceDetailResponse,
  TraceListResponse,
  VectorStatusResponse,
} from "../types";

export interface GuiApiLike {
  getHealth(): Promise<DoctorReport>;
  initialize(): Promise<Record<string, unknown>>;
  getReindexStatus(): Promise<VectorStatusResponse>;
  search(input: {
    query: string;
    mode: SearchMode;
    limit: number;
    maxPerDocument?: number;
  }): Promise<SearchResponse>;
  createJob(input: Record<string, unknown>): Promise<CreateJobResponse>;
  listJobs(): Promise<GuiJobsResponse>;
  getJob(jobId: number): Promise<GuiJobDetailResponse>;
  cancelJob(jobId: number): Promise<CreateJobResponse>;
  listTraces(limit?: number): Promise<TraceListResponse>;
  getTrace(traceType: string, name: string): Promise<TraceDetailResponse>;
  jobsWebSocketUrl(): string;
}

export class GuiApiClient implements GuiApiLike {
  constructor(private readonly session: SessionBootstrap) {}

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${this.session.apiBaseUrl}${path}`, {
      ...init,
      headers: {
        "content-type": "application/json",
        "x-gpt-rag-session-token": this.session.sessionToken,
        ...(init?.headers ?? {}),
      },
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(body || `Request failed: ${response.status}`);
    }

    return (await response.json()) as T;
  }

  getHealth(): Promise<DoctorReport> {
    return this.request<DoctorReport>("/health");
  }

  initialize(): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>("/init", { method: "POST" });
  }

  getReindexStatus(): Promise<VectorStatusResponse> {
    return this.request<VectorStatusResponse>("/reindex/status");
  }

  search(input: {
    query: string;
    mode: SearchMode;
    limit: number;
    maxPerDocument?: number;
  }): Promise<SearchResponse> {
    return this.request<SearchResponse>("/search", {
      method: "POST",
      body: JSON.stringify({
        query: input.query,
        mode: input.mode,
        limit: input.limit,
        max_per_document: input.maxPerDocument,
      }),
    });
  }

  createJob(input: Record<string, unknown>): Promise<CreateJobResponse> {
    return this.request<CreateJobResponse>("/jobs", {
      method: "POST",
      body: JSON.stringify(input),
    });
  }

  listJobs(): Promise<GuiJobsResponse> {
    return this.request<GuiJobsResponse>("/jobs");
  }

  getJob(jobId: number): Promise<GuiJobDetailResponse> {
    return this.request<GuiJobDetailResponse>(`/jobs/${jobId}`);
  }

  cancelJob(jobId: number): Promise<CreateJobResponse> {
    return this.request<CreateJobResponse>(`/jobs/${jobId}/cancel`, { method: "POST" });
  }

  listTraces(limit = 50): Promise<TraceListResponse> {
    return this.request<TraceListResponse>(`/traces?limit=${limit}`);
  }

  getTrace(traceType: string, name: string): Promise<TraceDetailResponse> {
    return this.request<TraceDetailResponse>(`/traces/${traceType}/${name}`);
  }

  jobsWebSocketUrl(): string {
    const websocketBase = this.session.apiBaseUrl.replace(/^http/, "ws");
    return `${websocketBase}/ws/jobs?token=${encodeURIComponent(this.session.sessionToken)}`;
  }
}

export function isJobEventMessage(value: unknown): value is JobEventMessage {
  return (
    typeof value === "object" &&
    value !== null &&
    "type" in value &&
    (value as { type?: unknown }).type === "job_event"
  );
}
