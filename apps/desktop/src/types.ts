export type SearchMode = "lexical" | "semantic" | "hybrid";
export type TraceType = "inspect" | "ask" | "debug-bundle";
export type GuiJobKind =
  | "runtime_check"
  | "ingest_preview"
  | "ingest_run"
  | "reindex_vectors"
  | "inspect"
  | "ask";

export interface SessionBootstrap {
  apiBaseUrl: string;
  sessionToken: string;
  version: string;
  gptRagHome: string;
  runtimeMode: "dev" | "packaged";
  runtimeSource: string;
}

export interface DesktopBackendStatus {
  sessionPresent: boolean;
  apiAlive: boolean;
  workerAlive: boolean;
  apiBaseUrl: string | null;
  gptRagHome: string;
  runtimeMode: "dev" | "packaged";
  runtimeSource: string;
}

export interface DoctorReport {
  version: string;
  runtime_ready: boolean;
  paths: Record<string, { path: string; exists: boolean }>;
  models: Record<string, string>;
  ollama: {
    base_url: string;
    is_local_endpoint: boolean;
    reachable: boolean;
    error: string | null;
    available_models: string[];
    embedding_model_available: boolean | null;
    generator_model_available: boolean | null;
  };
  reranker_cache: {
    available: boolean;
    dependencies_available: boolean;
    dependency_error: string | null;
    missing_files: string[];
  };
  sqlite: {
    all_required_tables_present: boolean;
    required_tables: Record<string, boolean>;
  };
}

export interface SearchResult {
  chunk_id: number;
  title?: string | null;
  source_path: string;
  section_title?: string | null;
  page_number?: number | null;
  chunk_index?: number;
  lexical_score?: number | null;
  semantic_score?: number | null;
  fusion_score?: number | null;
  reranker_score?: number | null;
  final_rank?: number | null;
  exact_title_match?: boolean;
  exact_source_name_match?: boolean;
  phrase_match?: boolean;
  text?: string;
  [key: string]: unknown;
}

export interface SearchResponse {
  query: string;
  mode: SearchMode;
  results: SearchResult[];
}

export interface VectorStatusResponse {
  status: string;
  sqlite_path: string;
  lancedb_path: string;
  embedding_model: string;
  chunk_count: number;
  vector_count: number;
  remaining_count: number;
  completion_percentage: number;
}

export interface GuiJob {
  id: number;
  kind: GuiJobKind | string;
  status: string;
  request_json: Record<string, unknown>;
  result_json: Record<string, unknown> | null;
  error_json: Record<string, unknown> | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  heartbeat_at: string | null;
  cancel_requested: boolean;
  worker_id: string | null;
}

export interface GuiJobEventRow {
  id: number;
  job_id: number;
  sequence: number;
  created_at: string;
  event_type: string;
  payload_json: {
    status?: string;
    stage?: string;
    message?: string;
    counts?: Record<string, number>;
    timestamp?: string;
    [key: string]: unknown;
  };
}

export interface JobProgressMetric {
  label: string;
  complete: number;
  total: number;
  percent: number;
}

export interface GuiJobDetailResponse {
  job: GuiJob;
  events: GuiJobEventRow[];
}

export interface GuiJobsResponse {
  jobs: GuiJob[];
}

export interface CreateJobResponse {
  job: GuiJob;
}

export interface TraceMetadata {
  path: string;
  name: string;
  type: TraceType | string;
  timestamp: string | null;
  query: string | null;
  size_bytes: number;
}

export interface TraceListResponse {
  trace_path: string;
  count: number;
  traces: TraceMetadata[];
}

export interface TraceDetailResponse {
  metadata: TraceMetadata;
  payload: Record<string, unknown>;
}

export interface JobEventMessage {
  type: "job_event";
  event: GuiJobEventRow;
}
