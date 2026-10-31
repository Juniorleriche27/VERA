export type DatasetColumnRole = "time" | "dimension" | "measure" | (string & {});

export interface DatasetColumnProfile {
  name: string;
  dtype: string;
  role: DatasetColumnRole;
  missing_count: number;
  missing_rate: number;
  unique_count: number;
}

export interface DatasetProfile {
  row_count: number;
  column_count: number;
  columns: DatasetColumnProfile[];
}

export interface DatasetUploadResponse {
  dataset_id: string;
  filename: string;
  profile: DatasetProfile;
}

export type VerificationStatus = "not_verified" | "verified" | "rejected" | "needs_more_analysis";
export type TraceStatus = "planned" | "completed" | "failed" | "running";
export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

export interface VerificationCheck {
  code: string;
  passed: boolean;
  severity: "critical" | "warning" | "info";
  message: string;
}

export interface VerificationReport {
  status: VerificationStatus;
  trace_hash: string;
  checks: VerificationCheck[];
  next_decision: string;
}

export interface ReplanningOutcome {
  status: "not_needed" | "scheduled" | "no_fallback" | "unsupported_decision" | "max_steps_reached" | string;
  source_call_id: string;
  decision: string;
  reason: string;
  source_trace_hash: string;
  fallback_tool: string;
  next_call_id: string;
}

export interface ToolCallTrace {
  call_id: string;
  tool: string;
  objective: string;
  reason: string;
  parameters: Record<string, JsonValue>;
  bindings: Record<string, JsonValue>;
  source_call_ids: string[];
  status: TraceStatus;
  verification_status: VerificationStatus;
  trace_hash: string;
  deterministic_result: Record<string, JsonValue> | null;
  next_decision: string;
}

export interface AnalysisStep {
  index: number;
  trace: ToolCallTrace;
  report: VerificationReport | null;
  replanning: ReplanningOutcome | null;
}

export interface EvidenceItem {
  call_id: string;
  tool: string;
  trace_hash: string;
  objective: string;
  result: Record<string, JsonValue>;
  source_call_ids: string[];
}

export interface DecisionBrief {
  question: string;
  headline: string;
  findings: string[];
  caveats: string[];
  next_action: string;
  evidence: { question: string; items: EvidenceItem[] };
  tools_used: string[];
  evidence_count: number;
}

export interface VisualizationSeries {
  dataKey: string;
  label: string;
  axisLabel?: string;
  role?: string;
}

export interface VisualizationSpec {
  chartType: "line" | "bar" | string;
  meta: { title: string; description: string; footer: string };
  xKey: string;
  series: VisualizationSeries[];
  data: Record<string, string | number | null>[];
  evidence: { call_id: string; trace_hash: string; tool: string };
  annotations?: { type: string; label: string; value: string | number }[];
  bands?: { lowerKey: string; upperKey: string; label: string }[];
}

export interface AnalysisResponse {
  analysis_id: string;
  dataset_id: string;
  question: string;
  succeeded: boolean;
  exhausted: boolean;
  final_status: VerificationStatus | null;
  steps: AnalysisStep[];
  decision_brief: DecisionBrief | null;
  visualizations: VisualizationSpec[];
  unresolved_call_ids?: string[];
}

export interface HealthResponse { status: string; service: string }
