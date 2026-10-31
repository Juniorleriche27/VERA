import { consumeAnalysisStream, type AnalysisEvent } from "./analysis-stream";
import type { AnalysisResponse, DatasetUploadResponse, HealthResponse } from "@/types/vera";

const FALLBACK_API_URL = "http://127.0.0.1:8000";
export const API_BASE_URL = (process.env.NEXT_PUBLIC_VERA_API_URL || FALLBACK_API_URL).replace(/\/+$/, "");

export class VeraApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = "VeraApiError";
  }
}

function detailMessage(detail: unknown): string | undefined {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const parts = detail.flatMap((item: unknown) => {
      if (item && typeof item === "object" && "msg" in item && typeof item.msg === "string") return [item.msg];
      return [];
    });
    return parts.length ? parts.join(". ") : undefined;
  }
  return undefined;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store", ...options });
  } catch {
    throw new VeraApiError("Could not reach VERA API. Check that the backend is running and its CORS origin is configured.");
  }
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    throw new VeraApiError("The API returned an unexpected response format.", response.status);
  }
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new VeraApiError("The API returned invalid JSON.", response.status);
  }
  if (!response.ok) {
    const detail = payload && typeof payload === "object" && "detail" in payload ? payload.detail : null;
    throw new VeraApiError(detailMessage(detail) || `The request failed (HTTP ${response.status}).`, response.status);
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new VeraApiError("The API returned an invalid response body.", response.status);
  }
  return payload as T;
}

export const veraApi = {
  health: () => request<HealthResponse>("/health"),
  uploadDataset: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DatasetUploadResponse>("/datasets", { method: "POST", body: form });
  },
  runAnalysis: (datasetId: string, question: string) =>
    request<AnalysisResponse>("/analyses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: datasetId, question }),
    }),
  streamAnalysis: async (datasetId: string, question: string, onEvent: (event: AnalysisEvent) => void, signal: AbortSignal): Promise<AnalysisResponse> => {
    let response: Response;
    try {
      response = await fetch(`${API_BASE_URL}/analyses/stream`, {
        method: "POST", cache: "no-store", signal,
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ dataset_id: datasetId, question }),
      });
    } catch (error) {
      if (signal.aborted) throw error;
      throw new VeraApiError("Could not reach VERA API. Check that the backend is running and its CORS origin is configured.");
    }
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      throw new VeraApiError(detailMessage(payload?.detail) || `The request failed (HTTP ${response.status}).`, response.status);
    }
    if (!response.headers.get("content-type")?.includes("text/event-stream") || !response.body)
      throw new VeraApiError("The API did not return a live event stream.");
    const analysisId = await consumeAnalysisStream(response.body, onEvent);
    return request<AnalysisResponse>(`/analyses/${encodeURIComponent(analysisId)}`, { signal });
  },
  getAnalysis: (analysisId: string) => request<AnalysisResponse>(`/analyses/${encodeURIComponent(analysisId)}`),
};

export function readableError(error: unknown): string {
  return error instanceof Error ? error.message : "An unexpected error occurred. Please try again.";
}
