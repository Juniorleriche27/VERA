import type { VerificationStatus } from "../types/vera";

export const eventFields = {
  analysis_started: [], planning_started: [], planning_completed: ["planning_turns"],
  tools_selected: ["calls"], execution_started: [], execution_completed: ["exhausted"],
  tool_planned: ["call"], tool_started: ["tool", "call_id"], tool_completed: ["tool", "call_id"],
  tool_failed: ["tool", "call_id"], verification_started: ["tool", "call_id"],
  verification_completed: ["tool", "call_id", "status"], tool_verified: ["tool", "call_id"],
  replanning_completed: ["call_id", "status"], decision_brief_started: [], decision_brief_built: [],
  visualizations_started: [], visualizations_built: ["count"],
  analysis_completed: ["succeeded", "analyses", "verified", "visualizations", "unresolved", "exhausted"],
  analysis_failed: ["code"],
} as const;
export type EventName = keyof typeof eventFields;
export interface ToolReference { tool: string; call_id: string; source_call_ids: string[]; selection: "ai" | "deterministic" }
export interface AnalysisEvent {
  version: 1; analysis_id: string; sequence: number; event: EventName; elapsed_ms: number;
  data: { calls?: ToolReference[]; call?: ToolReference; tool?: string; call_id?: string;
    status?: VerificationStatus | string; code?: string; planning_turns?: number; count?: number;
    succeeded?: boolean; analyses?: number; verified?: number; visualizations?: number; unresolved?: number; exhausted?: boolean };
}
export const streamErrors: Record<string, string> = {
  planning_failed: "The AI planner could not complete a valid investigation. Please try again.",
  configuration_error: "The AI planner is not configured on the backend.",
  analysis_error: "The investigation could not complete. Please try again.",
};
const token = /^[A-Za-z0-9_-]{1,256}$/;
const object = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === "object" && !Array.isArray(value);
function reference(value: unknown): boolean {
  return object(value) && Object.keys(value).sort().join() === "call_id,selection,source_call_ids,tool" &&
    typeof value.tool === "string" && token.test(value.tool) && typeof value.call_id === "string" && token.test(value.call_id) &&
    ["ai", "deterministic"].includes(String(value.selection)) && Array.isArray(value.source_call_ids) &&
    value.source_call_ids.every(id => typeof id === "string" && token.test(id));
}
export function parseAnalysisEvent(value: unknown): AnalysisEvent {
  const invalid = () => { throw new Error("The API returned an invalid live event."); };
  if (!object(value) || value.version !== 1 || typeof value.analysis_id !== "string" || !token.test(value.analysis_id) ||
      !Number.isInteger(value.sequence) || Number(value.sequence) < 1 || typeof value.elapsed_ms !== "number" ||
      !Number.isFinite(value.elapsed_ms) || value.elapsed_ms < 0 || typeof value.event !== "string" ||
      !Object.hasOwn(eventFields, value.event) || !object(value.data)) return invalid();
  if (Object.keys(value).sort().join() !== "analysis_id,data,elapsed_ms,event,sequence,version") return invalid();
  const fields = eventFields[value.event as EventName];
  if (Object.keys(value.data).sort().join() !== [...fields].sort().join()) return invalid();
  for (const [key, child] of Object.entries(value.data)) {
    if (["tool", "call_id"].includes(key) && (typeof child !== "string" || !token.test(child))) return invalid();
    if (["analyses", "verified", "visualizations", "unresolved", "count", "planning_turns"].includes(key) &&
        (typeof child !== "number" || !Number.isInteger(child) || child < 0)) return invalid();
    if (["succeeded", "exhausted"].includes(key) && typeof child !== "boolean") return invalid();
    if (key === "calls" && (!Array.isArray(child) || child.length > 4 || !child.every(reference))) return invalid();
    if (key === "call" && !reference(child)) return invalid();
    if (key === "status" && !["verified", "rejected", "needs_more_analysis", "not_verified", "scheduled", "no_fallback", "unsupported_decision", "max_steps_reached"].includes(String(child))) return invalid();
    if (key === "code" && !(typeof child === "string" && Object.hasOwn(streamErrors, child))) return invalid();
  }
  return value as unknown as AnalysisEvent;
}

export async function consumeAnalysisStream(body: ReadableStream<Uint8Array>, onEvent: (event: AnalysisEvent) => void): Promise<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", sequence = 0, analysisId = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer = (buffer + decoder.decode(value, { stream: !done })).replace(/\r\n/g, "\n");
      if (buffer.length > 65536) throw new Error("The live event exceeded its size limit.");
      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 2);
        const lines = frame.split("\n");
        const data = lines.filter(line => line.startsWith("data:")).map(line => line.slice(5).trimStart()).join("\n");
        if (!data) continue; // Transport comments are never UI progress.
        let parsed: unknown;
        try { parsed = JSON.parse(data); } catch { throw new Error("The API returned invalid live JSON."); }
        const event = parseAnalysisEvent(parsed);
        const wireEvent = lines.find(line => line.startsWith("event:"))?.slice(6).trim();
        const wireId = lines.find(line => line.startsWith("id:"))?.slice(3).trim();
        if (event.sequence !== sequence + 1 || (analysisId && analysisId !== event.analysis_id) ||
            (!sequence && event.event !== "analysis_started") || (wireEvent && wireEvent !== event.event) ||
            (wireId && wireId !== String(event.sequence))) throw new Error("The live event sequence is inconsistent.");
        if (event.sequence > 128) throw new Error("The live event limit was exceeded.");
        sequence = event.sequence; analysisId = event.analysis_id;
        onEvent(event);
        if (event.event === "analysis_failed") throw new Error(streamErrors[event.data.code!] || streamErrors.analysis_error);
        if (event.event === "analysis_completed") return analysisId;
      }
      if (done) throw new Error("The live connection ended before investigation completion. Please retry.");
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

export interface LiveTool extends ToolReference { status: "planned" | "running" | "completed" | "failed" | VerificationStatus; started: boolean; executed: boolean }
export interface LiveState { events: AnalysisEvent[]; tools: LiveTool[]; complete: boolean; failed: boolean }
export const initialLiveState = (): LiveState => ({ events: [], tools: [], complete: false, failed: false });
export function reduceLiveEvent(state: LiveState, event: AnalysisEvent): LiveState {
  let tools = state.tools;
  if (event.event === "tools_selected") tools = event.data.calls!.map(call => ({ ...call, status: "planned", started: false, executed: false }));
  if (event.event === "tool_planned") tools = [...tools, { ...event.data.call!, status: "planned", started: false, executed: false }];
  if (event.data.call_id) tools = tools.map(tool => {
    if (tool.call_id !== event.data.call_id) return tool;
    switch (event.event) {
      case "tool_started": return { ...tool, status: "running", started: true };
      case "tool_completed": return { ...tool, status: "completed", executed: true };
      case "tool_failed": return { ...tool, status: "failed", executed: false };
      case "verification_completed": return { ...tool, status: event.data.status as VerificationStatus };
      case "tool_verified": return { ...tool, status: "verified" };
      default: return tool;
    }
  });
  return { events: [...state.events, event], tools, complete: event.event === "analysis_completed" || state.complete,
    failed: event.event === "analysis_failed" || state.failed };
}
