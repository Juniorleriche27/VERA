import { Check, Circle, CircleAlert, GitBranch, LoaderCircle, Radio } from "lucide-react";
import type { AnalysisEvent, LiveState } from "@/lib/analysis-stream";
import { toolLabel } from "@/lib/result-utils";

const labels: Record<string, string> = {
  analysis_started: "Request accepted", planning_started: "AI planning started", planning_completed: "Planning completed",
  tools_selected: "Tools selected", execution_started: "Deterministic execution started", execution_completed: "Execution finished",
  tool_planned: "Deterministic follow-up planned", tool_started: "Running", tool_completed: "Deterministically executed",
  tool_failed: "Execution failed", verification_started: "Verification started", verification_completed: "Verification completed",
  tool_verified: "VERIFIED", replanning_completed: "Replanning decision recorded", decision_brief_started: "Building Decision Brief",
  decision_brief_built: "Decision Brief built", visualizations_started: "Building visualizations", visualizations_built: "Visualizations built",
  analysis_completed: "Investigation finished", analysis_failed: "Investigation could not complete",
};
function eventLabel(event: AnalysisEvent) {
  const name = event.data.tool ? `${toolLabel(event.data.tool)} · ` : "";
  const verdict = event.event === "verification_completed" ? ` · ${event.data.status!.replaceAll("_", " ")}` : "";
  return name + labels[event.event] + verdict;
}
export function LiveOrchestration({ state, busy }: { state: LiveState; busy: boolean }) {
  const seen = (name: string) => state.events.some(event => event.event === name);
  const terminal = state.complete || state.failed || !busy;
  const executionComplete = state.events.some(event => event.event === "execution_completed" && !event.data.exhausted);
  const stages = [
    { label: "Planning", active: seen("planning_started"), done: seen("planning_completed") },
    { label: "Tool Selection", active: seen("planning_completed"), done: seen("tools_selected") },
    { label: "Deterministic Execution", active: seen("execution_started"), done: executionComplete },
    { label: "Verification", active: seen("verification_started"), done: executionComplete && seen("verification_completed") },
    { label: "Decision Brief", active: seen("decision_brief_started"), done: seen("decision_brief_built") },
  ];
  const last = state.events.at(-1);
  return <section className="live-panel mt-5 overflow-hidden rounded-2xl border border-[#28493d] bg-[#142b23] text-[#e9f4ed]" aria-labelledby="live-title">
    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 px-5 py-4">
      <div><h2 id="live-title" className="flex items-center gap-2 text-[12px] font-bold tracking-[.12em]"><Radio size={15}/> LIVE AI ORCHESTRATION</h2>
        <p role="status" className="mt-1 text-[12px] text-[#b1c8bb]">{state.failed || (!busy && !state.complete) ? "Investigation interrupted" : state.complete ? "Received event history retained" : last ? eventLabel(last) : "Connecting to the investigation event stream…"}</p></div>
      <span className="text-[11px] tabular-nums text-[#b1c8bb]">{state.events.length} real events{last ? ` · last event ${(last.elapsed_ms / 1000).toFixed(2)}s` : ""}</span>
    </div>
    <ol aria-label="Analysis stages" className="grid grid-cols-2 gap-2 border-b border-white/10 px-5 py-3 sm:grid-cols-5">
      {stages.map(stage => <li key={stage.label} className="flex items-start gap-1.5 text-[10px] leading-4 text-[#c7d8cd]">
        {stage.done ? <Check size={13} className="shrink-0 text-[#86dfb0]"/> : stage.active && !terminal ? <LoaderCircle size={13} className="shrink-0 animate-spin"/> : <Circle size={12} className="shrink-0 opacity-50"/>}
        <span>{stage.label}<span className="block text-[9px] text-[#95b2a1]">{stage.done ? "Complete" : stage.active && !terminal ? "In progress" : terminal ? "Not completed" : "Pending"}</span></span>
      </li>)}
    </ol>
    {state.tools.length > 0 && <ul aria-label="Live analytical tools" className="grid gap-px bg-white/10 sm:grid-cols-2">
      {state.tools.map(tool => <li key={tool.call_id} className="min-w-0 bg-[#142b23] px-5 py-3">
        <div className="flex items-center justify-between gap-3"><span className="truncate text-xs font-bold" title={tool.tool}>{tool.tool}</span>
          <span className={`shrink-0 text-[10px] font-bold ${tool.status === "verified" ? "text-[#86dfb0]" : tool.status === "failed" || tool.status === "rejected" ? "text-[#ffb4a8]" : "text-[#e6d59e]"}`}>{tool.status.toUpperCase().replaceAll("_", " ")}</span></div>
        <p className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[10px] text-[#b1c8bb]">
          {tool.selection === "ai" ? "AI Selected" : <><GitBranch size={11}/>Deterministic follow-up</>}
          {tool.started && <span>→ Running ✓</span>}
          {tool.executed && <span>→ Deterministically Executed ✓</span>}
          {tool.status === "failed" && <span className="inline-flex items-center gap-1 text-[#ffb4a8]"><CircleAlert size={11}/>→ Failed</span>}
        </p>
      </li>)}
    </ul>}
    <details className="border-t border-white/10 px-5 py-3">
      <summary className="cursor-pointer text-[11px] font-semibold text-[#bad0c2]">Event history · {state.events.length} received</summary>
      <ol aria-label="Received event history" className="mt-3 max-h-64 space-y-1.5 overflow-y-auto text-[11px]">
        {state.events.map(event => <li key={event.sequence} className="flex gap-3"><time className="w-14 shrink-0 tabular-nums text-[#8eb49f]">{(event.elapsed_ms / 1000).toFixed(3)}s</time><span>{eventLabel(event)}</span></li>)}
      </ol>
    </details>
  </section>;
}
