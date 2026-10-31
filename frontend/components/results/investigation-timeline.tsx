import { ArrowRight, GitBranch, Info, ListChecks } from "lucide-react";
import type { AnalysisStep, ReplanningOutcome } from "@/types/vera";
import { shortHash, toolLabel } from "@/lib/result-utils";
import { ExecutionStatus, VerificationBadge } from "@/components/results/verification-badge";

function ReplanningNotice({ outcome }: { outcome: ReplanningOutcome }) {
  const heading: Record<string, string> = {
    scheduled: "Robustness check scheduled",
    no_fallback: "No safe deterministic fallback available",
    unsupported_decision: "Additional evidence required",
    max_steps_reached: "Investigation step limit reached",
    not_needed: "No additional analysis scheduled",
  };
  return <div className="mt-3 rounded-lg border border-[#dce5dc] bg-[#f7f9f6] px-3 py-3 text-xs text-[#55685b]">
    <div className="flex gap-2 font-bold text-[#3c6250]"><GitBranch size={14} className="mt-0.5 shrink-0"/>{heading[outcome.status] || outcome.status.replaceAll("_", " ")}</div>
    {outcome.status === "scheduled" && <p className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px]"><code>{outcome.source_call_id}</code><ArrowRight size={12}/><code>{outcome.fallback_tool}</code><span className="text-[#87968c]">({outcome.next_call_id})</span></p>}
    {outcome.reason && <p className="mt-1.5 text-[11px] leading-5 text-[#728277]">{outcome.reason}</p>}
  </div>;
}

export function InvestigationTimeline({ steps }: { steps: AnalysisStep[] }) {
  return <section id="investigation-path" className="surface px-5 py-6 sm:px-8 sm:py-7" aria-labelledby="path-title">
    <div className="mb-6 flex items-start justify-between gap-4"><div><p className="eyebrow mb-2">EXECUTED ANALYSES</p><h3 id="path-title" className="text-xl font-bold tracking-[-.04em]">Investigation path</h3><p className="mt-1.5 text-xs text-[#849187]">A verified record of completed tool calls and their decisions.</p></div><ListChecks size={19} className="mt-1 shrink-0 text-[#6b917f]"/></div>
    {steps.length === 0 ? <p className="rounded-xl bg-[#f5f7f5] p-5 text-sm text-[#7d897f]">No analytical steps were returned by the API.</p> :
      <ol className="relative space-y-0">{steps.map((step, i) => <li key={`${step.trace.call_id}-${i}`} className="relative flex gap-3 pb-3 last:pb-0 sm:gap-4">
        {i < steps.length - 1 && <span className="absolute left-[17px] top-10 bottom-0 w-px bg-[#dce6dd]" aria-hidden="true"/>}
        <span className="relative z-[1] flex size-9 shrink-0 items-center justify-center rounded-full border border-[#cbded1] bg-[#eff6f0] text-xs font-bold text-[#327257]">{step.index}</span>
        <div className="min-w-0 flex-1 rounded-xl border border-[#e6ece6] bg-[#fcfdfb] p-3 sm:p-4">
          <div className="flex flex-wrap items-start justify-between gap-3"><div className="min-w-0"><p className="text-[14px] font-bold text-[#253a2e]">{toolLabel(step.trace.tool)}</p><p className="mt-1 break-all font-mono text-[10px] text-[#829186]">{step.trace.tool}</p></div><div className="flex flex-wrap items-center gap-2"><ExecutionStatus status={step.trace.status}/>{step.trace.status !== "failed" && <VerificationBadge status={step.trace.verification_status}/>}</div></div>
          <p className="mt-3 text-[12px] leading-[1.6] text-[#637568]">{step.trace.objective}</p>
          <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[10px] text-[#9aa69e]"><span>ID: {step.trace.call_id}</span>{step.trace.trace_hash && <span>Hash: {shortHash(step.trace.trace_hash)}</span>}</div>
          {step.trace.source_call_ids?.length > 0 && <p className="mt-3 flex flex-wrap items-center gap-1 text-[11px] text-[#4f775f]"><GitBranch size={13}/>Derived from: <code className="break-all">{step.trace.source_call_ids.join(", ")}</code></p>}
          {step.trace.status === "failed" && <p className="mt-3 flex gap-2 rounded-lg bg-[#fff2ef] px-3 py-2.5 text-[11px] text-[#9d5042]"><Info size={14} className="shrink-0"/>This call failed during execution. No verification report is assumed.</p>}
          {step.replanning && <ReplanningNotice outcome={step.replanning}/>}
        </div>
      </li>)}</ol>}
  </section>;
}
