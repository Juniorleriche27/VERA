import { ArrowRight, CheckCircle2, ChevronRight, Cpu, Fingerprint, GitBranch, Info, Layers, ShieldCheck } from "lucide-react";
import type { AnalysisStep, ReplanningOutcome } from "@/types/vera";
import { shortHash, toolLabel } from "@/lib/result-utils";
import { ExecutionStatus, VerificationBadge } from "@/components/results/verification-badge";

function ReplanningNotice({ outcome }: { outcome: ReplanningOutcome }) {
  const heading: Record<string, string> = {
    scheduled: "Deterministic robustness check scheduled",
    no_fallback: "No safe deterministic fallback available",
    unsupported_decision: "Additional evidence required",
    max_steps_reached: "Investigation step limit reached",
    not_needed: "No additional analysis scheduled",
  };
  const isScheduled = outcome.status === "scheduled";
  return (
    <div className={`mt-3.5 rounded-xl border p-3.5 text-xs ${isScheduled ? "border-[#cce5d4] bg-[#f2f8f4] text-[#2c533c]" : "border-[#eddcc4] bg-[#fffaf3] text-[#7d5e38]"}`}>
      <div className="flex items-center gap-2 font-bold">
        <GitBranch size={15} className="shrink-0 text-[#256c4d]" />
        <span>{heading[outcome.status] || outcome.status.replaceAll("_", " ")}</span>
      </div>
      {isScheduled && (
        <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
          <span className="font-semibold text-[#4f6f5a]">Adaptation:</span>
          <code className="rounded bg-white px-1.5 py-0.5 border border-[#c4ded0]">{outcome.source_call_id}</code>
          <ArrowRight size={12} className="text-[#327253]" />
          <code className="rounded bg-white px-1.5 py-0.5 border border-[#c4ded0] font-bold text-[#1d6344]">{outcome.fallback_tool}</code>
          <span className="text-[#748779]">({outcome.next_call_id})</span>
        </div>
      )}
      {outcome.reason && <p className="mt-2 text-[11px] leading-relaxed text-[#5a7263]">{outcome.reason}</p>}
    </div>
  );
}

export function InvestigationTimeline({ steps }: { steps: AnalysisStep[] }) {
  return (
    <section id="investigation-path" className="surface px-5 py-6 sm:px-8 sm:py-8" aria-labelledby="path-title">
      <div className="mb-7 flex flex-wrap items-start justify-between gap-4 border-b border-[#e5ebe5] pb-5">
        <div>
          <p className="eyebrow mb-2 flex items-center gap-1.5 text-[#2e6d50]">
            <Cpu size={14} /> AI TOOL ORCHESTRATION
          </p>
          <h3 id="path-title" className="text-xl font-bold tracking-[-.04em] text-[#203629]">
            Investigation path & tool execution
          </h3>
          <p className="mt-1.5 text-xs leading-relaxed text-[#78887d]">
            The AI selected analytical tools based on dataset metadata; deterministic code executed them with full provenance.
          </p>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-[#d2e2d6] bg-[#f4f8f5] px-3.5 py-1.5 text-[11px] font-bold text-[#2d684d]">
          <Layers size={14} />
          <span>{steps.length} tool{steps.length !== 1 ? "s" : ""} orchestrated</span>
        </div>
      </div>

      {steps.length === 0 ? (
        <p className="rounded-xl bg-[#f5f7f5] p-5 text-sm text-[#7d897f]">No analytical steps were returned by the API.</p>
      ) : (
        <ol className="relative space-y-0" aria-label="Tool execution steps">
          {steps.map((step, i) => {
            const bindingsEntries = Object.entries(step.trace.bindings || {});
            const paramEntries = Object.entries(step.trace.parameters || {}).filter(
              ([k]) => !["data", "raw_data", "records", "rows"].includes(k.toLowerCase())
            );
            const isVerified = step.trace.verification_status === "verified";
            const isFailed = step.trace.status === "failed";

            return (
              <li key={`${step.trace.call_id}-${i}`} className="relative flex gap-3 pb-6 last:pb-0 sm:gap-5">
                {i < steps.length - 1 && (
                  <span className="absolute left-[17px] top-11 bottom-0 w-px bg-[#d5e2d7]" aria-hidden="true" />
                )}
                <span
                  className={`relative z-[1] flex size-9 shrink-0 items-center justify-center rounded-full border text-xs font-bold ${
                    isVerified
                      ? "border-[#b6dac0] bg-[#e8f6ec] text-[#206948]"
                      : isFailed
                      ? "border-[#f4c8c2] bg-[#fef1ef] text-[#a44233]"
                      : "border-[#eddcc4] bg-[#fffaf3] text-[#8e683a]"
                  }`}
                  aria-label={`Step ${step.index}`}
                >
                  {step.index}
                </span>

                <div className="min-w-0 flex-1 rounded-xl border border-[#e2ece3] bg-[#fcfdfc] p-4 sm:p-5 shadow-[0_1px_3px_rgba(20,50,30,.03)]">
                  {/* Tool Title & Badges */}
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <p className="text-[15px] font-bold tracking-tight text-[#21382b]">
                          {toolLabel(step.trace.tool)}
                        </p>
                        <span className="rounded bg-[#eef4f0] px-2 py-0.5 font-mono text-[10px] font-semibold text-[#4b6d58]">
                          {step.trace.tool}
                        </span>
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <ExecutionStatus status={step.trace.status} />
                      {!isFailed && <VerificationBadge status={step.trace.verification_status} />}
                    </div>
                  </div>

                  {/* 3-Stage Pipeline Status Indicator */}
                  <div className="mt-3 flex flex-wrap items-center gap-1.5 rounded-lg border border-[#e7eee8] bg-[#f8faf8] px-3 py-2 text-[10.5px]">
                    <span className="inline-flex items-center gap-1 font-semibold text-[#486352]">
                      <CheckCircle2 size={12} className="text-[#2b7854]" /> AI Selected
                    </span>
                    <ChevronRight size={12} className="text-[#99aa9f]" />
                    <span className="inline-flex items-center gap-1 font-semibold text-[#486352]">
                      {step.trace.status === "completed" ? <CheckCircle2 size={12} className="text-[#2b7854]" /> : <Info size={12} />}
                      {step.trace.status === "completed" ? "Deterministically Executed" : isFailed ? "Execution Failed" : step.trace.status === "running" ? "Execution Running" : "Execution Planned"}
                    </span>
                    <ChevronRight size={12} className="text-[#99aa9f]" />
                    <span
                      className={`inline-flex items-center gap-1 font-bold ${
                        isVerified
                          ? "text-[#1f6b49]"
                          : isFailed
                          ? "text-[#a44233]"
                          : "text-[#8e683a]"
                      }`}
                    >
                      <ShieldCheck size={12} />
                      {isFailed ? "FAILED" : step.trace.verification_status.toUpperCase().replaceAll("_", " ")}
                    </span>
                  </div>

                  {/* Objective */}
                  <p className="mt-3 text-[13px] leading-relaxed text-[#4f6456]">
                    <span className="font-semibold text-[#253d2f]">Objective: </span>
                    {step.trace.objective}
                  </p>

                  {/* Reason - Why VERA selected it */}
                  {step.trace.reason && (
                    <div className="mt-2.5 rounded-lg border border-[#e0ebe2] bg-[#f4f9f5] px-3 py-2 text-[11.5px] leading-relaxed text-[#3b5946]">
                      <span className="font-bold text-[#1f6243]">Why VERA selected it: </span>
                      {step.trace.reason}
                    </div>
                  )}

                  {/* Safe Semantic Bindings */}
                  {bindingsEntries.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-[#edf2ed]">
                      <p className="eyebrow mb-1.5 text-[9px] text-[#6d8474]">Safe Semantic Bindings</p>
                      <div className="flex flex-wrap gap-1.5 font-mono text-[11px]">
                        {bindingsEntries.map(([key, val]) => (
                          <span
                            key={key}
                            className="inline-flex items-center gap-1.5 rounded-md border border-[#d6e4d8] bg-white px-2.5 py-1 text-[#2d4b38] shadow-xs"
                          >
                            <span className="text-[#597863]">{key}</span>
                            <span className="text-[#9cb0a1]">→</span>
                            <span className="font-bold text-[#1a6549]">
                              {Array.isArray(val) ? `[${val.join(", ")}]` : String(val ?? "null")}
                            </span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Safe Method Parameters if any */}
                  {paramEntries.length > 0 && (
                    <div className="mt-2.5 flex flex-wrap items-center gap-1.5 text-[11px] text-[#627768]">
                      <span className="font-semibold text-[#486352]">Configuration:</span>
                      {paramEntries.map(([k, v]) => (
                        <code key={k} className="rounded bg-[#f0f5f1] px-1.5 py-0.5 font-mono text-[10px] text-[#335641]">
                          {k}={typeof v === "number" ? v : JSON.stringify(v)}
                        </code>
                      ))}
                    </div>
                  )}

                  {/* Provenance & Lineage */}
                  <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 border-t border-[#edf2ed] pt-2.5 font-mono text-[10px] text-[#86978c]">
                    <span className="inline-flex items-center gap-1">
                      <Fingerprint size={12} className="text-[#517a61]" />
                      Call ID: <strong className="text-[#3b5946]">{step.trace.call_id}</strong>
                    </span>
                    {step.trace.trace_hash && (
                      <span>
                        Trace Hash: <strong className="text-[#3b5946]">{shortHash(step.trace.trace_hash)}</strong>
                      </span>
                    )}
                  </div>

                  {step.trace.source_call_ids?.length > 0 && (
                    <p className="mt-2.5 flex flex-wrap items-center gap-1.5 text-[11px] font-medium text-[#3b6d4f]">
                      <GitBranch size={13} />
                      <span>Derived from:</span>
                      <code className="rounded bg-[#e8f3ec] px-1.5 py-0.5 font-mono text-[10px] text-[#245b3d]">
                        {step.trace.source_call_ids.join(", ")}
                      </code>
                    </p>
                  )}

                  {isFailed && (
                    <p className="mt-3 flex gap-2 rounded-lg bg-[#fff2ef] px-3.5 py-2.5 text-[11px] text-[#9d5042]">
                      <Info size={14} className="mt-0.5 shrink-0" />
                      <span>This call failed during execution: {step.trace.execution_error || "Unknown error"}. No verification report is assumed.</span>
                    </p>
                  )}

                  {/* Replanning / Adaptive Self-Correction */}
                  {step.replanning && <ReplanningNotice outcome={step.replanning} />}
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
