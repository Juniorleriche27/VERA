"use client";

import { useEffect, useState } from "react";
import { Check, ChevronDown, CircleAlert, Copy, Fingerprint, Info, ShieldCheck, X } from "lucide-react";
import type { AnalysisStep, EvidenceItem } from "@/types/vera";
import { evidenceStep, shortHash, toolLabel } from "@/lib/result-utils";
import { VerificationBadge } from "@/components/results/verification-badge";

function EvidenceRecord({ item, steps }: { item: EvidenceItem; steps: AnalysisStep[] }) {
  const [copied, setCopied] = useState(false);
  const step = evidenceStep(item, steps);
  const checks = step?.report?.checks ?? [];

  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(false), 2200);
    return () => window.clearTimeout(timer);
  }, [copied]);

  async function copyHash() {
    try {
      await navigator.clipboard.writeText(item.trace_hash);
      setCopied(true);
    } catch { setCopied(false); }
  }

  return <details className="group overflow-hidden rounded-xl border border-[#e1e9e2] bg-white open:border-[#b7d2c0]">
    <summary className="flex cursor-pointer list-none items-center gap-3 p-4 marker:hidden focus-visible:outline-offset-[-3px] sm:p-5 [&::-webkit-details-marker]:hidden">
      <div className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-[#edf6ef] text-[#3b7b58]"><ShieldCheck size={18}/></div>
      <div className="min-w-0 flex-1"><p className="text-[13px] font-bold text-[#243a2e]">{toolLabel(item.tool)}</p><p className="mt-1 truncate font-mono text-[10px] text-[#8a998d]">{item.tool} · {item.call_id}</p></div>
      {step && <span className="hidden sm:inline-flex"><VerificationBadge status={step.trace.verification_status}/></span>}
      <span className="ml-1 font-mono text-[10px] text-[#93a299]">{shortHash(item.trace_hash)}</span>
      <ChevronDown size={17} className="shrink-0 text-[#77937e] transition group-open:rotate-180"/>
    </summary>
    <div className="space-y-5 border-t border-[#e8eee8] bg-[#fbfdfb] px-4 py-5 sm:px-6">
      <div><p className="eyebrow mb-2">Analytical objective</p><p className="text-[13px] leading-6 text-[#455a4b]">{item.objective}</p></div>
      {!!item.source_call_ids.length && <div><p className="eyebrow mb-1.5">Derived from</p><p className="break-all font-mono text-[11px] text-[#548066]">{item.source_call_ids.join(", ")}</p></div>}
      <div><p className="eyebrow mb-2">Provenance hash · SHA-256</p><div className="flex flex-col gap-2 rounded-lg border border-[#dbe7de] bg-white px-3 py-3 sm:flex-row sm:items-center"><Fingerprint size={16} className="shrink-0 text-[#4c8467]"/><code className="min-w-0 flex-1 break-all font-mono text-[11px] leading-5 text-[#526d5b]">{item.trace_hash}</code><button type="button" onClick={() => void copyHash()} className="inline-flex shrink-0 items-center justify-center gap-1.5 rounded-md border border-[#d7e4d9] px-3 py-2 text-[11px] font-bold text-[#326c50] hover:bg-[#eff7f0]">{copied ? <Check size={13}/> : <Copy size={13}/>} {copied ? "Copied" : "Copy hash"}</button></div></div>
      <div><p className="eyebrow mb-3">Verification checks {step?.report ? `(${checks.length})` : ""}</p>
        {step?.report ? <div className="space-y-2">{checks.map((check,index) => {
          const criticalFailure = !check.passed && check.severity === "critical";
          const Icon = criticalFailure ? X : check.severity === "info" ? Info : check.passed ? Check : CircleAlert;
          return <div key={`${check.code}-${index}`} className="flex gap-3 rounded-lg border border-[#e7ece5] bg-white p-3"><Icon size={15} className={`mt-0.5 shrink-0 ${criticalFailure ? "text-[#b45043]" : !check.passed ? "text-[#a88439]" : check.severity === "info" ? "text-[#648596]" : "text-[#32825b]"}`}/><div className="min-w-0"><div className="flex flex-wrap items-center gap-2"><code className="break-all text-[11px] font-semibold text-[#435b49]">{check.code}</code><span className="text-[9px] uppercase tracking-[.07em] text-[#91a096]">{check.severity} · {check.passed ? "passed" : "not passed"}</span></div><p className="mt-1 text-[11px] leading-[1.65] text-[#708174]">{check.message}</p></div></div>;
        })}</div> : <p className="text-xs text-[#8a968d]">No verification report was included for this evidence.</p>}
      </div>
      <div><p className="eyebrow mb-2">Original deterministic result · unmodified</p><pre className="max-h-[350px] overflow-auto rounded-lg border border-[#253b31] bg-[#172923] p-4 font-mono text-[11px] leading-[1.75] text-[#e0f0e5]" tabIndex={0}>{JSON.stringify(item.result, null, 2)}</pre><p className="mt-2 text-[10px] leading-5 text-[#87958a]">Complete executor result shown without truncation; scroll to inspect arrays and diagnostics.</p></div>
    </div>
  </details>;
}

export function EvidenceInspector({ items, steps }: { items: EvidenceItem[]; steps: AnalysisStep[] }) {
  return <section id="provenance" aria-labelledby="evidence-title" className="surface p-5 sm:p-8">
    <div className="mb-6"><p className="eyebrow mb-2">AUDIT TRAIL</p><h3 id="evidence-title" className="text-xl font-bold tracking-[-.04em] sm:text-2xl">Evidence & provenance</h3><p className="mt-1.5 text-xs leading-5 text-[#849187]">Inspect original deterministic results, verification checks and source hashes. All provenance comes from the VERA API.</p></div>
    {items.length ? <div className="space-y-3">{items.map((item,index)=><EvidenceRecord key={`${item.call_id}-${index}`} item={item} steps={steps}/>)}</div> : <p className="text-sm text-[#718176]">No final verified evidence was included in this response.</p>}
  </section>;
}
