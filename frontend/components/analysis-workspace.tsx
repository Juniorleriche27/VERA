import { Activity, AlertCircle, ArrowUpRight, Fingerprint } from "lucide-react";
import type { AnalysisResponse } from "@/types/vera";
import { InvestigationOverview } from "@/components/results/investigation-overview";
import { InvestigationTimeline } from "@/components/results/investigation-timeline";
import { DecisionBriefPanel } from "@/components/results/decision-brief-panel";
import { VisualizationGrid } from "@/components/results/visualization-grid";
import { EvidenceInspector } from "@/components/results/evidence-inspector";

export function AnalysisWorkspace({ analysis, busy, error }: { analysis: AnalysisResponse | null; busy: boolean; error: string | null }) {
  if (!analysis && !busy && !error) return <section className="mt-7 flex min-h-[135px] items-center gap-4 rounded-2xl border border-[#dce3dc] bg-[#eef2ed] p-6 sm:p-8"><div className="flex size-11 shrink-0 items-center justify-center rounded-xl bg-white text-[#718579]"><Activity size={20}/></div><div><p className="text-[13px] font-bold text-[#43564a]">Evidence workspace</p><p className="mt-1 text-[12px] leading-6 text-[#819186]">Completed investigations appear here, directly from the VERA API. No simulated metrics.</p></div></section>;

  if (busy && !error) return null;

  if (error) return <section className="surface mt-7 overflow-hidden" aria-live="polite">
    {error && <div role="alert" className="flex items-start gap-3 px-6 py-5 text-[#9d433b]"><AlertCircle size={19} className="mt-0.5 shrink-0"/><div><p className="text-sm font-bold">Investigation could not complete</p><p className="mt-1 text-xs leading-5">{error}</p></div></div>}
    {busy && <div role="status" className="flex items-center gap-4 px-6 py-7"><Activity size={23} className="animate-pulse text-[#28785e]"/><div><p className="text-sm font-bold">Running verified investigation…</p><p className="mt-1 text-xs leading-5 text-[#819087]">VERA is planning and executing deterministic checks. Real backend events appear in Live AI Orchestration.</p></div></div>}
  </section>;

  if (!analysis) return null;
  return <div className="mt-5 space-y-5 scroll-mt-7" aria-label="Completed analysis workspace">
    <InvestigationOverview analysis={analysis}/>
    <nav aria-label="Result sections" className="sticky top-0 z-10 -mx-2 flex flex-wrap items-center gap-1.5 rounded-xl border border-[#e0e8df] bg-[#f8faf7]/95 p-2 text-[11px] font-bold shadow-sm backdrop-blur-sm sm:gap-2 sm:px-4">
      <span className="hidden text-[#88958c] lg:mr-2 lg:inline">EXPLORE</span>
      {[["#decision-brief", "Decision Brief"], ["#investigation-path", "Tool Orchestration"], ["#visual-evidence", "Visual evidence"], ["#provenance", "Provenance"]].map(([href,label]) => <a key={href} href={href} onClick={() => document.getElementById(href.slice(1))?.closest("details")?.setAttribute("open", "")} className="rounded-lg px-3 py-2 text-[#547361] transition hover:bg-[#e8f2e9] focus-visible:bg-[#e8f2e9]">{label}</a>)}
    </nav>
    {analysis.succeeded && analysis.decision_brief ? <DecisionBriefPanel brief={analysis.decision_brief}/> :
      <section id="decision-brief" className="rounded-2xl border border-[#efddc4] bg-[#fffaf3] p-6 sm:p-8" aria-label="Decision status"><h3 className="text-lg font-bold text-[#80603a]">No final Decision Brief was produced</h3><p className="mt-2 text-[13px] leading-6 text-[#866e52]">The investigation contains unresolved evidence or an execution failure. A verified business conclusion cannot be presented.</p>{!!analysis.unresolved_call_ids?.length && <div className="mt-5"><p className="eyebrow mb-2">Unresolved evidence</p><div className="flex flex-wrap gap-2">{analysis.unresolved_call_ids.map(id=><code key={id} className="rounded-lg border border-[#eadcc5] bg-white px-3 py-2 text-xs text-[#826a4b]">{id}</code>)}</div></div>}</section>}
    <VisualizationGrid visuals={analysis.visualizations}/>
    <details className="surface px-5 py-4"><summary className="cursor-pointer text-sm font-bold">Tool details · bindings, parameters & execution</summary><InvestigationTimeline steps={analysis.steps}/></details>
    {analysis.decision_brief?.evidence?.items ? <details className="surface px-5 py-4"><summary className="cursor-pointer text-sm font-bold">Evidence & Provenance · hashes, lineage & verification</summary><EvidenceInspector items={analysis.decision_brief.evidence.items} steps={analysis.steps}/></details> :
      <section id="provenance" className="surface px-6 py-7"><p className="eyebrow mb-2">AUDIT TRAIL</p><h3 className="text-xl font-bold">Evidence & provenance</h3><p className="mt-2 text-[13px] text-[#78877a]">No final verified evidence package was returned. Executed call IDs and their statuses remain visible in the investigation path.</p></section>}
    <div className="flex flex-wrap items-center justify-between gap-2 border-t border-[#dce5dc] pt-5 text-[10px] text-[#88998d]"><p className="flex items-center gap-2"><Fingerprint size={14}/>Verification reports and trace hashes originate from the backend.</p><p className="inline-flex items-center gap-1">VERA · EVIDENCE-FIRST WORKSPACE <ArrowUpRight size={13}/></p></div>
  </div>;
}
