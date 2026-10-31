import { AlertTriangle, CheckCircle2, FileBarChart2, GitBranch, Layers3, ShieldCheck } from "lucide-react";
import type { AnalysisResponse } from "@/types/vera";

export function InvestigationOverview({ analysis }: { analysis: AnalysisResponse }) {
  const verified = analysis.succeeded;
  const metrics = [
    { label: "Analytical steps", value: analysis.steps.length, Icon: GitBranch },
    { label: "Verified evidence", value: analysis.decision_brief?.evidence_count ?? 0, Icon: ShieldCheck },
    { label: "Tools used", value: analysis.decision_brief?.tools_used.length ?? 0, Icon: Layers3 },
    { label: "Visualizations", value: analysis.visualizations.length, Icon: FileBarChart2 },
  ];
  return <section aria-labelledby="overview-title" className="surface overflow-hidden">
    <div className="border-b border-[#e4ebe5] px-5 py-7 sm:px-8 sm:py-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex min-w-0 items-start gap-3">
          <div className={`mt-0.5 flex size-11 shrink-0 items-center justify-center rounded-xl ${verified ? "bg-[#e7f5ec] text-[#217653]" : "bg-[#fff3e7] text-[#a97637]"}`}>{verified ? <CheckCircle2 size={22}/> : <AlertTriangle size={22}/>}</div>
          <div><p className="eyebrow mb-2">03 / INVESTIGATION RESULT</p><h2 id="overview-title" className="text-xl font-bold tracking-[-.05em] sm:text-2xl">{verified ? "Verified investigation" : "Investigation needs review"}</h2><p className="mt-1.5 text-xs text-[#809087]">Completed analytical execution · Not a live progress feed</p></div>
        </div>
        <span className={`rounded-full border px-3 py-1.5 text-[10px] font-extrabold tracking-[.07em] ${verified ? "border-[#cce8d7] bg-[#e7f6eb] text-[#226f50]" : "border-[#f0dcc3] bg-[#fff5e8] text-[#93672f]"}`}>{verified ? "VERIFIED INVESTIGATION" : "INVESTIGATION NEEDS REVIEW"}</span>
      </div>
      <div className="mt-7 max-w-[950px] border-l-[3px] border-[#3b906f] pl-4"><p className="eyebrow mb-2">Business question</p><p className="text-[16px] font-semibold leading-[1.7] tracking-[-.015em] text-[#283b31] sm:text-lg">{analysis.question}</p></div>
    </div>
    <div className="grid grid-cols-2 gap-px bg-[#e5ebe5] lg:grid-cols-4">{metrics.map(({label,value,Icon}) => <div key={label} className="flex min-w-0 items-center gap-3 bg-[#fafcf9] px-4 py-5 sm:px-7"><div className="hidden size-9 shrink-0 items-center justify-center rounded-lg bg-white text-[#538473] shadow-sm sm:flex"><Icon size={17}/></div><div><p className="eyebrow text-[9px]">{label}</p><p className="metric-number mt-1 text-2xl font-bold sm:text-[29px]">{value.toLocaleString()}</p></div></div>)}</div>
  </section>;
}
