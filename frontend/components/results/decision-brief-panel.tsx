import { ArrowUpRight, BookOpenCheck, CircleAlert, Target } from "lucide-react";
import type { DecisionBrief } from "@/types/vera";
import { toolLabel } from "@/lib/result-utils";

export function DecisionBriefPanel({ brief }: { brief: DecisionBrief }) {
  return <section id="decision-brief" aria-labelledby="decision-title" className="overflow-hidden rounded-2xl border border-[#cadcd0] bg-white shadow-[0_5px_25px_rgba(20,65,48,.04)]">
    <div className="border-b border-[#e1eae3] bg-[#f0f7f1] px-5 py-5 sm:px-6 sm:py-5">
      <div className="flex items-center justify-between gap-4"><p className="eyebrow flex items-center gap-2 text-[#3e795c]"><BookOpenCheck size={15}/> VERIFIED DECISION BRIEF</p><span className="hidden rounded-full bg-white px-3 py-1.5 text-[10px] font-semibold text-[#4c705b] sm:inline">{brief.evidence_count} verified analyses</span></div>
      <h3 id="decision-title" className="mt-4 max-w-6xl text-[clamp(1.25rem,1.7vw,1.6rem)] font-semibold leading-[1.45] tracking-[-.045em] text-[#213e2c]">{brief.headline}</h3>
      {brief.tools_used.length > 0 && <p className="mt-5 text-[11px] leading-6 text-[#6c8572]">Evidence from: {brief.tools_used.map(toolLabel).join(" · ")}</p>}
    </div>
    <div className="grid gap-8 px-5 py-7 sm:px-8 sm:py-9 lg:grid-cols-[1.45fr_1fr] lg:gap-12">
      <div><p className="eyebrow mb-5">VERIFIED FINDINGS</p>
        <ol className="space-y-5">{brief.findings.map((finding, i) => <li key={i} className="flex gap-4"><span className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-[#eaf2eb] font-mono text-[11px] font-bold text-[#397253]">{String(i + 1).padStart(2, "0")}</span><p className="pt-0.5 text-[13px] leading-[1.8] text-[#44574b]">{finding}</p></li>)}</ol>
      </div>
      <div className="space-y-6"><div className="rounded-xl border border-[#e8e6d8] bg-[#faf9f4] px-5 py-5"><h4 className="flex items-center gap-2 text-[12px] font-bold text-[#746b42]"><CircleAlert size={16}/> What the evidence does not prove</h4>
        {brief.caveats.length ? <ul className="mt-4 space-y-3">{brief.caveats.map((caveat,i) => <li key={i} className="border-l-2 border-[#d8d1ad] pl-3 text-[12px] leading-[1.75] text-[#706f5c]">{caveat}</li>)}</ul> : <p className="mt-3 text-xs leading-6 text-[#737467]">No additional caveats were supplied in the verified report.</p>}
      </div>
      <div className="rounded-xl border border-[#d8e7d9] bg-[#f5faf5] px-5 py-5"><h4 className="flex items-center gap-2 text-[12px] font-bold text-[#286c50]"><Target size={16}/> Recommended next action</h4><p className="mt-3 text-[13px] leading-[1.85] text-[#385745]">{brief.next_action}</p><span className="mt-3 inline-flex items-center gap-1 text-[10px] font-bold text-[#70927a]">Evidence-based guidance <ArrowUpRight size={12}/></span></div></div>
    </div>
  </section>;
}
