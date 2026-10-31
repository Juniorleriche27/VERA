import { ChartNoAxesCombined } from "lucide-react";
import type { VisualizationSpec } from "@/types/vera";
import { VerifiedChart } from "@/components/results/verified-chart";

export function VisualizationGrid({ visuals }: { visuals: VisualizationSpec[] }) {
  return <section id="visual-evidence" aria-labelledby="visuals-title">
    <div className="mb-5 flex items-center justify-between gap-4"><div><p className="eyebrow mb-2">THE DATA BEHIND THE DECISION</p><h3 id="visuals-title" className="text-xl font-bold tracking-[-.04em] sm:text-2xl">Verified visual evidence</h3><p className="mt-1 text-xs text-[#849187]">Charts render only values supplied in verified analytical results.</p></div><ChartNoAxesCombined size={21} className="shrink-0 text-[#64937a]"/></div>
    {visuals.length ? <div className="grid min-w-0 gap-5 xl:grid-cols-2">{visuals.map((spec,index)=><VerifiedChart key={`${spec.evidence.call_id}-${index}`} spec={spec}/>)}</div> : <div className="rounded-xl border border-[#dce5dd] bg-white px-6 py-7 text-[13px] text-[#758579]">The API returned no verified chart specifications for this investigation.</div>}
  </section>;
}
