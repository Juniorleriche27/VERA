import { CheckCircle2, Database, ShieldCheck } from "lucide-react";
import type { DatasetUploadResponse } from "@/types/vera";

export function DatasetProfile({ dataset }: { dataset: DatasetUploadResponse }) {
  const { profile } = dataset;
  return <section aria-labelledby="profile-heading" className="surface mt-5 overflow-hidden">
    <div className="flex flex-wrap items-start justify-between gap-3 border-b border-[#e9efea] px-5 py-5 sm:px-7">
      <div className="min-w-0"><div className="mb-2 flex items-center gap-2 text-[11px] font-bold text-[#257654]"><CheckCircle2 size={15}/> DATASET READY</div><h2 id="profile-heading" className="max-w-[260px] truncate text-sm font-bold" title={dataset.filename}>{dataset.filename}</h2></div>
      <Database size={17} className="text-[#a4b2a8]"/>
      <div className="grid w-full grid-cols-2 gap-3 pt-2">
        <div className="rounded-lg bg-[#f5f8f5] p-4"><p className="eyebrow">Rows</p><p className="metric-number mt-2 text-[26px] font-bold">{profile.row_count.toLocaleString()}</p></div>
        <div className="rounded-lg bg-[#f5f8f5] p-4"><p className="eyebrow">Columns</p><p className="metric-number mt-2 text-[26px] font-bold">{profile.column_count.toLocaleString()}</p></div>
      </div>
    </div>
    <div className="px-5 pb-5 pt-5 sm:px-7">
      <p className="eyebrow mb-3">Schema overview</p>
      <div className="max-w-full overflow-x-auto rounded-lg border border-[#e7ebe6]">
        <table className="w-full min-w-[435px] text-left text-xs" aria-label="Safe dataset column profile"><thead className="bg-[#f6f8f5] text-[10px] uppercase tracking-[.09em] text-[#87938a]"><tr><th className="px-3 py-3 font-semibold">Column</th><th className="px-2 py-3 font-semibold">Role</th><th className="px-2 py-3 font-semibold">Type</th><th className="px-2 py-3 text-right font-semibold">Missing</th><th className="px-3 py-3 text-right font-semibold">Unique</th></tr></thead>
          <tbody className="divide-y divide-[#ebefea] text-[#3d4b42]">{profile.columns.map((col, index) => <tr key={`${col.name}-${index}`}><td className="max-w-[145px] truncate px-3 py-3 font-semibold" title={col.name}>{col.name}</td><td className="px-2 py-3"><span className={`rounded px-1.5 py-1 text-[10px] font-semibold ${col.role === "time" ? "bg-[#e7f0fc] text-[#4168a1]" : col.role === "measure" ? "bg-[#e5f4eb] text-[#297451]" : "bg-[#f2efe8] text-[#7e7154]"}`}>{col.role}</span></td><td className="px-2 py-3 text-[10px] text-[#829087]">{col.dtype}</td><td className="px-2 py-3 text-right tabular-nums">{col.missing_count}</td><td className="px-3 py-3 text-right tabular-nums">{col.unique_count}</td></tr>)}</tbody></table>
      </div>
      <div className="mt-5 flex items-start gap-2.5 rounded-lg bg-[#f2f8f4] p-3.5 text-[#49685a]"><ShieldCheck size={16} className="mt-0.5 shrink-0 text-[#28805d]"/><p className="text-[11px] leading-[1.65]"><strong>Privacy by design.</strong> Raw rows are not sent to the planning model. Only safe schema and profile metadata are shared for analytical planning.</p></div>
    </div>
  </section>;
}
