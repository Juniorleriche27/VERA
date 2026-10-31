"use client";

import { Area, Bar, BarChart, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CalendarRange, Fingerprint } from "lucide-react";
import type { VisualizationSeries, VisualizationSpec } from "@/types/vera";
import { formatAxisLabel, formatMeasure, shortHash } from "@/lib/result-utils";

const ROLE_COLORS: Record<string, string> = {
  observed: "#18765a", fitted: "#18765a", counterfactual: "#5b7284",
  interval_lower: "#b2c0cb", interval_upper: "#b2c0cb",
  mean: "#2a8265", median: "#2a8265", variance: "#62887c",
};
const FALLBACK_COLORS = ["#18765a", "#597c78", "#839e92", "#abbcb2"];

function seriesColor(series: VisualizationSeries, index: number) {
  return (series.role && ROLE_COLORS[series.role]) || FALLBACK_COLORS[index % FALLBACK_COLORS.length];
}

const tickStyle = { fontSize: 10, fill: "#89968c" };

function ChartTooltip({ spec }: { spec: VisualizationSpec }) {
  const isPeriod = spec.xKey === "post_period";
  return <Tooltip
    contentStyle={{ background: "#fff", border: "1px solid #dbe5de", borderRadius: 10, boxShadow: "0 10px 30px rgba(20,40,27,.08)", fontSize: 12 }}
    cursor={{ stroke: "#a0b8ab", strokeDasharray: "3 3" }}
    labelFormatter={(value) => formatAxisLabel(String(value), isPeriod)}
    formatter={(value, name) => [typeof value === "number" ? formatMeasure(value) : String(value ?? "—"), String(name)]}
  />;
}

export function VerifiedChart({ spec }: { spec: VisualizationSpec }) {
  const isLine = spec.chartType === "line";
  const isBar = spec.chartType === "bar";
  const isPeriod = spec.xKey === "post_period";
  const xTick = (value: string | number) => formatAxisLabel(value, isPeriod);
  const chartMargin = { top: 12, right: 12, left: -12, bottom: 8 };
  const axes = <>
    <CartesianGrid stroke="#e7ede7" strokeDasharray="3 5" vertical={false} />
    <XAxis dataKey={spec.xKey} tick={tickStyle} tickFormatter={xTick} tickLine={false} axisLine={false} minTickGap={18} />
    <YAxis tick={tickStyle} tickFormatter={(value: number) => formatMeasure(value)} width={51} tickLine={false} axisLine={false} />
    <ChartTooltip spec={spec}/>
  </>;

  return <article className="surface min-w-0 overflow-hidden" aria-label={spec.meta.title}>
    <div className="border-b border-[#eef1ec] px-5 pb-4 pt-5 sm:px-6 sm:pt-6"><p className="eyebrow mb-2">VERIFIED VISUAL EVIDENCE</p><h4 className="text-[17px] font-bold tracking-[-.035em] text-[#293d31]">{spec.meta.title}</h4><p className="mt-2 text-[12px] leading-5 text-[#7b8a80]">{spec.meta.description}</p></div>
    <div className="px-2 pb-3 pt-5 sm:px-4">
      {isLine || isBar ? <div className="h-[260px] w-full min-w-0 sm:h-[315px]" role="img" aria-label={`${spec.meta.title}. Chart based on ${spec.data.length} verified data points.`}>
        <ResponsiveContainer width="100%" height="100%" minWidth={0}>
          {isLine ? <ComposedChart data={spec.data} margin={chartMargin} accessibilityLayer>
            {axes}
            {spec.bands?.map((band, index) => <Area key={`${band.lowerKey}-${band.upperKey}-${index}`} name={band.label} dataKey={(datum: Record<string, string | number | null>) => [datum[band.lowerKey], datum[band.upperKey]]} stroke="none" fill="#9bb9aa" fillOpacity={0.25} type="monotone" isAnimationActive={false} legendType="none" tooltipType="none" connectNulls={false}/>)}
            {spec.series.map((series, index) => <Line key={series.dataKey} type="monotone" dataKey={series.dataKey} name={series.label} stroke={seriesColor(series, index)} strokeWidth={series.role === "interval_lower" || series.role === "interval_upper" ? 1 : 2.5} strokeDasharray={series.role === "counterfactual" ? "6 4" : series.role?.startsWith("interval_") ? "3 4" : undefined} dot={false} activeDot={series.role?.startsWith("interval_") ? false : { r: 4 }} isAnimationActive={false}/>) }
          </ComposedChart> : <BarChart data={spec.data} margin={chartMargin} accessibilityLayer barCategoryGap="24%">
            {axes}
            {spec.series.map((series, index) => <Bar key={series.dataKey} dataKey={series.dataKey} name={series.label} fill={seriesColor(series, index)} radius={[5,5,0,0]} maxBarSize={75} isAnimationActive={false}/>)}
          </BarChart>}
        </ResponsiveContainer>
      </div> : <p className="rounded-lg bg-[#f5f7f5] p-5 text-sm text-[#68796c]">This verified visualization uses an unsupported chart type: {spec.chartType}.</p>}
      {(isLine || isBar) && <div className="mt-2 flex flex-wrap items-center justify-center gap-x-5 gap-y-2 px-5">{spec.series.filter((series) => !series.role?.startsWith("interval_")).map((series, index) => <span key={series.dataKey} className="inline-flex items-center gap-2 text-[11px] text-[#718077]"><span className="inline-block h-[3px] w-5 rounded-full" style={{background:seriesColor(series, index)}}/>{series.label}</span>)}{(spec.bands || []).map((band, i) => <span key={i} className="inline-flex items-center gap-2 text-[11px] text-[#718077]"><span className="inline-block h-3 w-5 rounded bg-[#9bb9aa]/30"/>{band.label}</span>)}</div>}
    </div>
    {!!spec.annotations?.length && <div className="mx-5 mb-3 flex flex-wrap gap-2 sm:mx-6">{spec.annotations.map((note,i) => <span key={i} className="inline-flex items-center gap-2 rounded-lg border border-[#e2e9e1] bg-[#f6f9f5] px-3 py-2 text-[11px] text-[#59755f]"><CalendarRange size={13}/>{note.label}: {String(note.value)}</span>)}</div>}
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-[#eef1ec] bg-[#fafcf9] px-5 py-3 text-[10px] text-[#85958a] sm:px-6"><Fingerprint size={13} className="text-[#638971]"/><span>{spec.meta.footer}</span><span aria-label="Evidence call ID" className="ml-auto font-mono text-[#668472]">{spec.evidence.call_id} · {shortHash(spec.evidence.trace_hash)}</span></div>
  </article>;
}
