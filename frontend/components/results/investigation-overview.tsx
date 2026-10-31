import { AlertTriangle, CheckCircle2, FileBarChart2, GitBranch, Layers3, ShieldCheck } from "lucide-react";
import type { AnalysisResponse } from "@/types/vera";
import { toolLabel } from "@/lib/result-utils";

export function InvestigationOverview({ analysis }: { analysis: AnalysisResponse }) {
  const verified = analysis.succeeded;
  const metrics = [
    { label: "Analytical steps", value: analysis.steps.length, Icon: GitBranch },
    { label: "Verified evidence", value: analysis.decision_brief?.evidence_count ?? 0, Icon: ShieldCheck },
    { label: "Tools used", value: analysis.decision_brief?.tools_used?.length ?? 0, Icon: Layers3 },
    { label: "Visualizations", value: analysis.visualizations.length, Icon: FileBarChart2 },
  ];

  return (
    <section aria-labelledby="overview-title" className="surface overflow-hidden">
      <div className="border-b border-[#e4ebe5] px-5 py-7 sm:px-8 sm:py-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex min-w-0 items-start gap-3">
            <div
              className={`mt-0.5 flex size-11 shrink-0 items-center justify-center rounded-xl ${
                verified ? "bg-[#e7f5ec] text-[#217653]" : "bg-[#fff3e7] text-[#a97637]"
              }`}
            >
              {verified ? <CheckCircle2 size={22} /> : <AlertTriangle size={22} />}
            </div>
            <div>
              <p className="eyebrow mb-2">03 / INVESTIGATION RESULT</p>
              <h2 id="overview-title" className="text-xl font-bold tracking-[-.05em] sm:text-2xl">
                {verified ? "Verified investigation" : "Investigation needs review"}
              </h2>
              <p className="mt-1.5 text-xs text-[#809087]">
                Deterministic analytical execution · Cryptographic trace provenance
              </p>
            </div>
          </div>
          <span
            className={`rounded-full border px-3 py-1.5 text-[10px] font-extrabold tracking-[.07em] ${
              verified
                ? "border-[#cce8d7] bg-[#e7f6eb] text-[#226f50]"
                : "border-[#f0dcc3] bg-[#fff5e8] text-[#93672f]"
            }`}
          >
            {verified ? "VERIFIED INVESTIGATION" : "INVESTIGATION NEEDS REVIEW"}
          </span>
        </div>

        {/* Business question */}
        <div className="mt-7 max-w-[950px] border-l-[3px] border-[#3b906f] pl-4">
          <p className="eyebrow mb-2">Business question</p>
          <p className="text-[16px] font-semibold leading-[1.7] tracking-[-.015em] text-[#283b31] sm:text-lg">
            {analysis.question}
          </p>
        </div>

        {/* AI Tool Orchestration Chain Banner */}
        {analysis.steps.length > 0 && (
          <div className="mt-6 flex flex-wrap items-center gap-2 rounded-xl border border-[#dfeae1] bg-[#f8fbf8] p-3 text-xs">
            <span className="eyebrow text-[9.5px] font-bold text-[#2d6f4f]">AI TOOL ORCHESTRATION:</span>
            <div className="flex flex-wrap items-center gap-1.5">
              {analysis.steps.map((step, idx) => {
                const isVer = step.trace.verification_status === "verified";
                const isFailed = step.trace.status === "failed";
                return (
                  <span key={`${step.trace.call_id}-${idx}`} className="inline-flex items-center gap-1">
                    <span className="inline-flex items-center gap-1.5 rounded-md border border-[#d6e4da] bg-white px-2 py-1 text-[11px] font-medium text-[#264332] shadow-2xs">
                      <span className="flex size-4 items-center justify-center rounded-full bg-[#206f4d] text-[9px] font-bold text-white">
                        {step.index}
                      </span>
                      <span>{toolLabel(step.trace.tool)}</span>
                      <span
                        className={`text-[9.5px] font-bold ${
                          isVer
                            ? "text-[#1d6b47]"
                            : isFailed
                            ? "text-[#b03b29]"
                            : "text-[#976023]"
                        }`}
                      >
                        • {isVer ? "Verified" : step.trace.verification_status.replaceAll("_", " ")}
                      </span>
                    </span>
                    {idx < analysis.steps.length - 1 && (
                      <span className="text-[#a5b9aa] font-bold text-xs">→</span>
                    )}
                  </span>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 gap-px bg-[#e5ebe5] lg:grid-cols-4">
        {metrics.map(({ label, value, Icon }) => (
          <div key={label} className="flex min-w-0 items-center gap-3 bg-[#fafcf9] px-4 py-5 sm:px-7">
            <div className="hidden size-9 shrink-0 items-center justify-center rounded-lg bg-white text-[#538473] shadow-xs sm:flex">
              <Icon size={17} />
            </div>
            <div>
              <p className="eyebrow text-[9px]">{label}</p>
              <p className="metric-number mt-1 text-2xl font-bold sm:text-[29px]">{value.toLocaleString()}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
