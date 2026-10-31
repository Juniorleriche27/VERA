import type { AnalysisStep, EvidenceItem, VerificationStatus } from "@/types/vera";

const TOOL_LABELS: Record<string, string> = {
  linear_trend: "Sales trend",
  mean_shift_change_point: "Mean-level shift",
  variance_shift_change_point: "Variance shift",
  one_way_anova: "Regional comparison",
  kruskal_wallis_test: "Robust regional comparison",
  mann_whitney_u_test: "Two-group comparison",
  bayesian_structural_causal_impact: "Campaign impact",
  moving_average_trend: "Moving average",
  additive_seasonal_decomposition: "Seasonal decomposition",
};

export function toolLabel(tool: string): string {
  return TOOL_LABELS[tool] || tool.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function verificationLabel(status: VerificationStatus): string {
  switch (status) {
    case "verified": return "Verified";
    case "needs_more_analysis": return "Needs more analysis";
    case "rejected": return "Rejected";
    default: return "Not verified";
  }
}

const formatter = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
export function formatMeasure(value: number): string {
  return Number.isFinite(value) ? formatter.format(value) : String(value);
}

export function formatAxisLabel(value: string | number, isPostPeriod = false): string {
  if (isPostPeriod) return `Period ${value}`;
  if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}/.test(value)) {
    const short = value.slice(5, 10);
    return `${short.slice(0, 2)}/${short.slice(3, 5)}`;
  }
  return String(value);
}

export function evidenceStep(item: EvidenceItem, steps: AnalysisStep[]): AnalysisStep | undefined {
  return steps.find((step) => step.trace.call_id === item.call_id);
}

export function shortHash(hash: string): string {
  return hash.slice(0, 12);
}
