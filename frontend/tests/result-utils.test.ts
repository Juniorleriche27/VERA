import assert from "node:assert/strict";
import test from "node:test";
import { evidenceStep, formatAxisLabel, formatMeasure, shortHash, toolLabel, verificationLabel } from "../lib/result-utils";
import type { AnalysisStep, EvidenceItem } from "../types/vera";

test("friendly tool labels preserve unknown tools", () => {
  assert.equal(toolLabel("linear_trend"), "Sales trend");
  assert.equal(toolLabel("mean_shift_change_point"), "Mean-level shift");
  assert.equal(toolLabel("kruskal_wallis_test"), "Robust regional comparison");
  assert.equal(toolLabel("bayesian_structural_causal_impact"), "Campaign impact");
  assert.equal(toolLabel("new_future_method"), "New Future Method");
});

test("verification states are faithfully labeled without implicit success", () => {
  assert.equal(verificationLabel("verified"), "Verified");
  assert.equal(verificationLabel("rejected"), "Rejected");
  assert.equal(verificationLabel("needs_more_analysis"), "Needs more analysis");
  assert.equal(verificationLabel("not_verified"), "Not verified");
});

test("numeric formatting is presentational only", () => {
  assert.equal(formatMeasure(6928.472987123), "6,928.47");
  assert.equal(formatMeasure(-123.4), "-123.4");
  assert.equal(formatAxisLabel("2026-09-01T00:00:00"), "09/01");
  assert.equal(formatAxisLabel(1, true), "Period 1");
  assert.equal(shortHash("abc123def456rest"), "abc123def456");
});

test("evidence maps to the actual report by call ID, including replans", () => {
  const trace = {
    call_id: "anova-replan-1", tool: "kruskal_wallis_test", source_call_ids: ["anova"],
    objective: "Check group differences", reason: "Robustness check", parameters: {}, bindings: {},
    status: "completed", verification_status: "verified", trace_hash: "9".repeat(64),
    deterministic_result: {}, next_decision: "interpret_result",
  } as const;
  const step: AnalysisStep = { index: 2, trace: { ...trace, source_call_ids: [...trace.source_call_ids] }, report: null, replanning: null };
  const item: EvidenceItem = {
    call_id: trace.call_id, tool: trace.tool, trace_hash: trace.trace_hash,
    objective: trace.objective, result: {}, source_call_ids: ["anova"],
  };
  assert.equal(evidenceStep(item, [step]), step);
  assert.equal(evidenceStep({ ...item, call_id: "other" }, [step]), undefined);
});
