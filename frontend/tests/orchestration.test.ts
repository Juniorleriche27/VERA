import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { InvestigationTimeline } from "../components/results/investigation-timeline";
import { InvestigationOverview } from "../components/results/investigation-overview";
import type { AnalysisResponse, AnalysisStep, TraceStatus } from "../types/vera";

function step(status: TraceStatus): AnalysisStep {
  return { index: 1, report: null, replanning: null, trace: {
    call_id: "call", tool: "linear_trend", objective: "Trend", reason: "Check trend",
    bindings: { time_column: "date", value_column: "sales" }, parameters: {},
    source_call_ids: [], status, verification_status: status === "completed" ? "verified" : "not_verified",
    deterministic_result: null, trace_hash: "a".repeat(64), next_decision: "",
    execution_error: status === "failed" ? "Invalid input" : "",
  }};
}

for (const status of ["failed", "planned", "running", "completed"] as const) {
  test(`timeline faithfully renders ${status} execution`, () => {
    const html = renderToStaticMarkup(createElement(InvestigationTimeline, { steps: [step(status)] }));
    assert.equal(html.includes("Deterministically Executed"), status === "completed");
    assert.equal(html.includes(">VERIFIED</span>"), status === "completed");
    if (status === "failed") {
      assert.ok(html.includes("Execution Failed"));
      assert.ok(html.includes("FAILED"));
      assert.ok(html.includes("Invalid input"));
    }
    assert.ok(html.includes("time_column"));
    assert.ok(html.includes("date"));
  });
}

test("overview identifies failed tool explicitly", () => {
  const analysis: AnalysisResponse = {
    analysis_id: "a", dataset_id: "d", question: "Trend", succeeded: false, exhausted: false,
    final_status: "not_verified", steps: [step("failed")], decision_brief: null, visualizations: [],
  };
  const html = renderToStaticMarkup(createElement(InvestigationOverview, { analysis }));
  assert.ok(html.includes("Failed"));
  assert.ok(!html.includes("Verified investigation"));
});
