import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { consumeAnalysisStream, initialLiveState, parseAnalysisEvent, reduceLiveEvent, type AnalysisEvent, type EventName } from "../lib/analysis-stream";
import { LiveOrchestration } from "../components/live-orchestration";
import { AnalysisWorkspace } from "../components/analysis-workspace";

const ref = { tool: "linear_trend", call_id: "trend", source_call_ids: [], selection: "ai" as const };
function event(event: EventName, data: AnalysisEvent["data"] = {}, sequence = 1): AnalysisEvent {
  return { version: 1, analysis_id: "analysis", sequence, event, elapsed_ms: sequence * .25, data };
}
const frame = (value: AnalysisEvent) => `id: ${value.sequence}\nevent: ${value.event}\ndata: ${JSON.stringify(value)}\n\n`;
function stream(text: string, size = text.length): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream({ start(controller) { for (let n = 0; n < bytes.length; n += size) controller.enqueue(bytes.slice(n, n + size)); controller.close(); } });
}
const complete = (sequence = 2) => event("analysis_completed", { succeeded: true, analyses: 1, verified: 1, visualizations: 1, unresolved: 0, exhausted: false }, sequence);

test("initial live state contains no invented backend activity", () => {
  const html = renderToStaticMarkup(createElement(LiveOrchestration, { state: initialLiveState(), busy: true }));
  assert.ok(html.includes("Connecting to the investigation event stream"));
  assert.ok(html.includes("0 real events"));
  assert.ok(!html.includes("AI Selected"));
});

test("planning through verified history comes only from received events", () => {
  let state = initialLiveState();
  for (const value of [event("analysis_started"), event("planning_started"), event("planning_completed", { planning_turns: 1 }), event("tools_selected", { calls: [ref] })]) state = reduceLiveEvent(state, value);
  assert.equal(state.tools[0].status, "planned");
  state = reduceLiveEvent(state, event("tool_started", { tool: ref.tool, call_id: ref.call_id }));
  let html = renderToStaticMarkup(createElement(LiveOrchestration, { state, busy: true }));
  assert.ok(html.includes("RUNNING")); assert.ok(!html.includes("Deterministically Executed"));
  state = reduceLiveEvent(state, event("tool_completed", { tool: ref.tool, call_id: ref.call_id }));
  state = reduceLiveEvent(state, event("verification_completed", { tool: ref.tool, call_id: ref.call_id, status: "verified" }));
  state = reduceLiveEvent(state, event("tool_verified", { tool: ref.tool, call_id: ref.call_id }));
  state = reduceLiveEvent(state, event("decision_brief_started"));
  state = reduceLiveEvent(state, event("decision_brief_built"));
  state = reduceLiveEvent(state, complete());
  html = renderToStaticMarkup(createElement(LiveOrchestration, { state, busy: false }));
  assert.ok(html.includes("VERIFIED")); assert.ok(html.includes("Deterministically Executed"));
  assert.ok(html.includes("Decision Brief built")); assert.ok(html.includes("Received event history retained"));
  assert.equal(state.events.length, 11);
});

test("failed tool never claims executed or verified", () => {
  let state = reduceLiveEvent(initialLiveState(), event("tools_selected", { calls: [ref] }));
  state = reduceLiveEvent(state, event("tool_started", { tool: ref.tool, call_id: ref.call_id }));
  state = reduceLiveEvent(state, event("tool_failed", { tool: ref.tool, call_id: ref.call_id }));
  const html = renderToStaticMarkup(createElement(LiveOrchestration, { state, busy: false }));
  assert.ok(html.includes("FAILED")); assert.ok(!html.includes("Deterministically Executed")); assert.ok(!html.includes(">VERIFIED<"));
});

test("deterministic follow-ups are not labeled as AI selections", () => {
  const state = reduceLiveEvent(initialLiveState(), event("tool_planned", { call: { ...ref, selection: "deterministic" } }));
  const html = renderToStaticMarkup(createElement(LiveOrchestration, { state, busy: true }));
  assert.ok(html.includes("Deterministic follow-up")); assert.ok(!html.includes("AI Selected"));
});

for (const size of [1, 7, 1024]) test(`SSE decodes fragmented UTF-8 and CRLF at chunk size ${size}`, async () => {
  const seen: AnalysisEvent[] = [];
  const id = await consumeAnalysisStream(stream((': keepalive\n\n' + frame(event("analysis_started")) + frame(complete())).replaceAll('\n', '\r\n'), size), e => seen.push(e));
  assert.equal(id, "analysis"); assert.equal(seen.length, 2);
});

test("a truncated connection cannot be shown as completed", async () => {
  const seen: AnalysisEvent[] = [];
  await assert.rejects(consumeAnalysisStream(stream(frame(event("analysis_started"))), e => seen.push(e)), /ended before/);
  assert.equal(seen.length, 1);
});

test("provider failures retain the real terminal event and only the safe error", async () => {
  const seen: AnalysisEvent[] = [];
  await assert.rejects(consumeAnalysisStream(stream(frame(event("analysis_started")) + frame(event("analysis_failed", { code: "planning_failed" }, 2))), e => seen.push(e)), /AI planner/);
  assert.equal(seen.at(-1)?.event, "analysis_failed");
});

test("out-of-order events and switched analysis IDs are rejected", async () => {
  await assert.rejects(consumeAnalysisStream(stream(frame(event("analysis_started")) + frame(complete(3))), () => {}), /inconsistent/);
  await assert.rejects(consumeAnalysisStream(stream(frame(event("analysis_started")) + frame({ ...complete(), analysis_id: "other" })), () => {}), /inconsistent/);
});

test("unknown fields and raw numerical payloads are rejected", () => {
  assert.throws(() => parseAnalysisEvent({ ...event("analysis_started"), data: { rows: [1, 2] } }), /invalid/);
  assert.throws(() => parseAnalysisEvent(event("tools_selected", { calls: [{ ...ref, bindings: [1, 2] }] as never })), /invalid/);
});

test("zero-unresolved summary and Decision Brief precede collapsed provenance", () => {
  const analysis = { analysis_id: "a", dataset_id: "d", question: "Compare", succeeded: true, exhausted: false, final_status: "verified" as const,
    unresolved_call_ids: [], steps: [], visualizations: [], decision_brief: { question: "Compare", headline: "Verified finding", findings: ["Finding"], caveats: [], next_action: "Review", tools_used: [], evidence_count: 0, evidence: { question: "Compare", items: [] } } };
  const html = renderToStaticMarkup(createElement(AnalysisWorkspace, { analysis, busy: false, error: null }));
  assert.ok(html.includes("Unresolved")); assert.ok(html.indexOf("Verified finding") < html.indexOf("Evidence &amp; Provenance")); assert.ok(html.includes("<details"));
});

test("a lost connection stops all progress indicators without inventing a terminal event", () => {
  const state = reduceLiveEvent(initialLiveState(), event("planning_started"));
  const html = renderToStaticMarkup(createElement(LiveOrchestration, { state, busy: false }));
  assert.ok(html.includes("Investigation interrupted"));
  assert.ok(!html.includes("animate-spin"));
  assert.equal(state.events.length, 1);
  assert.equal(state.complete, false);
});

test("an exhausted execution budget is not labeled a completed execution stage", () => {
  let state = reduceLiveEvent(initialLiveState(), event("execution_started"));
  state = reduceLiveEvent(state, event("execution_completed", { exhausted: true }));
  state = reduceLiveEvent(state, { ...complete(), data: { ...complete().data, succeeded: false, exhausted: true } });
  const html = renderToStaticMarkup(createElement(LiveOrchestration, { state, busy: false }));
  assert.ok(html.includes("Deterministic Execution"));
  assert.ok(html.includes("Not completed"));
});

test("prototype names and extra envelope data are not accepted as stream events", () => {
  assert.throws(() => parseAnalysisEvent({ ...event("analysis_started"), event: "constructor" }), /invalid/);
  assert.throws(() => parseAnalysisEvent({ ...event("analysis_started"), raw_data: [1, 2] }), /invalid/);
});
