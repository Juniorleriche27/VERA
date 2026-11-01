# Live verifiable orchestration

## Architecture and API compatibility

- `agent/events.py`: immutable JSON event records and a closed, metadata-only
  emitter, with no FastAPI dependency.
- `agent/pipeline.py`: optional observer calls immediately before/after actual
  planning, binding/execution, verification and deterministic replanning.
- `backend/analysis.py`: one shared service for both API execution paths. The
  final response and evidence serialization are unchanged.
- `backend/streaming.py`: a worker runs the synchronous analytics pipeline; an
  async queue forwards emitted events as SSE without blocking the event loop.
- Added `POST /analyses/stream`, consumed with fetch because the request has a
  JSON body. Existing `POST /analyses` and `GET /analyses/{id}` remain supported.
- SSE contains metadata only. After `analysis_completed`, the client fetches the
  existing final JSON result separately. Raw analytical result arrays therefore
  do not enter the event transport.
- `Cache-Control: no-store, no-transform` and `X-Accel-Buffering: no` discourage
  intermediary caching/buffering. No polling or analysis replay is used.

The observer is independent of the planning provider. The default runtime uses
`MetaModelPlanner`; see [Meta integration](META_MODEL_API.md).

## Event contract and true generation points

Every SSE `data` object contains `version: 1`, an `analysis_id`, monotonically
increasing `sequence`, backend monotonic `elapsed_ms`, `event` and `data`.
The SSE event/id fields agree with those JSON fields. Event data has an exact
allowlist; arbitrary nested mappings and error messages are not accepted.

| Event | Actual production boundary / safe payload |
| --- | --- |
| analysis_started | Valid request begins service execution |
| planning_started | Pipeline is about to select its catalog and invoke planner |
| planning_completed | Returned planned traces have passed pipeline validation; planning-turn count |
| tools_selected | Actual planned call references: tool, call ID, source IDs, selection origin |
| execution_started | Nonempty execution queue begins |
| tool_started | Local binding succeeded and trace entered running, immediately before executor |
| tool_completed | Executor returned and its result was accepted by the trace |
| tool_failed | Binding/execution raised; only tool and call ID are streamed |
| verification_started | Completed trace is about to enter verifier |
| verification_completed | Actual verifier decision, including rejected / needs-more-analysis |
| tool_verified | Emitted only for a VERIFIED trace |
| tool_planned | Real deterministic follow-up queued, marked deterministic rather than AI-selected |
| replanning_completed | Actual replanning outcome, without internal diagnostic text |
| execution_completed | Pipeline queue/budget processing ended; exhaustion flag |
| decision_brief_started / built | Actual final brief construction, only for successful evidence |
| visualizations_started / built | Actual chart-spec construction and count |
| analysis_completed | Final JSON stored; actual success, analysis/verified/chart/unresolved counts |
| analysis_failed | Service exception: safe fixed code, no raw exception or provider body |

Validation failures such as unknown datasets or blank questions use regular HTTP
errors before streaming headers. Failures after headers terminate with a safe
SSE event. A statistical tool failure remains a genuine failed trace and the
analysis can finish with `succeeded=false`; it does not invent verified evidence.

Events do not contain raw rows, bound numerical arrays, parameters, bindings,
questions, planner prompts, provider responses or keys. Final semantic bindings
and original results remain available through the unchanged final JSON API and
collapsible inspectors. The event count is bounded by the four planned calls
and five-step pipeline budget. The browser additionally enforces ordering,
analysis identity, typed fields, frame size and event-count limits.

## No simulated work

No artificial sleeps, UI timers, percentages or delayed replay were added to
execution. Elapsed values come from actual monotonic backend event times.
The timestamp shown in the panel explicitly denotes the last received event;
it is not a continuously simulated execution clock. Spinners represent active
backend stages. The only initial local state says the event stream is connecting.

Very fast execution/verification events can arrive in the same browser paint.
The UI does not slow them down: tool rows retain their actual selected, started,
executed and verified history, and the expandable timestamped log keeps every
event. A failed call never claims deterministic execution. Deterministic
follow-ups are identified separately from AI-selected calls.

## UX, responsive layout and provenance

- Existing VERA identity, green palette and analytical typography retained.
- Compact desktop introduction, dataset-ready card, collapsed dataset profile,
  prominent CTA, and no duplicate official-question suggestion after selection.
- During an investigation the desktop setup becomes compact; VERA, question,
  CTA and the live panel fit together at 1366×768.
- Live panel uses a restrained dark background, stage checklist and per-tool
  states; event history remains after completion or interruption.
- First result metrics: Analyses / Verified / Visualizations / Unresolved.
- Decision Brief appears before charts and technical details; headline spacing
  is compact enough for the video viewport. Statistical wording/rules unchanged.
- Bindings, parameters, execution errors, statuses, hashes and lineage remain
  inspectable. Provenance and tool-detail navigation opens collapsed inspectors.
- No chart calculations or source data changed. Existing accessible status,
  alert, keyboard, native disclosure and reduced-motion behavior retained.
- AbortController cleanup prevents stale stream updates after unmount; an
  immediate ref guard prevents duplicate submits before React state rerenders.

## Verification scope

Offline and injected-client tests cover causal event ordering, four-tool execution,
matching observed/unobserved hashes, safe planning failures, binding/execution
failures, empty plans, disconnection and deterministic replanning origins.
Frontend tests cover SSE parsing, event ordering and identity, retained progress,
safe errors, interruption, and final evidence/provenance presentation.

These checks exercise actual pipeline events with injected planning responses.
They do not establish live Meta availability or model-selection quality. Run the
optional Meta diagnostic with official credentials for that validation. The offline
demo separately verifies reproducible traces, Decision Brief and chart specifications.

## Remaining operational limits

- Use one API worker for the local demo. Dataset/analysis storage is still
  process-local and ephemeral; multiple workers need shared storage or sticky
  routing for the final GET. No database was introduced.
- Disconnect cancels at the next event boundary and prevents subsequent binding
  or result persistence when cancellation is observed. A provider HTTP request
  already running in a Python thread cannot be forcibly interrupted; it may
  finish before the worker exits. There is no background execution replay.
- A dropped stream is reported and its received history retained. No automatic
  reconnect duplicates an analysis. Proxy buffering/idle timeout settings still
  matter outside the local environment; no public-production sign-off implied.
- Model availability, network and account quota still determine live planning
  availability. Injected-client validation does not certify a live Meta request.
