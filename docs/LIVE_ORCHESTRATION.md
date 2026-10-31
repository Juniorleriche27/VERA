# Live verifiable orchestration — final UX validation

Starting SHA: `809bbc67de92f4daef063030f0e1399e66bcf983`, branch `main`.
The initial working tree contained only a generated `frontend/next-env.d.ts`
dev-path modification; it is preserved outside this change.

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

The same observer works with every `PlanningProvider`, including the existing
Meta planner. Groq configuration remains Qwen, system, parallel true, temperature
zero and bounded planning. No planner implementation or setting was changed.

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

## Verification evidence

Backend: **469 tests passed** (457 retained + 12 new cases). New tests cover
causal ordering, callback placement before execution, four tools, unchanged
observed-vs-unobserved hashes, safe planning failures, binding failures, executor
failures, no tools, response compatibility, closed payload fields, validation
before headers, disconnection and deterministic replanning origin/verdicts.
Two warnings refer to the existing deprecated FastAPI 422 alias.

Frontend: **27 tests passed** (12 retained + 15 new cases). New tests cover
initial/connecting, planning, selection, running, completed/verified/failed,
deterministic origin, retained history, interrupted connections, exhausted
execution, SSE chunk/CRLF decoding, ordering, analysis identity, safe errors,
unknown/prototype fields, and final Brief/provenance hierarchy.

`npm install`, lint, typecheck, test, demo integrity and production build passed.
Source and production frontend bundle secret-pattern scans found zero candidates;
`.env` remains ignored. Captured live events contain no bindings or provider keys.
No production dependency or lockfile changed. Browser tooling and fixtures were
installed only under `/tmp`, not added to the application.

The offline demo still produces four VERIFIED analyses and four charts. Full
trace/result/hash, Decision Brief and visualization snapshots compare identical
to the historical baseline. Both CSV copies retain SHA-256:
`65a3c77850475f248f455d87a2a8b2852c772a1988dcbaeacce636229a8764f6`.

Real Chromium browser QA exercised:

- 1920×1080, 1440×900, 1366×768; mobile 390×844 without horizontal overflow.
- Demo upload → exact official question → SSE → final metrics, brief, charts
  and expandable provenance.
- API unavailable, actual invalid-date CSV, planning exception, failed binding
  and empty plan. Failure fixtures use the actual pipeline and event emitter,
  never pre-recorded or fabricated SSE. No QA controls enter production code.
- Real Groq calls through frontend/backend, including the usual ports 3000/8000.
  All four expected analyses VERIFIED, four charts, zero unresolved.
- Two immediate button clicks produced one analysis request. No uncaught browser
  errors in the passing scenarios. Trace/report and chart/evidence hashes match.

Recorded real run on ports 3000/8000 (browser receive times from the CTA action):

| Measurement | Observed |
| --- | ---: |
| First event | 11.0 ms |
| planning_started | 433.4 ms |
| First tools_selected | 3775.0 ms |
| First tool_started | 3791.2 ms |
| First VERIFIED | 3796.2 ms |
| Terminal analysis_completed | 3803.4 ms |
| Events received | 31 |
| Distinct rendered live-panel states | 10 |

An additional warm isolated live run completed in 2988.6 ms. These are observed
samples, not performance guarantees. Several tool events shared a network read;
backend timestamps still distinguish their real execution boundaries. The final
result is fetched only after the terminal event.

Screenshots were inspected locally under `/tmp/vera-live-ui-captures/` and final
views copied to `~/.codex/artifacts/vera-live-orchestration/`, including
planning, retained completed history, final results, each desktop/mobile size,
and failure states. No video was started. The normal frontend on port 3000 now serves the validated
production build; the backend on port 8000 was restarted to load the new route.

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
- Existing model/network/quota variability and 16 moderate npm development-tool
  advisories remain as documented in the preceding audit; no forced upgrades.
- The pre-existing generated `frontend/next-env.d.ts` local change is preserved
  outside the commit, so it can remain the sole working-tree difference.

**READY FOR VIDEO RECORDING** after successful final validation. Statistical
engines, verifier, Decision Brief rules, official CSV and historical hashes are
unchanged. Groq retained; no Meta-only migration and no recording performed.
