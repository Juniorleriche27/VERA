# Independent final technical audit

Historical audit at `93b0678`. The subsequent live Groq investigation and current
planner configuration are documented in [GROQ_LIVE_DIAGNOSTIC.md](GROQ_LIVE_DIAGNOSTIC.md).

Starting point: main, `473e57f46f6e0245e5620c3d06e5694d06ac20f2`, clean.
Baseline: 380 Python tests pass; official demo: four verified calls, four charts.

## Findings recorded before implementation

21 adversarial regressions in `tests/test_final_hardening.py` failed on the baseline.
The existing suite was not removed or weakened.

| Severity | Location | Proven behavior and impact | Scoped correction |
| --- | --- | --- | --- |
| HIGH | agent/meta_planner.py, agent/pipeline.py | A `preview` field or column metadata extension can send raw rows to the provider; blacklist-only protection. API-created profiles are safe, but direct callers are not protected. | Explicit typed metadata allowlist, applied before any planning provider. |
| HIGH | agent/meta_planner.py, agent/pipeline.py | Initial calls can invent ancestors; duplicate IDs and a replan ID matching a queued initial call corrupt provenance/resolution. | Require independent initial calls and unique IDs; reserve all queued IDs when generating descendants. |
| MEDIUM | agent/data_binding.py | Explicit unknown grouped columns silently infer different columns; a single explicit Mann-Whitney group is overwritten, reversing directional hypotheses. | Reject unknown explicit references; preserve either supplied group and infer only its counterpart. |
| MEDIUM | verification/verifier.py | Negative/fractional/string sample counts, non-array p-values, half intervals and causal interval length mismatch can be VERIFIED. | Reject malformed evidence and check declared causal array lengths without recalculating statistics. |
| MEDIUM | agent/tool_call_trace.py | Direct dataclass construction accepts NaN/Infinity JSON and bypasses factory validation. | Validate and canonicalize every stored JSON string at construction. |
| MEDIUM | backend/app.py | Upload reads all bytes before enforcing limit. | Read at most limit + 1; multipart disk spooling / upstream body limits remain deployment concerns. |
| MEDIUM | backend/app.py | Provider failures produce an unstructured 500; unresolved IDs omitted although frontend supports them. CSV exception strings may contain user data. | Safe JSON errors for known provider/planning failures, generic parse detail, include unresolved IDs. |
| MEDIUM | frontend/components/results/investigation-timeline.tsx | Failed calls unconditionally display `Deterministically Executed` with a success icon. Overview calls them `not verified`. | Render execution and failed state from actual trace status; render actual components in regression tests. |
| LOW | README.md, agent/decision_brief.py | `.env` is documented but not loaded by uvicorn command; selector is shown after planner though it executes first; catalog size omits two core overlaps; non-significant rank next-action refers to means. | Correct startup command, diagram, counts and rank-test wording; document local-only storage constraints. |
| MEDIUM (deferred) | frontend/package-lock.json | npm audit reports 16 moderate development-tool findings cascading from minimatch <11.0.1. No automatic compatible fix offered; proposed fixes include major downgrades. | Do not force dependency changes; record advisory and production-only audit separately. |

No BLOCKER established for the final local live test. Public multi-user deployment
is out of scope: unauthenticated, process-local, unbounded dataset/analysis storage;
CORS is not authentication. Do not treat this application as an isolated tenant service.

## Final assessment

- Architecture retained: local catalog preselection precedes the LLM; only the
  binder sees local values; executor computes; verifier evaluates; presentation
  uses deterministic evidence. No provider migration or business feature added.
- Seven strategic contracts: real model-response parsing (injected response,
  no network), semantic validation, binding, execution and verification tested.
  Non-default execution settings are asserted in the actual result, not just
  forwarded mock arguments. Initial lineage is empty in both schema and parser;
  IDs and generated descendants are checked in the pipeline.
- Schemas remain closed for the seven strategic tools. Optional binding types,
  required top-level objects and numerical schema bounds match runtime checks.
  Other catalog tools retain generic schemas: these are not a guarantee that
  all 102 exposed names have fully certified semantic bindings. Expanding that
  contract coverage is deferred rather than changing the demo's scope.
- Groq remains the API default, with its own key, model and provider metadata;
  `parallel_tool_calls=False` is forwarded. Meta compatibility is tested with
  an injected OpenAI-compatible client, not a live endpoint. Endpoints unchanged.
- Privacy: outbound context is constructed from typed metadata fields at both
  pipeline and planner boundaries. Profile generation contains no cell examples
  or raw rows. The frontend renders `ToolCallTrace.bindings`, never local
  `BoundToolCall.bindings`. Question text and schema labels remain untrusted
  user text; the allowlist cannot determine whether someone puts private text
  into a column name. No LLM evaluates statistics or authorizes code execution.
- Trace/hash: stored JSON is validated/canonicalized even through direct
  dataclass construction. JSON accessors return copies. Re-verification hashes
  the resulting trace and report consistently. Official demo snapshots from
  the starting commit and the final implementation compare byte-for-byte equal
  for all trace fields (including full hashes), brief and visualization specs.
  A trace hash is a content digest, not a signature or a digest of the input CSV;
  it does not authenticate dataset identity by itself.
- Pipeline/replanning: failed branches prevent success; unresolved independent
  branches are retained; max_steps/exhaustion and empty plans remain explicit.
  ANOVA → Kruskal and Pearson → Spearman mappings are preserved. No new fallback.
- Verifier: rejects the reproduced malformed evidence; causal arrays provided
  must match declared post counts. Existing interval, probability, warning and
  diagnostic checks remain. This is a rule-based consistency verifier, not an
  independent recomputation of every statistic or proof of causal validity.
- Decision Brief: conditional causal wording and zero-crossing intervals retained;
  robust descendants replace non-verified ancestors in final evidence. The
  official brief and chart evidence hashes are unchanged. Non-significant rank
  tests now use group-difference wording rather than claiming to test means.
- Visualization: uses existing fitted/observed/counterfactual arrays; no invented
  observed trend. Causal range bands use actual lower and upper keys. Unknown
  chart types and empty arrays have explicit UI states. Backend validates finite
  chart values and evidence source hashes are preserved.
- API: bounded application-level upload read, safe known provider errors, explicit
  unresolved IDs, and dataclass/as_dict serialization regression. CSV MIME is not
  trusted as validation: extension plus actual parsing is authoritative. IDs are
  random UUIDs and filenames are not used for filesystem writes. Storage remains
  unauthenticated, unbounded, per-process and ephemeral. Synchronous CSV profiling
  runs in the async upload handler; high concurrent upload load remains unsuitable
  for a public deployment. No database or new infrastructure introduced.
- Frontend: existing busy/reset guards retained; actual orchestration components
  rendered in tests for all four execution states. Failed calls no longer claim
  execution success. Accessibility status/alert semantics and responsive structure
  retained. React best-practices checklist applied to the changed components;
  no new hooks, effects, network waterfall or bundle dependency. Historical browser
  QA was reviewed in code; this audit did not repeat that browser matrix.
- Security: tracked/untracked nonignored review files scanned for private-key,
  provider-key and token patterns without printing matched contents: zero
  candidates. Only `.env.example` and `frontend/.env.example` are environment
  files in review; `.env.*` and frontend local environments are now ignored.
  This is a reasonable pattern scan, not a guarantee against all secret formats
  or an audit of the complete Git history.
- Dependencies: pinned core engine SHA preserved; requirements install and
  `pip check` succeed. No dependency or lockfile changes. `npm audit --omit=dev`
  reports zero vulnerabilities. Full npm audit reports 16 moderate development
  dependency findings through minimatch, advisory
  https://github.com/advisories/GHSA-g3hv-75wg-m249; no forced major fix applied.
  pip-audit reports advisories only in the temporary environment's bootstrap pip
  24.0 (12 entries, 6 unique IDs); no application dependency advisory identified.
  The Git-sourced koryxa engine is not in the PyPI advisory database and is skipped
  by that scanner; its local source and pinned revision were inspected directly.
- Documentation: corrected the actual catalog count (97 core + 7 extensions −
  2 overridden names = 102 unique tools), selector/planner order, environment loading, frontend
  URL default and validation counts; documented trusted-local runtime limits and
  actual Meta/Groq configuration. Dockerfile's pip upgrade and explicit env-file
  startup were inspected; no Docker daemon/build validation performed.

## Validation and changed files

- Python: `PYTHONPATH=. python -m pytest -q -p no:asyncio`: **416 passed**;
  one pre-existing FastAPI deprecated-422-constant warning.
- Demo runner: four VERIFIED tools, one Decision Brief, four visualizations.
- Full baseline/final demo snapshots: identical traces, hashes, brief and specs.
- CSV SHA-256 unchanged in both copies:
  `65a3c77850475f248f455d87a2a8b2852c772a1988dcbaeacce636229a8764f6`.
- Frontend: npm install, lint, typecheck, **12 tests**, integrity, production build
  all successful. No dependency tree changes from npm install.
- `git diff --check`: clean. Secret-pattern scan: zero candidate files.
- New tests: **36 Python cases**, **5 React rendering cases**. Existing 380 Python
  and 7 frontend cases retained. No test removed or weakened.
- Changed production files: `.gitignore`, `README.md`, `agent/data_binding.py`,
  `agent/meta_planner.py`, `agent/pipeline.py`, `agent/tool_call_trace.py`,
  `agent/decision_brief.py`,
  `backend/app.py`, `verification/verifier.py`,
  `frontend/components/results/investigation-overview.tsx`,
  `frontend/components/results/investigation-timeline.tsx`.
- Added: this report, `tests/test_final_hardening.py`,
  `frontend/tests/orchestration.test.ts`.

## Remaining risks and recommendation

BLOCKER: none established for local testing. HIGH: reproduced findings corrected.
MEDIUM remaining: development dependency advisories; public-service memory,
concurrency and isolation limits documented above. LOW: the existing deprecated constant warning; catalog-count and rank-test
wording inaccuracies corrected. OPTIONAL: broader generic tool
schemas/contract coverage, a complete dependency lock for Python, and repeating
browser QA before a public release.

**READY for the final local live Groq test**, not a public production sign-off.
No live Groq or Meta request was made during this audit. The offline demo uses
an explicit deterministic planner and cannot certify actual model selection.
Check credentials, provider/model tool support, real response shape and the four
expected analyses live. The official question says "September" without supplying
an exact year/date; safe profile metadata contains no temporal bounds or group
labels. A live model may lack enough information for an exact intervention date
or named group comparison; supply an explicit date in a follow-up if requested,
never introduce a hidden fabricated-date fallback. Prompt/tool examples do not
prove that a date exists in a particular dataset. The binder fails explicitly
when dates or columns do not exist.

No video recorded. Groq retained. Official CSV untouched.
