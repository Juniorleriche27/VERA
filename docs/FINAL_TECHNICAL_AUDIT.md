# Technical guarantees and audit scope

The implementation retains these independently tested boundaries:

- The local catalog selector runs before Meta planning; only typed, allowlisted
  schema/profile metadata and the user question enter the provider request.
  No rows, group examples, bound arrays or analytical results are transmitted.
- Meta chooses tools, parameters and semantic bindings. The local binder resolves
  column references; the deterministic engine computes every statistical result.
- Initial calls are independent; closed strategic tool schemas reject malformed
  bindings, fabricated lineage, unknown tools and invalid numerical parameters.
- Bounded planning rejects duplicates and incomplete coverage, with at most four
  planning turns and four initial analytical calls. Continuations contain selected
  tool names and missing scopes, without execution results.
- The verifier checks numerical consistency and selected structural rules. It
  does not recompute every statistic or establish causal validity. A trace hash
  is a content digest, not a signature or authentication of dataset identity.
- Verified evidence supplies the deterministic Decision Brief and visualizations.
  Provenance hashes link traces, reports, evidence and chart specifications.
- HTTP and SSE provider errors use fixed safe messages/codes. Missing Meta
  credentials do not prevent import, health checks or dataset upload.

The official offline demo must retain four VERIFIED analyses, one Decision Brief,
four visualizations, and unchanged full trace/brief/chart snapshots. Its synthetic
CSV SHA-256 is `65a3c77850475f248f455d87a2a8b2852c772a1988dcbaeacce636229a8764f6`.
It uses fixed plans and cannot certify model selection or live API access.

The seven strategic contracts receive focused parser/binder/executor/verifier
coverage; generic schemas for other catalog tools do not certify all 102 tools
for every possible semantic binding. Column labels and questions are supplied
text: users must not put private cell contents into these fields.

The application is intended for a trusted local demonstration. Dataset/analysis
storage is unauthenticated, ephemeral, process-local and unbounded in total size.
Use one API worker. CORS is not authentication; public deployment needs separate
isolation, retention, rate limiting and request/body/memory controls. SSE disconnect
cancels at event boundaries but cannot terminate an in-flight provider HTTP call.

No statistical engine, extension, binder, verifier, trace/hash implementation,
Decision Brief rule, visualization, demo dataset or frontend business component
is changed by the Meta provider consolidation. Dependencies remain pinned as before.
See [Meta contract and validation](META_MODEL_API.md) and the root README for
current setup, test commands and optional live diagnostics.

## Consolidation validation — 2026-11-01

- Python suite: 472 passed, zero failed; two pre-existing FastAPI constant warnings.
- Frontend: lint, typecheck, 27 tests, demo integrity and production build passed.
- Full offline trace/result/hash, brief and chart snapshots match the saved baseline
  byte-for-byte; both demo CSV copies retain the SHA-256 recorded above.
- Missing-key HTTP/SSE and complete injected Meta-client evidence paths passed.
- No live Meta request was made: MODEL_API_KEY was unavailable in this session.
