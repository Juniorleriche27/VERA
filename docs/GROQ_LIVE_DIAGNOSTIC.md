# Groq live planner diagnosis — 2026-10-31

Starting commit: `93b06786dacdac6064120b34de66cee7f8c62bf1`, branch `main`, clean.
All provider calls used the authorized local environment; no credential value was
printed, persisted in diagnostics or added to the repository. Raw CSV rows were
not sent to a provider. Only the official synthetic dataset was used, plus a
local date-shifted copy for the generalization check.

## Account endpoint and outbound catalog

`GET https://api.groq.com/openai/v1/models` returned these 11 IDs:

- `allam-2-7b`
- `canopylabs/orpheus-arabic-saudi`
- `canopylabs/orpheus-v1-english`
- `meta-llama/llama-prompt-guard-2-22m`
- `meta-llama/llama-prompt-guard-2-86m`
- `openai/gpt-oss-120b`
- `openai/gpt-oss-20b`
- `openai/gpt-oss-safeguard-20b`
- `qwen/qwen3.8-27b`
- `whisper-large-v3`
- `whisper-large-v3-turbo`

Only the three requested planning candidates were exercised as chat planners;
a listing does not certify chat/tool capabilities for audio or safeguard models.

For every initial official-question call, the real selector supplied, in order:

```
linear_trend
mean_shift_change_point
one_way_anova
bayesian_structural_causal_impact
kruskal_wallis_test
descriptive_statistics
constant_column_diagnostic
automatic_forecast
automatic_forecast_with_intervals
automatic_strategy_forecast
```

The selector was not modified. The diagnostic emits the exact tool-name list
before **each** provider request, including reduced continuation catalogs.

## Controlled live observations

The exact official question remained:

> Why did our sales decline, and did the September campaign actually help?

| Experiment | Model / setting | Observed result |
| --- | --- | --- |
| Starting request | Qwen, developer, parallel false | Provider HTTP 400, `BadRequestError`, unexpected message role |
| Starting request | Qwen, developer, parallel true | Same HTTP 400 before model tool output |
| Role-only ablation | Qwen, system, parallel false | Accepted; one trend call, then binder rejected missing aggregation |
| Role-only ablation | Qwen, system, parallel true | Accepted; four calls, three temporal calls omitted aggregation |
| Starting one-shot request | GPT-OSS 20B, parallel false | One `linear_trend`, VERIFIED, one visualization |
| Starting one-shot request | GPT-OSS 120B, parallel false | One `linear_trend`, VERIFIED, one visualization |
| Retest after quota refill | GPT-OSS 20B, parallel true | Accepted parameter; still one trend call |
| Retest after quota refill | GPT-OSS 120B, parallel true | Accepted parameter; still one trend call |
| Minimal two-empty-tool probe | GPT-OSS 20B, parallel true | One of the two explicitly requested probes |
| Minimal two-empty-tool probe | GPT-OSS 120B, parallel true | One of the two explicitly requested probes |
| Same minimal probe | Qwen, parallel true | Both requested probes in one response |
| Prompt + temporal bounds, aggregation still optional | Qwen | Still omitted aggregation; stronger prose alone was insufficient |
| Required aggregation from metadata, one-shot repetitions | Qwen, parallel true | One four-call success, one two-call partial success, one transport failure |
| Bounded planner, provider sampling default | Qwen | Two complete four-call runs (one chose Kruskal), one no-call response rejected as incomplete |
| Temperature-zero ablation | Qwen | One complete four-call result; two transport failures with retries disabled |
| Final API configuration, three consecutive runs | Qwen, system, parallel true, temperature zero | **3/3 HTTP 201, four expected tools VERIFIED, four charts, matching hashes** |

Final API runs used actual CSV upload and `POST /analyses` through FastAPI's
TestClient, with real provider requests, binding, executor, verifier and API
serialization. No model response or analytical result was mocked in those runs.
All three selected `linear_trend`, `mean_shift_change_point`, `one_way_anova`,
and `bayesian_structural_causal_impact`, each in one native four-call response.

Two initial GPT parallel-true attempts hit HTTP 429 (8,000 tokens/minute);
they were not counted as evidence of unsupported tool calling. Later valid
responses and the minimal probe supplied the comparison. Additional transport
failures were recorded explicitly, not converted into successful analyses.
A final GPT-OSS bounded test returned no calls and was rejected as incomplete.

## Causes and corrections

1. **Qwen 502:** VERA safely wrapped Groq's underlying HTTP 400. The isolated
   change from `developer` to `system` removed the request-template error.
   Groq now explicitly uses `system`; Meta retains `developer` by default.
2. **GPT one-tool behavior:** the model/provider responses themselves contain
   one call. VERA's parser does not truncate them and the required tools were
   available. Switching the parallel flag alone did not solve this in live
   tests, even with two trivial tools. The original one-shot planner had no
   way to request the missing objectives, and computational success was
   mistaken for sufficient question coverage. These observations do not
   establish the provider's undocumented internal reason for this behavior.
3. **Temporal preparation contract:** aggregation was optional in the schema
   even when safe cardinalities already proved timestamps repeat. It is now
   required in the exposed schema when all candidate time columns repeat,
   and checked against the selected time column at parsing time. With unique
   timestamps it remains optional. The LLM chooses sum or mean; no value is
   injected on its behalf. The binder retains its independent validation.
4. **Missing year:** translating all dates by one year left the old model-visible
   profile identical. This proves the year could not be inferred from that
   profile and a month name. Typed, ordered `min_timestamp`/`max_timestamp`
   metadata now distinguish these cases. Empty time columns have no invented
   bounds. A fixed example intervention date was removed from tool guidance.
   A month boundary remains an interpretation, not proof of an observed event
   date; the prompt requires the reason to explain such an interpretation and
   asks for clarification when timing is ambiguous.
5. **Native multi-call variability:** Qwen can emit four calls, but one response
   is not reliably complete at provider-default sampling. The default Groq
   configuration is now Qwen, parallel true, temperature zero, with a bounded
   completion policy. No endpoint or statistical implementation changed.

No JSON Schema keywords were removed. Groq accepted the actual nested schemas,
including `additionalProperties`, `oneOf`, `required`, `maxItems`, numerical
bounds and arrays. Acceptance does not imply the provider enforces every rule;
VERA runtime validation remains authoritative.

## Bounded planning contract and provenance

The planner reuses the selector's existing general investigation-intent policy
and safe column roles to identify supported analytical scopes. It first permits
native multi-call selection. Only uncovered scopes can be offered in another
request. The LLM selects a justified tool or requests clarification; local code
never manufactures a tool call, a date or an aggregation.

Limits: at most **four planning responses** and **four analytical calls total**.
A complete first response stops immediately. A targeted scope does not cause
four requests. Identical analytical calls, excess calls and uncompleted scopes
are rejected. A no-call clarification stops planning; it is not retried until
the model agrees. No partial plan is quietly treated as complete by the default
bounded Groq configuration.

Continuations send only the original question, safe profile, already-selected
tool names and remaining scope names. They never send observations or execution
results. Tools execute after planning has completed. IDs are scoped as
`planning-<turn>-<provider-id>` to avoid cross-response ID collisions;
`source_call_ids` remain empty for these independent planned calls. Execution
replanning retains its separate lineage. Token usage is summed and
`MetaPlanningResult.planning_turns` records the response count.

Meta's default model, endpoint, instruction role and single-response behavior
remain unchanged. The same bounded mechanism is opt-in through
`MetaModelPlanner(max_planning_turns=4)`; no Meta key or live Meta request was
needed. Groq remains temporary, not the final platform target.

## Reproduction and verification

The diagnostic is `scripts/diagnose_groq_planner.py`. It lists account models,
logs safe request/response metadata, distinguishes provider errors from VERA
parser errors, and exits nonzero on failed cases. It never prints request
messages, authorization headers, full exception representations or raw results.

For the final API matrix, with the local environment already loaded:

```bash
PYTHONPATH=. python scripts/diagnose_groq_planner.py \
  --models qwen/qwen3.8-27b --parallel true --repeats 3 \
  --via-api --transport-retries 2
```

The diagnostic timeout is 45 seconds. Its default transport retries are zero for
observing provider failures; final API runs explicitly used the SDK's existing
production retry count of two. Transport retries repeat the same request and
are distinct from planning responses: at most 12 transport attempts for four
planning requests with two retries. There is no provider/model fallback.
`--planning-turns 1` isolates single-response behavior.

## Generalization checks with real provider calls

| Question / setting | Observed plan | Outcome |
| --- | --- | --- |
| Official question, Qwen parallel false | Four sequential planning responses: trend → change → causal → ANOVA | Four VERIFIED results, four charts; each continuation catalog excluded covered scopes |
| How are sales evolving over time? | Trend + descriptive statistics, one response | Both VERIFIED, one chart; no four-tool expansion |
| Are sales significantly different across regions? | ANOVA only | VERIFIED, one chart |
| Did the September campaign improve sales? | Causal impact only | VERIFIED, one chart |
| Official question, all local dates shifted to 2027 | Four native calls, causal binding `2027-09-01` | Four VERIFIED results, four charts |

The shifted frame was constructed in memory; the official CSV and its frontend
copy were not changed. This tests year resolution without a demo-specific date
mapping. Metadata bounds do not prove actual campaign timing or causal validity;
ambiguous questions can still require clarification. Intent coverage is a
heuristic over supported analytical families, not proof that a business question
has been fully understood.

## Validation and remaining limits

- Full Python suite: **457 passed**, one pre-existing FastAPI deprecated-constant
  warning. This retains the previous 416 cases and adds 41 live-planner and
  diagnostic regressions. Provider responses are mocked only in unit tests;
  the live experiments described above use the actual account.
- Frontend lint, typecheck, **12 tests**, integrity check and production build:
  all successful. No frontend implementation change required.
- Offline official demo: four VERIFIED results and four visualizations. Full
  trace/result/hash, Decision Brief and visualization snapshots compare identical
  to the previous audit baseline.
- `git diff --check`: clean. Reasonable secret-pattern scan: zero candidates;
  `.env` remains ignored. No requirement or lockfile changes.
- The native model may choose Kruskal instead of ANOVA on other runs; the planner
  does not force one statistical method merely to match a demo list.
- Network errors, rate limits, provider/model changes and legitimate clarification
  remain possible. Temperature zero reduces variation; it is not a guarantee of
  deterministic provider responses. Failed/incomplete planning returns a safe
  error rather than manufactured evidence. Bounded completion can add latency.
- Meta remains the final target. No Meta endpoint/model change or live Meta
  certification is claimed. No video was recorded.

**READY for the final local live demonstration**, based on the final three-run
API matrix and the generalization checks. This is not a public-production or
causal-validity sign-off.
