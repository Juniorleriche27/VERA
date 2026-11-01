# Meta Model API integration

Official documentation checked on 2026-11-01:

- [Meta quickstart](https://dev.meta.ai/docs/quickstart): credential `MODEL_API_KEY`,
  base URL `https://api.meta.ai/v1`, model ID `muse-spark-1.3`.
- [Chat Completions](https://dev.meta.ai/docs/protocols/chat-completions): supported
  OpenAI-compatible protocol; `developer` is preferred, `system` is also accepted.
- [Tool calling](https://dev.meta.ai/docs/tool-calling): nested function definitions,
  `choices[0].message.tool_calls`, parallel calls enabled by default. Setting
  `parallel_tool_calls=False` restricts one call per response. `tool_choice` must
  be `auto`; forced/named choices are unsupported. `max_tool_calls` limits built-in
  tools, not VERA's client-side functions. Server `strict` validates supported
  schema structure; VERA independently parses and validates returned arguments.

VERA retains Chat Completions through the existing OpenAI SDK, directed to Meta.
Responses is recommended by Meta for loops that carry reasoning/results forward;
VERA instead makes bounded independent planning requests and never returns
analytical results to the model. No alternate inference route or fallback is used.

## Planning guarantees

`MetaModelPlanner` defaults to four planning turns, four total analytical calls,
`developer`, `parallel_tool_calls=True`, and no explicit sampling parameter.
Native multi-call responses are retained. A partial plan receives only a reduced
catalog for uncovered objectives. Duplicate semantic calls, budget overruns and
incomplete coverage fail explicitly; missing analyses are never fabricated.
Provider IDs are scoped by planning turn so restarted IDs cannot collide.

Temporal bounds are safe profile metadata. A month name alone cannot establish
an intervention year. Repeated timestamps require an explicit aggregation in both
the generated tool schema and parser; no hidden sum/mean default is inserted.
No raw rows, numerical arrays, group examples or executed results reach Meta.

## Safe diagnostics and validation

```bash
# With requirements installed and MODEL_API_KEY exported:
PYTHONPATH=. python scripts/diagnose_meta_planner.py
```

The diagnostic first makes a minimal text request, then exercises the actual
Meta planner through CSV upload and the full API pipeline. Output reports stages,
counts, verification statuses and hash consistency, with a final SUCCESS/FAILED.
Missing credentials exit 2; failures exit 1; success exits 0. Exception bodies,
messages, headers, request payloads and credentials are never printed.

Injected-client tests exercise metadata privacy, filtered catalogs, tool parsing,
multiple calls, bounded continuations, local execution, verified hashes, Decision
Brief and charts. No-key tests exercise lazy startup, upload and safe HTTP/SSE
configuration errors. The fixed-plan offline demo separately checks reproducibility.
None of these offline checks claims a live Meta call. No live Meta validation was
performed during consolidation because a Meta credential was unavailable.
