# VERA — Verifiable Analytics Agent

VERA is the submission for the **Meta Global AI Developer Hackathon**.

VERA answers business questions with **verified, traceable analytical evidence**. The
planner (LLM) chooses which statistical methods to run; it never computes the numbers.
All calculations are deterministic and machine-verifiable.

---

## Architecture

```
Business question
      ↓
Safe dataset metadata (no raw rows)
      ↓
ToolCatalogSelector  (local, deterministic pre-filter)
      ↓
Meta model analytical planning  (MetaModelPlanner)
      ↓
DatasetBinder  (semantic column → local raw values)
      ↓
Deterministic analytics engine  (102 unique tools; VERA extensions take precedence)
      ↓
ToolCallTrace  (immutable audit record + SHA-256 hash)
      ↓
DeterministicVerifier  (rule-based numerical verification)
      ↓
Optional deterministic replanning  (SelfCorrectionPlanner)
      ↓
Evidence Package  (verified, hash-linked)
      ↓
Decision Brief  (findings · caveats · next action)
      ↓
Verified visualisations
      ↓
FastAPI  →  Next.js frontend
```

**VERA uses Meta models for AI analytical planning.**

The Meta model selects analytical tools and their semantic bindings. It never computes the statistical results itself. All numerical analysis is executed deterministically by VERA and independently verified before being presented as evidence.

---

## Pre-existing infrastructure

VERA uses the [koryxa data-analysis-engine](https://github.com/Juniorleriche27/data-analysis-engine)
as its computational foundation. The original foundation had 95 native
capabilities; the pinned engine now exposes 97 entries. VERA registers seven
extensions, overriding the core ANOVA and Mann-Whitney implementations, for
**102 unique tools** (97 + 7 − 2).

## New hackathon work

All VERA-facing agent workflows, Meta Model API integration, verification, replanning,
visualization, UI/API work, and the following analytical extensions are new:

| # | Extension |
|---|-----------|
| 96 | `one_way_anova` |
| 97 | `mann_whitney_u_test` |
| 98 | `kruskal_wallis_test` |
| 99 | `multiple_testing_correction` |
| 100 | `extreme_value_tail_risk` |
| 101 | `bayesian_structural_causal_impact` |
| 102 | `directed_transfer_entropy` |

---

## Hackathon setup

Use Python 3.12 and Node.js 20.9 or newer. In the backend terminal:

```bash
git clone https://github.com/Juniorleriche27/VERA.git
cd VERA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MODEL_API_KEY="YOUR_META_MODEL_API_KEY"
PYTHONPATH=. uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

In a second terminal, from the cloned repository:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3000`, upload `demo/vera_sales_campaign_demo.csv`, and
ask: **Why did our sales decline, and did the September campaign actually help?**
The live application calls Meta for planning; analysis and verification run locally.
Use one API worker because dataset and analysis storage are process-local.

### Official Meta provider

The default is `MetaModelPlanner`, using Meta Model API at
`https://api.meta.ai/v1`, model **`muse-spark-1.3`**, and the backend-only
**`MODEL_API_KEY`** credential. The OpenAI Python SDK is an API-compatible
transport to Meta, not a separate inference provider.

The request uses Chat Completions, a `developer` instruction message, nested
function `tools`, `tool_choice="auto"` and `parallel_tool_calls=True`. Sampling
parameters are omitted by default. Official Meta documentation confirms this
[protocol and model](https://dev.meta.ai/docs/protocols/chat-completions) and
[tool-calling contract](https://dev.meta.ai/docs/tool-calling).

Planning is bounded to four responses and four analytical calls. Continuations
request only uncovered objectives; duplicate or incomplete plans are rejected.
Simple questions do not automatically expand to four tools. No observations or
execution results are sent back to Meta. Temporal profiles include timestamp
bounds; repeated timestamps require an explicit model-selected aggregation.
There is no automatic provider fallback.

`.env.example` contains the single required credential name. Export it before
starting the backend; uvicorn does not automatically load `.env`. Never put this
key in frontend configuration. Without it, import, health and CSV upload work;
analysis returns a safe configuration error (HTTP 503 or SSE `configuration_error`).

Optional application settings retain their defaults: `VERA_APP_NAME=VERA API`,
`VERA_APP_VERSION=0.1.0`, `VERA_ENV=development`,
`VERA_CORS_ORIGINS=http://localhost:3000`, `VERA_MAX_UPLOAD_BYTES=26214400`.
No extra model settings are required.

### Optional live Meta diagnostic

With the virtual environment active and `MODEL_API_KEY` exported:

```bash
PYTHONPATH=. python scripts/diagnose_meta_planner.py
```

This sends a minimal request, then tests real function calling through CSV upload,
planning, local execution and verification. It reports `SUCCESS` / `FAILED` with
nonzero exit on failure, without displaying credentials or provider payloads.
A real Meta credential is required; injected-client tests and the offline demo
do not certify a live Meta call. See [Meta integration notes](docs/META_MODEL_API.md).

---

## Quick start — frontend

```bash
cd frontend
npm install
npm run dev
```

The UI will be available at `http://localhost:3000`.

### Frontend environment

Copy `frontend/.env.example` to `frontend/.env.local` and set `NEXT_PUBLIC_VERA_API_URL`
to the backend URL (default: `http://127.0.0.1:8000`).

---

## Running tests

### Backend (Python)

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=. python -m pytest -q -p no:asyncio
```

The suite includes provider-contract, bounded-planning, no-key startup, injected
Meta-client integration and analytical regression tests. See
[technical guarantees](docs/FINAL_TECHNICAL_AUDIT.md).

### Frontend

```bash
cd frontend
npm run lint
npm run typecheck
npm run test
npm run test:integrity
npm run build
```

Expected: **27 frontend tests** passing, no lint errors, clean build.

---

## Offline deterministic demo (no LLM or network required)

```bash
cd ~/VERA
PYTHONPATH=. python scripts/run_demo_scenario.py
```

This fixed-plan demo checks analytical reproducibility. It does not call Meta or
replace validation of the live application with a real Meta credential.

**Dataset:** `demo/vera_sales_campaign_demo.csv` (synthetic data)

**Question:** *Why did our sales decline, and did the September campaign actually help?*

**Expected output:**
- 4 analyses: `linear_trend`, `mean_shift_change_point`, `one_way_anova`, `bayesian_structural_causal_impact`
- All 4 results: `VERIFIED`
- 1 Decision Brief
- 4 visualisations
- Reproducible trace hashes

---

## Docker

```bash
docker build -t vera-api .
docker run -p 8000:8000 --env-file .env vera-api
```

---

## Repository structure

```
agent/                  VERA agent runtime (planner, binder, pipeline, verifier bridge)
analytics_extensions/   7 hackathon-specific analytical extensions
backend/                FastAPI application, data preparation, configuration
demo/                   Official synthetic demonstration dataset
frontend/               Next.js 16 + TypeScript + Recharts UI
scripts/                Reproducible demo runner
tests/                  Python unit, integration and adversarial regressions
verification/           DeterministicVerifier (rule-based numerical checks)
visualization/          VerifiedVisualizationBuilder (chart specs from evidence)
```

---

## Key guarantees

- Raw dataset values are **never transmitted to the LLM**.
- Every analytical result carries a **SHA-256 trace hash** for provenance.
- The verifier checks **numerical consistency** — it does not claim causal proof.
- The demo is **fully reproducible** without a live LLM or network connection.

The API constructs metadata explicitly; direct pipeline/planner contexts also
accept only typed schema/profile metadata. Column names and the business question
are supplied text, so do not put private cell contents in either. Verification
checks numerical consistency and selected structural rules; it does not rerun
every statistic or validate every statistical assumption.

The API is intended for a trusted local demo: datasets and analyses remain in
process memory without authentication, retention limits or tenant isolation.
Restarting loses them, and multiple workers do not share them. The upload reader
is bounded, but multipart spooling, request body limits, rate limits and total
memory limits still require deployment controls before public use.

---

*The pre-existing engine and the hackathon extensions remain explicitly separated for auditability.*

## Live orchestration

The frontend observes real backend events through `POST /analyses/stream` (SSE
consumed using fetch). The existing synchronous `POST /analyses` remains supported.
A completed stream supplies an analysis ID and safe counts; the existing
`GET /analyses/{analysis_id}` supplies the final evidence, brief and chart specs.
No dataset rows, bound arrays, provider prompts or provider errors appear in the
SSE stream. The UI retains the received event history even for very fast tools.
There are no simulated execution delays or progress percentages.

See [event contract, validation and browser results](docs/LIVE_ORCHESTRATION.md).
Run the local API with one worker: analysis storage remains process-local, so
multiple workers would need shared storage or sticky routing for the final GET.
Disconnect cancels work at the next event boundary; it cannot forcibly stop a
provider request already in flight. A dropped stream is reported explicitly and
is not automatically replayed as a duplicate investigation.
