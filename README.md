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
LLM planner  (GroqModelPlanner / MetaModelPlanner)
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

**Core principle:** The LLM selects analyses. It does not compute statistics.

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

## Quick start — backend

### Prerequisites
- Python 3.12
- A Python virtual environment with the packages from `requirements.txt` installed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

```bash
# With the virtual environment activated and GROQ_API_KEY exported:
cd ~/VERA
PYTHONPATH=. uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### Environment variables

`.env.example` lists backend settings. Export these variables in the backend
process environment before starting uvicorn; this command does not load `.env`
automatically. Docker's `--env-file .env` loads them explicitly.

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Groq API key for the LLM planner |
| `VERA_APP_NAME` | `VERA API` | Displayed application name |
| `VERA_APP_VERSION` | `0.1.0` | API version string |
| `VERA_ENV` | `development` | Runtime environment |
| `VERA_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed CORS origins |
| `VERA_MAX_UPLOAD_BYTES` | `26214400` | Maximum CSV upload size (bytes) |

The default API planner remains Groq. `MetaModelPlanner` is available for explicit
integration and uses `MODEL_API_KEY`; its configured endpoint and model still
require a live provider check. Switching the API provider is not automatic.

Groq currently uses `qwen/qwen3.8-27b`, the `system` instruction role,
`parallel_tool_calls=True` and temperature zero. Planning uses at most four
responses and four analytical calls: it requests only uncovered objectives after
a partial response, and rejects incomplete or duplicate plans. Simple questions
do not automatically expand to four tools. No observations or execution results
are sent back to the planner. Temporal profiles include timestamp bounds;
repeated timestamps require an explicit model-selected aggregation. See the
[live diagnosis and measured results](docs/GROQ_LIVE_DIAGNOSTIC.md).

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
cd ~/VERA
PYTHONPATH=. python -m pytest -q -p no:asyncio
```

The original baseline was **380 passed**; final audit and live-planner regressions are included
in the current suite (see `docs/FINAL_TECHNICAL_AUDIT.md`).

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

## Official demo scenario (no LLM or network required)

```bash
cd ~/VERA
PYTHONPATH=. python scripts/run_demo_scenario.py
```

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
