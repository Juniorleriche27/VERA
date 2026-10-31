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
LLM planner  (GroqModelPlanner / MetaModelPlanner)
      ↓
ToolCatalogSelector  (local, deterministic pre-filter)
      ↓
DatasetBinder  (semantic column → local raw values)
      ↓
Deterministic analytics engine  (95 core + 7 hackathon extensions)
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
(95 native analytical capabilities) as its computational foundation.

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
# From an environment that already has the data-analysis-engine installed:
cd ~/VERA
PYTHONPATH=. uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### Environment variables

Copy `.env.example` to `.env` and fill in the required values:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Groq API key for the LLM planner |
| `VERA_APP_NAME` | `VERA API` | Displayed application name |
| `VERA_APP_VERSION` | `0.1.0` | API version string |
| `VERA_ENV` | `development` | Runtime environment |
| `VERA_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed CORS origins |
| `VERA_MAX_UPLOAD_BYTES` | `26214400` | Maximum CSV upload size (bytes) |

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
to the backend URL (default: `http://localhost:8000`).

---

## Running tests

### Backend (Python)

```bash
cd ~/VERA
PYTHONPATH=. python -m pytest -q
```

Expected: **380 passed**.

### Frontend

```bash
cd frontend
npm run lint
npm run typecheck
npm run test
npm run test:integrity
npm run build
```

Expected: **7 frontend tests** passing, no lint errors, clean build.

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
tests/                  380 Python test cases
verification/           DeterministicVerifier (rule-based numerical checks)
visualization/          VerifiedVisualizationBuilder (chart specs from evidence)
```

---

## Key guarantees

- Raw dataset values are **never transmitted to the LLM**.
- Every analytical result carries a **SHA-256 trace hash** for provenance.
- The verifier checks **numerical consistency** — it does not claim causal proof.
- The demo is **fully reproducible** without a live LLM or network connection.

---

*The pre-existing engine and the hackathon extensions remain explicitly separated for auditability.*
