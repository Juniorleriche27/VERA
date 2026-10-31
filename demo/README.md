# VERA official hackathon demonstration — synthetic data

**Question:** Why did our sales decline, and did the September campaign actually help?

**Important:** `vera_sales_campaign_demo.csv` is a **synthetic demonstration dataset**, not real customer or business data. It is deliberately designed to illustrate the hypotheses and limitations of VERA's deterministic analysis. Statistical significance and positive conditional effects in synthetic data do not establish that an actual marketing campaign worked.

## Data contract

- **Period:** 2026-08-01 through 2026-09-30, inclusive (61 consecutive days).
- **Grain:** date × region, with one observation per day per region; 183 rows, 3 regions: North, South, West.
- **Columns:** `date` (ISO date), `region` (category), `sales` (synthetic sales value), `control_sales` (synthetic untreated control value), `campaign_spend` (synthetic daily regional campaign spend).
- **Intervention date:** 2026-09-01. `campaign_spend=0` in August and `45` per region per day in September. The causal model receives `control_sales`, **not** `campaign_spend`, as its control variable.
- **Synthetic story:** strong early August sales; a level drop around August 16; persistent downward drift; regional baseline differences; a positive sales lift in September relative to an untreated control. The campaign does not eliminate every underlying challenge.

## Reproducible construction

The committed CSV is the single source of truth; tests always read that file. It was constructed without random sampling. For zero-based day index `i` in `0..60`, region index `r` in `0..2` for North/South/West, and region baseline `base` of `330/260/190`, respectively:

```text
wave = (i % 7 - 3) * 1.1 + (r - 1) * 0.4 * ((i % 5) - 2)
common = -1.45 * i - (48 if i >= 15 else 0) + wave
control_sales = 0.72 * base + 0.72 * common
sales = base + common + (76 if i >= 31 else 0)
campaign_spend = 0 if i < 31 else 45
```

`sales` and `control_sales` were rounded to three decimal places for the versioned CSV. The `control_sales` formula deliberately excludes the campaign lift. The control is simulated and cannot independently demonstrate the validity of a real-world control group.

## Official four-step workflow

1. `linear_trend` — aggregate daily `sales` with `sum`.
2. `mean_shift_change_point` — use the same daily series, `min_segment_size=10` to guard against very short comparison segments.
3. `one_way_anova` — compare sales across `region`.
4. `bayesian_structural_causal_impact` — aggregate `sales` and `control_sales` by day with `sum`; intervention index resolved **exactly** from 2026-09-01.

The offline runner uses `DemoPlanner` with fixed IDs `demo-trend`, `demo-change`, `demo-regions` and `demo-campaign`. It runs the real Binder, Executor, Verifier, deterministic Decision Brief and verified Visualization Builder. It does **not** exercise the external LLM; the runtime Groq planner and providers are unchanged. The E2E tests additionally exercise `POST /datasets`, `POST /analyses` and `GET /analyses/{analysis_id}` with this exact CSV.

```bash
PYTHONPATH=. python scripts/run_demo_scenario.py
PYTHONPATH=. python -m pytest -q -p no:asyncio tests/test_demo_scenario_e2e.py
```

The resulting campaign estimate is **conditional on model specification and counterfactual assumptions**; the Verifier checks numerical consistency, not independent causal validity. No synthetic finding should be passed off as a real business claim.
