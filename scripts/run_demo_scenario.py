"""Official reproducible VERA hackathon demonstration (SYNTHETIC data).

Run: PYTHONPATH=. python scripts/run_demo_scenario.py
No LLM, network connection, or secret is required for this deterministic run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from agent.analytics_executor import AnalyticsToolExecutor
from agent.pipeline import VerifiedAnalyticsPipeline
from agent.tool_call_trace import ToolCallTrace
from agent.tool_catalog_selector import ToolCatalogSelector
from backend.app import _safe_dataset_profile
from backend.data_preparation import prepare_dataset

QUESTION = "Why did our sales decline, and did the September campaign actually help?"
INTERVENTION_DATE = "2026-09-01"
CSV_PATH = Path(__file__).resolve().parents[1] / "demo" / "vera_sales_campaign_demo.csv"


class DemoPlanner:
    """Offline, fixed four-call plan; not a replacement for the runtime LLM."""

    SPECS = (
        (
            "demo-trend", "linear_trend", "Quantify the sales trend",
            {}, {"time_column": "date", "value_column": "sales", "aggregation": "sum"},
        ),
        (
            "demo-change", "mean_shift_change_point", "Find a change in sales level",
            {"min_segment_size": 10},
            {"time_column": "date", "value_column": "sales", "aggregation": "sum"},
        ),
        (
            "demo-regions", "one_way_anova", "Compare sales by region",
            {}, {"grouping_column": "region", "value_column": "sales"},
        ),
        (
            "demo-campaign", "bayesian_structural_causal_impact",
            "Estimate the campaign lift conditional on model assumptions",
            {}, {
                "time_column": "date", "value_column": "sales", "aggregation": "sum",
                "intervention_date": INTERVENTION_DATE, "control_columns": ["control_sales"],
            },
        ),
    )

    def plan(self, question, *, dataset_context, tool_catalog):
        from types import SimpleNamespace
        names = {entry["name"] for entry in tool_catalog}
        if not {entry[1] for entry in self.SPECS} <= names:
            raise ValueError("The selected catalog omits an official demo analysis tool.")
        traces = tuple(
            ToolCallTrace.planned(
                call_id=call_id, tool=tool, objective=objective,
                reason="Official deterministic synthetic demonstration",
                parameters=parameters, bindings=bindings, source_call_ids=(),
            )
            for call_id, tool, objective, parameters, bindings in self.SPECS
        )
        return SimpleNamespace(traces=traces)


def load_demo_dataset():
    """Read the single versioned source of synthetic demo data."""
    return prepare_dataset(pd.read_csv(CSV_PATH)).frame


def run_demo():
    """Execute actual VERA analytics and return results without printing raw data."""
    frame = load_demo_dataset()
    profile = _safe_dataset_profile(frame)
    selected = ToolCatalogSelector().select(
        QUESTION, dataset_context=profile,
        tool_catalog=AnalyticsToolExecutor().available_tools(),
    )
    result = VerifiedAnalyticsPipeline(DemoPlanner()).run(
        QUESTION, dataset=frame, dataset_context=profile,
    )
    if not result.succeeded:
        raise RuntimeError(
            "Official demo did not verify fully: "
            f"unresolved={result.unresolved_call_ids}, exhausted={result.exhausted}"
        )
    brief = result.build_decision_brief()
    visuals = result.build_visualizations()
    return frame, profile, selected, result, brief, visuals


def main() -> None:
    frame, profile, selected, result, brief, visuals = run_demo()
    print("VERA DEMO — SYNTHETIC DEMONSTRATION DATASET")
    print(f"Question: {QUESTION}")
    print(f"Dataset: {len(frame)} rows / {len(frame.columns)} columns")
    print("Investigation:")
    for step in result.steps:
        print(f"  {step.index}. {step.trace.tool} — {step.trace.verification_status.value.upper()}")
    print("Decision:")
    print(f"  {brief.headline}")
    print("Next action:")
    print(f"  {brief.next_action}")
    print(f"Visualizations: {len(visuals)}")
    print("Evidence: call_id | tool | trace hash prefix")
    for evidence in brief.evidence.items:
        print(f"  {evidence.call_id} | {evidence.tool} | {evidence.trace_hash[:12]}")
    print("This is synthetic data; modeled campaign effects are conditional, not causal proof.")


if __name__ == "__main__":
    main()
