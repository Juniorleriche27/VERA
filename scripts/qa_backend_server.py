"""QA backend server simulating all evaluation scenarios for graphical browser inspection."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

# Ensure root is in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.pipeline import VerifiedAnalyticsPipeline
from agent.replanning import SelfCorrectionPlanner
from agent.tool_call_trace import ToolCallTrace, TraceStatus, VerificationStatus
from backend.app import _brief_payload, _safe_dataset_profile, create_app
from scripts.run_demo_scenario import DemoPlanner, INTERVENTION_DATE
from verification.verifier import DeterministicVerifier


class AnalysisRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class ControlledQAVerifier:
    def __init__(self):
        self.real_verifier = DeterministicVerifier()

    def verify(self, trace: ToolCallTrace):
        if trace.call_id == "call-anova-replan":
            return trace.verify(VerificationStatus.NEEDS_MORE_ANALYSIS, next_decision="run_robustness_check"), None
        if trace.call_id == "call-causal-unresolved":
            return trace.verify(VerificationStatus.NEEDS_MORE_ANALYSIS, next_decision="unresolved"), None
        return self.real_verifier.verify(trace)


class ScenarioPlanner:
    def plan(self, question: str, *, dataset_context: Any, tool_catalog: Any):
        q = question.lower()
        if "replan" in q or "kruskal" in q:
            trace = ToolCallTrace.planned(
                call_id="call-anova-replan",
                tool="one_way_anova",
                objective="Compare regional sales with ANOVA",
                reason="Initial group comparison test under normality assumption",
                parameters={},
                bindings={"grouping_column": "region", "value_column": "sales"},
            )
            return SimpleNamespace(traces=(trace,))

        if "unresolved" in q:
            trace = ToolCallTrace.planned(
                call_id="call-causal-unresolved",
                tool="bayesian_structural_causal_impact",
                objective="Assess September campaign impact",
                reason="Evaluate causal lift of September marketing",
                parameters={"credibility": 0.95},
                bindings={
                    "time_column": "date",
                    "value_column": "sales",
                    "aggregation": "sum",
                    "intervention_date": INTERVENTION_DATE,
                    "control_columns": ["control_sales"],
                },
            )
            return SimpleNamespace(traces=(trace,))

        if "fail" in q or "error" in q:
            trace = ToolCallTrace.planned(
                call_id="call-failed-exec",
                tool="one_way_anova",
                objective="Analyze invalid column",
                reason="Testing graceful UI error handling without crash",
                parameters={},
                bindings={"grouping_column": "non_existent_column", "value_column": "sales"},
            )
            return SimpleNamespace(traces=(trace,))

        if "one-tool" in q or "single" in q or "trend only" in q:
            trace = ToolCallTrace.planned(
                call_id="call-trend-single",
                tool="linear_trend",
                objective="Quantify single sales trend",
                reason="Targeted univariant trend analysis",
                parameters={},
                bindings={"time_column": "date", "value_column": "sales", "aggregation": "sum"},
            )
            return SimpleNamespace(traces=(trace,))

        if "no-chart" in q or "zero-vis" in q:
            trace = ToolCallTrace.planned(
                call_id="call-mann-whitney-nochart",
                tool="mann_whitney_u_test",
                objective="Compare sales distribution across two groups",
                reason="Rank comparison without chart specification",
                parameters={"alternative": "two-sided"},
                bindings={"grouping_column": "region", "value_column": "sales", "group_a": "North", "group_b": "South"},
            )
            return SimpleNamespace(traces=(trace,))

        # Default: official 4-tool demo plan
        return DemoPlanner().plan(question, dataset_context=dataset_context, tool_catalog=tool_catalog)


def build_qa_app() -> FastAPI:
    app = create_app(planner_factory=ScenarioPlanner)
    verifier = ControlledQAVerifier()

    # Filter out original /analyses route to inject custom verifier logic
    app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/analyses"]

    @app.post("/analyses", status_code=status.HTTP_201_CREATED)
    def create_analysis(request: AnalysisRequest) -> dict[str, Any]:
        dataset = app.state.datasets.get(request.dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")

        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="Question must not be blank.")

        planner = ScenarioPlanner()
        pipeline = VerifiedAnalyticsPipeline(planner, verifier=verifier)
        result = pipeline.run(
            question,
            dataset=dataset["frame"],
            dataset_context=dataset["profile"],
        )

        analysis_id = uuid4().hex
        payload: dict[str, Any] = {
            "analysis_id": analysis_id,
            "dataset_id": request.dataset_id,
            "question": question,
            "succeeded": result.succeeded,
            "exhausted": result.exhausted,
            "steps": [step.as_dict() for step in result.steps],
            "unresolved_call_ids": list(result.unresolved_call_ids),
            "final_status": (
                result.final_trace.verification_status.value
                if result.final_trace is not None
                else None
            ),
            "decision_brief": None,
            "visualizations": [],
        }

        if result.succeeded:
            brief = result.build_decision_brief()
            visuals = result.build_visualizations()
            payload["decision_brief"] = _brief_payload(brief)
            payload["visualizations"] = [spec.as_dict() for spec in visuals]

        encoded = jsonable_encoder(payload)
        app.state.analyses[analysis_id] = encoded
        return encoded

    return app


if __name__ == "__main__":
    app = build_qa_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)
