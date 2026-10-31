"""Evidence-aware replanning regressions, including real analytical execution."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.analytics_executor import AnalyticsToolExecutor
from agent.data_binding import DatasetBinder
from agent.pipeline import PipelineResult, PipelineStep, VerifiedAnalyticsPipeline
from agent.replanning import SelfCorrectionPlanner
from agent.tool_call_trace import ToolCallTrace, TraceStatus, VerificationStatus
from verification.verifier import DeterministicVerifier
from tests.test_binder_integration import dataset
from tests.test_pipeline import FakePlanner


def planned(tool="one_way_anova", call_id="call-a", bindings=None, sources=()):
    if bindings is None:
        bindings = {"grouping_column": "region", "value_column": "sales"}
    return ToolCallTrace.planned(
        call_id=call_id, tool=tool, objective="Investigate sales",
        reason="Deterministic evidence", parameters={},
        bindings=bindings, source_call_ids=sources,
    )


class ControlledVerifier:
    def __init__(self, statuses=None):
        self.statuses = statuses or {}
        self.real = DeterministicVerifier()

    def verify(self, trace):
        state, decision = self.statuses.get(trace.call_id, (VerificationStatus.VERIFIED, "interpret_result"))
        verified, report = self.real.verify(trace)
        if state is verified.verification_status and decision == verified.next_decision:
            return verified, report
        return trace.verify(state, next_decision=decision), report


def run(traces, *, statuses=None, max_steps=5, analytics=None, binder=None, replanner=None):
    return VerifiedAnalyticsPipeline(
        FakePlanner(traces), binder=binder or DatasetBinder(),
        analytics=analytics or AnalyticsToolExecutor(),
        verifier=ControlledVerifier(statuses),
        replanner=replanner or SelfCorrectionPlanner(), max_steps=max_steps,
    ).run("Why did sales decline?", dataset=dataset(), dataset_context={"row_count": 48})


def causal(call_id="causal"):
    return planned("bayesian_structural_causal_impact", call_id, {
        "time_column": "date", "value_column": "sales", "aggregation": "sum",
        "intervention_date": "2026-08-13", "control_columns": ["control"],
    })


def temporal(call_id="trend"):
    return planned("linear_trend", call_id, {
        "time_column": "date", "value_column": "sales", "aggregation": "sum",
    })


def test_verified_needs_no_replanning():
    result = run([temporal()])
    assert result.succeeded and result.unresolved_call_ids == ()
    assert result.steps[0].replanning is None


def test_anova_to_kruskal_real_executor_and_provenance():
    result = run([planned()], statuses={
        "call-a": (VerificationStatus.NEEDS_MORE_ANALYSIS, "run_robustness_check"),
    })
    assert len(result.steps) == 2
    first, second = result.steps
    assert first.trace.tool == "one_way_anova"
    assert second.trace.tool == "kruskal_wallis_test"
    assert second.trace.status is TraceStatus.COMPLETED
    assert second.trace.verification_status is VerificationStatus.VERIFIED
    assert second.trace.call_id != first.trace.call_id
    assert first.trace.call_id in second.trace.source_call_ids
    assert first.trace.trace_hash != second.trace.trace_hash
    assert first.replanning.status == "scheduled"
    assert first.replanning.source_trace_hash == first.trace.trace_hash
    assert first.replanning.next_call_id == second.trace.call_id
    assert result.unresolved_call_ids == ()
    assert result.succeeded
    assert result.exhausted is False
    assert result.as_dict()["steps"][0]["replanning"]["status"] == "scheduled"
    assert result.as_dict()["unresolved_call_ids"] == []


def test_no_causal_fallback_does_not_raise_or_exhaust():
    result = run([causal()], statuses={
        "causal": (VerificationStatus.NEEDS_MORE_ANALYSIS, "run_robustness_check"),
    })
    assert len(result.steps) == 1
    assert result.steps[0].replanning.status == "no_fallback"
    assert result.unresolved_call_ids == ("causal",)
    assert result.succeeded is False
    assert result.exhausted is False
    assert result.as_dict()["unresolved_call_ids"] == ["causal"]
    with pytest.raises(Exception):
        result.build_decision_brief()


def test_causal_rejected_and_following_independent_verified():
    result = run([causal(), temporal()], statuses={
        "causal": (VerificationStatus.REJECTED, "rerun_tool_with_valid_output"),
    })
    assert len(result.steps) == 2
    assert result.steps[0].trace.verification_status is VerificationStatus.REJECTED
    assert result.steps[0].replanning.status == "unsupported_decision"
    assert result.steps[1].trace.verification_status is VerificationStatus.VERIFIED
    assert result.unresolved_call_ids == ("causal",)
    assert not result.succeeded and not result.exhausted


@pytest.mark.parametrize("decision,expected", [
    ("collect_more_data_or_reduce_scope", "unsupported_decision"),
    ("rerun_tool_with_valid_output", "unsupported_decision"),
    ("run_data_quality_and_missingness_analysis", "unsupported_decision"),
    ("unexpected_new_decision", "unsupported_decision"),
    ("run_robustness_check", "no_fallback"),
])
def test_unsupported_decisions_do_not_crash_or_queue_false_tools(decision, expected):
    result = run([causal()], statuses={
        "causal": (VerificationStatus.NEEDS_MORE_ANALYSIS, decision),
    })
    assert len(result.steps) == 1
    assert result.steps[0].replanning.status == expected
    assert result.unresolved_call_ids == ("causal",)
    assert result.exhausted is False


def test_fallback_not_in_executor_catalog_is_not_executed():
    class RestrictedAnalytics:
        def __init__(self):
            self.real = AnalyticsToolExecutor()

        def available_tools(self):
            return [{"name": "one_way_anova"}]

        def execute(self, tool, parameters, bindings):
            return self.real.execute(tool, parameters, bindings)

    result = run([planned()], analytics=RestrictedAnalytics(), statuses={
        "call-a": (VerificationStatus.NEEDS_MORE_ANALYSIS, "run_robustness_check"),
    })
    assert len(result.steps) == 1
    assert result.steps[0].replanning.status == "no_fallback"
    assert result.unresolved_call_ids == ("call-a",)


def test_max_steps_prevents_extra_fallback_execution():
    result = run([planned()], max_steps=1, statuses={
        "call-a": (VerificationStatus.NEEDS_MORE_ANALYSIS, "run_robustness_check"),
    })
    assert len(result.steps) == 1
    assert result.steps[0].replanning.status == "max_steps_reached"
    assert result.exhausted and not result.succeeded


def test_remaining_queue_exhausts_without_silent_success():
    result = run([temporal("first"), temporal("second")], max_steps=1)
    assert len(result.steps) == 1
    assert result.exhausted and not result.succeeded


def test_failed_executor_continues_without_replanning():
    class Failure:
        def __init__(self):
            self.real = AnalyticsToolExecutor()

        def available_tools(self):
            return self.real.available_tools()

        def execute(self, name, parameters, bindings):
            if name == "linear_trend":
                raise ValueError("intentional execution failure")
            return self.real.execute(name, parameters, bindings)

    result = run([temporal("first"), planned(call_id="second")], analytics=Failure())
    assert result.steps[0].trace.status is TraceStatus.FAILED
    assert result.steps[0].replanning is None
    assert result.steps[1].trace.verification_status is VerificationStatus.VERIFIED
    assert not result.succeeded


def test_failed_binder_continues_without_replanning():
    class Failure:
        def bind(self, trace, data):
            if trace.call_id == "first":
                raise ValueError("intentional bind failure")
            return DatasetBinder().bind(trace, data)

    result = run([temporal("first"), planned(call_id="second")], binder=Failure())
    assert result.steps[0].trace.status is TraceStatus.FAILED
    assert result.steps[0].replanning is None
    assert result.steps[1].trace.verification_status is VerificationStatus.VERIFIED
    assert not result.succeeded


def test_verified_descendant_resolves_multiple_generations():
    root = planned(call_id="root").complete({"p_value": 0.9}).verify(
        VerificationStatus.NEEDS_MORE_ANALYSIS, next_decision="run_robustness_check"
    )
    child = planned(call_id="child", sources=("root",)).complete({"p_value": 0.5}).verify(
        VerificationStatus.REJECTED, next_decision="rerun_tool_with_valid_output"
    )
    grandchild = planned(call_id="grandchild", sources=("child",)).complete({"p_value": 0.4}).verify(
        VerificationStatus.VERIFIED, next_decision="interpret_result"
    )
    result = PipelineResult("question", None, (
        PipelineStep(1, root), PipelineStep(2, child), PipelineStep(3, grandchild),
    ))
    assert result.unresolved_call_ids == ()
    assert result.succeeded


def test_unverified_descendant_does_not_resolve_ancestor():
    root = planned(call_id="root").complete({"p_value": 0.9}).verify(
        VerificationStatus.REJECTED, next_decision="rerun_tool_with_valid_output"
    )
    child = planned(call_id="child", sources=("root",)).complete({"p_value": 0.6}).verify(
        VerificationStatus.NEEDS_MORE_ANALYSIS, next_decision="collect_more_data_or_reduce_scope"
    )
    result = PipelineResult("question", None, (PipelineStep(1, root), PipelineStep(2, child)))
    assert result.unresolved_call_ids == ("root", "child")
    assert not result.succeeded


def test_fallback_api_names_and_pearson_provenance():
    replanner = SelfCorrectionPlanner()
    assert replanner.fallback_for("pearson_correlation") == "spearman_correlation"
    assert replanner.fallback_for("one_way_anova") == "kruskal_wallis_test"
    assert replanner.fallback_for("linear_regression") is None
    assert replanner.fallback_for("ordinary_least_squares") is None
    source = planned("pearson_correlation", "pearson", {"value_column": "sales"}).complete({"p_value": .2}).verify(
        VerificationStatus.NEEDS_MORE_ANALYSIS, next_decision="run_robustness_check"
    )
    assert replanner.can_automatically_replan(source)
    directive = replanner.automatic_robustness_check(source, next_call_id="spearman")
    assert directive.tool == "spearman_correlation"
    assert directive.source_call_ids == ("pearson",)
    assert directive.source_trace_hash == source.trace_hash
    assert directive.parameters == source.parameters
    assert directive.bindings == source.bindings
