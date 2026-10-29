
from types import SimpleNamespace

import pytest

from agent.pipeline import PipelineError, VerifiedAnalyticsPipeline
from agent.tool_call_trace import ToolCallTrace, VerificationStatus


def _trace(call_id="call-1", tool="one_way_anova"):
    return ToolCallTrace.planned(
        call_id=call_id,
        tool=tool,
        objective="Compare regional sales",
        reason="Compare independent groups",
        parameters={"alpha": 0.05},
        bindings={"grouping_column": "region", "value_column": "sales"},
    )


class FakePlanner:
    def __init__(self, traces):
        self.traces = tuple(traces)
        self.received_context = None
        self.received_catalog = None

    def plan(self, question, *, dataset_context, tool_catalog):
        self.received_context = dataset_context
        self.received_catalog = tool_catalog
        return SimpleNamespace(
            traces=self.traces,
            assistant_content="",
            finish_reason="tool_calls" if self.traces else "stop",
            model="fake-model",
            usage={},
        )


class FakeBound:
    def __init__(self, kwargs):
        self._kwargs = kwargs

    def executor_kwargs(self):
        return dict(self._kwargs)


class FakeBinder:
    def __init__(self):
        self.calls = []

    def bind(self, trace, dataset):
        self.calls.append((trace, dataset))
        return FakeBound(
            {
                "parameters": trace.parameters,
                "bindings": {"data": {"north": [1, 2], "south": [3, 4]}},
            }
        )


class FakeAnalytics:
    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def available_tools(self):
        return [{"name": "one_way_anova"}]

    def execute(self, name, *, parameters, bindings):
        self.calls.append((name, parameters, bindings))
        if self.fail:
            raise ValueError("execution failed")
        return {"statistic": 4.2, "p_value": 0.03}


class VerifiedVerifier:
    def __init__(self):
        self.calls = []

    def verify(self, trace):
        self.calls.append(trace)
        verified = trace.verify(
            VerificationStatus.VERIFIED,
            next_decision="interpret_result",
        )
        return verified, SimpleNamespace(as_dict=lambda: {"status": "verified"})


class NeedsMoreThenVerified:
    def __init__(self):
        self.count = 0

    def verify(self, trace):
        self.count += 1
        if self.count == 1:
            updated = trace.verify(
                VerificationStatus.NEEDS_MORE_ANALYSIS,
                next_decision="run_robustness_check",
            )
            return updated, SimpleNamespace(as_dict=lambda: {"status": "needs_more_analysis"})
        updated = trace.verify(
            VerificationStatus.VERIFIED,
            next_decision="interpret_result",
        )
        return updated, SimpleNamespace(as_dict=lambda: {"status": "verified"})


class FakeDirective:
    def __init__(self, trace):
        self._trace = trace

    def to_trace(self):
        return self._trace


class FakeReplanner:
    def __init__(self):
        self.calls = []

    def automatic_robustness_check(self, trace, *, next_call_id):
        self.calls.append((trace, next_call_id))
        return FakeDirective(
            ToolCallTrace.planned(
                call_id=next_call_id,
                tool="one_way_anova",
                objective="Robustness check",
                reason="Verifier requested more analysis",
                parameters={"alpha": 0.05},
                bindings=trace.bindings,
                source_call_ids=[trace.call_id],
            )
        )


def test_pipeline_keeps_raw_dataset_local_to_binder():
    dataset = object()
    context = {"row_count": 120, "columns": [{"name": "region"}, {"name": "sales"}]}
    planner = FakePlanner([_trace()])
    binder = FakeBinder()
    analytics = FakeAnalytics()

    result = VerifiedAnalyticsPipeline(
        planner,
        binder=binder,
        analytics=analytics,
        verifier=VerifiedVerifier(),
        replanner=FakeReplanner(),
    ).run("Compare sales", dataset=dataset, dataset_context=context)

    assert planner.received_context == context
    assert planner.received_context is not dataset
    assert binder.calls[0][1] is dataset
    assert result.succeeded is True


def test_pipeline_executes_bound_parameters_and_bindings():
    analytics = FakeAnalytics()
    result = VerifiedAnalyticsPipeline(
        FakePlanner([_trace()]),
        binder=FakeBinder(),
        analytics=analytics,
        verifier=VerifiedVerifier(),
        replanner=FakeReplanner(),
    ).run("Compare sales", dataset={"secret": "raw"}, dataset_context={"row_count": 4})

    assert len(analytics.calls) == 1
    name, parameters, bindings = analytics.calls[0]
    assert name == "one_way_anova"
    assert parameters == {"alpha": 0.05}
    assert "data" in bindings
    assert result.final_trace.verification_status is VerificationStatus.VERIFIED


def test_pipeline_records_execution_failure_without_fake_result():
    result = VerifiedAnalyticsPipeline(
        FakePlanner([_trace()]),
        binder=FakeBinder(),
        analytics=FakeAnalytics(fail=True),
        verifier=VerifiedVerifier(),
        replanner=FakeReplanner(),
    ).run("Compare sales", dataset={}, dataset_context={})

    assert result.succeeded is False
    assert result.final_trace.status.value == "failed"
    assert result.final_trace.deterministic_result is None
    assert "execution failed" in result.final_trace.execution_error


def test_pipeline_replans_after_needs_more_analysis():
    replanner = FakeReplanner()
    result = VerifiedAnalyticsPipeline(
        FakePlanner([_trace()]),
        binder=FakeBinder(),
        analytics=FakeAnalytics(),
        verifier=NeedsMoreThenVerified(),
        replanner=replanner,
        max_steps=3,
    ).run("Compare sales", dataset={}, dataset_context={})

    assert len(result.steps) == 2
    assert result.steps[0].trace.verification_status is VerificationStatus.NEEDS_MORE_ANALYSIS
    assert result.steps[1].trace.verification_status is VerificationStatus.VERIFIED
    assert result.steps[1].trace.source_call_ids == ("call-1",)
    assert len(replanner.calls) == 1
    assert result.exhausted is False
    assert result.succeeded is True


def test_pipeline_honors_max_steps():
    class AlwaysNeedsMore:
        def verify(self, trace):
            return (
                trace.verify(
                    VerificationStatus.NEEDS_MORE_ANALYSIS,
                    next_decision="run_robustness_check",
                ),
                None,
            )

    result = VerifiedAnalyticsPipeline(
        FakePlanner([_trace()]),
        binder=FakeBinder(),
        analytics=FakeAnalytics(),
        verifier=AlwaysNeedsMore(),
        replanner=FakeReplanner(),
        max_steps=2,
    ).run("Compare sales", dataset={}, dataset_context={})

    assert len(result.steps) == 2
    assert result.exhausted is True
    assert result.succeeded is False


def test_pipeline_allows_text_only_planner_response():
    result = VerifiedAnalyticsPipeline(
        FakePlanner([]),
        binder=FakeBinder(),
        analytics=FakeAnalytics(),
        verifier=VerifiedVerifier(),
        replanner=FakeReplanner(),
    ).run("Hello", dataset={}, dataset_context={})

    assert result.steps == ()
    assert result.final_trace is None
    assert result.succeeded is False


@pytest.mark.parametrize("question", ["", "   ", None])
def test_pipeline_rejects_invalid_question(question):
    pipeline = VerifiedAnalyticsPipeline(
        FakePlanner([]),
        binder=FakeBinder(),
        analytics=FakeAnalytics(),
        verifier=VerifiedVerifier(),
        replanner=FakeReplanner(),
    )
    with pytest.raises(PipelineError):
        pipeline.run(question, dataset={}, dataset_context={})


def test_pipeline_rejects_non_mapping_dataset_context():
    pipeline = VerifiedAnalyticsPipeline(
        FakePlanner([]),
        binder=FakeBinder(),
        analytics=FakeAnalytics(),
        verifier=VerifiedVerifier(),
        replanner=FakeReplanner(),
    )
    with pytest.raises(PipelineError):
        pipeline.run("Question", dataset={}, dataset_context=["bad"])


def test_pipeline_requires_positive_max_steps():
    with pytest.raises(PipelineError):
        VerifiedAnalyticsPipeline(
            FakePlanner([]),
            binder=FakeBinder(),
            analytics=FakeAnalytics(),
            verifier=VerifiedVerifier(),
            replanner=FakeReplanner(),
            max_steps=0,
        )

def test_real_anova_pipeline_keeps_parameters_and_bindings_separate():
    from types import SimpleNamespace

    import pandas as pd

    from agent.pipeline import VerifiedAnalyticsPipeline
    from agent.tool_call_trace import ToolCallTrace

    class FixedPlanner:
        def plan(self, *args, **kwargs):
            trace = ToolCallTrace(
                call_id="regression-anova",
                tool="one_way_anova",
                objective="Compare average sales across regions.",
                reason="Regression test for Binder to AnalyticsExecutor contract.",
                _parameters_json='{"alpha":0.05}',
                _bindings_json='{"region":"region","sales":"sales"}',
            )
            return SimpleNamespace(traces=(trace,))

    dataset = pd.DataFrame(
        {
            "region": ["North"] * 5 + ["South"] * 5 + ["West"] * 5,
            "sales": [10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33, 34],
        }
    )
    context = {
        "row_count": len(dataset),
        "columns": [
            {"name": "region", "role": "dimension", "dtype": "string"},
            {"name": "sales", "role": "measure", "dtype": "numeric"},
        ],
    }

    result = VerifiedAnalyticsPipeline(FixedPlanner()).run(
        "Are average sales significantly different across the three regions?",
        dataset=dataset,
        dataset_context=context,
    )

    assert result.steps
    assert result.steps[0].trace.status.value == "completed"
    assert result.steps[0].trace.execution_error == ""


def test_pipeline_result_builds_decision_brief_through_verified_builder(monkeypatch):
    from agent.pipeline import PipelineResult
    import agent.decision_brief as decision_brief

    sentinel = object()

    class FakeBuilder:
        def build(self, pipeline_result):
            assert pipeline_result.question == "Why did sales change?"
            return sentinel

    monkeypatch.setattr(decision_brief, "VerifiedDecisionBriefBuilder", FakeBuilder)

    result = PipelineResult(
        question="Why did sales change?",
        planning=None,
        steps=(),
    )

    assert result.build_decision_brief() is sentinel
