from dataclasses import dataclass
from typing import Any

import pytest

from agent.runtime import AgentRuntime, AgentRuntimeError
from agent.tool_call_trace import ToolCallTrace, VerificationStatus


@dataclass(frozen=True)
class _Report:
    decision: str

    def as_dict(self):
        return {"decision": self.decision}


class _Verifier:
    def verify(self, trace):
        result = trace.deterministic_result
        if result.get("needs_more_analysis"):
            verified = trace.verify(
                VerificationStatus.NEEDS_MORE_ANALYSIS,
                next_decision="run_robustness_check",
            )
            return verified, _Report("needs_more_analysis")

        verified = trace.verify(
            VerificationStatus.VERIFIED,
            next_decision="interpret_result",
        )
        return verified, _Report("verified")


class _Directive:
    def __init__(self, trace):
        self._trace = trace

    def to_trace(self):
        return self._trace


class _Replanner:
    def __init__(self, tool="robust_tool"):
        self.tool = tool
        self.calls = []

    def propose(self, trace, *, next_call_id, **kwargs):
        self.calls.append((trace.call_id, next_call_id))
        return _Directive(
            ToolCallTrace.planned(
                call_id=next_call_id,
                tool=self.tool,
                objective=f"Robustness check for {trace.objective}",
                reason="Verifier requested more analysis",
                parameters={"alpha": 0.05},
                bindings=trace.bindings,
                source_call_ids=[trace.call_id],
            )
        )


def _trace(tool="primary_tool"):
    return ToolCallTrace.planned(
        call_id="call-1",
        tool=tool,
        objective="Explain sales decline",
        reason="Run the selected deterministic analysis",
        parameters={"alpha": 0.05},
        bindings={"sales": [10, 9, 8, 7]},
    )


def test_runtime_executes_verifies_and_stops_when_evidence_passes():
    seen = []

    def executor(parameters, bindings):
        seen.append((parameters, bindings))
        return {"effect": -3.0, "needs_more_analysis": False}

    runtime = AgentRuntime(
        {"primary_tool": executor},
        verifier=_Verifier(),
        replanner=_Replanner(),
    )

    result = runtime.run(_trace())

    assert result.completed is True
    assert result.stop_reason == "verified"
    assert result.call_count == 1
    assert result.terminal_trace.verification_status is VerificationStatus.VERIFIED
    assert seen[0][0] == {"alpha": 0.05}
    assert seen[0][1] == {"sales": [10, 9, 8, 7]}


def test_runtime_replans_and_executes_second_tool():
    calls = []

    def primary(parameters, bindings):
        calls.append("primary")
        return {"needs_more_analysis": True}

    def robust(parameters, bindings):
        calls.append("robust")
        return {"needs_more_analysis": False, "robust": True}

    replanner = _Replanner()
    runtime = AgentRuntime(
        {
            "primary_tool": primary,
            "robust_tool": robust,
        },
        verifier=_Verifier(),
        replanner=replanner,
        max_steps=3,
    )

    result = runtime.run(_trace())

    assert calls == ["primary", "robust"]
    assert result.completed is True
    assert result.call_count == 2
    assert result.steps[0].trace.verification_status is VerificationStatus.NEEDS_MORE_ANALYSIS
    assert result.steps[1].trace.verification_status is VerificationStatus.VERIFIED
    assert result.steps[1].trace.source_call_ids == ("call-1",)
    assert result.steps[1].trace.call_id == "call-1-replan-1"
    assert replanner.calls == [("call-1", "call-1-replan-1")]


def test_runtime_stops_at_max_steps_without_infinite_loop():
    def executor(parameters, bindings):
        return {"needs_more_analysis": True}

    runtime = AgentRuntime(
        {
            "primary_tool": executor,
            "robust_tool": executor,
        },
        verifier=_Verifier(),
        replanner=_Replanner(),
        max_steps=2,
    )

    result = runtime.run(_trace())

    assert result.completed is False
    assert result.stop_reason == "max_steps_reached"
    assert result.call_count == 2
    assert result.terminal_trace.verification_status is VerificationStatus.NEEDS_MORE_ANALYSIS


def test_executor_exception_becomes_failed_trace():
    def executor(parameters, bindings):
        raise ValueError("bad input")

    runtime = AgentRuntime(
        {"primary_tool": executor},
        verifier=_Verifier(),
        replanner=_Replanner(),
    )

    result = runtime.run(_trace())

    assert result.completed is False
    assert result.stop_reason == "execution_failed"
    assert result.call_count == 0
    assert result.terminal_trace.status.value == "failed"
    assert result.terminal_trace.execution_error == "ValueError: bad input"


def test_missing_executor_is_configuration_error():
    runtime = AgentRuntime(
        {"another_tool": lambda parameters, bindings: {}},
        verifier=_Verifier(),
        replanner=_Replanner(),
    )

    with pytest.raises(AgentRuntimeError, match="No executor registered"):
        runtime.run(_trace())


def test_runtime_rejects_non_planned_initial_trace():
    runtime = AgentRuntime(
        {"primary_tool": lambda parameters, bindings: {}},
        verifier=_Verifier(),
        replanner=_Replanner(),
    )

    with pytest.raises(AgentRuntimeError, match="planned state"):
        runtime.run(_trace().mark_running())


def test_runtime_validates_executor_registry():
    with pytest.raises(AgentRuntimeError):
        AgentRuntime({})

    with pytest.raises(AgentRuntimeError):
        AgentRuntime({"primary_tool": object()})


def test_runtime_validates_max_steps():
    with pytest.raises(AgentRuntimeError):
        AgentRuntime({"primary_tool": lambda p, b: {}}, max_steps=0)


def test_available_tools_are_sorted_and_immutable():
    runtime = AgentRuntime(
        {
            "z_tool": lambda p, b: {},
            "a_tool": lambda p, b: {},
        },
        verifier=_Verifier(),
        replanner=_Replanner(),
    )

    assert runtime.available_tools == ("a_tool", "z_tool")


def test_runtime_result_serializes_full_audit_chain():
    def primary(parameters, bindings):
        return {"needs_more_analysis": True}

    def robust(parameters, bindings):
        return {"needs_more_analysis": False}

    runtime = AgentRuntime(
        {
            "primary_tool": primary,
            "robust_tool": robust,
        },
        verifier=_Verifier(),
        replanner=_Replanner(),
    )

    payload = runtime.run(_trace()).as_dict()

    assert payload["completed"] is True
    assert payload["stop_reason"] == "verified"
    assert payload["call_count"] == 2
    assert len(payload["steps"]) == 2
    assert payload["steps"][0]["trace"]["call_id"] == "call-1"
    assert payload["steps"][1]["trace"]["call_id"] == "call-1-replan-1"
    assert payload["terminal_trace"]["verification_status"] == "verified"
