from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from agent.replanning import SelfCorrectionPlanner
from agent.tool_call_trace import ToolCallTrace, TraceStatus, VerificationStatus
from verification.verifier import DeterministicVerifier, VerificationReport


class AgentRuntimeError(RuntimeError):
    """Raised when the agent runtime contract is invalid."""


class ToolExecutor(Protocol):
    def __call__(
        self,
        parameters: Mapping[str, Any],
        bindings: Mapping[str, Any],
    ) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class RuntimeStep:
    """One fully executed and verified agent step."""

    trace: ToolCallTrace
    verification_report: VerificationReport

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace": self.trace.as_dict(),
            "verification_report": self.verification_report.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    """Immutable result of one VERA agent execution loop."""

    steps: tuple[RuntimeStep, ...]
    terminal_trace: ToolCallTrace
    stop_reason: str

    @property
    def completed(self) -> bool:
        return self.stop_reason == "verified"

    @property
    def call_count(self) -> int:
        return len(self.steps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.as_dict() for step in self.steps],
            "terminal_trace": self.terminal_trace.as_dict(),
            "stop_reason": self.stop_reason,
            "completed": self.completed,
            "call_count": self.call_count,
        }


class AgentRuntime:
    """
    Deterministic orchestration loop for VERA.

    The runtime never invents analytical values. Each executor returns the
    deterministic result, the verifier judges the evidence, and the replanner
    may create a new planned trace when more analysis is required.
    """

    def __init__(
        self,
        executors: Mapping[str, ToolExecutor],
        *,
        verifier: DeterministicVerifier | None = None,
        replanner: SelfCorrectionPlanner | None = None,
        max_steps: int = 5,
    ) -> None:
        if not isinstance(executors, Mapping) or not executors:
            raise AgentRuntimeError("'executors' must be a non-empty mapping.")

        normalized: dict[str, ToolExecutor] = {}
        for name, executor in executors.items():
            if not isinstance(name, str) or not name.strip():
                raise AgentRuntimeError("Executor names must be non-empty strings.")
            if not callable(executor):
                raise AgentRuntimeError(
                    f"Executor for {name!r} must be callable."
                )
            normalized[name.strip()] = executor

        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise AgentRuntimeError("'max_steps' must be an integer >= 1.")

        self._executors = normalized
        self._verifier = verifier or DeterministicVerifier()
        self._replanner = replanner or SelfCorrectionPlanner()
        self._max_steps = max_steps

    @property
    def available_tools(self) -> tuple[str, ...]:
        return tuple(sorted(self._executors))

    def run(self, initial_trace: ToolCallTrace) -> RuntimeResult:
        if not isinstance(initial_trace, ToolCallTrace):
            raise AgentRuntimeError("'initial_trace' must be a ToolCallTrace.")
        if initial_trace.status is not TraceStatus.PLANNED:
            raise AgentRuntimeError(
                "AgentRuntime requires an initial trace in planned state."
            )

        current = initial_trace
        steps: list[RuntimeStep] = []

        for index in range(self._max_steps):
            executor = self._executors.get(current.tool)
            if executor is None:
                raise AgentRuntimeError(
                    f"No executor registered for tool {current.tool!r}."
                )

            running = current.mark_running()
            try:
                deterministic_result = executor(
                    running.parameters,
                    running.bindings,
                )
            except Exception as exc:
                failed = running.fail(
                    f"{type(exc).__name__}: {exc}"
                )
                return RuntimeResult(
                    steps=tuple(steps),
                    terminal_trace=failed,
                    stop_reason="execution_failed",
                )

            completed = running.complete(deterministic_result)
            verified, report = self._verifier.verify(completed)
            steps.append(
                RuntimeStep(
                    trace=verified,
                    verification_report=report,
                )
            )

            if verified.verification_status is VerificationStatus.VERIFIED:
                return RuntimeResult(
                    steps=tuple(steps),
                    terminal_trace=verified,
                    stop_reason="verified",
                )

            if verified.verification_status not in {
                VerificationStatus.REJECTED,
                VerificationStatus.NEEDS_MORE_ANALYSIS,
            }:
                raise AgentRuntimeError(
                    "Verifier returned a completed trace without a terminal "
                    "verification decision."
                )

            if index + 1 >= self._max_steps:
                return RuntimeResult(
                    steps=tuple(steps),
                    terminal_trace=verified,
                    stop_reason="max_steps_reached",
                )

            next_call_id = self._next_call_id(
                initial_trace.call_id,
                index + 1,
            )
            directive = self._replanner.propose(
                verified,
                next_call_id=next_call_id,
            )
            current = directive.to_trace()

        raise AgentRuntimeError("Unreachable agent runtime state.")

    @staticmethod
    def _next_call_id(root_call_id: str, replan_index: int) -> str:
        return f"{root_call_id}-replan-{replan_index}"
