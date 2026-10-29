
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from agent.analytics_executor import AnalyticsToolExecutor
from agent.data_binding import DatasetBinder
from agent.replanning import SelfCorrectionPlanner
from agent.tool_call_trace import ToolCallTrace, TraceStatus, VerificationStatus
from verification.verifier import DeterministicVerifier


class PipelineError(RuntimeError):
    """Raised when the verified analytics pipeline cannot continue safely."""


class PlanningProvider(Protocol):
    def plan(
        self,
        question: str,
        *,
        dataset_context: Mapping[str, Any],
        tool_catalog: Sequence[Mapping[str, Any]],
    ) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class PipelineStep:
    index: int
    trace: ToolCallTrace
    report: Any | None = None

    @property
    def status(self) -> str:
        return self.trace.status.value

    def as_dict(self) -> dict[str, Any]:
        report_payload = None
        if self.report is not None:
            as_dict = getattr(self.report, "as_dict", None)
            report_payload = as_dict() if callable(as_dict) else self.report
        return {
            "index": self.index,
            "trace": self.trace.as_dict(),
            "report": report_payload,
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    question: str
    planning: Any
    steps: tuple[PipelineStep, ...]
    exhausted: bool = False

    @property
    def traces(self) -> tuple[ToolCallTrace, ...]:
        return tuple(step.trace for step in self.steps)

    @property
    def succeeded(self) -> bool:
        if not self.steps:
            return False
        final = self.steps[-1].trace
        return (
            final.status is TraceStatus.COMPLETED
            and final.verification_status is VerificationStatus.VERIFIED
            and all(step.trace.status is TraceStatus.COMPLETED for step in self.steps)
        )

    @property
    def final_trace(self) -> ToolCallTrace | None:
        return self.steps[-1].trace if self.steps else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "steps": [step.as_dict() for step in self.steps],
            "exhausted": self.exhausted,
            "succeeded": self.succeeded,
        }


class VerifiedAnalyticsPipeline:
    """
    End-to-end VERA execution path.

    The planning provider receives only caller-supplied safe dataset context
    and the analytics catalog. Raw dataset values remain local and are bound
    only after the model has selected a tool and semantic column references.
    """

    def __init__(
        self,
        planner: PlanningProvider,
        *,
        binder: DatasetBinder | None = None,
        analytics: AnalyticsToolExecutor | None = None,
        verifier: DeterministicVerifier | None = None,
        replanner: SelfCorrectionPlanner | None = None,
        max_steps: int = 5,
    ) -> None:
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise PipelineError("'max_steps' must be a positive integer.")
        self.planner = planner
        self.binder = binder or DatasetBinder()
        self.analytics = analytics or AnalyticsToolExecutor()
        self.verifier = verifier or DeterministicVerifier()
        self.replanner = replanner or SelfCorrectionPlanner()
        self.max_steps = max_steps

    def run(
        self,
        question: str,
        *,
        dataset: Any,
        dataset_context: Mapping[str, Any],
    ) -> PipelineResult:
        if not isinstance(question, str) or not question.strip():
            raise PipelineError("'question' must be a non-empty string.")
        if not isinstance(dataset_context, Mapping):
            raise PipelineError("'dataset_context' must be a mapping.")

        planning = self.planner.plan(
            question.strip(),
            dataset_context=dict(dataset_context),
            tool_catalog=self.analytics.available_tools(),
        )
        planned_traces = tuple(getattr(planning, "traces", ()) or ())
        if not all(isinstance(trace, ToolCallTrace) for trace in planned_traces):
            raise PipelineError("Planner returned an invalid trace collection.")

        queue = list(planned_traces)
        steps: list[PipelineStep] = []
        next_replan_number = 1

        while queue and len(steps) < self.max_steps:
            trace = queue.pop(0)
            if trace.status is not TraceStatus.PLANNED:
                raise PipelineError("Pipeline accepts only planned traces for execution.")

            try:
                bound = self.binder.bind(trace, dataset)
                kwargs = bound.executor_kwargs()
                running = trace.mark_running()
                deterministic_result = self.analytics.execute(trace.tool, **kwargs)
                completed = running.complete(deterministic_result)
            except Exception as exc:
                failed = trace.fail(str(exc))
                steps.append(PipelineStep(len(steps) + 1, failed, None))
                continue

            verified_trace, report = self.verifier.verify(completed)
            steps.append(PipelineStep(len(steps) + 1, verified_trace, report))

            if verified_trace.verification_status in {
                VerificationStatus.NEEDS_MORE_ANALYSIS,
                VerificationStatus.REJECTED,
            }:
                if len(steps) >= self.max_steps:
                    break

                directive = self.replanner.automatic_robustness_check(
                    verified_trace,
                    next_call_id=f"{verified_trace.call_id}-replan-{next_replan_number}",
                )
                next_replan_number += 1
                replanned_trace = directive.to_trace()
                if not isinstance(replanned_trace, ToolCallTrace):
                    raise PipelineError("Replanner returned an invalid trace.")
                queue.insert(0, replanned_trace)

        exhausted = bool(queue) or bool(
            steps
            and len(steps) >= self.max_steps
            and steps[-1].trace.verification_status
            in {VerificationStatus.NEEDS_MORE_ANALYSIS, VerificationStatus.REJECTED}
        )
        return PipelineResult(
            question=question.strip(),
            planning=planning,
            steps=tuple(steps),
            exhausted=exhausted,
        )
