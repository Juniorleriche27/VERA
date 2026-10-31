
from __future__ import annotations

from agent.tool_catalog_selector import ToolCatalogSelector

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from agent.analytics_executor import AnalyticsToolExecutor
from agent.data_binding import DatasetBinder
from agent.replanning import ReplanningError, SelfCorrectionPlanner
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
class ReplanningOutcome:
    status: str
    source_call_id: str
    decision: str
    reason: str
    source_trace_hash: str = ""
    fallback_tool: str = ""
    next_call_id: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "status": self.status,
            "source_call_id": self.source_call_id,
            "decision": self.decision,
            "reason": self.reason,
            "source_trace_hash": self.source_trace_hash,
            "fallback_tool": self.fallback_tool,
            "next_call_id": self.next_call_id,
        }


@dataclass(frozen=True, slots=True)
class PipelineStep:
    index: int
    trace: ToolCallTrace
    report: Any | None = None
    replanning: ReplanningOutcome | None = None

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
            "replanning": self.replanning.as_dict() if self.replanning else None,
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
    def unresolved_call_ids(self) -> tuple[str, ...]:
        """Non-verified evidence not resolved by a later verified descendant."""
        ancestors = {step.trace.call_id: step.trace.source_call_ids for step in self.steps}
        verified_sources: set[str] = set()
        for step in self.steps:
            trace = step.trace
            if trace.status is TraceStatus.COMPLETED and trace.verification_status is VerificationStatus.VERIFIED:
                pending = list(trace.source_call_ids)
                while pending:
                    source = pending.pop()
                    if source in verified_sources:
                        continue
                    verified_sources.add(source)
                    pending.extend(ancestors.get(source, ()))
        return tuple(
            step.trace.call_id for step in self.steps
            if step.trace.verification_status in {
                VerificationStatus.REJECTED, VerificationStatus.NEEDS_MORE_ANALYSIS,
            } and step.trace.call_id not in verified_sources
        )

    @property
    def succeeded(self) -> bool:
        if not self.steps or self.exhausted or self.unresolved_call_ids:
            return False
        if any(step.trace.status is not TraceStatus.COMPLETED for step in self.steps):
            return False
        if not any(step.trace.verification_status is VerificationStatus.VERIFIED for step in self.steps):
            return False
        # Every terminal branch must end in verified evidence.
        parent_ids = {
            source for step in self.steps for source in step.trace.source_call_ids
        }
        return all(
            step.trace.verification_status is VerificationStatus.VERIFIED
            for step in self.steps if step.trace.call_id not in parent_ids
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
            "unresolved_call_ids": list(self.unresolved_call_ids),
        }

    def build_decision_brief(self) -> Any:
        from agent.decision_brief import VerifiedDecisionBriefBuilder

        return VerifiedDecisionBriefBuilder().build(self)

    def build_visualizations(self) -> Any:
        from visualization.spec import VerifiedVisualizationBuilder

        brief = self.build_decision_brief()
        return VerifiedVisualizationBuilder().build(brief.evidence)


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
        tool_selector: ToolCatalogSelector | None = None,
        max_steps: int = 5,
    ) -> None:
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise PipelineError("'max_steps' must be a positive integer.")
        self.planner = planner
        self.binder = binder or DatasetBinder()
        self.analytics = analytics or AnalyticsToolExecutor()
        self.verifier = verifier or DeterministicVerifier()
        self.replanner = replanner or SelfCorrectionPlanner()
        self.tool_selector = tool_selector or ToolCatalogSelector(max_tools=10)
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

        safe_context = dict(dataset_context)
        selection = self.tool_selector.select(
            question.strip(),
            dataset_context=safe_context,
            tool_catalog=self.analytics.available_tools(),
        )
        planning = self.planner.plan(
            question.strip(),
            dataset_context=safe_context,
            tool_catalog=list(selection.tools),
        )
        planned_traces = tuple(getattr(planning, "traces", ()) or ())
        if not all(isinstance(trace, ToolCallTrace) for trace in planned_traces):
            raise PipelineError("Planner returned an invalid trace collection.")
        if len(planned_traces) > 4:
            raise PipelineError("Planner returned more than four initial tool calls; refusing to truncate silently.")

        queue = list(planned_traces)
        steps: list[PipelineStep] = []
        next_replan_number = 1

        while queue and len(steps) < self.max_steps:
            trace = queue.pop(0)
            if trace.status is not TraceStatus.PLANNED:
                raise PipelineError("Pipeline accepts only planned traces for execution.")

            try:
                bound = self.binder.bind(trace, dataset)
                running = trace.mark_running()

                if hasattr(bound, "parameters") and hasattr(bound, "bindings"):
                    deterministic_result = self.analytics.execute(
                        trace.tool,
                        parameters=bound.parameters,
                        bindings=bound.bindings,
                    )
                else:
                    deterministic_result = self.analytics.execute(
                        trace.tool,
                        **bound.executor_kwargs(),
                    )
                completed = running.complete(deterministic_result)
            except Exception as exc:
                failed = trace.fail(str(exc))
                steps.append(PipelineStep(len(steps) + 1, failed, None))
                continue

            verified_trace, report = self.verifier.verify(completed)
            outcome: ReplanningOutcome | None = None
            if verified_trace.verification_status in {
                VerificationStatus.NEEDS_MORE_ANALYSIS,
                VerificationStatus.REJECTED,
            }:
                decision = verified_trace.next_decision
                fallback = (
                    self.replanner.fallback_for(verified_trace.tool)
                    if decision == "run_robustness_check" else None
                )
                if decision == "run_robustness_check" and fallback:
                    available = {item["name"] for item in self.analytics.available_tools()}
                    if fallback not in available:
                        outcome = ReplanningOutcome(
                            "no_fallback", verified_trace.call_id, decision,
                            f"Fallback {fallback!r} is not in the analytics catalog.",
                            verified_trace.trace_hash,
                        )
                    elif len(steps) + 1 >= self.max_steps:
                        outcome = ReplanningOutcome(
                            "max_steps_reached", verified_trace.call_id, decision,
                            "Cannot schedule another tool call within max_steps.",
                            verified_trace.trace_hash, fallback,
                        )
                    else:
                        next_id = f"{verified_trace.call_id}-replan-{next_replan_number}"
                        try:
                            directive = self.replanner.automatic_robustness_check(
                                verified_trace, next_call_id=next_id,
                            )
                            replanned_trace = directive.to_trace()
                            if not isinstance(replanned_trace, ToolCallTrace):
                                raise ReplanningError("Replanner returned an invalid trace.")
                            if replanned_trace.tool != fallback:
                                raise ReplanningError("Replanner produced an unexpected fallback tool.")
                        except (ReplanningError, ValueError) as exc:
                            outcome = ReplanningOutcome(
                                "no_fallback", verified_trace.call_id, decision,
                                str(exc), verified_trace.trace_hash,
                            )
                        else:
                            queue.insert(0, replanned_trace)
                            next_replan_number += 1
                            outcome = ReplanningOutcome(
                                "scheduled", verified_trace.call_id, decision,
                                directive.reason if hasattr(directive, "reason") else "Deterministic robustness fallback scheduled.",
                                verified_trace.trace_hash, fallback, next_id,
                            )
                elif decision == "run_robustness_check":
                    outcome = ReplanningOutcome(
                        "no_fallback", verified_trace.call_id, decision,
                        "No registered safe deterministic fallback.",
                        verified_trace.trace_hash,
                    )
                else:
                    outcome = ReplanningOutcome(
                        "unsupported_decision", verified_trace.call_id, decision,
                        "Decision recorded; no automatic execution is defined.",
                        verified_trace.trace_hash,
                    )
            steps.append(PipelineStep(len(steps) + 1, verified_trace, report, outcome))

        exhausted = bool(queue) or any(
            step.replanning is not None
            and step.replanning.status == "max_steps_reached"
            for step in steps
        )
        return PipelineResult(
            question=question.strip(),
            planning=planning,
            steps=tuple(steps),
            exhausted=exhausted,
        )
