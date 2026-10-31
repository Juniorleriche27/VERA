from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from agent.tool_call_trace import (
    ToolCallTrace,
    ToolCallTraceError,
    TraceStatus,
    VerificationStatus,
)


class ReplanningError(ValueError):
    """Raised when a self-correction directive violates the replanning contract."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplanningError(f"'{field}' must be a non-empty string.")
    return value.strip()


def _mapping(value: Mapping[str, Any] | None, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ReplanningError(f"'{field}' must be a mapping.")
    return dict(value)


@dataclass(frozen=True, slots=True)
class ReplanDirective:
    """Immutable instruction describing why and how VERA should self-correct."""

    source_call_id: str
    source_trace_hash: str
    next_call_id: str
    tool: str
    objective: str
    reason: str
    parameters: dict[str, Any]
    bindings: dict[str, Any]
    source_call_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_call_id", _required_text(self.source_call_id, "source_call_id")
        )
        object.__setattr__(
            self,
            "source_trace_hash",
            _required_text(self.source_trace_hash, "source_trace_hash"),
        )
        object.__setattr__(
            self, "next_call_id", _required_text(self.next_call_id, "next_call_id")
        )
        object.__setattr__(self, "tool", _required_text(self.tool, "tool"))
        object.__setattr__(
            self, "objective", _required_text(self.objective, "objective")
        )
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))

        parameters = _mapping(self.parameters, "parameters")
        bindings = _mapping(self.bindings, "bindings")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "bindings", bindings)

        sources = tuple(_required_text(x, "source_call_ids") for x in self.source_call_ids)
        if len(set(sources)) != len(sources):
            raise ReplanningError("'source_call_ids' must be unique.")
        if self.next_call_id in sources:
            raise ReplanningError("The next call cannot depend on itself.")
        if self.source_call_id not in sources:
            raise ReplanningError(
                "The rejected/incomplete call must be included in 'source_call_ids'."
            )
        object.__setattr__(self, "source_call_ids", sources)

    def to_trace(self) -> ToolCallTrace:
        """Materialize the directive as a new planned ToolCallTrace."""
        try:
            return ToolCallTrace.planned(
                call_id=self.next_call_id,
                tool=self.tool,
                objective=self.objective,
                reason=self.reason,
                parameters=self.parameters,
                bindings=self.bindings,
                source_call_ids=self.source_call_ids,
            )
        except ToolCallTraceError as exc:
            raise ReplanningError(str(exc)) from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_call_id": self.source_call_id,
            "source_trace_hash": self.source_trace_hash,
            "next_call_id": self.next_call_id,
            "tool": self.tool,
            "objective": self.objective,
            "reason": self.reason,
            "parameters": dict(self.parameters),
            "bindings": dict(self.bindings),
            "source_call_ids": list(self.source_call_ids),
        }


class SelfCorrectionPlanner:
    """Create explicit follow-up tool calls from failed verification decisions."""

    _ROBUSTNESS_FALLBACKS = {
        "pearson_correlation": "spearman_correlation",
        "one_way_anova": "kruskal_wallis_test",
    }

    def fallback_for(self, tool: str) -> str | None:
        """Return a registered deterministic fallback, without raising."""
        return self._ROBUSTNESS_FALLBACKS.get(tool)

    def can_automatically_replan(self, trace: ToolCallTrace) -> bool:
        return (
            trace.status is TraceStatus.COMPLETED
            and trace.verification_status in {
                VerificationStatus.REJECTED, VerificationStatus.NEEDS_MORE_ANALYSIS
            }
            and trace.next_decision == "run_robustness_check"
            and self.fallback_for(trace.tool) is not None
        )

    def propose(
        self,
        trace: ToolCallTrace,
        *,
        next_call_id: str,
        tool: str | None = None,
        objective: str | None = None,
        reason: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        bindings: Mapping[str, Any] | None = None,
        additional_source_call_ids: tuple[str, ...] | list[str] = (),
    ) -> ReplanDirective:
        if not isinstance(trace, ToolCallTrace):
            raise ReplanningError("'trace' must be a ToolCallTrace.")
        if trace.status is not TraceStatus.COMPLETED:
            raise ReplanningError("Only completed tool calls can be replanned.")
        if trace.verification_status not in {
            VerificationStatus.REJECTED,
            VerificationStatus.NEEDS_MORE_ANALYSIS,
        }:
            raise ReplanningError(
                "Replanning requires a rejected or needs-more-analysis verification."
            )
        if not trace.next_decision:
            raise ReplanningError(
                "The source trace must contain an explicit 'next_decision'."
            )

        normalized_call_id = _required_text(next_call_id, "next_call_id")
        if normalized_call_id == trace.call_id:
            raise ReplanningError("A replanned call must use a new call_id.")

        selected_tool = (
            _required_text(tool, "tool")
            if tool is not None
            else self._tool_from_decision(trace.next_decision, trace.tool)
        )

        selected_objective = (
            _required_text(objective, "objective")
            if objective is not None
            else trace.objective
        )
        selected_reason = (
            _required_text(reason, "reason")
            if reason is not None
            else self._default_reason(trace, selected_tool)
        )

        selected_parameters = (
            _mapping(parameters, "parameters")
            if parameters is not None
            else trace.parameters
        )
        selected_bindings = (
            _mapping(bindings, "bindings")
            if bindings is not None
            else trace.bindings
        )

        sources: list[str] = [trace.call_id]
        for source in trace.source_call_ids:
            if source not in sources:
                sources.append(source)
        for source in additional_source_call_ids:
            normalized = _required_text(source, "additional_source_call_ids")
            if normalized not in sources:
                sources.append(normalized)

        return ReplanDirective(
            source_call_id=trace.call_id,
            source_trace_hash=trace.trace_hash,
            next_call_id=normalized_call_id,
            tool=selected_tool,
            objective=selected_objective,
            reason=selected_reason,
            parameters=selected_parameters,
            bindings=selected_bindings,
            source_call_ids=tuple(sources),
        )

    def automatic_robustness_check(
        self,
        trace: ToolCallTrace,
        *,
        next_call_id: str,
    ) -> ReplanDirective:
        fallback = self._ROBUSTNESS_FALLBACKS.get(trace.tool)
        if fallback is None:
            raise ReplanningError(
                f"No deterministic robustness fallback is registered for '{trace.tool}'."
            )
        return self.propose(
            trace,
            next_call_id=next_call_id,
            tool=fallback,
            reason=(
                f"Verification of '{trace.tool}' requested a robustness check; "
                f"run '{fallback}' as an independent follow-up."
            ),
        )

    @staticmethod
    def _tool_from_decision(next_decision: str, current_tool: str) -> str:
        decision = _required_text(next_decision, "next_decision")
        prefixes = ("run_", "use_", "execute_")
        lowered = decision.lower()
        for prefix in prefixes:
            if lowered.startswith(prefix):
                candidate = decision[len(prefix):].strip()
                if candidate:
                    return candidate
        if decision in SelfCorrectionPlanner._ROBUSTNESS_FALLBACKS:
            return SelfCorrectionPlanner._ROBUSTNESS_FALLBACKS[decision]
        raise ReplanningError(
            "Could not infer the next tool from 'next_decision'; provide 'tool' explicitly."
        )

    @staticmethod
    def _default_reason(trace: ToolCallTrace, selected_tool: str) -> str:
        status = trace.verification_status.value
        return (
            f"Previous evidence from '{trace.tool}' was marked '{status}'. "
            f"Follow verifier decision '{trace.next_decision}' with '{selected_tool}'."
        )
