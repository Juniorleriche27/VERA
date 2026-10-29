from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from agent.pipeline import PipelineResult
from agent.tool_call_trace import TraceStatus, VerificationStatus


class DecisionBriefError(ValueError):
    """Raised when a verified decision brief cannot be produced safely."""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if isfinite(value) else None


def _format_number(value: float) -> str:
    magnitude = abs(value)
    if magnitude != 0 and (magnitude < 0.001 or magnitude >= 1_000_000):
        return f"{value:.3e}"
    if value.is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    call_id: str
    tool: str
    trace_hash: str
    objective: str
    result: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool": self.tool,
            "trace_hash": self.trace_hash,
            "objective": self.objective,
            "result": dict(self.result),
        }


@dataclass(frozen=True, slots=True)
class EvidencePackage:
    question: str
    items: tuple[EvidenceItem, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "items": [item.as_dict() for item in self.items],
        }


@dataclass(frozen=True, slots=True)
class DecisionBrief:
    question: str
    headline: str
    findings: tuple[str, ...]
    caveats: tuple[str, ...]
    next_action: str
    evidence: EvidencePackage

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "headline": self.headline,
            "findings": list(self.findings),
            "caveats": list(self.caveats),
            "next_action": self.next_action,
            "evidence": self.evidence.as_dict(),
        }


class VerifiedDecisionBriefBuilder:
    """Build an inspectable business brief exclusively from verified results."""

    def build(self, pipeline_result: PipelineResult) -> DecisionBrief:
        if not isinstance(pipeline_result, PipelineResult):
            raise DecisionBriefError("'pipeline_result' must be a PipelineResult.")
        if not pipeline_result.steps:
            raise DecisionBriefError("Cannot build a decision brief without executed steps.")
        if not pipeline_result.succeeded:
            raise DecisionBriefError(
                "Decision briefs require a fully completed and verified pipeline."
            )

        items: list[EvidenceItem] = []
        findings: list[str] = []
        caveats: list[str] = []

        for step in pipeline_result.steps:
            trace = step.trace
            if (
                trace.status is not TraceStatus.COMPLETED
                or trace.verification_status is not VerificationStatus.VERIFIED
            ):
                raise DecisionBriefError(
                    "Every evidence item must be completed and verified."
                )

            result = trace.deterministic_result
            if not isinstance(result, Mapping):
                raise DecisionBriefError(
                    f"Verified trace {trace.call_id!r} does not expose a mapping result."
                )

            result_copy = dict(result)
            items.append(
                EvidenceItem(
                    call_id=trace.call_id,
                    tool=trace.tool,
                    trace_hash=trace.trace_hash,
                    objective=trace.objective,
                    result=result_copy,
                )
            )
            findings.extend(self._interpret(trace.tool, result_copy))

            report = step.report
            if report is not None:
                checks = getattr(report, "checks", ()) or ()
                for check in checks:
                    if not getattr(check, "passed", True):
                        message = str(getattr(check, "message", "")).strip()
                        if message:
                            caveats.append(message)

        evidence = EvidencePackage(
            question=pipeline_result.question,
            items=tuple(items),
        )
        unique_findings = tuple(dict.fromkeys(findings))
        unique_caveats = tuple(dict.fromkeys(caveats))

        if not unique_findings:
            unique_findings = (
                f"{len(items)} verified analytical result(s) are available for inspection.",
            )

        return DecisionBrief(
            question=pipeline_result.question,
            headline=self._headline(items[-1].tool, items[-1].result),
            findings=unique_findings,
            caveats=unique_caveats,
            next_action=self._next_action(items[-1].tool, items[-1].result),
            evidence=evidence,
        )

    def _headline(self, tool: str, result: Mapping[str, Any]) -> str:
        if tool == "one_way_anova":
            p_value = _number(result.get("p_value"))
            if p_value is not None:
                if p_value < 0.05:
                    return "The verified analysis finds statistically significant differences between group means."
                return "The verified analysis does not find statistically significant differences between group means."
        return "VERA produced a verified analytical result."

    def _interpret(self, tool: str, result: Mapping[str, Any]) -> list[str]:
        if tool == "one_way_anova":
            return self._interpret_anova(result)
        return self._interpret_generic(tool, result)

    def _interpret_anova(self, result: Mapping[str, Any]) -> list[str]:
        findings: list[str] = []

        p_value = _number(result.get("p_value"))
        f_stat = _number(result.get("f_statistic"))
        eta_squared = _number(result.get("eta_squared"))
        sample_size = _number(result.get("sample_size"))
        groups = result.get("group_statistics")

        if p_value is not None:
            relation = "below" if p_value < 0.05 else "at or above"
            findings.append(
                f"The ANOVA p-value is {_format_number(p_value)}, {relation} the 0.05 decision threshold."
            )

        if f_stat is not None:
            findings.append(
                f"The verified F statistic is {_format_number(f_stat)}."
            )

        if eta_squared is not None:
            findings.append(
                f"The reported eta-squared effect size is {_format_number(eta_squared)}."
            )

        if sample_size is not None:
            findings.append(
                f"The analysis used {_format_number(sample_size)} observations."
            )

        if isinstance(groups, Mapping) and groups:
            group_means: list[tuple[str, float]] = []
            for name, stats in groups.items():
                if isinstance(stats, Mapping):
                    mean = _number(stats.get("mean"))
                    if mean is not None:
                        group_means.append((str(name), mean))
            if group_means:
                group_means.sort(key=lambda item: item[1])
                low_name, low_mean = group_means[0]
                high_name, high_mean = group_means[-1]
                findings.append(
                    f"Verified group means range from {low_name}={_format_number(low_mean)} "
                    f"to {high_name}={_format_number(high_mean)}."
                )

        return findings

    def _interpret_generic(
        self,
        tool: str,
        result: Mapping[str, Any],
    ) -> list[str]:
        findings: list[str] = []
        preferred = (
            "p_value",
            "statistic",
            "estimate",
            "effect_size",
            "sample_size",
            "r_squared",
            "rmse",
            "mae",
        )
        for key in preferred:
            value = _number(result.get(key))
            if value is not None:
                findings.append(
                    f"{tool} reports {key}={_format_number(value)}."
                )
        if not findings:
            findings.append(
                f"{tool} completed successfully and its deterministic result passed verification."
            )
        return findings

    def _next_action(self, tool: str, result: Mapping[str, Any]) -> str:
        if tool == "one_way_anova":
            p_value = _number(result.get("p_value"))
            if p_value is not None and p_value < 0.05:
                return (
                    "Inspect verified group-level differences and, when the business question "
                    "requires it, run an appropriate post-hoc comparison before acting on a "
                    "specific pair of groups."
                )
            if p_value is not None:
                return (
                    "Do not claim a group-mean difference from this test; review power, sample "
                    "size, and alternative explanations before further action."
                )
        return (
            "Use the verified evidence as the basis for interpretation, and run additional "
            "analysis only if the business decision requires stronger or more specific evidence."
        )
