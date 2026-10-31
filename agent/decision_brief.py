"""Deterministic, provenance-preserving synthesis of verified analytical evidence.

Interpretations describe calculated results, not causality beyond the model's
assumptions. Original executor outputs remain unmodified in the evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from agent.pipeline import PipelineResult
from agent.tool_call_trace import TraceStatus, VerificationStatus


class DecisionBriefError(ValueError):
    """Raised when a final, verified decision brief cannot be produced safely."""


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


def _analytical_payload(tool: str, result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read the engine's nested payload; do not alter the original result."""
    if (result.get("status") == "success" and result.get("method") == tool
            and isinstance(result.get("result"), Mapping)):
        return result["result"]
    return result


def _family_order(tool: str) -> int:
    return {
        "linear_trend": 0,
        "mean_shift_change_point": 1,
        "variance_shift_change_point": 1,
        "one_way_anova": 2,
        "kruskal_wallis_test": 2,
        "mann_whitney_u_test": 2,
        "bayesian_structural_causal_impact": 3,
    }.get(tool, 4)


def _interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    left, right = _number(value[0]), _number(value[1])
    if left is None or right is None or left > right:
        return None
    return left, right


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    call_id: str
    tool: str
    trace_hash: str
    objective: str
    result: Mapping[str, Any]
    source_call_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool": self.tool,
            "trace_hash": self.trace_hash,
            "objective": self.objective,
            "result": dict(self.result),
            "source_call_ids": list(self.source_call_ids),
        }


@dataclass(frozen=True, slots=True)
class EvidencePackage:
    question: str
    items: tuple[EvidenceItem, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"question": self.question, "items": [item.as_dict() for item in self.items]}


@dataclass(frozen=True, slots=True)
class DecisionBrief:
    question: str
    headline: str
    findings: tuple[str, ...]
    caveats: tuple[str, ...]
    next_action: str
    evidence: EvidencePackage

    @property
    def tools_used(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.tool for item in self.evidence.items))

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "headline": self.headline,
            "findings": list(self.findings),
            "caveats": list(self.caveats),
            "next_action": self.next_action,
            "evidence": self.evidence.as_dict(),
            "tools_used": list(self.tools_used),
            "evidence_count": len(self.evidence.items),
        }


class VerifiedDecisionBriefBuilder:
    """A deterministic synthesis from completed, verified traces only."""

    def build(self, pipeline_result: PipelineResult) -> DecisionBrief:
        if not isinstance(pipeline_result, PipelineResult):
            raise DecisionBriefError("'pipeline_result' must be a PipelineResult.")
        if not pipeline_result.steps:
            raise DecisionBriefError("Cannot build a decision brief without executed steps.")
        if not pipeline_result.succeeded:
            raise DecisionBriefError("Decision briefs require a fully completed and verified pipeline.")

        items: list[EvidenceItem] = []
        caveats: list[str] = []
        for step in pipeline_result.steps:
            trace = step.trace
            # A non-verified ancestor may be resolved by a verified descendant;
            # it remains in pipeline history but is not final decision evidence.
            if trace.status is not TraceStatus.COMPLETED or trace.verification_status is not VerificationStatus.VERIFIED:
                continue
            result = trace.deterministic_result
            if not isinstance(result, Mapping):
                raise DecisionBriefError(f"Verified trace {trace.call_id!r} does not expose a mapping result.")
            original = dict(result)
            items.append(EvidenceItem(
                call_id=trace.call_id, tool=trace.tool,
                trace_hash=trace.trace_hash, objective=trace.objective,
                result=original, source_call_ids=trace.source_call_ids,
            ))
            if result.get("status") == "success" and isinstance(result.get("warnings"), (list, tuple)):
                caveats.extend(str(w).strip() for w in result["warnings"] if str(w).strip())
            payload = _analytical_payload(trace.tool, result)
            limitations = payload.get("limitations", ())
            if isinstance(limitations, str):
                limitations = (limitations,)
            if isinstance(limitations, (list, tuple)):
                caveats.extend(str(l).strip() for l in limitations if isinstance(l, str) and l.strip())
            if trace.tool == "bayesian_structural_causal_impact":
                caveats.append(
                    "Causal impact is conditional on model specification and counterfactual assumptions; numerical verification does not independently prove causality."
                )
            if step.report is not None:
                for check in getattr(step.report, "checks", ()) or ():
                    if not getattr(check, "passed", True):
                        message = str(getattr(check, "message", "")).strip()
                        if message:
                            caveats.append(message)
        if not items:
            raise DecisionBriefError("No completed verified evidence is available.")

        # Findings are sorted by analytical purpose, not accidental execution order.
        sorted_items = sorted(enumerate(items), key=lambda x: (_family_order(x[1].tool), x[0]))
        findings: list[str] = []
        summaries: list[str] = []
        for _, item in sorted_items:
            payload = _analytical_payload(item.tool, item.result)
            findings.extend(self._interpret(item.tool, payload))
            summary = self._summary(item.tool, payload)
            if summary:
                summaries.append(summary)
        if not findings:
            findings = [f"{len(items)} verified analytical result(s) are available for inspection."]
        headline = (
            "; ".join(summaries[:4]) + "."
            if summaries else "VERA produced a verified analytical result."
        )
        return DecisionBrief(
            question=pipeline_result.question,
            headline=headline,
            findings=tuple(dict.fromkeys(findings)),
            caveats=tuple(dict.fromkeys(caveats)),
            next_action=self._combined_next_action(items),
            evidence=EvidencePackage(pipeline_result.question, tuple(items)),
        )

    def _summary(self, tool: str, result: Mapping[str, Any]) -> str:
        if tool == "linear_trend":
            slope = _number(result.get("slope"))
            return ("Verified sales trend is downward" if slope < 0 else
                    "Verified trend is upward" if slope > 0 else
                    "Verified linear trend is flat") if slope is not None else "Verified time trend is available"
        if tool == "mean_shift_change_point":
            return ("A mean-level change was detected" if result.get("detected") is True
                    else "No single mean-level change was detected")
        if tool == "variance_shift_change_point":
            return ("A variance change was detected" if result.get("detected") is True
                    else "No variance change was detected")
        if tool == "one_way_anova":
            p = _number(result.get("p_value"))
            if p is None:
                return "Verified group comparison is available"
            return ("The verified analysis finds statistically significant differences between group means" if p < .05
                    else "The verified analysis does not find statistically significant differences between group means")
        if tool == "kruskal_wallis_test":
            p = _number(result.get("p_value"))
            return ("Significant group-rank differences were detected" if p < .05 else
                    "No significant group-rank differences were detected") if p is not None else "Verified rank comparison is available"
        if tool == "mann_whitney_u_test":
            p = _number(result.get("p_value"))
            return ("Significant two-sample rank evidence was detected" if p < .05 else
                    "No significant two-sample rank evidence was detected") if p is not None else "Verified two-sample comparison is available"
        if tool == "bayesian_structural_causal_impact":
            bounds = _interval(result.get("absolute_effect_interval"))
            if bounds is None or bounds[0] <= 0 <= bounds[1]:
                return "Campaign impact remains uncertain under the counterfactual model"
            return ("The campaign has an estimated positive conditional effect" if bounds[0] > 0 else
                    "The campaign has an estimated negative conditional effect")
        return ""

    def _interpret(self, tool: str, result: Mapping[str, Any]) -> list[str]:
        if tool == "linear_trend":
            slope = _number(result.get("slope"))
            if slope is None:
                return ["The verified time-trend result is available without a numeric slope."]
            direction = "downward" if slope < 0 else "upward" if slope > 0 else "flat"
            text = f"The verified linear trend is {direction}, with a slope of {_format_number(slope)} per period."
            r2 = _number(result.get("r_squared"))
            if r2 is not None and result.get("r_squared_defined") is not False:
                text += f" Its descriptive R-squared is {_format_number(r2)}; this is not a causal explanation."
            return [text]
        if tool in ("mean_shift_change_point", "variance_shift_change_point"):
            kind = "mean level" if tool == "mean_shift_change_point" else "variance"
            if result.get("detected") is not True:
                return [f"The method did not identify a single {kind} shift above its configured threshold."]
            timestamp = result.get("change_timestamp")
            text = f"A verified {kind} shift is detected"
            if timestamp is not None:
                text += f" around {timestamp}"
            candidate = result.get("strongest_candidate")
            if kind == "mean level" and isinstance(candidate, Mapping):
                left, right = _number(candidate.get("left_mean")), _number(candidate.get("right_mean"))
                if left is not None and right is not None:
                    text += f", with the post-shift mean {'below' if right < left else 'above' if right > left else 'equal to'} the pre-shift mean ({_format_number(left)} vs {_format_number(right)})"
            if kind == "variance" and isinstance(result.get("direction"), str):
                text += f" (reported direction: {result['direction']})"
            return [text + "."]
        if tool == "one_way_anova":
            return self._interpret_anova(result)
        if tool == "kruskal_wallis_test":
            p = _number(result.get("p_value"))
            h = _number(result.get("h_statistic"))
            epsilon = _number(result.get("epsilon_squared"))
            parts = ["Kruskal-Wallis compares group distributions/ranks, not proven differences between specific pairs or their means."]
            if p is not None:
                parts.append(f"Kruskal-Wallis p-value {_format_number(p)} {'supports statistically significant rank/distribution differences' if p < .05 else 'does not show statistically significant rank/distribution differences'} across groups at 0.05.")
            if h is not None:
                parts.append(f"Verified H statistic is {_format_number(h)}.")
            if epsilon is not None:
                parts.append(f"Reported epsilon-squared effect size is {_format_number(epsilon)}.")
            groups = result.get("group_statistics")
            if isinstance(groups, Mapping):
                medians = [(str(name), _number(stats.get("median"))) for name, stats in groups.items() if isinstance(stats, Mapping)]
                medians = sorted(((name, median) for name, median in medians if median is not None), key=lambda x: x[1])
                if medians:
                    parts.append(f"Observed group medians range from {medians[0][0]}={_format_number(medians[0][1])} to {medians[-1][0]}={_format_number(medians[-1][1])}; no pairwise significance is implied.")
            return parts
        if tool == "mann_whitney_u_test":
            p = _number(result.get("p_value"))
            first, second = result.get("first_sample", "first sample"), result.get("second_sample", "second sample")
            alternative = result.get("alternative", "two-sided")
            text = f"Mann-Whitney U ({alternative}) compares {first} and {second} via ranks."
            if p is not None:
                text += f" p-value {_format_number(p)} {'indicates significant' if p < .05 else 'does not indicate significant'} rank evidence at 0.05."
            effect = _number(result.get("rank_biserial_effect_size"))
            if effect is not None:
                text += f" Rank-biserial effect size {_format_number(effect)}."
            if result.get("direction"):
                text += f" Reported direction: {result['direction']}."
            return [text]
        if tool == "bayesian_structural_causal_impact":
            bounds = _interval(result.get("absolute_effect_interval"))
            relative_bounds = _interval(result.get("relative_effect_interval_percent"))
            effect = _number(result.get("absolute_effect"))
            relative = _number(result.get("relative_effect_percent"))
            observed = _number(result.get("observed_post_total"))
            counterfactual = _number(result.get("counterfactual_post_total"))
            credibility = _number(result.get("credibility"))
            text = "Under model specification and counterfactual assumptions, the intervention"
            if bounds is None or bounds[0] <= 0 <= bounds[1]:
                text += " has an uncertain estimated effect because its verified interval includes zero or is unavailable"
            elif bounds[0] > 0:
                text += " has an estimated positive effect"
            else:
                text += " has an estimated negative effect"
            if effect is not None:
                text += f" (point estimate {_format_number(effect)})"
            if bounds is not None:
                text += f", interval [{_format_number(bounds[0])}, {_format_number(bounds[1])}]"
            if relative is not None:
                text += f", relative estimate {_format_number(relative)}%"
            if relative_bounds is not None:
                text += f" with relative interval [{_format_number(relative_bounds[0])}, {_format_number(relative_bounds[1])}]%"
            if observed is not None and counterfactual is not None:
                text += f"; observed post total {_format_number(observed)} versus estimated counterfactual {_format_number(counterfactual)}"
            if credibility is not None:
                text += f" at credibility {_format_number(credibility)}"
            rmse = _number(result.get("pre_intervention_rmse"))
            if rmse is not None:
                text += f" (pre-intervention RMSE {_format_number(rmse)})"
            return [text + ". This does not independently prove causality."]
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
            findings.append(f"The ANOVA p-value is {_format_number(p_value)}, {relation} the 0.05 decision threshold.")
        if f_stat is not None:
            findings.append(f"The verified F statistic is {_format_number(f_stat)}.")
        if eta_squared is not None:
            findings.append(f"The reported eta-squared effect size is {_format_number(eta_squared)}.")
        if sample_size is not None:
            findings.append(f"The analysis used {_format_number(sample_size)} observations.")
        if isinstance(groups, Mapping) and groups:
            group_means = []
            for name, stats in groups.items():
                if isinstance(stats, Mapping):
                    mean = _number(stats.get("mean"))
                    if mean is not None:
                        group_means.append((str(name), mean))
            if group_means:
                group_means.sort(key=lambda item: item[1])
                findings.append(f"Verified group means range from {group_means[0][0]}={_format_number(group_means[0][1])} to {group_means[-1][0]}={_format_number(group_means[-1][1])}.")
        return findings

    def _interpret_generic(self, tool: str, result: Mapping[str, Any]) -> list[str]:
        findings: list[str] = []
        for key in ("p_value", "statistic", "estimate", "effect_size", "sample_size", "r_squared", "rmse", "mae"):
            value = _number(result.get(key))
            if value is not None:
                findings.append(f"{tool} reports {key}={_format_number(value)}.")
        if not findings:
            findings.append(f"{tool} completed successfully and its deterministic result passed verification.")
        return findings

    def _combined_next_action(self, items: list[EvidenceItem]) -> str:
        payloads = {item.tool: _analytical_payload(item.tool, item.result) for item in items}
        causal = payloads.get("bayesian_structural_causal_impact")
        if causal is not None:
            bounds = _interval(causal.get("absolute_effect_interval"))
            if bounds is None or bounds[0] <= 0 <= bounds[1]:
                return "Collect more post-intervention evidence or strengthen the control series before claiming a campaign effect."
            return "Validate the counterfactual assumptions and assess cost-effectiveness before considering wider rollout; an estimated effect is not causal proof."
        for tool in ("linear_trend", "mean_shift_change_point"):
            payload = payloads.get(tool)
            if payload is not None and ((tool == "linear_trend" and (_number(payload.get("slope")) or 0) < 0) or (tool == "mean_shift_change_point" and payload.get("detected") is True)):
                return "Investigate business drivers around the verified decline and its possible change period before attributing a cause."
        for tool in ("one_way_anova", "kruskal_wallis_test", "mann_whitney_u_test"):
            payload = payloads.get(tool)
            if payload is not None:
                p = _number(payload.get("p_value"))
                if p is not None and p < .05:
                    return "Inspect verified group-level differences and, when the business question requires it, run an appropriate post-hoc comparison before acting on a specific pair of groups."
                if p is not None:
                    return "Do not claim a group-mean difference from this test; review power, sample size, and alternative explanations before further action."
        return "Use the verified evidence as the basis for interpretation, and run additional analysis only if the business decision requires stronger or more specific evidence."
