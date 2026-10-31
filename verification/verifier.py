from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Mapping

from agent.tool_call_trace import (
    ToolCallTrace,
    TraceStatus,
    VerificationStatus,
)


class VerifierError(ValueError):
    """Raised when a trace cannot be verified deterministically."""


class CheckSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    code: str
    passed: bool
    severity: CheckSeverity
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "passed": self.passed,
            "severity": self.severity.value,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class VerificationReport:
    call_id: str
    trace_hash: str
    status: VerificationStatus
    next_decision: str
    checks: tuple[VerificationCheck, ...]

    @property
    def passed(self) -> bool:
        return self.status is VerificationStatus.VERIFIED

    @property
    def failed_checks(self) -> tuple[VerificationCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "trace_hash": self.trace_hash,
            "status": self.status.value,
            "next_decision": self.next_decision,
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
        }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if isfinite(number) else None


def _iter_probability_fields(result: Mapping[str, Any]):
    for key, value in result.items():
        normalized = key.lower()
        if normalized == "p_value" or normalized.endswith("_p_value"):
            yield key, value
        elif normalized in {"p_values", "adjusted_p_values"} and isinstance(
            value, list
        ):
            for index, item in enumerate(value):
                yield f"{key}[{index}]", item


def _explicit_negative_flag(result: Mapping[str, Any]) -> tuple[str, str] | None:
    negative_flags = (
        ("assumptions_satisfied", "Statistical assumptions are not satisfied."),
        ("diagnostics_passed", "Model diagnostics did not pass."),
        ("fit_adequate", "Model fit is not adequate."),
        ("verification_ready", "Result reports that it is not verification-ready."),
    )
    for key, message in negative_flags:
        if key in result and result[key] is False:
            return key, message
    return None


def _extract_verification_payload(
    trace: ToolCallTrace, result: Mapping[str, Any]
) -> tuple[Mapping[str, Any], list[VerificationCheck], Mapping[str, Any]]:
    """Read the engine's nested analytical fields without rewriting evidence."""
    checks: list[VerificationCheck] = []
    if not {"status", "method", "result"}.intersection(result):
        return result, checks, {}
    # An engine envelope is identifiable by its status/method metadata.
    if "status" not in result and "method" not in result:
        return result, checks, {}
    status_ok = result.get("status") == "success"
    method_ok = result.get("method") == trace.tool
    payload = result.get("result")
    payload_ok = isinstance(payload, Mapping)
    for code, ok, message in (
        ("engine_status", status_ok, "Engine status must be 'success'."),
        ("engine_method", method_ok, "Engine method must match the trace tool."),
        ("engine_result", payload_ok, "Engine analytical result must be a mapping."),
    ):
        checks.append(VerificationCheck(code, ok, CheckSeverity.CRITICAL, message))
    warnings = result.get("warnings", [])
    checks.append(VerificationCheck(
        "engine_warnings", not bool(warnings), CheckSeverity.WARNING,
        "Engine warnings: " + str(warnings) if warnings else "No engine warnings.",
    ))
    diagnostics = result.get("diagnostics", {})
    return payload if payload_ok else {}, checks, diagnostics if isinstance(diagnostics, Mapping) else {}


def _check_causal_structure(
    result: Mapping[str, Any], *, tool: str
) -> list[VerificationCheck]:
    checks: list[VerificationCheck] = []
    if "causal_interpretation" in result:
        checks.append(VerificationCheck(
            "causal_assumptions_scope", True, CheckSeverity.INFO,
            "Numerical consistency does not establish causal validity independently of model and counterfactual assumptions.",
        ))
    fields = ("sample_size", "pre_intervention_count", "post_intervention_count", "intervention_index")
    if tool == "bayesian_structural_causal_impact":
        present = all(key in result for key in fields)
        numbers = [_number(result[key]) for key in fields] if present else []
        valid = present and all(v is not None and v >= 0 and v.is_integer() for v in numbers)
        if valid:
            sample, pre, post, index = numbers
            valid = pre + post == sample and index == pre and pre > 0 and post > 0
        checks.append(VerificationCheck(
            "causal_sample_structure", valid, CheckSeverity.CRITICAL,
            "Causal counts must exist, add up and intervention index must equal pre-intervention count.",
        ))
    return checks


def _check_change_index(result: Mapping[str, Any]) -> VerificationCheck | None:
    if result.get("detected") is not True or "change_index" not in result or "sample_size" not in result:
        return None
    index = _number(result["change_index"])
    sample = _number(result["sample_size"])
    valid = (index is not None and sample is not None and index.is_integer()
             and sample.is_integer() and 0 <= index < sample)
    return VerificationCheck(
        "change_index_bounds", valid, CheckSeverity.CRITICAL,
        "Detected change index must be a valid zero-based position in the sample.",
    )


class DeterministicVerifier:
    """Rule-based verifier for deterministic analytical tool outputs."""

    def __init__(
        self,
        *,
        missingness_warning_threshold: float = 0.25,
        minimum_sample_size: int = 3,
    ) -> None:
        if not 0.0 <= missingness_warning_threshold <= 1.0:
            raise VerifierError(
                "'missingness_warning_threshold' must be between 0 and 1."
            )
        if minimum_sample_size < 1:
            raise VerifierError("'minimum_sample_size' must be at least 1.")

        self.missingness_warning_threshold = float(
            missingness_warning_threshold
        )
        self.minimum_sample_size = int(minimum_sample_size)

    def evaluate(self, trace: ToolCallTrace) -> VerificationReport:
        if not isinstance(trace, ToolCallTrace):
            raise VerifierError("'trace' must be a ToolCallTrace.")
        if trace.status is not TraceStatus.COMPLETED:
            raise VerifierError(
                "Only completed deterministic tool calls can be verified."
            )

        result = trace.deterministic_result
        checks: list[VerificationCheck] = []

        if result is None:
            checks.append(
                VerificationCheck(
                    code="result_present",
                    passed=False,
                    severity=CheckSeverity.CRITICAL,
                    message="Deterministic result is missing.",
                )
            )
            return self._report(
                trace,
                VerificationStatus.REJECTED,
                "rerun_tool_with_valid_output",
                checks,
            )

        checks.append(
            VerificationCheck(
                code="result_present",
                passed=True,
                severity=CheckSeverity.INFO,
                message="Deterministic result is present.",
            )
        )

        if not isinstance(result, Mapping):
            checks.append(
                VerificationCheck(
                    code="structured_result",
                    passed=True,
                    severity=CheckSeverity.INFO,
                    message="Result is JSON-safe but not a mapping; generic checks only.",
                )
            )
            return self._report(
                trace,
                VerificationStatus.VERIFIED,
                "interpret_result",
                checks,
            )

        payload, envelope_checks, diagnostics = _extract_verification_payload(trace, result)
        checks.extend(envelope_checks)
        if any(not check.passed and check.severity is CheckSeverity.CRITICAL for check in envelope_checks):
            return self._report(trace, VerificationStatus.REJECTED, "rerun_tool_with_valid_output", checks)

        probability_failures = []
        for field, value in _iter_probability_fields(payload):
            number = _number(value)
            if number is None or not 0.0 <= number <= 1.0:
                probability_failures.append(field)

        checks.append(
            VerificationCheck(
                code="probability_bounds",
                passed=not probability_failures,
                severity=CheckSeverity.CRITICAL,
                message=(
                    "Probability-like outputs are within [0, 1]."
                    if not probability_failures
                    else "Invalid probability-like outputs: "
                    + ", ".join(probability_failures)
                ),
            )
        )
        if probability_failures:
            return self._report(
                trace,
                VerificationStatus.REJECTED,
                "rerun_tool_with_valid_output",
                checks,
            )

        interval_check = self._check_intervals(payload)
        checks.append(interval_check)
        structural_checks = _check_causal_structure(payload, tool=trace.tool)
        change_check = _check_change_index(payload)
        if change_check is not None:
            structural_checks.append(change_check)
        checks.extend(structural_checks)
        if not interval_check.passed or any(
            not check.passed and check.severity is CheckSeverity.CRITICAL
            for check in structural_checks
        ):
            return self._report(
                trace,
                VerificationStatus.REJECTED,
                "rerun_tool_with_valid_output",
                checks,
            )

        missingness_check = self._check_missingness(payload)
        if missingness_check is not None:
            checks.append(missingness_check)
            if not missingness_check.passed:
                return self._report(
                    trace,
                    VerificationStatus.NEEDS_MORE_ANALYSIS,
                    "run_data_quality_and_missingness_analysis",
                    checks,
                )

        sample_check = self._check_sample_size(payload)
        if sample_check is not None:
            checks.append(sample_check)
            if not sample_check.passed:
                return self._report(
                    trace,
                    VerificationStatus.NEEDS_MORE_ANALYSIS,
                    "collect_more_data_or_reduce_scope",
                    checks,
                )

        negative_flag = _explicit_negative_flag(payload) or _explicit_negative_flag(diagnostics)
        if negative_flag is not None:
            key, message = negative_flag
            checks.append(
                VerificationCheck(
                    code=f"explicit_flag:{key}",
                    passed=False,
                    severity=CheckSeverity.WARNING,
                    message=message,
                )
            )
            return self._report(
                trace,
                VerificationStatus.NEEDS_MORE_ANALYSIS,
                "run_robustness_check",
                checks,
            )

        checks.append(
            VerificationCheck(
                code="explicit_diagnostics",
                passed=True,
                severity=CheckSeverity.INFO,
                message="No explicit failed diagnostic flag was reported.",
            )
        )

        return self._report(
            trace,
            VerificationStatus.VERIFIED,
            "interpret_result",
            checks,
        )

    def verify(self, trace: ToolCallTrace) -> tuple[ToolCallTrace, VerificationReport]:
        report = self.evaluate(trace)
        verified_trace = trace.verify(
            report.status,
            next_decision=report.next_decision,
        )
        final_report = VerificationReport(
            call_id=report.call_id,
            trace_hash=verified_trace.trace_hash,
            status=report.status,
            next_decision=report.next_decision,
            checks=report.checks,
        )
        return verified_trace, final_report

    def _check_missingness(
        self, result: Mapping[str, Any]
    ) -> VerificationCheck | None:
        missing = _number(result.get("missing_count"))
        total = _number(result.get("total_count"))
        if missing is None or total is None or total <= 0.0:
            return None

        rate = missing / total
        return VerificationCheck(
            code="missingness",
            passed=rate <= self.missingness_warning_threshold,
            severity=CheckSeverity.WARNING,
            message=(
                f"Missingness rate {rate:.3f} is within threshold "
                f"{self.missingness_warning_threshold:.3f}."
                if rate <= self.missingness_warning_threshold
                else f"Missingness rate {rate:.3f} exceeds threshold "
                f"{self.missingness_warning_threshold:.3f}."
            ),
        )

    def _check_sample_size(
        self, result: Mapping[str, Any]
    ) -> VerificationCheck | None:
        candidate_keys = (
            "sample_size",
            "n_observations",
            "observation_count",
            "exceedance_count",
        )
        for key in candidate_keys:
            value = _number(result.get(key))
            if value is None:
                continue
            return VerificationCheck(
                code=f"sample_size:{key}",
                passed=value >= self.minimum_sample_size,
                severity=CheckSeverity.WARNING,
                message=(
                    f"{key}={int(value)} satisfies the minimum "
                    f"{self.minimum_sample_size}."
                    if value >= self.minimum_sample_size
                    else f"{key}={int(value)} is below the minimum "
                    f"{self.minimum_sample_size}."
                ),
            )
        return None

    @staticmethod
    def _check_intervals(result: Mapping[str, Any]) -> VerificationCheck:
        pairs = (
            ("ci_lower", "ci_upper"),
            ("confidence_interval_lower", "confidence_interval_upper"),
            ("credible_interval_lower", "credible_interval_upper"),
            ("lower_bound", "upper_bound"),
        )
        invalid = []
        for lower_key, upper_key in pairs:
            if lower_key not in result or upper_key not in result:
                continue
            lower = _number(result[lower_key])
            upper = _number(result[upper_key])
            if lower is None or upper is None or lower > upper:
                invalid.append(f"{lower_key}/{upper_key}")

        scalar_intervals = (
            "absolute_effect_interval", "relative_effect_interval_percent",
        )
        collection_intervals = (
            "counterfactual_post_interval", "point_effect_interval",
        )

        def valid_pair(value: Any) -> bool:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return False
            lower, upper = (_number(value[0]), _number(value[1]))
            return lower is not None and upper is not None and lower <= upper

        for name in scalar_intervals:
            if name in result and not valid_pair(result[name]):
                invalid.append(name)
        for name in collection_intervals:
            if name not in result:
                continue
            intervals = result[name]
            if not isinstance(intervals, (list, tuple)) or not intervals or any(
                not valid_pair(pair) for pair in intervals
            ):
                invalid.append(name)

        return VerificationCheck(
            code="interval_order",
            passed=not invalid,
            severity=CheckSeverity.CRITICAL,
            message=(
                "Reported intervals have valid lower/upper ordering."
                if not invalid
                else "Invalid interval ordering: " + ", ".join(invalid)
            ),
        )

    @staticmethod
    def _report(
        trace: ToolCallTrace,
        status: VerificationStatus,
        next_decision: str,
        checks: list[VerificationCheck],
    ) -> VerificationReport:
        return VerificationReport(
            call_id=trace.call_id,
            trace_hash=trace.trace_hash,
            status=status,
            next_decision=next_decision,
            checks=tuple(checks),
        )
