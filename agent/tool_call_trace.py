from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Mapping


class ToolCallTraceError(ValueError):
    """Raised when a tool-call trace violates the audit contract."""


class TraceStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class VerificationStatus(str, Enum):
    NOT_VERIFIED = "not_verified"
    VERIFIED = "verified"
    REJECTED = "rejected"
    NEEDS_MORE_ANALYSIS = "needs_more_analysis"


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolCallTraceError(f"'{field}' must be a non-empty string.")
    return value.strip()


def _optional_text(value: Any, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ToolCallTraceError(f"'{field}' must be a string.")
    return value.strip()


def _canonical_json(value: Any, field: str) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ToolCallTraceError(
            f"'{field}' must contain only finite JSON-compatible values."
        ) from exc


def _decode_json(value: str) -> Any:
    return json.loads(value)


def _coerce_status(value: TraceStatus | str) -> TraceStatus:
    try:
        return value if isinstance(value, TraceStatus) else TraceStatus(value)
    except ValueError as exc:
        raise ToolCallTraceError(f"Unknown trace status: {value!r}.") from exc


def _coerce_verification(
    value: VerificationStatus | str,
) -> VerificationStatus:
    try:
        return (
            value
            if isinstance(value, VerificationStatus)
            else VerificationStatus(value)
        )
    except ValueError as exc:
        raise ToolCallTraceError(
            f"Unknown verification status: {value!r}."
        ) from exc


@dataclass(frozen=True, slots=True)
class ToolCallTrace:
    """Immutable, JSON-safe audit record for one deterministic tool call."""

    call_id: str
    tool: str
    objective: str
    reason: str
    _parameters_json: str
    _bindings_json: str
    status: TraceStatus = TraceStatus.PLANNED
    _result_json: str | None = None
    verification_status: VerificationStatus = VerificationStatus.NOT_VERIFIED
    next_decision: str = ""
    source_call_ids: tuple[str, ...] = ()
    execution_error: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "call_id", _required_text(self.call_id, "call_id"))
        object.__setattr__(self, "tool", _required_text(self.tool, "tool"))
        object.__setattr__(
            self, "objective", _required_text(self.objective, "objective")
        )
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
        object.__setattr__(self, "status", _coerce_status(self.status))
        object.__setattr__(
            self,
            "verification_status",
            _coerce_verification(self.verification_status),
        )
        object.__setattr__(
            self,
            "next_decision",
            _optional_text(self.next_decision, "next_decision"),
        )
        object.__setattr__(
            self,
            "execution_error",
            _optional_text(self.execution_error, "execution_error"),
        )

        try:
            parameters = _decode_json(self._parameters_json)
            bindings = _decode_json(self._bindings_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ToolCallTraceError(
                "Trace payloads must use canonical JSON strings."
            ) from exc

        if not isinstance(parameters, dict):
            raise ToolCallTraceError("'parameters' must be a mapping.")
        if not isinstance(bindings, dict):
            raise ToolCallTraceError("'bindings' must be a mapping.")

        normalized_sources = tuple(
            _required_text(value, "source_call_ids")
            for value in self.source_call_ids
        )
        if len(set(normalized_sources)) != len(normalized_sources):
            raise ToolCallTraceError("'source_call_ids' must be unique.")
        if self.call_id in normalized_sources:
            raise ToolCallTraceError("A trace cannot depend on itself.")
        object.__setattr__(self, "source_call_ids", normalized_sources)

        if self.status is TraceStatus.FAILED and not self.execution_error:
            raise ToolCallTraceError("Failed traces require 'execution_error'.")
        if self.status is not TraceStatus.FAILED and self.execution_error:
            raise ToolCallTraceError(
                "'execution_error' is only valid for failed traces."
            )
        if (
            self.verification_status is not VerificationStatus.NOT_VERIFIED
            and self.status is not TraceStatus.COMPLETED
        ):
            raise ToolCallTraceError(
                "Only completed traces may carry a verification decision."
            )

    @classmethod
    def planned(
        cls,
        *,
        call_id: str,
        tool: str,
        objective: str,
        reason: str,
        parameters: Mapping[str, Any] | None = None,
        bindings: Mapping[str, Any] | None = None,
        source_call_ids: tuple[str, ...] | list[str] = (),
    ) -> "ToolCallTrace":
        if parameters is not None and not isinstance(parameters, Mapping):
            raise ToolCallTraceError("'parameters' must be a mapping.")
        if bindings is not None and not isinstance(bindings, Mapping):
            raise ToolCallTraceError("'bindings' must be a mapping.")

        return cls(
            call_id=call_id,
            tool=tool,
            objective=objective,
            reason=reason,
            _parameters_json=_canonical_json(
                dict(parameters or {}), "parameters"
            ),
            _bindings_json=_canonical_json(dict(bindings or {}), "bindings"),
            source_call_ids=tuple(source_call_ids),
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _decode_json(self._parameters_json)

    @property
    def bindings(self) -> dict[str, Any]:
        return _decode_json(self._bindings_json)

    @property
    def deterministic_result(self) -> Any | None:
        if self._result_json is None:
            return None
        return _decode_json(self._result_json)

    def mark_running(self) -> "ToolCallTrace":
        if self.status is not TraceStatus.PLANNED:
            raise ToolCallTraceError(
                "Only a planned trace can transition to running."
            )
        return replace(self, status=TraceStatus.RUNNING)

    def complete(self, deterministic_result: Any) -> "ToolCallTrace":
        if self.status not in {TraceStatus.PLANNED, TraceStatus.RUNNING}:
            raise ToolCallTraceError(
                "Only a planned or running trace can complete."
            )
        return replace(
            self,
            status=TraceStatus.COMPLETED,
            _result_json=_canonical_json(
                deterministic_result, "deterministic_result"
            ),
            execution_error="",
        )

    def fail(self, error: str) -> "ToolCallTrace":
        if self.status not in {TraceStatus.PLANNED, TraceStatus.RUNNING}:
            raise ToolCallTraceError(
                "Only a planned or running trace can fail."
            )
        return replace(
            self,
            status=TraceStatus.FAILED,
            execution_error=_required_text(error, "error"),
        )

    def verify(
        self,
        verification_status: VerificationStatus | str,
        *,
        next_decision: str = "",
    ) -> "ToolCallTrace":
        if self.status is not TraceStatus.COMPLETED:
            raise ToolCallTraceError(
                "A trace must complete before verification."
            )

        decision = _coerce_verification(verification_status)
        if decision is VerificationStatus.NOT_VERIFIED:
            raise ToolCallTraceError(
                "Verification must produce an explicit decision."
            )

        normalized_next = _optional_text(next_decision, "next_decision")
        if (
            decision
            in {
                VerificationStatus.REJECTED,
                VerificationStatus.NEEDS_MORE_ANALYSIS,
            }
            and not normalized_next
        ):
            raise ToolCallTraceError(
                "Rejected or incomplete evidence requires 'next_decision'."
            )

        return replace(
            self,
            verification_status=decision,
            next_decision=normalized_next,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool": self.tool,
            "objective": self.objective,
            "reason": self.reason,
            "parameters": self.parameters,
            "bindings": self.bindings,
            "status": self.status.value,
            "deterministic_result": self.deterministic_result,
            "verification_status": self.verification_status.value,
            "next_decision": self.next_decision,
            "source_call_ids": list(self.source_call_ids),
            "execution_error": self.execution_error,
            "trace_hash": self.trace_hash,
        }

    @property
    def trace_hash(self) -> str:
        payload = {
            "call_id": self.call_id,
            "tool": self.tool,
            "objective": self.objective,
            "reason": self.reason,
            "parameters": self.parameters,
            "bindings": self.bindings,
            "status": self.status.value,
            "deterministic_result": self.deterministic_result,
            "verification_status": self.verification_status.value,
            "next_decision": self.next_decision,
            "source_call_ids": list(self.source_call_ids),
            "execution_error": self.execution_error,
        }
        canonical = _canonical_json(payload, "trace")
        return sha256(canonical.encode("utf-8")).hexdigest()
