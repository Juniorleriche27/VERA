from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from agent.tool_call_trace import ToolCallTrace


class DataBindingError(ValueError):
    """Raised when a planned tool call cannot be bound to local dataset values."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataBindingError(f"'{field}' must be a non-empty string.")
    return value.strip()


def _to_frame(dataset: Any) -> pd.DataFrame:
    if isinstance(dataset, pd.DataFrame):
        return dataset.copy(deep=False)
    if isinstance(dataset, Mapping):
        try:
            return pd.DataFrame(dict(dataset))
        except Exception as exc:
            raise DataBindingError("Dataset mapping could not be converted to a table.") from exc
    if isinstance(dataset, Sequence) and not isinstance(dataset, (str, bytes, bytearray)):
        try:
            return pd.DataFrame(list(dataset))
        except Exception as exc:
            raise DataBindingError("Dataset rows could not be converted to a table.") from exc
    raise DataBindingError("Dataset must be a pandas DataFrame, column mapping, or row sequence.")


def _native(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        missing = pd.isna(value)
    except Exception:
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return value


def _column(frame: pd.DataFrame, name: Any, field: str) -> str:
    column = _required_text(name, field)
    if column not in frame.columns:
        raise DataBindingError(f"Unknown dataset column {column!r} for '{field}'.")
    return column


_GROUP_HINTS = (
    "group",
    "grouping",
    "category",
    "segment",
    "region",
    "class",
    "treatment",
    "cohort",
)
_VALUE_HINTS = (
    "value",
    "measure",
    "sales",
    "revenue",
    "amount",
    "price",
    "score",
    "metric",
    "outcome",
    "target",
)


def _semantic_column_refs(
    frame: pd.DataFrame,
    semantic: Mapping[str, Any],
) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, value in semantic.items():
        if isinstance(value, str) and value in frame.columns and value not in seen:
            refs.append((str(key).strip().lower(), value))
            seen.add(value)
    return refs


def _hinted_column(
    refs: Sequence[tuple[str, str]],
    hints: Sequence[str],
    *,
    exclude: str | None = None,
) -> str | None:
    matches = [
        column
        for key, column in refs
        if column != exclude and any(hint in key.lower() for hint in hints)
    ]
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None


def _resolve_grouped_columns(
    frame: pd.DataFrame,
    semantic: Mapping[str, Any],
) -> tuple[str, str]:
    group_candidate = semantic.get("grouping_column") or semantic.get("group_column")
    value_candidate = semantic.get("value_column") or semantic.get("measure_column")
    refs = _semantic_column_refs(frame, semantic)

    if group_candidate is None:
        group_candidate = _hinted_column(refs, _GROUP_HINTS)
    if value_candidate is None:
        value_candidate = _hinted_column(
            refs,
            _VALUE_HINTS,
            exclude=group_candidate if isinstance(group_candidate, str) else None,
        )

    if group_candidate is None:
        categorical = [
            column
            for _, column in refs
            if column != value_candidate
            and not pd.api.types.is_numeric_dtype(frame[column])
        ]
        if len(categorical) == 1:
            group_candidate = categorical[0]

    if value_candidate is None:
        numeric = [
            column
            for _, column in refs
            if column != group_candidate
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if len(numeric) == 1:
            value_candidate = numeric[0]

    if group_candidate is None or value_candidate is None:
        raise DataBindingError(
            "Could not deterministically resolve grouped analysis columns. "
            "Provide semantic bindings for 'grouping_column' and 'value_column'."
        )

    group_column = _column(frame, group_candidate, "grouping_column")
    value_column = _column(frame, value_candidate, "value_column")
    if group_column == value_column:
        raise DataBindingError(
            "'grouping_column' and 'value_column' must reference different columns."
        )
    return group_column, value_column


def _resolve_value_column(
    frame: pd.DataFrame,
    semantic: Mapping[str, Any],
) -> str:
    candidate = (
        semantic.get("value_column")
        or semantic.get("measure_column")
        or semantic.get("column")
    )
    refs = _semantic_column_refs(frame, semantic)

    if candidate is None:
        candidate = _hinted_column(refs, _VALUE_HINTS)

    if candidate is None:
        numeric = [
            column
            for _, column in refs
            if pd.api.types.is_numeric_dtype(frame[column])
        ]
        if len(numeric) == 1:
            candidate = numeric[0]

    if candidate is None and len(refs) == 1:
        candidate = refs[0][1]

    if candidate is None:
        raise DataBindingError(
            "Could not deterministically resolve a value column. "
            "Provide a semantic 'value_column' binding."
        )

    return _column(frame, candidate, "value_column")


@dataclass(frozen=True, slots=True)
class BoundToolCall:
    call_id: str
    tool: str
    parameters: dict[str, Any]
    bindings: dict[str, Any]
    source_bindings: dict[str, Any]

    def executor_kwargs(self) -> dict[str, Any]:
        payload = dict(self.parameters)
        overlap = set(payload).intersection(self.bindings)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise DataBindingError(f"Parameters and bound data overlap: {names}.")
        payload.update(self.bindings)
        return payload


class DatasetBinder:
    """
    Converts semantic column references chosen by an LLM into local raw values.

    Raw dataset values are returned only in BoundToolCall.bindings. The original
    ToolCallTrace remains unchanged, so raw rows are not added to the model-facing
    audit trace.
    """

    GROUPED_TOOLS = frozenset({"one_way_anova", "kruskal_wallis_test"})
    VALUE_TOOLS = frozenset({"descriptive_statistics"})

    def bind(self, trace: ToolCallTrace, dataset: Any) -> BoundToolCall:
        if not isinstance(trace, ToolCallTrace):
            raise DataBindingError("'trace' must be a ToolCallTrace.")

        frame = _to_frame(dataset)
        semantic = trace.bindings
        parameters = trace.parameters

        if trace.tool in self.GROUPED_TOOLS:
            bound = self._bind_grouped(frame, semantic)
        elif trace.tool == "mann_whitney_u_test":
            bound = self._bind_mann_whitney(frame, semantic)
        elif trace.tool in self.VALUE_TOOLS:
            bound = self._bind_values(frame, semantic)
        else:
            bound = self._bind_generic(frame, semantic)

        return BoundToolCall(
            call_id=trace.call_id,
            tool=trace.tool,
            parameters=parameters,
            bindings=bound,
            source_bindings=semantic,
        )

    def _bind_grouped(
        self,
        frame: pd.DataFrame,
        semantic: Mapping[str, Any],
    ) -> dict[str, Any]:
        group_column, value_column = _resolve_grouped_columns(frame, semantic)

        groups: dict[str, list[Any]] = {}
        for group_value, value in zip(frame[group_column], frame[value_column]):
            native_group = _native(group_value)
            if native_group is None:
                continue
            key = str(native_group).strip()
            if not key:
                continue
            groups.setdefault(key, []).append(_native(value))

        if len(groups) < 2:
            raise DataBindingError(
                "Grouped analysis requires at least two non-empty groups."
            )
        return {"data": groups}

    def _bind_values(
        self,
        frame: pd.DataFrame,
        semantic: Mapping[str, Any],
    ) -> dict[str, Any]:
        value_column = _resolve_value_column(frame, semantic)
        return {"values": [_native(value) for value in frame[value_column]]}

    def _bind_mann_whitney(
        self,
        frame: pd.DataFrame,
        semantic: Mapping[str, Any],
    ) -> dict[str, Any]:
        group_column, value_column = _resolve_grouped_columns(frame, semantic)

        requested_a = semantic.get("group_a")
        requested_b = semantic.get("group_b")
        observed = [
            _native(value)
            for value in pd.unique(frame[group_column])
            if _native(value) is not None
        ]

        if requested_a is None or requested_b is None:
            if len(observed) != 2:
                raise DataBindingError(
                    "Mann-Whitney binding needs 'group_a' and 'group_b' when "
                    "the grouping column does not contain exactly two groups."
                )
            requested_a, requested_b = observed

        if requested_a == requested_b:
            raise DataBindingError("'group_a' and 'group_b' must be different.")

        sample_a = [
            _native(value)
            for group, value in zip(frame[group_column], frame[value_column])
            if _native(group) == requested_a
        ]
        sample_b = [
            _native(value)
            for group, value in zip(frame[group_column], frame[value_column])
            if _native(group) == requested_b
        ]

        if not sample_a or not sample_b:
            raise DataBindingError("Selected Mann-Whitney groups must exist in the dataset.")

        return {"sample_a": sample_a, "sample_b": sample_b}

    def _bind_generic(
        self,
        frame: pd.DataFrame,
        semantic: Mapping[str, Any],
    ) -> dict[str, Any]:
        bound: dict[str, Any] = {}

        for key, value in semantic.items():
            if key.endswith("_column") and isinstance(value, str):
                column = _column(frame, value, key)
                output_key = key[:-7]
                if output_key in {"value", "measure"}:
                    output_key = "values"
                bound[output_key] = [_native(item) for item in frame[column]]
            else:
                bound[key] = value

        return bound
