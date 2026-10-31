from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd

from agent.tool_call_trace import ToolCallTrace

if TYPE_CHECKING:
    # Local import at runtime avoids agent -> backend.__init__ -> app -> agent cycle.
    from backend.data_preparation import TimeSeriesView


class DataBindingError(ValueError):
    """Raised when a planned tool call cannot be bound to local dataset values."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataBindingError(f"'{field}' must be a non-empty string.")
    return value.strip()


def _to_frame(dataset: Any) -> pd.DataFrame:
    if isinstance(dataset, pd.DataFrame):
        # Deep copy prevents the binder from mutating the caller's source frame,
        # which is stored by reference in app.state.datasets.
        return dataset.copy(deep=True)
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



def _frame_column_lookup(frame: pd.DataFrame) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for column in frame.columns:
        name = str(column)
        lookup.setdefault(name.casefold(), []).append(name)
    return lookup


def _resolve_frame_column_reference(
    frame: pd.DataFrame,
    candidate: Any,
) -> str | None:
    if not isinstance(candidate, str):
        return None
    raw = candidate.strip()
    if not raw:
        return None
    if raw in frame.columns:
        return raw

    matches = _frame_column_lookup(frame).get(raw.casefold(), [])
    return matches[0] if len(matches) == 1 else None


def _semantic_column_refs(
    frame: pd.DataFrame,
    semantic: Mapping[str, Any],
) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    seen: set[str] = set()

    for key, value in semantic.items():
        resolved = _resolve_frame_column_reference(frame, value)
        if resolved is None:
            resolved = _resolve_frame_column_reference(frame, key)
        if resolved is not None and resolved not in seen:
            refs.append((str(key).strip().lower(), resolved))
            seen.add(resolved)

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
    group_candidate = (
        semantic.get("grouping_column")
        or semantic.get("group_column")
        or semantic.get("group_col")
    )
    value_candidate = (
        semantic.get("value_column")
        or semantic.get("measure_column")
        or semantic.get("value_col")
        or semantic.get("metric_column")
    )
    refs = _semantic_column_refs(frame, semantic)

    for candidate in (group_candidate, value_candidate):
        if candidate is not None and _resolve_frame_column_reference(frame, candidate) is None:
            raise DataBindingError(
                f"Could not deterministically resolve grouped analysis columns: unknown dataset column {candidate!r}."
            )

    group_candidate = _resolve_frame_column_reference(frame, group_candidate)
    value_candidate = _resolve_frame_column_reference(frame, value_candidate)

    if group_candidate is None:
        group_candidate = _hinted_column(refs, _GROUP_HINTS)
    if value_candidate is None:
        value_candidate = _hinted_column(
            refs,
            _VALUE_HINTS,
            exclude=group_candidate if isinstance(group_candidate, str) else None,
        )

    if group_candidate is None:
        categorical_refs = [
            column
            for _, column in refs
            if column != value_candidate
            and not pd.api.types.is_numeric_dtype(frame[column])
        ]
        if len(categorical_refs) == 1:
            group_candidate = categorical_refs[0]

    if value_candidate is None:
        numeric_refs = [
            column
            for _, column in refs
            if column != group_candidate
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if len(numeric_refs) == 1:
            value_candidate = numeric_refs[0]

    if group_candidate is None:
        categorical_frame = [
            str(column)
            for column in frame.columns
            if str(column) != value_candidate
            and not pd.api.types.is_numeric_dtype(frame[column])
        ]
        if len(categorical_frame) == 1:
            group_candidate = categorical_frame[0]

    if value_candidate is None:
        numeric_frame = [
            str(column)
            for column in frame.columns
            if str(column) != group_candidate
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if len(numeric_frame) == 1:
            value_candidate = numeric_frame[0]

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
    raw_candidate = (
        semantic.get("value_column")
        or semantic.get("measure_column")
        or semantic.get("value_col")
        or semantic.get("metric_column")
        or semantic.get("column")
    )
    refs = _semantic_column_refs(frame, semantic)

    candidate = _resolve_frame_column_reference(frame, raw_candidate)
    if raw_candidate is not None and candidate is None:
        raise DataBindingError(
            f"Unknown dataset column: {raw_candidate!r}."
        )

    if candidate is None:
        candidate = _hinted_column(refs, _VALUE_HINTS)

    if candidate is None:
        numeric_refs = [
            column
            for _, column in refs
            if pd.api.types.is_numeric_dtype(frame[column])
        ]
        if len(numeric_refs) == 1:
            candidate = numeric_refs[0]

    if candidate is None and len(refs) == 1:
        candidate = refs[0][1]

    if candidate is None:
        numeric_frame = [
            str(column)
            for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column])
        ]
        if len(numeric_frame) == 1:
            candidate = numeric_frame[0]

    if candidate is None:
        raise DataBindingError(
            "Could not deterministically resolve numeric analysis column. "
            "Provide a semantic binding for 'value_column'."
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
    TEMPORAL_TOOLS = frozenset({
        "linear_trend", "mean_shift_change_point", "variance_shift_change_point",
        "moving_average_trend", "additive_seasonal_decomposition",
    })

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
        elif trace.tool in self.TEMPORAL_TOOLS:
            bound = self._bind_temporal(frame, semantic)
        elif trace.tool == "bayesian_structural_causal_impact":
            bound = self._bind_causal(frame, semantic)
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
            if requested_a is None and requested_b is None:
                requested_a, requested_b = observed
            elif requested_a is None:
                requested_a = next((group for group in observed if group != requested_b), None)
            else:
                requested_b = next((group for group in observed if group != requested_a), None)

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

        return {"data": {str(requested_a): sample_a, str(requested_b): sample_b}}

    @staticmethod
    def _time_series(
        frame: pd.DataFrame,
        semantic: Mapping[str, Any],
        *,
        value_column: str | None = None,
    ) -> TimeSeriesView:
        """Resolve explicit semantic references and build one local chronological view."""
        time_name = _column(frame, semantic.get("time_column"), "time_column")
        value_name = _column(
            frame, value_column if value_column is not None else semantic.get("value_column"),
            "value_column",
        )
        aggregation = semantic.get("aggregation")
        if aggregation is not None and aggregation not in ("sum", "mean"):
            raise DataBindingError("'aggregation' must be 'sum' or 'mean'.")
        # Local import avoids agent -> backend.__init__ -> app -> agent cycle.
        from backend.data_preparation import DatasetPreparationError, build_time_series

        try:
            # With one row per timestamp, sum and mean have identical output.
            # Even when ISO representations differ, duplicates must not be
            # silently aggregated: compare post-parsing period count to rows.
            view = build_time_series(
                frame, time_column=time_name, value_column=value_name,
                aggregation=aggregation or "sum",
            )
            if aggregation is None and view.period_count != view.source_row_count:
                raise DataBindingError(
                    "Repeated timestamps require explicit 'aggregation': 'sum' or 'mean'."
                )
            return view
        except DatasetPreparationError as exc:
            raise DataBindingError(str(exc)) from exc

    def _bind_temporal(
        self, frame: pd.DataFrame, semantic: Mapping[str, Any]
    ) -> dict[str, Any]:
        view = self._time_series(frame, semantic)
        return {
            "time": [timestamp.isoformat() for timestamp in view.timestamps],
            "values": list(view.values),
        }

    def _bind_causal(
        self, frame: pd.DataFrame, semantic: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Build aligned local outcome and optional control arrays.

        Controls are separate numeric columns sharing the same timestamps.
        No fuzzy joins, missing-period fills or inferred intervention dates.
        """
        if "intervention_date" not in semantic:
            raise DataBindingError("Causal impact requires 'intervention_date'.")
        view = self._time_series(frame, semantic)
        from backend.data_preparation import DatasetPreparationError, resolve_intervention_index

        try:
            index = resolve_intervention_index(view, semantic["intervention_date"])
        except DatasetPreparationError as exc:
            raise DataBindingError(str(exc)) from exc
        controls: dict[str, list[float]] = {}
        requested_controls = semantic.get("control_columns", [])
        if not isinstance(requested_controls, list) or any(
            not isinstance(name, str) or not name.strip() for name in requested_controls
        ):
            raise DataBindingError("'control_columns' must be a list of column names.")
        if len(requested_controls) != len(set(requested_controls)):
            raise DataBindingError("Control column references must be unique.")
        outcome_column = _column(frame, semantic.get("value_column"), "value_column")
        for name in requested_controls:
            control_column = _column(frame, name, "control_columns")
            if control_column == outcome_column:
                raise DataBindingError("Control cannot reference the outcome column.")
            control_view = self._time_series(frame, semantic, value_column=control_column)
            if control_view.timestamps != view.timestamps:
                raise DataBindingError(
                    f"Control column {control_column!r} does not match outcome timestamps."
                )
            controls[control_column] = list(control_view.values)
        return {
            "outcome": list(view.values),
            "intervention_index": index,
            "controls": controls,
        }

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
