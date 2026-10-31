"""Deterministic CSV preparation and explicit analytical time-series views.

Upload preparation retains the original row grain and numeric measures. A
TimeSeriesView is created only by an explicit caller; future causal controls
must be aligned on identical timestamps, in identical order and length.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


class DatasetPreparationError(ValueError):
    """Invalid dataset or explicit time-series preparation request."""


_TEMPORAL_NAMES = frozenset({"date", "datetime", "timestamp", "time"})


@dataclass(frozen=True)
class PreparedDataset:
    frame: pd.DataFrame
    temporal_columns: tuple[str, ...]


@dataclass(frozen=True)
class TimeSeriesView:
    """Chronologically sorted unique timestamps and numeric aggregate values.

    No missing periods are generated, and no missing values are imputed.
    A control series must later match `timestamps` exactly (not by nearest date).
    """

    timestamps: tuple[pd.Timestamp, ...]
    values: tuple[float, ...]
    time_column: str
    value_column: str
    aggregation: Literal["sum", "mean"]
    source_row_count: int
    period_count: int
    first_period: pd.Timestamp
    last_period: pd.Timestamp
    inferred_frequency: str | None
    is_regular: bool
    has_temporal_gaps: bool | None


def _parse_temporal(series: pd.Series, name: str) -> pd.Series:
    """Parse only explicitly named temporal columns; distinguish bad vs blank."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.copy()
    if pd.api.types.is_numeric_dtype(series):
        raise DatasetPreparationError(
            f"Temporal column {name!r} must contain ISO date strings, not numeric timestamps."
        )
    try:
        # Mixed offsets / timezone-aware values must not yield object dtype.
        converted = pd.to_datetime(series, errors="coerce", format="ISO8601")
    except (ValueError, TypeError, OverflowError) as exc:
        raise DatasetPreparationError(f"Invalid temporal column {name!r}: {exc}") from exc
    invalid = series.notna() & converted.isna()
    if bool(invalid.any()):
        raise DatasetPreparationError(
            f"Invalid date in explicitly temporal column {name!r} "
            f"({int(invalid.sum())} non-missing value(s) could not be parsed)."
        )
    if not pd.api.types.is_datetime64_any_dtype(converted):
        raise DatasetPreparationError(f"Temporal column {name!r} has incompatible timezone values.")
    return converted


def prepare_dataset(frame: pd.DataFrame) -> PreparedDataset:
    """Copy a read_csv DataFrame, parsing strictly named ISO date columns only."""
    if not isinstance(frame, pd.DataFrame) or frame.empty or len(frame.columns) == 0:
        raise DatasetPreparationError("Dataset must contain at least one row and one column.")
    if not frame.columns.is_unique:
        raise DatasetPreparationError("Dataset column names must be unique.")
    result = frame.copy(deep=True)
    temporal: list[str] = []
    for column in result.columns:
        if not isinstance(column, str):
            continue
        if column.strip().casefold() in _TEMPORAL_NAMES or pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = _parse_temporal(result[column], column)
            temporal.append(column)
    return PreparedDataset(frame=result, temporal_columns=tuple(temporal))


def build_time_series(
    frame: pd.DataFrame,
    *,
    time_column: str,
    value_column: str,
    aggregation: Literal["sum", "mean"],
) -> TimeSeriesView:
    """Aggregate explicit values at identical timestamps without mutating source.

    Rejects null timestamps/values instead of silently dropping or summing
    them; no inferred dates are ever inserted.
    """
    if aggregation not in ("sum", "mean"):
        raise DatasetPreparationError("Aggregation must be 'sum' or 'mean'.")
    if time_column not in frame.columns:
        raise DatasetPreparationError(f"Unknown time column {time_column!r}.")
    if value_column not in frame.columns:
        raise DatasetPreparationError(f"Unknown value column {value_column!r}.")
    time = _parse_temporal(frame[time_column], time_column)
    values = frame[value_column]
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        raise DatasetPreparationError(f"Value column {value_column!r} must be numeric.")
    if bool(time.isna().any()) or bool(values.isna().any()):
        raise DatasetPreparationError("Time series has missing timestamps or values; explicit handling is required.")
    if not bool(pd.Series(pd.to_numeric(values, errors="coerce")).map(lambda x: pd.notna(x) and abs(float(x)) != float("inf")).all()):
        raise DatasetPreparationError("Time series values must be finite.")
    grouped = (
        pd.DataFrame({"_time": time, "_value": values})
        .groupby("_time", sort=True, dropna=False)["_value"]
        .agg(aggregation)
    )
    if len(grouped) < 2:
        raise DatasetPreparationError("Time series requires at least two distinct periods.")
    times = tuple(pd.Timestamp(t) for t in grouped.index)
    try:
        # Pandas 3.1 warns that a future version will return a BaseOffset instead of
        # a string from infer_freq. Opt into the new behaviour now so the result is
        # always a BaseOffset; call .freqstr for a normalised string representation.
        with pd.option_context("future.infer_freq_returns_offset", True):
            _raw_freq = pd.infer_freq(pd.DatetimeIndex(times)) if len(times) >= 3 else None
        frequency = _raw_freq.freqstr if _raw_freq is not None else None
    except (TypeError, ValueError, AttributeError):
        frequency = None
    # Regularity is only asserted when a supported frequency is inferable.
    # If frequency cannot be inferred, gaps cannot be established safely.
    return TimeSeriesView(
        timestamps=times,
        values=tuple(float(x) for x in grouped.to_numpy()),
        time_column=time_column,
        value_column=value_column,
        aggregation=aggregation,
        source_row_count=len(frame),
        period_count=len(times),
        first_period=times[0],
        last_period=times[-1],
        inferred_frequency=frequency,
        is_regular=frequency is not None,
        has_temporal_gaps=False if frequency is not None else None,
    )


def resolve_intervention_index(view: TimeSeriesView, intervention_date: str | pd.Timestamp) -> int:
    """Resolve an exact intervention timestamp; never snap to nearby dates."""
    try:
        parsed = pd.Timestamp(intervention_date)
    except (TypeError, ValueError) as exc:
        raise DatasetPreparationError("Intervention date must be a valid timestamp.") from exc
    try:
        return view.timestamps.index(parsed)
    except ValueError as exc:
        raise DatasetPreparationError("Intervention date must exactly match an existing period.") from exc
