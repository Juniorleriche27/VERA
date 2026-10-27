from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError
from data_analysis_engine.engines.inference.parametric_tests import _regularized_incomplete_beta


def _numeric_group(values: Any, label: str) -> tuple[np.ndarray, int, int]:
    try:
        series = pd.Series(values)
    except Exception as exc:
        raise InvalidAnalysisInputError(
            f"Group '{label}' must be convertible to a one-dimensional series."
        ) from exc

    total = int(len(series))
    if total == 0:
        raise InvalidAnalysisInputError(f"Group '{label}' requires observations.")

    missing = int(series.isna().sum())
    observed = series.dropna()

    for value in observed:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise InvalidAnalysisInputError(
                "One-way ANOVA requires genuinely numeric non-missing values."
            )

    numeric = np.asarray(observed, dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise InvalidAnalysisInputError(
            "One-way ANOVA requires finite non-missing values."
        )
    if len(numeric) < 2:
        raise InvalidAnalysisInputError(
            f"Group '{label}' requires at least two non-missing observations."
        )
    return numeric, total, missing


def _f_survival_probability(statistic: float, df_between: int, df_within: int) -> float:
    if statistic < 0.0:
        raise InvalidAnalysisInputError("F statistic cannot be negative.")
    if statistic == float("inf"):
        return 0.0
    if statistic == 0.0:
        return 1.0
    x = float(df_within / (df_within + df_between * statistic))
    return float(
        _regularized_incomplete_beta(
            x,
            df_within / 2.0,
            df_between / 2.0,
        )
    )


class OneWayANOVA(AnalysisMethod):
    name = "one_way_anova"
    category = "statistics"
    version = "1.0"
    description = (
        "Compare the means of two or more independent numeric groups using "
        "one-way ANOVA, including eta-squared and omega-squared effect sizes."
    )
    supported_roles = ("measure",)
    minimum_compatible_columns = 2
    selection_mode = "joint"
    analysis_unit = "dataset"
    role_requirements = (("measure", 2),)
    allow_constant_columns = True
    preserve_missing_values = True

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        data = kwargs.get("data")
        if not isinstance(data, dict):
            raise InvalidAnalysisInputError(
                "'data' must be a dictionary of group names to values."
            )
        if len(data) < 2:
            raise InvalidAnalysisInputError(
                "One-way ANOVA requires at least two groups."
            )

        groups: dict[str, np.ndarray] = {}
        stats: dict[str, dict[str, Any]] = {}
        total_count = 0
        missing_count = 0

        for raw_name, values in data.items():
            name = str(raw_name).strip()
            if not name or name in groups:
                raise InvalidAnalysisInputError(
                    "Group names must be unique non-empty strings."
                )
            numeric, total, missing = _numeric_group(values, name)
            groups[name] = numeric
            stats[name] = {
                "total_count": total,
                "sample_size": int(len(numeric)),
                "missing_count": missing,
                "mean": float(np.mean(numeric)),
                "standard_deviation": float(np.std(numeric, ddof=1)),
            }
            total_count += total
            missing_count += missing

        sample_size = int(sum(len(values) for values in groups.values()))
        group_count = int(len(groups))
        df_between = group_count - 1
        df_within = sample_size - group_count
        if df_within <= 0:
            raise InvalidAnalysisInputError(
                "One-way ANOVA requires positive within-group degrees of freedom."
            )

        grand_mean = float(
            sum(len(values) * float(np.mean(values)) for values in groups.values())
            / sample_size
        )
        ss_between = float(
            sum(
                len(values) * (float(np.mean(values)) - grand_mean) ** 2
                for values in groups.values()
            )
        )
        ss_within = float(
            sum(
                float(np.sum((values - np.mean(values)) ** 2))
                for values in groups.values()
            )
        )
        ss_total = float(ss_between + ss_within)
        ms_between = float(ss_between / df_between)
        ms_within = float(ss_within / df_within)

        if ms_within == 0.0:
            statistic = 0.0 if ms_between == 0.0 else float("inf")
        else:
            statistic = float(ms_between / ms_within)

        p_value = _f_survival_probability(statistic, df_between, df_within)
        eta_squared = None if ss_total == 0.0 else float(ss_between / ss_total)
        omega_denominator = ss_total + ms_within
        omega_squared = (
            None
            if omega_denominator == 0.0
            else float(
                max(
                    0.0,
                    (ss_between - df_between * ms_within) / omega_denominator,
                )
            )
        )

        return {
            "statistic_type": "one_way_anova",
            "group_count": group_count,
            "total_count": total_count,
            "sample_size": sample_size,
            "missing_count": missing_count,
            "grand_mean": grand_mean,
            "group_statistics": stats,
            "df_between": df_between,
            "df_within": df_within,
            "sum_squares_between": ss_between,
            "sum_squares_within": ss_within,
            "sum_squares_total": ss_total,
            "mean_square_between": ms_between,
            "mean_square_within": ms_within,
            "f_statistic": None if not np.isfinite(statistic) else statistic,
            "f_statistic_infinite": bool(np.isinf(statistic)),
            "p_value": p_value,
            "eta_squared": eta_squared,
            "omega_squared": omega_squared,
        }
