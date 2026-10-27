from math import erfc, exp, lgamma, log, sqrt
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError


def _numeric_sample(values: Any, label: str) -> tuple[np.ndarray, int, int]:
    try:
        series = pd.Series(values)
    except Exception as exc:
        raise InvalidAnalysisInputError(
            f"Sample '{label}' must be convertible to a one-dimensional series."
        ) from exc

    total = int(len(series))
    if total == 0:
        raise InvalidAnalysisInputError(f"Sample '{label}' requires observations.")

    missing = int(series.isna().sum())
    observed = series.dropna()

    for value in observed:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise InvalidAnalysisInputError(
                "Mann-Whitney U requires genuinely numeric non-missing values."
            )

    numeric = np.asarray(observed, dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise InvalidAnalysisInputError(
            "Mann-Whitney U requires finite non-missing values."
        )
    if len(numeric) < 2:
        raise InvalidAnalysisInputError(
            f"Sample '{label}' requires at least two non-missing observations."
        )

    return numeric, total, missing


def _normal_cdf(value: float) -> float:
    return 0.5 * erfc(-value / sqrt(2.0))


def _average_ranks(values: np.ndarray) -> tuple[np.ndarray, list[int]]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    tie_sizes: list[int] = []

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1

        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank

        tie_size = end - start
        if tie_size > 1:
            tie_sizes.append(tie_size)

        start = end

    return ranks, tie_sizes


class MannWhitneyUTest(AnalysisMethod):
    name = "mann_whitney_u_test"
    category = "statistics"
    version = "1.0"
    description = (
        "Compare two independent numeric samples using the Mann-Whitney U "
        "rank test with tie correction and rank-biserial effect size."
    )
    supported_roles = ("measure",)
    minimum_compatible_columns = 2
    selection_mode = "pairwise"
    analysis_unit = "column_pair"
    role_requirements = (("measure", 2),)
    allow_constant_columns = True
    preserve_missing_values = True

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        data = kwargs.get("data")
        alternative = kwargs.get("alternative", "two-sided")

        if alternative not in {"two-sided", "less", "greater"}:
            raise InvalidAnalysisInputError(
                "'alternative' must be 'two-sided', 'less', or 'greater'."
            )

        if not isinstance(data, dict) or len(data) != 2:
            raise InvalidAnalysisInputError(
                "'data' must contain exactly two independent samples."
            )

        raw_names = list(data.keys())
        names = [str(name).strip() for name in raw_names]
        if any(not name for name in names) or len(set(names)) != 2:
            raise InvalidAnalysisInputError(
                "Sample names must be unique non-empty strings."
            )

        first, first_total, first_missing = _numeric_sample(
            data[raw_names[0]],
            names[0],
        )
        second, second_total, second_missing = _numeric_sample(
            data[raw_names[1]],
            names[1],
        )

        n1 = int(len(first))
        n2 = int(len(second))
        combined = np.concatenate([first, second])
        ranks, tie_sizes = _average_ranks(combined)

        rank_sum_first = float(np.sum(ranks[:n1]))
        u_first = float(rank_sum_first - n1 * (n1 + 1) / 2.0)
        u_second = float(n1 * n2 - u_first)
        mean_u = float(n1 * n2 / 2.0)

        total_n = n1 + n2
        tie_term = float(sum(size**3 - size for size in tie_sizes))
        variance_u = float(
            (n1 * n2 / 12.0)
            * (
                (total_n + 1)
                - tie_term / (total_n * (total_n - 1))
            )
        )

        if variance_u <= 0.0:
            z_statistic = 0.0
            p_value = 1.0
        else:
            standard_error = sqrt(variance_u)
            difference = u_first - mean_u

            if alternative == "two-sided":
                corrected = max(0.0, abs(difference) - 0.5)
                z_statistic = float(
                    (1.0 if difference >= 0.0 else -1.0)
                    * corrected
                    / standard_error
                )
                p_value = float(
                    min(1.0, 2.0 * (1.0 - _normal_cdf(abs(z_statistic))))
                )
            elif alternative == "greater":
                z_statistic = float((difference - 0.5) / standard_error)
                p_value = float(1.0 - _normal_cdf(z_statistic))
            else:
                z_statistic = float((difference + 0.5) / standard_error)
                p_value = float(_normal_cdf(z_statistic))

        rank_biserial = float(
            (2.0 * u_first) / (n1 * n2) - 1.0
        )

        if rank_biserial > 0.0:
            direction = f"{names[0]}_higher"
        elif rank_biserial < 0.0:
            direction = f"{names[0]}_lower"
        else:
            direction = "no_rank_direction"

        return {
            "statistic_type": "mann_whitney_u_test",
            "alternative": alternative,
            "first_sample": names[0],
            "second_sample": names[1],
            "first_total_count": first_total,
            "second_total_count": second_total,
            "first_sample_size": n1,
            "second_sample_size": n2,
            "first_missing_count": first_missing,
            "second_missing_count": second_missing,
            "u_statistic": float(min(u_first, u_second))
            if alternative == "two-sided"
            else u_first,
            "u_first": u_first,
            "u_second": u_second,
            "z_statistic": z_statistic,
            "p_value": p_value,
            "rank_biserial_effect_size": rank_biserial,
            "direction": direction,
            "tie_corrected": bool(tie_sizes),
            "tie_group_count": int(len(tie_sizes)),
            "approximation": "asymptotic_normal_with_continuity_correction",
        }
