from numbers import Real
from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError


def _validated_p_values(values: Any) -> np.ndarray:
    if isinstance(values, (str, bytes)) or values is None:
        raise InvalidAnalysisInputError(
            "'p_values' must be a non-empty sequence of probabilities."
        )

    try:
        raw = list(values)
    except TypeError as exc:
        raise InvalidAnalysisInputError(
            "'p_values' must be a non-empty sequence of probabilities."
        ) from exc

    if not raw:
        raise InvalidAnalysisInputError(
            "'p_values' must contain at least one probability."
        )

    numeric: list[float] = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise InvalidAnalysisInputError(
                "Multiple-testing correction requires numeric p-values."
            )

        number = float(value)
        if not np.isfinite(number) or number < 0.0 or number > 1.0:
            raise InvalidAnalysisInputError(
                "Each p-value must be finite and between 0 and 1 inclusive."
            )
        numeric.append(number)

    return np.asarray(numeric, dtype=float)


def _validated_alpha(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise InvalidAnalysisInputError("'alpha' must be numeric.")

    alpha = float(value)
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
        raise InvalidAnalysisInputError(
            "'alpha' must be finite and in the interval (0, 1]."
        )

    return alpha


def _bonferroni(p_values: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, p_values * len(p_values))


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    count = len(p_values)
    order = np.argsort(p_values, kind="mergesort")
    sorted_values = p_values[order]

    adjusted_sorted = np.empty(count, dtype=float)
    running_minimum = 1.0

    for reverse_index in range(count - 1, -1, -1):
        rank = reverse_index + 1
        candidate = float(sorted_values[reverse_index] * count / rank)
        running_minimum = min(running_minimum, candidate, 1.0)
        adjusted_sorted[reverse_index] = running_minimum

    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


class MultipleTestingCorrection(AnalysisMethod):
    name = "multiple_testing_correction"
    category = "statistics"
    version = "1.0"
    description = (
        "Correct a family of p-values for multiple hypothesis testing using "
        "Bonferroni family-wise error control or Benjamini-Hochberg false "
        "discovery rate control."
    )
    supported_roles = ("measure",)
    minimum_compatible_columns = 1
    selection_mode = "joint"
    analysis_unit = "result_set"
    role_requirements = ()
    allow_constant_columns = True
    preserve_missing_values = True

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        p_values = _validated_p_values(kwargs.get("p_values"))
        alpha = _validated_alpha(kwargs.get("alpha", 0.05))

        raw_method = kwargs.get("method", "benjamini_hochberg")
        if not isinstance(raw_method, str) or not raw_method.strip():
            raise InvalidAnalysisInputError(
                "'method' must be 'bonferroni' or 'benjamini_hochberg'."
            )

        method = raw_method.strip().lower().replace("-", "_")
        aliases = {
            "bonferroni": "bonferroni",
            "benjamini_hochberg": "benjamini_hochberg",
            "bh": "benjamini_hochberg",
            "fdr_bh": "benjamini_hochberg",
        }

        if method not in aliases:
            raise InvalidAnalysisInputError(
                "'method' must be 'bonferroni' or 'benjamini_hochberg'."
            )

        canonical_method = aliases[method]

        if canonical_method == "bonferroni":
            adjusted = _bonferroni(p_values)
            error_control = "family_wise_error_rate"
        else:
            adjusted = _benjamini_hochberg(p_values)
            error_control = "false_discovery_rate"

        reject = adjusted <= alpha

        return {
            "statistic_type": "multiple_testing_correction",
            "method": canonical_method,
            "error_control": error_control,
            "alpha": alpha,
            "hypothesis_count": int(len(p_values)),
            "p_values": [float(value) for value in p_values],
            "adjusted_p_values": [float(value) for value in adjusted],
            "reject": [bool(value) for value in reject],
            "rejection_count": int(np.sum(reject)),
        }
