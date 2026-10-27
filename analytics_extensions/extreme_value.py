from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError


def _numeric_values(values: Any) -> tuple[np.ndarray, int, int]:
    try:
        series = pd.Series(values)
    except Exception as exc:
        raise InvalidAnalysisInputError(
            "'values' must be convertible to a one-dimensional series."
        ) from exc

    total = int(len(series))
    missing = int(series.isna().sum())
    observed = series.dropna()

    for value in observed:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise InvalidAnalysisInputError(
                "Extreme-value analysis requires genuinely numeric non-missing values."
            )

    numeric = np.asarray(observed, dtype=float)
    if len(numeric) < 3:
        raise InvalidAnalysisInputError(
            "Extreme-value analysis requires at least three non-missing observations."
        )
    if not np.all(np.isfinite(numeric)):
        raise InvalidAnalysisInputError(
            "Extreme-value analysis requires finite non-missing values."
        )
    return numeric, total, missing


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise InvalidAnalysisInputError(f"'{label}' must be numeric.")
    number = float(value)
    if not np.isfinite(number):
        raise InvalidAnalysisInputError(f"'{label}' must be finite.")
    return number


def _probability(value: Any, label: str) -> float:
    number = _finite_number(value, label)
    if number <= 0.0 or number >= 1.0:
        raise InvalidAnalysisInputError(
            f"'{label}' must be in the open interval (0, 1)."
        )
    return number


def _positive(value: Any, label: str) -> float:
    number = _finite_number(value, label)
    if number <= 0.0:
        raise InvalidAnalysisInputError(f"'{label}' must be positive.")
    return number


def _gpd_survival(excess: float, shape: float, scale: float) -> float:
    if excess <= 0.0:
        return 1.0
    if abs(shape) < 1e-12:
        return float(np.exp(-excess / scale))
    support = 1.0 + shape * excess / scale
    if support <= 0.0:
        return 0.0
    return float(support ** (-1.0 / shape))


def _tail_quantile(
    threshold: float,
    exceedance_rate: float,
    shape: float,
    scale: float,
    tail_probability: float,
) -> float:
    if tail_probability >= exceedance_rate:
        raise InvalidAnalysisInputError(
            "Requested tail probability must be smaller than the empirical exceedance rate."
        )

    ratio = exceedance_rate / tail_probability
    if abs(shape) < 1e-12:
        return float(threshold + scale * np.log(ratio))

    value = threshold + (scale / shape) * (ratio**shape - 1.0)
    if shape < 0.0:
        value = min(value, threshold - scale / shape)
    return float(value)


class ExtremeValueTailRisk(AnalysisMethod):
    name = "extreme_value_tail_risk"
    category = "statistics"
    version = "1.0"
    description = (
        "Estimate upper-tail risk using Peaks Over Threshold and a generalized Pareto "
        "distribution, including extreme quantiles, return levels, and tail diagnostics."
    )
    supported_roles = ("measure",)
    minimum_compatible_columns = 1
    selection_mode = "per_column"
    analysis_unit = "column"
    role_requirements = ()
    allow_constant_columns = False
    preserve_missing_values = True

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        values, total_count, missing_count = _numeric_values(
            kwargs.get("values")
        )

        raw_threshold = kwargs.get("threshold")
        threshold_quantile = kwargs.get("threshold_quantile", 0.90)

        if raw_threshold is None:
            threshold_quantile = _probability(
                threshold_quantile,
                "threshold_quantile",
            )
            threshold = float(
                np.quantile(values, threshold_quantile)
            )
            threshold_source = "quantile"
        else:
            threshold = _finite_number(
                raw_threshold,
                "threshold",
            )
            threshold_quantile = None
            threshold_source = "explicit"

        exceedances = values[values > threshold] - threshold
        exceedance_count = int(len(exceedances))
        sample_size = int(len(values))

        if exceedance_count < 3:
            raise InvalidAnalysisInputError(
                "Extreme-value analysis requires at least three threshold exceedances."
            )

        exceedance_rate = float(
            exceedance_count / sample_size
        )
        mean_excess = float(np.mean(exceedances))
        variance_excess = float(
            np.var(exceedances, ddof=1)
        )

        if mean_excess <= 0.0 or variance_excess <= 0.0:
            raise InvalidAnalysisInputError(
                "Threshold exceedances must have positive mean and variance."
            )

        shape = float(
            0.5
            * (
                1.0
                - (mean_excess**2)
                / variance_excess
            )
        )
        scale = float(
            mean_excess * (1.0 - shape)
        )

        if (
            not np.isfinite(shape)
            or not np.isfinite(scale)
            or scale <= 0.0
        ):
            raise InvalidAnalysisInputError(
                "Generalized Pareto parameter estimation failed."
            )

        requested_probabilities = kwargs.get(
            "tail_probabilities",
            [0.05, 0.01],
        )
        try:
            raw_probabilities = list(
                requested_probabilities
            )
        except TypeError as exc:
            raise InvalidAnalysisInputError(
                "'tail_probabilities' must be a sequence."
            ) from exc
        if not raw_probabilities:
            raise InvalidAnalysisInputError(
                "'tail_probabilities' must not be empty."
            )

        extreme_quantiles: dict[str, float] = {}
        for raw_probability in raw_probabilities:
            probability = _probability(
                raw_probability,
                "tail_probability",
            )
            extreme_quantiles[str(probability)] = (
                _tail_quantile(
                    threshold,
                    exceedance_rate,
                    shape,
                    scale,
                    probability,
                )
            )

        requested_periods = kwargs.get(
            "return_periods",
            [20.0, 100.0],
        )
        try:
            raw_periods = list(requested_periods)
        except TypeError as exc:
            raise InvalidAnalysisInputError(
                "'return_periods' must be a sequence."
            ) from exc
        if not raw_periods:
            raise InvalidAnalysisInputError(
                "'return_periods' must not be empty."
            )

        return_levels: dict[str, float] = {}
        for raw_period in raw_periods:
            period = _positive(
                raw_period,
                "return_period",
            )
            probability = 1.0 / period
            return_levels[str(period)] = (
                _tail_quantile(
                    threshold,
                    exceedance_rate,
                    shape,
                    scale,
                    probability,
                )
            )

        requested_points = kwargs.get(
            "evaluate_at",
            [],
        )
        try:
            raw_points = list(requested_points)
        except TypeError as exc:
            raise InvalidAnalysisInputError(
                "'evaluate_at' must be a sequence."
            ) from exc

        tail_probabilities: dict[str, float] = {}
        for raw_point in raw_points:
            point = _finite_number(
                raw_point,
                "evaluate_at",
            )
            if point < threshold:
                raise InvalidAnalysisInputError(
                    "'evaluate_at' values must be at or above the threshold."
                )
            conditional = _gpd_survival(
                point - threshold,
                shape,
                scale,
            )
            tail_probabilities[str(point)] = float(
                exceedance_rate * conditional
            )

        if shape > 0.05:
            tail_regime = "heavy_tail"
        elif shape < -0.05:
            tail_regime = "bounded_tail"
        else:
            tail_regime = "approximately_exponential_tail"

        risk_flags = []
        if shape > 0.25:
            risk_flags.append("elevated_tail_heaviness")
        if exceedance_count < 20:
            risk_flags.append("limited_exceedance_sample")

        upper_endpoint = (
            None
            if shape >= 0.0
            else float(threshold - scale / shape)
        )

        return {
            "statistic_type": "extreme_value_tail_risk",
            "total_count": total_count,
            "sample_size": sample_size,
            "missing_count": missing_count,
            "threshold": threshold,
            "threshold_source": threshold_source,
            "threshold_quantile": threshold_quantile,
            "exceedance_count": exceedance_count,
            "exceedance_rate": exceedance_rate,
            "mean_excess": mean_excess,
            "variance_excess": variance_excess,
            "gpd_shape": shape,
            "gpd_scale": scale,
            "tail_regime": tail_regime,
            "upper_endpoint": upper_endpoint,
            "tail_probabilities": tail_probabilities,
            "extreme_quantiles": extreme_quantiles,
            "return_levels": return_levels,
            "risk_flags": risk_flags,
            "fit_diagnostics": {
                "fit_method": "method_of_moments",
                "exceedance_count": exceedance_count,
                "exceedance_rate": exceedance_rate,
                "finite_variance_assumption": True,
                "parameter_valid": True,
            },
        }
