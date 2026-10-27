from numbers import Integral, Real
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError


def _numeric_series(values: Any, label: str) -> np.ndarray:
    try:
        series = pd.Series(values)
    except Exception as exc:
        raise InvalidAnalysisInputError(
            f"'{label}' must be convertible to a one-dimensional series."
        ) from exc

    if len(series) == 0:
        raise InvalidAnalysisInputError(f"'{label}' requires observations.")
    if series.isna().any():
        raise InvalidAnalysisInputError(
            f"'{label}' cannot contain missing values for causal-impact analysis."
        )

    for value in series:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise InvalidAnalysisInputError(
                f"'{label}' must contain genuinely numeric values."
            )

    numeric = np.asarray(series, dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise InvalidAnalysisInputError(f"'{label}' must contain finite values.")
    return numeric


def _validated_intervention_index(value: Any, sample_size: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise InvalidAnalysisInputError("'intervention_index' must be an integer.")

    index = int(value)
    if index < 5:
        raise InvalidAnalysisInputError(
            "Causal-impact analysis requires at least five pre-intervention observations."
        )
    if sample_size - index < 2:
        raise InvalidAnalysisInputError(
            "Causal-impact analysis requires at least two post-intervention observations."
        )
    return index


def _validated_probability(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise InvalidAnalysisInputError(f"'{label}' must be numeric.")

    probability = float(value)
    if not np.isfinite(probability) or probability <= 0.0 or probability >= 1.0:
        raise InvalidAnalysisInputError(f"'{label}' must be strictly between 0 and 1.")
    return probability


def _validated_positive(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise InvalidAnalysisInputError(f"'{label}' must be numeric.")

    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise InvalidAnalysisInputError(f"'{label}' must be finite and positive.")
    return number


def _control_matrix(controls: Any, sample_size: int) -> tuple[np.ndarray, list[str]]:
    if controls is None:
        return np.empty((sample_size, 0), dtype=float), []

    if not isinstance(controls, dict):
        raise InvalidAnalysisInputError(
            "'controls' must be a dictionary of control-series names to values."
        )

    names: list[str] = []
    columns: list[np.ndarray] = []

    for raw_name, values in controls.items():
        name = str(raw_name).strip()
        if not name or name in names:
            raise InvalidAnalysisInputError(
                "Control names must be unique non-empty strings."
            )
        numeric = _numeric_series(values, f"control:{name}")
        if len(numeric) != sample_size:
            raise InvalidAnalysisInputError(
                "Every control series must have the same length as the outcome."
            )
        names.append(name)
        columns.append(numeric)

    if not columns:
        return np.empty((sample_size, 0), dtype=float), names

    return np.column_stack(columns), names


def _design_matrix(sample_size: int, controls: np.ndarray) -> np.ndarray:
    time = np.arange(sample_size, dtype=float)
    if sample_size > 1:
        time = (time - np.mean(time)) / np.std(time, ddof=0)

    intercept = np.ones(sample_size, dtype=float)
    parts = [intercept, time]
    if controls.shape[1]:
        centered = controls.copy()
        means = np.mean(centered, axis=0)
        scales = np.std(centered, axis=0, ddof=0)
        scales = np.where(scales == 0.0, 1.0, scales)
        centered = (centered - means) / scales
        parts.append(centered)

    return np.column_stack(parts)


def _safe_percentage(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return float(100.0 * numerator / denominator)


class BayesianStructuralCausalImpact(AnalysisMethod):
    name = "bayesian_structural_causal_impact"
    category = "statistics"
    version = "1.0"
    description = (
        "Estimate a post-intervention counterfactual with Bayesian regularized regression "
        "fit only on pre-intervention observations, returning conditional absolute and "
        "relative impact with credible intervals and diagnostics."
    )
    supported_roles = ("time", "measure")
    minimum_compatible_columns = 2
    selection_mode = "time_series"
    analysis_unit = "series"
    role_requirements = (("time", 1), ("measure", 1))
    allow_constant_columns = True
    preserve_missing_values = True

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        outcome = _numeric_series(kwargs.get("outcome"), "outcome")
        sample_size = int(len(outcome))
        intervention_index = _validated_intervention_index(
            kwargs.get("intervention_index"),
            sample_size,
        )
        credibility = _validated_probability(kwargs.get("credibility", 0.95), "credibility")
        prior_scale = _validated_positive(kwargs.get("prior_scale", 10.0), "prior_scale")
        prior_alpha = _validated_positive(kwargs.get("prior_alpha", 2.0), "prior_alpha")
        prior_beta = _validated_positive(kwargs.get("prior_beta", 1.0), "prior_beta")

        controls, control_names = _control_matrix(
            kwargs.get("controls"),
            sample_size,
        )
        design = _design_matrix(sample_size, controls)

        x_pre = design[:intervention_index]
        y_pre = outcome[:intervention_index]
        x_post = design[intervention_index:]
        y_post = outcome[intervention_index:]

        parameter_count = int(design.shape[1])
        prior_precision = np.eye(parameter_count, dtype=float) / (prior_scale ** 2)
        prior_mean = np.zeros(parameter_count, dtype=float)

        posterior_precision = prior_precision + x_pre.T @ x_pre
        try:
            posterior_covariance_factor = np.linalg.inv(posterior_precision)
        except np.linalg.LinAlgError as exc:
            raise InvalidAnalysisInputError(
                "Unable to estimate a stable causal-impact model from the pre-intervention data."
            ) from exc

        posterior_mean = posterior_covariance_factor @ (
            prior_precision @ prior_mean + x_pre.T @ y_pre
        )

        posterior_alpha = float(prior_alpha + len(y_pre) / 2.0)
        quadratic = float(
            y_pre @ y_pre
            + prior_mean @ prior_precision @ prior_mean
            - posterior_mean @ posterior_precision @ posterior_mean
        )
        posterior_beta = float(prior_beta + 0.5 * max(0.0, quadratic))
        posterior_variance_mean = float(posterior_beta / max(posterior_alpha - 1.0, 1e-12))

        pre_fitted = x_pre @ posterior_mean
        pre_residuals = y_pre - pre_fitted
        pre_rmse = float(np.sqrt(np.mean(pre_residuals ** 2)))
        pre_mae = float(np.mean(np.abs(pre_residuals)))

        counterfactual_mean = x_post @ posterior_mean
        parameter_covariance = (
            posterior_variance_mean * posterior_covariance_factor
        )
        predictive_covariance = (
            posterior_variance_mean * np.eye(len(y_post), dtype=float)
            + x_post @ parameter_covariance @ x_post.T
        )
        predictive_variance = np.diag(predictive_covariance)
        predictive_sd = np.sqrt(np.maximum(0.0, predictive_variance))

        z = float(NormalDist().inv_cdf(0.5 + credibility / 2.0))
        point_lower = counterfactual_mean - z * predictive_sd
        point_upper = counterfactual_mean + z * predictive_sd

        point_effect = y_post - counterfactual_mean
        point_effect_lower = y_post - point_upper
        point_effect_upper = y_post - point_lower

        observed_post_total = float(np.sum(y_post))
        counterfactual_post_total = float(np.sum(counterfactual_mean))
        absolute_effect = float(np.sum(point_effect))

        aggregate_variance = float(np.sum(predictive_covariance))
        aggregate_sd = float(np.sqrt(max(0.0, aggregate_variance)))
        absolute_effect_lower = float(absolute_effect - z * aggregate_sd)
        absolute_effect_upper = float(absolute_effect + z * aggregate_sd)
        relative_effect_percent = _safe_percentage(
            absolute_effect,
            counterfactual_post_total,
        )
        relative_bounds = [
            _safe_percentage(absolute_effect_lower, counterfactual_post_total),
            _safe_percentage(absolute_effect_upper, counterfactual_post_total),
        ]
        if None in relative_bounds:
            relative_effect_lower_percent = None
            relative_effect_upper_percent = None
        else:
            relative_effect_lower_percent = float(min(relative_bounds))
            relative_effect_upper_percent = float(max(relative_bounds))

        direction = (
            "positive"
            if absolute_effect > 0.0
            else "negative"
            if absolute_effect < 0.0
            else "none"
        )

        return {
            "statistic_type": "bayesian_structural_causal_impact",
            "sample_size": sample_size,
            "pre_intervention_count": int(intervention_index),
            "post_intervention_count": int(sample_size - intervention_index),
            "intervention_index": intervention_index,
            "control_names": control_names,
            "parameter_count": parameter_count,
            "credibility": credibility,
            "prior_scale": prior_scale,
            "posterior_alpha": posterior_alpha,
            "posterior_beta": posterior_beta,
            "posterior_noise_variance_mean": posterior_variance_mean,
            "coefficient_names": ["intercept", "time", *control_names],
            "posterior_coefficients": [float(value) for value in posterior_mean],
            "pre_intervention_rmse": pre_rmse,
            "pre_intervention_mae": pre_mae,
            "observed_post_total": observed_post_total,
            "counterfactual_post_total": counterfactual_post_total,
            "absolute_effect": absolute_effect,
            "absolute_effect_interval": [
                absolute_effect_lower,
                absolute_effect_upper,
            ],
            "relative_effect_percent": relative_effect_percent,
            "relative_effect_interval_percent": [
                relative_effect_lower_percent,
                relative_effect_upper_percent,
            ],
            "direction": direction,
            "observed_post": [float(value) for value in y_post],
            "counterfactual_post_mean": [
                float(value) for value in counterfactual_mean
            ],
            "counterfactual_post_interval": [
                [float(lower), float(upper)]
                for lower, upper in zip(point_lower, point_upper)
            ],
            "point_effect": [float(value) for value in point_effect],
            "point_effect_interval": [
                [float(lower), float(upper)]
                for lower, upper in zip(point_effect_lower, point_effect_upper)
            ],
            "causal_interpretation": (
                "conditional_on_model_specification_and_counterfactual_assumptions"
            ),
            "interval_method": "bayesian_normal_posterior_predictive_approximation",
        }
