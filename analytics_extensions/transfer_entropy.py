from collections import Counter
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError


def _numeric_series(values: Any, label: str) -> tuple[np.ndarray, int, int]:
    try:
        series = pd.Series(values)
    except Exception as exc:
        raise InvalidAnalysisInputError(
            f"Series '{label}' must be convertible to a one-dimensional series."
        ) from exc

    total = int(len(series))
    if total == 0:
        raise InvalidAnalysisInputError(
            f"Series '{label}' requires observations."
        )

    missing = int(series.isna().sum())
    observed = series.dropna()

    for value in observed:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise InvalidAnalysisInputError(
                "Directed transfer entropy requires genuinely numeric non-missing values."
            )

    numeric = np.asarray(observed, dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise InvalidAnalysisInputError(
            "Directed transfer entropy requires finite non-missing values."
        )

    return numeric, total, missing


def _validate_positive_integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise InvalidAnalysisInputError(
            f"'{name}' must be an integer greater than or equal to {minimum}."
        )
    parsed = int(value)
    if parsed < minimum:
        raise InvalidAnalysisInputError(
            f"'{name}' must be an integer greater than or equal to {minimum}."
        )
    return parsed


def _quantile_discretize(values: np.ndarray, bins: int) -> tuple[np.ndarray, int]:
    unique_count = int(len(np.unique(values)))
    actual_bins = min(bins, unique_count)

    if actual_bins <= 1:
        return np.zeros(len(values), dtype=int), 1

    quantiles = np.linspace(0.0, 1.0, actual_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)

    if len(edges) <= 2:
        threshold_edges = edges[1:-1]
    else:
        threshold_edges = edges[1:-1]

    discrete = np.searchsorted(
        threshold_edges,
        values,
        side="right",
    ).astype(int)

    return discrete, int(len(np.unique(discrete)))


def _transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int,
) -> float:
    if len(source) != len(target):
        raise InvalidAnalysisInputError(
            "Source and target must have the same aligned length."
        )
    if lag >= len(source):
        raise InvalidAnalysisInputError(
            "'lag' must be smaller than the aligned sample size."
        )

    source_past = source[:-lag]
    target_past = target[:-lag]
    target_now = target[lag:]

    triple = Counter(
        zip(target_now.tolist(), target_past.tolist(), source_past.tolist())
    )
    target_source_past = Counter(
        zip(target_past.tolist(), source_past.tolist())
    )
    target_transition = Counter(
        zip(target_now.tolist(), target_past.tolist())
    )
    target_past_counts = Counter(target_past.tolist())

    sample_size = len(target_now)
    if sample_size == 0:
        return 0.0

    te = 0.0
    for (current, past, source_state), count in triple.items():
        p_joint = count / sample_size
        denominator_with_source = target_source_past[(past, source_state)]
        denominator_without_source = target_past_counts[past]
        target_pair_count = target_transition[(current, past)]

        p_current_given_both = count / denominator_with_source
        p_current_given_target = target_pair_count / denominator_without_source

        if p_current_given_both > 0.0 and p_current_given_target > 0.0:
            te += p_joint * np.log2(
                p_current_given_both / p_current_given_target
            )

    return float(max(0.0, te))


def _surrogate_distribution(
    source: np.ndarray,
    target: np.ndarray,
    lag: int,
    surrogate_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sample_size = len(source)
    if sample_size <= 2:
        return np.zeros(surrogate_count, dtype=float)

    possible_shifts = np.arange(1, sample_size, dtype=int)
    surrogates = np.empty(surrogate_count, dtype=float)

    for index in range(surrogate_count):
        shift = int(rng.choice(possible_shifts))
        shifted = np.roll(source, shift)
        surrogates[index] = _transfer_entropy(
            shifted,
            target,
            lag,
        )

    return surrogates


def _direction_summary(
    observed: float,
    surrogates: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    mean_surrogate = float(np.mean(surrogates))
    std_surrogate = float(np.std(surrogates, ddof=1)) if len(surrogates) > 1 else 0.0
    effective = float(observed - mean_surrogate)
    p_value = float(
        (1 + int(np.sum(surrogates >= observed)))
        / (len(surrogates) + 1)
    )

    if std_surrogate == 0.0:
        z_score = 0.0 if observed == mean_surrogate else (
            float("inf") if observed > mean_surrogate else float("-inf")
        )
    else:
        z_score = float((observed - mean_surrogate) / std_surrogate)

    return {
        "transfer_entropy_bits": float(observed),
        "surrogate_mean_bits": mean_surrogate,
        "surrogate_std_bits": std_surrogate,
        "effective_transfer_entropy_bits": effective,
        "empirical_p_value": p_value,
        "significant": bool(p_value <= alpha and effective > 0.0),
        "surrogate_z_score": None if not np.isfinite(z_score) else z_score,
        "surrogate_z_score_infinite": bool(np.isinf(z_score)),
    }


class DirectedTransferEntropy(AnalysisMethod):
    name = "directed_transfer_entropy"
    category = "statistics"
    version = "1.0"
    description = (
        "Estimate directional predictive information transfer between two aligned "
        "numeric series using discretized transfer entropy and deterministic "
        "circular-shift surrogate significance testing. This is not proof of causality."
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
        if not isinstance(data, dict) or len(data) != 2:
            raise InvalidAnalysisInputError(
                "'data' must contain exactly two aligned numeric series."
            )

        lag = _validate_positive_integer(
            kwargs.get("lag", 1),
            "lag",
            1,
        )
        bins = _validate_positive_integer(
            kwargs.get("bins", 4),
            "bins",
            2,
        )
        surrogate_count = _validate_positive_integer(
            kwargs.get("surrogate_count", 100),
            "surrogate_count",
            10,
        )

        seed = kwargs.get("seed", 0)
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
            raise InvalidAnalysisInputError("'seed' must be an integer.")
        seed = int(seed)

        alpha_raw = kwargs.get("alpha", 0.05)
        if isinstance(alpha_raw, (bool, np.bool_)) or not isinstance(alpha_raw, Real):
            raise InvalidAnalysisInputError("'alpha' must be numeric.")
        alpha = float(alpha_raw)
        if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
            raise InvalidAnalysisInputError(
                "'alpha' must be finite and in the interval (0, 1]."
            )

        raw_names = list(data.keys())
        names = [str(name).strip() for name in raw_names]
        if any(not name for name in names) or len(set(names)) != 2:
            raise InvalidAnalysisInputError(
                "Series names must be unique non-empty strings."
            )

        try:
            first_series = pd.Series(data[raw_names[0]])
            second_series = pd.Series(data[raw_names[1]])
        except Exception as exc:
            raise InvalidAnalysisInputError(
                "Both series must be convertible to one-dimensional series."
            ) from exc

        if len(first_series) != len(second_series):
            raise InvalidAnalysisInputError(
                "Directed transfer entropy requires equal-length aligned series."
            )

        total_count = int(len(first_series))
        if total_count < 12:
            raise InvalidAnalysisInputError(
                "Directed transfer entropy requires at least 12 aligned observations."
            )

        pair_frame = pd.DataFrame({
            names[0]: first_series,
            names[1]: second_series,
        })

        missing_mask = pair_frame.isna().any(axis=1)
        missing_pair_count = int(missing_mask.sum())
        observed_frame = pair_frame.loc[~missing_mask]

        first, _, _ = _numeric_series(
            observed_frame[names[0]].tolist(),
            names[0],
        )
        second, _, _ = _numeric_series(
            observed_frame[names[1]].tolist(),
            names[1],
        )

        aligned_size = int(len(first))
        if aligned_size < 12:
            raise InvalidAnalysisInputError(
                "Directed transfer entropy requires at least 12 complete aligned observations."
            )
        if lag >= aligned_size:
            raise InvalidAnalysisInputError(
                "'lag' must be smaller than the complete aligned sample size."
            )
        effective_sample_size = aligned_size - lag
        if effective_sample_size < 10:
            raise InvalidAnalysisInputError(
                "The selected lag leaves fewer than 10 effective observations."
            )

        first_discrete, first_actual_bins = _quantile_discretize(first, bins)
        second_discrete, second_actual_bins = _quantile_discretize(second, bins)

        first_to_second = _transfer_entropy(
            first_discrete,
            second_discrete,
            lag,
        )
        second_to_first = _transfer_entropy(
            second_discrete,
            first_discrete,
            lag,
        )

        rng_first = np.random.default_rng(seed)
        rng_second = np.random.default_rng(seed + 1)

        first_surrogates = _surrogate_distribution(
            first_discrete,
            second_discrete,
            lag,
            surrogate_count,
            rng_first,
        )
        second_surrogates = _surrogate_distribution(
            second_discrete,
            first_discrete,
            lag,
            surrogate_count,
            rng_second,
        )

        first_summary = _direction_summary(
            first_to_second,
            first_surrogates,
            alpha,
        )
        second_summary = _direction_summary(
            second_to_first,
            second_surrogates,
            alpha,
        )

        balance = float(
            first_summary["effective_transfer_entropy_bits"]
            - second_summary["effective_transfer_entropy_bits"]
        )

        if first_summary["significant"] and not second_summary["significant"]:
            dominant_direction = f"{names[0]}_to_{names[1]}"
        elif second_summary["significant"] and not first_summary["significant"]:
            dominant_direction = f"{names[1]}_to_{names[0]}"
        elif first_summary["significant"] and second_summary["significant"]:
            dominant_direction = (
                f"{names[0]}_to_{names[1]}"
                if balance > 0.0
                else f"{names[1]}_to_{names[0]}"
                if balance < 0.0
                else "bidirectional_balanced"
            )
        else:
            dominant_direction = "no_significant_direction"

        return {
            "statistic_type": "directed_transfer_entropy",
            "interpretation_label": "directed_information_transfer",
            "series": names,
            "total_count": total_count,
            "aligned_sample_size": aligned_size,
            "missing_pair_count": missing_pair_count,
            "effective_sample_size": effective_sample_size,
            "lag": lag,
            "requested_bins": bins,
            "actual_bins": {
                names[0]: first_actual_bins,
                names[1]: second_actual_bins,
            },
            "surrogate_count": surrogate_count,
            "alpha": alpha,
            "seed": seed,
            "directions": {
                f"{names[0]}_to_{names[1]}": first_summary,
                f"{names[1]}_to_{names[0]}": second_summary,
            },
            "effective_transfer_balance_bits": balance,
            "dominant_direction": dominant_direction,
            "causal_claim": False,
            "interpretation_warning": (
                "Transfer entropy measures directional predictive information, "
                "not proof of causality."
            ),
            "estimation": {
                "discretization": "empirical_quantile_bins",
                "logarithm_base": 2,
                "surrogate_method": "circular_shift",
            },
        }
