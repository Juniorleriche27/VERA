import numpy as np
import pytest

from analytics_extensions.transfer_entropy import DirectedTransferEntropy
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError


def _lagged_driver_sample(size=240, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=size)
    y = np.zeros(size, dtype=float)
    noise = rng.normal(scale=0.15, size=size)
    for index in range(1, size):
        y[index] = 1.8 * x[index - 1] + noise[index]
    return x, y


def test_transfer_entropy_detects_known_x_to_y_information_flow():
    x, y = _lagged_driver_sample()

    result = DirectedTransferEntropy().execute(
        data={"x": x, "y": y},
        lag=1,
        bins=4,
        surrogate_count=120,
        seed=11,
    )

    xy = result["directions"]["x_to_y"]
    yx = result["directions"]["y_to_x"]

    assert xy["transfer_entropy_bits"] > yx["transfer_entropy_bits"]
    assert xy["effective_transfer_entropy_bits"] > 0.0
    assert xy["empirical_p_value"] <= 0.05
    assert xy["significant"] is True
    assert result["dominant_direction"] == "x_to_y"
    assert result["causal_claim"] is False


def test_transfer_entropy_is_deterministic_for_same_seed():
    x, y = _lagged_driver_sample(size=160)

    first = DirectedTransferEntropy().execute(
        data={"x": x, "y": y},
        surrogate_count=60,
        seed=19,
    )
    second = DirectedTransferEntropy().execute(
        data={"x": x, "y": y},
        surrogate_count=60,
        seed=19,
    )

    assert first["directions"] == second["directions"]
    assert first["dominant_direction"] == second["dominant_direction"]


def test_transfer_entropy_tracks_pairwise_missing_values():
    x, y = _lagged_driver_sample(size=40)
    x = x.tolist()
    y = y.tolist()
    x[3] = None
    y[8] = np.nan

    result = DirectedTransferEntropy().execute(
        data={"x": x, "y": y},
        surrogate_count=20,
    )

    assert result["total_count"] == 40
    assert result["missing_pair_count"] == 2
    assert result["aligned_sample_size"] == 38
    assert result["effective_sample_size"] == 37


def test_transfer_entropy_handles_constant_series_without_false_causal_claim():
    x = np.ones(40)
    y = np.ones(40)

    result = DirectedTransferEntropy().execute(
        data={"x": x, "y": y},
        surrogate_count=20,
    )

    assert result["directions"]["x_to_y"]["transfer_entropy_bits"] == pytest.approx(0.0)
    assert result["directions"]["y_to_x"]["transfer_entropy_bits"] == pytest.approx(0.0)
    assert result["dominant_direction"] == "no_significant_direction"
    assert result["causal_claim"] is False


def test_transfer_entropy_rejects_unequal_lengths():
    with pytest.raises(InvalidAnalysisInputError):
        DirectedTransferEntropy().execute(
            data={
                "x": list(range(20)),
                "y": list(range(19)),
            }
        )


def test_transfer_entropy_rejects_too_few_observations():
    with pytest.raises(InvalidAnalysisInputError):
        DirectedTransferEntropy().execute(
            data={
                "x": list(range(10)),
                "y": list(range(10)),
            }
        )


def test_transfer_entropy_rejects_non_numeric_values():
    x = list(range(20))
    y = list(range(20))
    y[5] = "bad"

    with pytest.raises(InvalidAnalysisInputError):
        DirectedTransferEntropy().execute(
            data={"x": x, "y": y}
        )


def test_transfer_entropy_rejects_invalid_lag():
    with pytest.raises(InvalidAnalysisInputError):
        DirectedTransferEntropy().execute(
            data={
                "x": list(range(20)),
                "y": list(range(20)),
            },
            lag=0,
        )


def test_transfer_entropy_rejects_invalid_bins():
    with pytest.raises(InvalidAnalysisInputError):
        DirectedTransferEntropy().execute(
            data={
                "x": list(range(20)),
                "y": list(range(20)),
            },
            bins=1,
        )


def test_transfer_entropy_rejects_too_few_surrogates():
    with pytest.raises(InvalidAnalysisInputError):
        DirectedTransferEntropy().execute(
            data={
                "x": list(range(20)),
                "y": list(range(20)),
            },
            surrogate_count=5,
        )
