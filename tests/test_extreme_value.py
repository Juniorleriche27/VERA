import numpy as np
import pytest

from analytics_extensions.extreme_value import ExtremeValueTailRisk
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError


def test_evt_quantile_threshold_and_tail_outputs():
    result = ExtremeValueTailRisk().execute(
        values=list(range(1, 101)),
        threshold_quantile=0.80,
        tail_probabilities=[0.10, 0.05],
        return_periods=[20, 50],
        evaluate_at=[90, 95],
    )

    assert result["sample_size"] == 100
    assert result["exceedance_count"] == 20
    assert result["exceedance_rate"] == pytest.approx(0.20)
    assert result["threshold_source"] == "quantile"
    assert result["gpd_scale"] > 0.0
    assert set(result["extreme_quantiles"]) == {"0.1", "0.05"}
    assert set(result["return_levels"]) == {"20.0", "50.0"}
    assert (
        result["tail_probabilities"]["95.0"]
        < result["tail_probabilities"]["90.0"]
    )


def test_evt_explicit_threshold_is_preserved():
    result = ExtremeValueTailRisk().execute(
        values=list(range(-50, 51)),
        threshold=-10.0,
        tail_probabilities=[0.05],
        return_periods=[25],
    )

    assert result["threshold"] == pytest.approx(-10.0)
    assert result["threshold_source"] == "explicit"
    assert result["threshold_quantile"] is None


def test_evt_tracks_missing_values():
    values = list(range(1, 41)) + [None, np.nan]

    result = ExtremeValueTailRisk().execute(
        values=values,
        threshold=30.0,
        tail_probabilities=[0.05],
        return_periods=[25],
    )

    assert result["total_count"] == 42
    assert result["sample_size"] == 40
    assert result["missing_count"] == 2


def test_evt_return_levels_increase_with_return_period():
    result = ExtremeValueTailRisk().execute(
        values=list(range(1, 101)),
        threshold=75.0,
        tail_probabilities=[0.10],
        return_periods=[10, 100],
    )

    assert (
        result["return_levels"]["100.0"]
        > result["return_levels"]["10.0"]
    )


def test_evt_reports_fit_diagnostics_and_tail_regime():
    result = ExtremeValueTailRisk().execute(
        values=list(range(1, 101)),
        threshold=70.0,
        tail_probabilities=[0.10],
        return_periods=[20],
    )

    assert result["fit_diagnostics"]["parameter_valid"] is True
    assert result["fit_diagnostics"]["fit_method"] == "method_of_moments"
    assert result["tail_regime"] in {
        "heavy_tail",
        "bounded_tail",
        "approximately_exponential_tail",
    }


def test_evt_requires_three_exceedances():
    with pytest.raises(InvalidAnalysisInputError):
        ExtremeValueTailRisk().execute(
            values=[1, 2, 3, 4, 5],
            threshold=3.5,
            tail_probabilities=[0.1],
            return_periods=[20],
        )


def test_evt_rejects_non_numeric_values():
    with pytest.raises(InvalidAnalysisInputError):
        ExtremeValueTailRisk().execute(
            values=[1, 2, "bad", 4, 5, 6],
            threshold=2.0,
        )


def test_evt_rejects_probability_above_exceedance_rate():
    with pytest.raises(InvalidAnalysisInputError):
        ExtremeValueTailRisk().execute(
            values=list(range(1, 101)),
            threshold=80.0,
            tail_probabilities=[0.30],
            return_periods=[20],
        )


def test_evt_rejects_evaluation_below_threshold():
    with pytest.raises(InvalidAnalysisInputError):
        ExtremeValueTailRisk().execute(
            values=list(range(1, 101)),
            threshold=80.0,
            tail_probabilities=[0.10],
            return_periods=[20],
            evaluate_at=[70.0],
        )
