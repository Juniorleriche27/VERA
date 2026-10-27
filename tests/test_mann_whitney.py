import numpy as np
import pytest

from analytics_extensions.nonparametric import MannWhitneyUTest
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError


def test_mann_whitney_detects_separated_samples():
    result = MannWhitneyUTest().execute(
        data={
            "control": [1.0, 2.0, 3.0, 4.0, 5.0],
            "treatment": [10.0, 11.0, 12.0, 13.0, 14.0],
        },
    )

    assert result["u_first"] == pytest.approx(0.0)
    assert result["u_second"] == pytest.approx(25.0)
    assert result["rank_biserial_effect_size"] == pytest.approx(-1.0)
    assert result["direction"] == "control_lower"
    assert result["p_value"] < 0.05


def test_mann_whitney_identical_samples_have_no_rank_direction():
    result = MannWhitneyUTest().execute(
        data={
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [1.0, 2.0, 3.0, 4.0],
        },
    )

    assert result["u_first"] == pytest.approx(8.0)
    assert result["rank_biserial_effect_size"] == pytest.approx(0.0)
    assert result["direction"] == "no_rank_direction"
    assert result["p_value"] == pytest.approx(1.0)
    assert result["tie_corrected"] is True


def test_mann_whitney_tracks_missing_values():
    result = MannWhitneyUTest().execute(
        data={
            "a": [1.0, None, 2.0, 3.0],
            "b": [4.0, 5.0, np.nan, 6.0],
        },
    )

    assert result["first_total_count"] == 4
    assert result["second_total_count"] == 4
    assert result["first_sample_size"] == 3
    assert result["second_sample_size"] == 3
    assert result["first_missing_count"] == 1
    assert result["second_missing_count"] == 1


def test_mann_whitney_supports_directional_alternative():
    result = MannWhitneyUTest().execute(
        data={
            "a": [10.0, 11.0, 12.0, 13.0],
            "b": [1.0, 2.0, 3.0, 4.0],
        },
        alternative="greater",
    )

    assert result["u_first"] == pytest.approx(16.0)
    assert result["p_value"] < 0.05
    assert result["rank_biserial_effect_size"] == pytest.approx(1.0)


def test_mann_whitney_requires_exactly_two_samples():
    with pytest.raises(InvalidAnalysisInputError):
        MannWhitneyUTest().execute(
            data={
                "a": [1.0, 2.0],
                "b": [3.0, 4.0],
                "c": [5.0, 6.0],
            },
        )


def test_mann_whitney_rejects_non_numeric_values():
    with pytest.raises(InvalidAnalysisInputError):
        MannWhitneyUTest().execute(
            data={
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, "bad", 6.0],
            },
        )


def test_mann_whitney_rejects_invalid_alternative():
    with pytest.raises(InvalidAnalysisInputError):
        MannWhitneyUTest().execute(
            data={
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
            },
            alternative="invalid",
        )
