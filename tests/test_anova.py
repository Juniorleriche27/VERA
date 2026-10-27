import numpy as np
import pytest

from analytics_extensions.anova import OneWayANOVA
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError


def test_one_way_anova_computes_known_example():
    result = OneWayANOVA().execute(
        data={
            "A": [1.0, 2.0, 3.0],
            "B": [4.0, 5.0, 6.0],
            "C": [7.0, 8.0, 9.0],
        },
    )

    assert result["f_statistic"] == pytest.approx(27.0)
    assert result["p_value"] == pytest.approx(0.001)
    assert result["df_between"] == 2
    assert result["df_within"] == 6
    assert result["eta_squared"] == pytest.approx(0.9)
    assert result["omega_squared"] == pytest.approx(0.8524590163934426)


def test_one_way_anova_identical_groups_have_p_one():
    result = OneWayANOVA().execute(
        data={
            "A": [1.0, 2.0, 3.0],
            "B": [1.0, 2.0, 3.0],
            "C": [1.0, 2.0, 3.0],
        },
    )

    assert result["f_statistic"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["eta_squared"] == pytest.approx(0.0)
    assert result["omega_squared"] == pytest.approx(0.0)


def test_one_way_anova_tracks_missing_values():
    result = OneWayANOVA().execute(
        data={
            "A": [1.0, None, 2.0, 3.0],
            "B": [4.0, 5.0, np.nan, 6.0],
        },
    )

    assert result["total_count"] == 8
    assert result["sample_size"] == 6
    assert result["missing_count"] == 2
    assert result["group_statistics"]["A"]["missing_count"] == 1
    assert result["group_statistics"]["B"]["missing_count"] == 1


def test_one_way_anova_requires_two_groups():
    with pytest.raises(InvalidAnalysisInputError):
        OneWayANOVA().execute(
            data={"A": [1.0, 2.0, 3.0]},
        )


def test_one_way_anova_rejects_non_numeric_values():
    with pytest.raises(InvalidAnalysisInputError):
        OneWayANOVA().execute(
            data={
                "A": [1.0, 2.0, 3.0],
                "B": [4.0, "bad", 6.0],
            },
        )


def test_one_way_anova_normalizes_and_rejects_duplicate_group_names():
    with pytest.raises(InvalidAnalysisInputError):
        OneWayANOVA().execute(
            data={
                "A": [1.0, 2.0, 3.0],
                " A ": [4.0, 5.0, 6.0],
            },
        )
