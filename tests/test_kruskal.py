import numpy as np
import pytest

from analytics_extensions.kruskal import KruskalWallisTest
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError


def test_kruskal_separated_groups():
    result = KruskalWallisTest().execute(
        data={
            "a": [1, 2, 3, 4, 5],
            "b": [10, 11, 12, 13, 14],
            "c": [20, 21, 22, 23, 24],
        }
    )

    assert result["h_statistic"] == pytest.approx(12.5)
    assert result["degrees_of_freedom"] == 2
    assert result["p_value"] < 0.01
    assert result["epsilon_squared"] == pytest.approx(0.875)


def test_kruskal_identical_groups():
    result = KruskalWallisTest().execute(
        data={
            "a": [1, 2, 3, 4],
            "b": [1, 2, 3, 4],
            "c": [1, 2, 3, 4],
        }
    )

    assert result["h_statistic"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["epsilon_squared"] == pytest.approx(0.0)


def test_kruskal_tracks_missing():
    result = KruskalWallisTest().execute(
        data={
            "a": [1, None, 2, 3],
            "b": [4, 5, np.nan, 6],
            "c": [7, 8, 9, None],
        }
    )

    assert result["total_count"] == 12
    assert result["sample_size"] == 9
    assert result["missing_count"] == 3


def test_kruskal_reports_rank_summaries():
    result = KruskalWallisTest().execute(
        data={
            "low": [1, 2, 3],
            "high": [7, 8, 9],
        }
    )

    assert (
        result["group_statistics"]["low"]["mean_rank"]
        < result["group_statistics"]["high"]["mean_rank"]
    )
    assert result["group_statistics"]["low"]["rank_sum"] == pytest.approx(6.0)
    assert result["group_statistics"]["high"]["rank_sum"] == pytest.approx(15.0)


def test_kruskal_requires_two_groups():
    with pytest.raises(InvalidAnalysisInputError):
        KruskalWallisTest().execute(data={"a": [1, 2, 3]})


def test_kruskal_rejects_non_numeric():
    with pytest.raises(InvalidAnalysisInputError):
        KruskalWallisTest().execute(
            data={
                "a": [1, 2, 3],
                "b": [4, "bad", 6],
            }
        )


def test_kruskal_rejects_duplicate_normalized_group_names():
    with pytest.raises(InvalidAnalysisInputError):
        KruskalWallisTest().execute(
            data={
                "A": [1, 2, 3],
                " A ": [4, 5, 6],
            }
        )
