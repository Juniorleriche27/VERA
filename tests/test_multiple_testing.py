import pytest

from analytics_extensions.multiple_testing import MultipleTestingCorrection
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError


def test_bonferroni_adjusts_and_rejects_in_original_order():
    result = MultipleTestingCorrection().execute(
        p_values=[0.01, 0.04, 0.20],
        alpha=0.05,
        method="bonferroni",
    )

    assert result["method"] == "bonferroni"
    assert result["error_control"] == "family_wise_error_rate"
    assert result["adjusted_p_values"] == pytest.approx([0.03, 0.12, 0.60])
    assert result["reject"] == [True, False, False]
    assert result["rejection_count"] == 1


def test_benjamini_hochberg_known_example():
    result = MultipleTestingCorrection().execute(
        p_values=[0.01, 0.04, 0.03, 0.002],
        alpha=0.05,
        method="benjamini_hochberg",
    )

    assert result["adjusted_p_values"] == pytest.approx(
        [0.02, 0.04, 0.04, 0.008]
    )
    assert result["reject"] == [True, True, True, True]
    assert result["rejection_count"] == 4


def test_benjamini_hochberg_preserves_original_order():
    result = MultipleTestingCorrection().execute(
        p_values=[0.20, 0.01, 0.04],
        method="bh",
    )

    assert result["p_values"] == pytest.approx([0.20, 0.01, 0.04])
    assert result["adjusted_p_values"] == pytest.approx([0.20, 0.03, 0.06])


def test_fdr_bh_alias_is_normalized():
    result = MultipleTestingCorrection().execute(
        p_values=[0.01, 0.50],
        method="fdr-bh",
    )

    assert result["method"] == "benjamini_hochberg"
    assert result["error_control"] == "false_discovery_rate"


def test_multiple_testing_rejects_empty_p_values():
    with pytest.raises(InvalidAnalysisInputError):
        MultipleTestingCorrection().execute(p_values=[])


def test_multiple_testing_rejects_invalid_p_value():
    with pytest.raises(InvalidAnalysisInputError):
        MultipleTestingCorrection().execute(p_values=[0.01, 1.2])


def test_multiple_testing_rejects_invalid_alpha():
    with pytest.raises(InvalidAnalysisInputError):
        MultipleTestingCorrection().execute(
            p_values=[0.01, 0.02],
            alpha=0.0,
        )


def test_multiple_testing_rejects_unknown_method():
    with pytest.raises(InvalidAnalysisInputError):
        MultipleTestingCorrection().execute(
            p_values=[0.01, 0.02],
            method="holm",
        )
