import numpy as np
import pytest

from analytics_extensions.causal_impact import BayesianStructuralCausalImpact
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError


def test_causal_impact_detects_positive_intervention_effect():
    outcome = [10, 11, 12, 13, 14, 20, 21, 22]
    result = BayesianStructuralCausalImpact().execute(
        outcome=outcome,
        intervention_index=5,
        prior_scale=100.0,
    )

    assert result["pre_intervention_count"] == 5
    assert result["post_intervention_count"] == 3
    assert result["absolute_effect"] > 10.0
    assert result["relative_effect_percent"] > 20.0
    assert result["direction"] == "positive"
    assert len(result["counterfactual_post_mean"]) == 3


def test_causal_impact_near_zero_without_intervention():
    outcome = [10, 11, 12, 13, 14, 15, 16, 17]
    result = BayesianStructuralCausalImpact().execute(
        outcome=outcome,
        intervention_index=5,
        prior_scale=100.0,
    )

    assert abs(result["absolute_effect"]) < 1.0
    assert result["pre_intervention_rmse"] < 0.1


def test_causal_impact_uses_controls():
    control = [1, 2, 3, 4, 5, 6, 7, 8]
    outcome = [2, 4, 6, 8, 10, 17, 19, 21]
    result = BayesianStructuralCausalImpact().execute(
        outcome=outcome,
        intervention_index=5,
        controls={"market": control},
        prior_scale=100.0,
    )

    assert result["control_names"] == ["market"]
    assert result["parameter_count"] == 3
    assert result["absolute_effect"] > 10.0


def test_causal_impact_returns_ordered_intervals():
    result = BayesianStructuralCausalImpact().execute(
        outcome=[10, 11, 12, 13, 14, 20, 21],
        intervention_index=5,
    )

    low, high = result["absolute_effect_interval"]
    assert low <= result["absolute_effect"] <= high

    for low, high in result["counterfactual_post_interval"]:
        assert low <= high


def test_causal_impact_rejects_too_short_pre_period():
    with pytest.raises(InvalidAnalysisInputError):
        BayesianStructuralCausalImpact().execute(
            outcome=[1, 2, 3, 4, 10, 11],
            intervention_index=4,
        )


def test_causal_impact_rejects_too_short_post_period():
    with pytest.raises(InvalidAnalysisInputError):
        BayesianStructuralCausalImpact().execute(
            outcome=[1, 2, 3, 4, 5, 6],
            intervention_index=5,
        )


def test_causal_impact_rejects_missing_outcome():
    with pytest.raises(InvalidAnalysisInputError):
        BayesianStructuralCausalImpact().execute(
            outcome=[1, 2, 3, None, 5, 6, 7],
            intervention_index=5,
        )


def test_causal_impact_rejects_control_length_mismatch():
    with pytest.raises(InvalidAnalysisInputError):
        BayesianStructuralCausalImpact().execute(
            outcome=[1, 2, 3, 4, 5, 6, 7],
            intervention_index=5,
            controls={"market": [1, 2, 3]},
        )


def test_causal_impact_rejects_invalid_credibility():
    with pytest.raises(InvalidAnalysisInputError):
        BayesianStructuralCausalImpact().execute(
            outcome=[1, 2, 3, 4, 5, 6, 7],
            intervention_index=5,
            credibility=1.0,
        )


def test_causal_impact_is_explicitly_conditional():
    result = BayesianStructuralCausalImpact().execute(
        outcome=[1, 2, 3, 4, 5, 8, 9],
        intervention_index=5,
    )

    assert (
        result["causal_interpretation"]
        == "conditional_on_model_specification_and_counterfactual_assumptions"
    )
