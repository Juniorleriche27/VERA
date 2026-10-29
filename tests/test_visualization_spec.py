import pytest

from agent.decision_brief import EvidenceItem, EvidencePackage
from visualization.spec import (
    VerifiedVisualizationBuilder,
    VisualizationSeries,
    VisualizationSpec,
    VisualizationSpecError,
)


def _evidence(result=None, *, tool="one_way_anova"):
    if result is None:
        result = {
            "group_statistics": {
                "North": {"mean": 12.0, "sample_size": 5},
                "South": {"mean": 22.0, "sample_size": 5},
                "West": {"mean": 32.0, "sample_size": 5},
            }
        }
    return EvidencePackage(
        question="Are average sales different?",
        items=(
            EvidenceItem(
                call_id="call-1",
                tool=tool,
                trace_hash="a" * 64,
                objective="Compare regional sales.",
                result=result,
            ),
        ),
    )


def test_builds_bar_chart_from_verified_anova_group_means():
    specs = VerifiedVisualizationBuilder().build(_evidence())
    assert len(specs) == 1
    spec = specs[0]
    assert spec.chart_type == "bar"
    assert spec.x_key == "group"
    assert spec.data == (
        {"group": "North", "mean": 12.0},
        {"group": "South", "mean": 22.0},
        {"group": "West", "mean": 32.0},
    )


def test_chart_values_are_copied_from_evidence_not_invented():
    evidence = _evidence(
        {"group_statistics": {"A": {"mean": 1.25}, "B": {"mean": 9.75}}}
    )
    spec = VerifiedVisualizationBuilder().build(evidence)[0]
    assert [row["mean"] for row in spec.data] == [1.25, 9.75]


def test_preserves_evidence_provenance():
    spec = VerifiedVisualizationBuilder().build(_evidence())[0]
    assert spec.call_id == "call-1"
    assert spec.trace_hash == "a" * 64
    assert spec.evidence_tool == "one_way_anova"


def test_as_dict_is_frontend_chart_ready():
    payload = VerifiedVisualizationBuilder().build(_evidence())[0].as_dict()
    assert payload["chartType"] == "bar"
    assert payload["xKey"] == "group"
    assert payload["series"][0]["dataKey"] == "mean"
    assert payload["evidence"]["trace_hash"] == "a" * 64


def test_unsupported_verified_tool_produces_no_fake_chart():
    assert VerifiedVisualizationBuilder().build(
        _evidence(tool="descriptive_statistics")
     ) == ()


def test_rejects_non_evidence_package():
    with pytest.raises(VisualizationSpecError):
        VerifiedVisualizationBuilder().build(object())


def test_rejects_missing_group_statistics():
    with pytest.raises(VisualizationSpecError):
        VerifiedVisualizationBuilder().build(_evidence({}))


def test_rejects_non_numeric_group_mean():
    with pytest.raises(VisualizationSpecError):
        VerifiedVisualizationBuilder().build(
            _evidence({"group_statistics": {"North": {"mean": "12"}}})
        )


def test_rejects_non_finite_group_mean():
    with pytest.raises(VisualizationSpecError):
        VerifiedVisualizationBuilder().build(
            _evidence({"group_statistics": {"North": {"mean": float("inf")}}})
        )


def test_visualization_spec_and_series_are_immutable_dataclasses():
    spec = VerifiedVisualizationBuilder().build(_evidence())[0]
    assert isinstance(spec, VisualizationSpec)
    assert isinstance(spec.series[0], VisualizationSeries)
    with pytest.raises(Exception):
        spec.chart_type = "line"
