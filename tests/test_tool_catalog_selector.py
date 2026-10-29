from __future__ import annotations

from agent.meta_planner import MetaPlanningResult
from agent.pipeline import VerifiedAnalyticsPipeline
from agent.tool_catalog_selector import (
    ToolCatalogSelector,
    ToolSelectionError,
)


def _tool(name: str, description: str = "", category: str = "statistics") -> dict:
    return {
        "name": name,
        "description": description,
        "category": category,
        "source": "test",
    }


CATALOG = [
    _tool("descriptive_statistics", "Summarize numeric values."),
    _tool("one_way_anova", "Compare means across independent groups."),
    _tool("mann_whitney_u_test", "Compare two independent groups using ranks."),
    _tool("kruskal_wallis_test", "Compare multiple independent groups using ranks."),
    _tool("multiple_testing_correction", "Correct p-values for multiple tests."),
    _tool("extreme_value_tail_risk", "Estimate extreme tail risk."),
    _tool("bayesian_structural_causal_impact", "Estimate intervention or campaign impact over time."),
    _tool("directed_transfer_entropy", "Estimate directional information flow."),
    _tool("correlation_analysis", "Measure correlation between variables."),
    _tool("linear_regression", "Explain or predict an outcome from drivers."),
    _tool("change_point_detection", "Detect structural breaks and shifts in time."),
    _tool("time_series_forecast", "Forecast a measure over time."),
]


def test_selector_rejects_invalid_limit():
    try:
        ToolCatalogSelector(max_tools=0)
    except ToolSelectionError:
        return
    raise AssertionError("Expected ToolSelectionError.")


def test_selector_rejects_blank_question():
    selector = ToolCatalogSelector()
    try:
        selector.select("   ", dataset_context={}, tool_catalog=CATALOG)
    except ToolSelectionError:
        return
    raise AssertionError("Expected ToolSelectionError.")


def test_selector_respects_max_tools():
    result = ToolCatalogSelector(max_tools=5).select(
        "Summarize sales and identify useful patterns.",
        dataset_context={"row_count": 100},
        tool_catalog=CATALOG,
    )
    assert len(result.tools) == 5
    assert result.total_available == len(CATALOG)
    assert result.max_tools == 5


def test_selector_is_deterministic():
    selector = ToolCatalogSelector(max_tools=6)
    kwargs = {
        "dataset_context": {
            "row_count": 100,
            "columns": [
                {"name": "region", "role": "dimension"},
                {"name": "sales", "role": "measure"},
            ],
        },
        "tool_catalog": CATALOG,
    }
    first = selector.select(
        "Are average sales significantly different across three regions?",
        **kwargs,
    )
    second = selector.select(
        "Are average sales significantly different across three regions?",
        **kwargs,
    )
    assert first.ranked_names == second.ranked_names


def test_selector_prioritizes_anova_for_three_group_mean_question():
    result = ToolCatalogSelector(max_tools=4).select(
        "Are average sales significantly different across the three regions?",
        dataset_context={
            "columns": [
                {"name": "region", "role": "dimension"},
                {"name": "sales", "role": "measure"},
            ]
        },
        tool_catalog=CATALOG,
    )
    assert result.ranked_names[0] == "one_way_anova"


def test_selector_prioritizes_causal_impact_for_campaign_question():
    result = ToolCatalogSelector(max_tools=4).select(
        "Did the September campaign actually have a causal impact on sales over time?",
        dataset_context={"columns": [{"name": "date"}, {"name": "sales"}]},
        tool_catalog=CATALOG,
    )
    assert result.ranked_names[0] == "bayesian_structural_causal_impact"


def test_selector_prioritizes_kruskal_for_nonparametric_many_groups():
    result = ToolCatalogSelector(max_tools=4).select(
        "Use a nonparametric rank test to compare three independent groups.",
        dataset_context={},
        tool_catalog=CATALOG,
    )
    assert result.ranked_names[0] == "kruskal_wallis_test"


def test_selector_prioritizes_tail_risk():
    result = ToolCatalogSelector(max_tools=3).select(
        "Estimate extreme tail risk and expected shortfall.",
        dataset_context={},
        tool_catalog=CATALOG,
    )
    assert result.ranked_names[0] == "extreme_value_tail_risk"


def test_selector_does_not_mutate_catalog():
    catalog = [_tool("one_way_anova", "Compare means."), _tool("linear_regression")]
    snapshot = [dict(item) for item in catalog]
    ToolCatalogSelector(max_tools=1).select(
        "Compare average values across three groups.",
        dataset_context={},
        tool_catalog=catalog,
    )
    assert catalog == snapshot


def test_pipeline_sends_only_preselected_tools_to_planner():
    class CapturingPlanner:
        def __init__(self):
            self.catalog = None

        def plan(self, question, *, dataset_context, tool_catalog):
            self.catalog = list(tool_catalog)
            return MetaPlanningResult(
                traces=(),
                assistant_content="",
                finish_reason="stop",
                model="fake",
                usage={},
            )

    class FakeAnalytics:
        def available_tools(self):
            extras = [
                _tool(f"irrelevant_method_{index}", "Unrelated analytics method.")
                for index in range(20)
            ]
            return CATALOG + extras

    planner = CapturingPlanner()
    pipeline = VerifiedAnalyticsPipeline(
        planner,
        analytics=FakeAnalytics(),
        tool_selector=ToolCatalogSelector(max_tools=5),
    )
    pipeline.run(
        "Are average sales significantly different across three regions?",
        dataset={},
        dataset_context={
            "row_count": 15,
            "columns": [
                {"name": "region", "role": "dimension"},
                {"name": "sales", "role": "measure"},
            ],
        },
    )

    assert planner.catalog is not None
    assert len(planner.catalog) == 5
    assert any(item["name"] == "one_way_anova" for item in planner.catalog)
