
import pytest

from agent.analytics_executor import (
    AnalyticsExecutorError,
    AnalyticsToolExecutor,
)
from agent.runtime import AgentRuntime
from agent.tool_call_trace import ToolCallTrace


def test_catalog_has_102_unique_tools():
    bridge = AnalyticsToolExecutor()
    tools = bridge.available_tools()
    names = [item["name"] for item in tools]

    assert len(tools) == 102
    assert len(names) == len(set(names))


def test_all_seven_vera_extensions_are_present():
    bridge = AnalyticsToolExecutor()
    names = {item["name"] for item in bridge.available_tools()}

    expected = {
        "one_way_anova",
        "mann_whitney_u_test",
        "kruskal_wallis_test",
        "multiple_testing_correction",
        "extreme_value_tail_risk",
        "bayesian_structural_causal_impact",
        "directed_transfer_entropy",
    }
    assert expected <= names


def test_vera_extensions_override_core_duplicates():
    bridge = AnalyticsToolExecutor()

    assert bridge.tool_source("one_way_anova") == "vera_extension"
    assert bridge.tool_source("mann_whitney_u_test") == "vera_extension"


def test_regular_core_method_keeps_core_source():
    bridge = AnalyticsToolExecutor()

    assert bridge.tool_source("descriptive_statistics") == "core_engine"


def test_available_tools_exposes_source_metadata():
    bridge = AnalyticsToolExecutor()
    by_name = {item["name"]: item for item in bridge.available_tools()}

    assert by_name["one_way_anova"]["source"] == "vera_extension"
    assert by_name["descriptive_statistics"]["source"] == "core_engine"


def test_extension_executes_through_bridge():
    bridge = AnalyticsToolExecutor()

    result = bridge.execute(
        "one_way_anova",
        parameters={"alpha": 0.05},
        bindings={
            "data": {
                "north": [1, 2, 3],
                "south": [4, 5, 6],
                "west": [7, 8, 9],
            }
        },
    )

    assert result["statistic_type"] == "one_way_anova"
    assert result["f_statistic"] == pytest.approx(27.0)


def test_core_result_is_normalized_to_dict():
    bridge = AnalyticsToolExecutor()

    result = bridge.execute(
        "descriptive_statistics",
        parameters={},
        bindings={"values": [1.0, 2.0, 3.0, 4.0]},
    )

    assert isinstance(result, dict)
    assert result["method"] == "descriptive_statistics"
    assert result["status"] == "success"
    assert isinstance(result["result"], dict)


def test_conflicting_parameter_and_binding_is_rejected():
    bridge = AnalyticsToolExecutor()

    with pytest.raises(AnalyticsExecutorError):
        bridge.execute(
            "one_way_anova",
            parameters={"data": {"a": [1, 2], "b": [3, 4]}},
            bindings={"data": {"a": [9, 9], "b": [8, 8]}},
        )


def test_unknown_tool_is_rejected():
    bridge = AnalyticsToolExecutor()

    with pytest.raises(AnalyticsExecutorError):
        bridge.executor_for("does_not_exist")


def test_executor_map_is_agent_runtime_compatible():
    bridge = AnalyticsToolExecutor()
    runtime = AgentRuntime(bridge.executor_map())

    assert len(runtime.available_tools) == 102
    assert "one_way_anova" in runtime.available_tools
    assert "descriptive_statistics" in runtime.available_tools


def test_runtime_can_execute_real_vera_extension():
    bridge = AnalyticsToolExecutor()
    runtime = AgentRuntime(bridge.executor_map())

    trace = ToolCallTrace.planned(
        call_id="call-anova",
        tool="one_way_anova",
        objective="Compare three regional means",
        reason="Test whether regional means differ",
        parameters={"alpha": 0.05},
        bindings={
            "data": {
                "north": [1, 2, 3, 4],
                "south": [5, 6, 7, 8],
                "west": [9, 10, 11, 12],
            }
        },
    )

    result = runtime.run(trace)

    assert result.steps
    assert result.steps[0].trace.tool == "one_way_anova"
    assert result.steps[0].trace.deterministic_result is not None
