import json
from types import SimpleNamespace

import pytest

from agent.meta_planner import (
    MetaModelError,
    MetaModelPlanner,
    build_meta_tools,
)


def _catalog():
    return [
        {
            "name": "descriptive_statistics",
            "category": "statistics",
            "description": "Summarize a numeric measure.",
            "selection_mode": "per_column",
            "analysis_unit": "column",
            "supported_roles": ["measure"],
        },
        {
            "name": "one_way_anova",
            "category": "statistics",
            "description": "Compare independent group means.",
            "selection_mode": "joint",
            "analysis_unit": "dataset",
            "supported_roles": ["measure"],
        },
    ]


class _FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(response)
        )


def _tool_call(
    call_id="call-1",
    name="one_way_anova",
    arguments=None,
):
    if arguments is None:
        arguments = {
            "objective": "Compare regional sales",
            "reason": "Test whether regional means differ",
            "parameters": {"alpha": 0.05},
            "bindings": {
                "measure": "sales",
                "group": "region",
            },
            "source_call_ids": [],
        }
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


def _response(
    *,
    tool_calls=None,
    content=None,
    finish_reason="tool_calls",
):
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
    )
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason=finish_reason,
            )
        ],
        model="muse-spark-1.3",
        usage=usage,
    )


def test_build_meta_tools_preserves_analytics_names():
    tools = build_meta_tools(_catalog())

    assert [tool["function"]["name"] for tool in tools] == [
        "descriptive_statistics",
        "one_way_anova",
    ]
    assert all(tool["type"] == "function" for tool in tools)


def test_tool_schema_requires_auditable_planning_fields():
    schema = build_meta_tools(_catalog())[0]["function"]["parameters"]

    assert set(schema["required"]) == {
        "objective",
        "reason",
        "parameters",
        "bindings",
    }
    assert schema["additionalProperties"] is False


def test_duplicate_tool_names_are_rejected():
    catalog = _catalog() + [_catalog()[0]]

    with pytest.raises(MetaModelError, match="Duplicate tool name"):
        build_meta_tools(catalog)


def test_raw_dataset_rows_are_rejected_before_api_call():
    client = _FakeClient(_response(content="unused", finish_reason="stop"))
    planner = MetaModelPlanner(client=client)

    with pytest.raises(MetaModelError, match="Raw dataset payload"):
        planner.plan(
            "Summarize sales",
            dataset_context={
                "dataset_id": "d1",
                "rows": [{"sales": 10}],
            },
            tool_catalog=_catalog(),
        )

    assert client.chat.completions.calls == []


def test_missing_key_is_rejected_without_injected_client(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)

    with pytest.raises(MetaModelError, match="MODEL_API_KEY"):
        MetaModelPlanner()


def test_plain_text_response_is_supported_because_tool_choice_is_auto():
    client = _FakeClient(
        _response(
            content="The question does not require analysis.",
            finish_reason="stop",
        )
    )
    planner = MetaModelPlanner(client=client)

    result = planner.plan(
        "What columns exist?",
        dataset_context={
            "dataset_id": "d1",
            "row_count": 100,
            "columns": ["sales", "region"],
        },
        tool_catalog=_catalog(),
    )

    assert result.traces == ()
    assert result.assistant_content == (
        "The question does not require analysis."
    )
    assert result.finish_reason == "stop"


def test_single_meta_tool_call_becomes_planned_trace():
    client = _FakeClient(
        _response(tool_calls=[_tool_call()])
    )
    planner = MetaModelPlanner(client=client)

    result = planner.plan(
        "Do regions have different sales?",
        dataset_context={
            "dataset_id": "d1",
            "row_count": 500,
            "columns": ["sales", "region"],
        },
        tool_catalog=_catalog(),
    )

    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.call_id == "call-1"
    assert trace.tool == "one_way_anova"
    assert trace.parameters == {"alpha": 0.05}
    assert trace.bindings == {
        "group": "region",
        "measure": "sales",
    }


def test_parallel_meta_tool_calls_are_preserved():
    calls = [
        _tool_call(call_id="call-1", name="descriptive_statistics"),
        _tool_call(call_id="call-2", name="one_way_anova"),
    ]
    client = _FakeClient(_response(tool_calls=calls))
    planner = MetaModelPlanner(client=client)

    result = planner.plan(
        "Analyze regional sales",
        dataset_context={
            "dataset_id": "d1",
            "columns": ["sales", "region"],
        },
        tool_catalog=_catalog(),
    )

    assert [trace.call_id for trace in result.traces] == [
        "call-1",
        "call-2",
    ]
    assert [trace.tool for trace in result.traces] == [
        "descriptive_statistics",
        "one_way_anova",
    ]


def test_unknown_tool_requested_by_model_is_rejected():
    client = _FakeClient(
        _response(
            tool_calls=[
                _tool_call(name="invented_magic_test")
            ]
        )
    )
    planner = MetaModelPlanner(client=client)

    with pytest.raises(MetaModelError, match="unknown analytics tool"):
        planner.plan(
            "Analyze sales",
            dataset_context={"columns": ["sales"]},
            tool_catalog=_catalog(),
        )


def test_invalid_tool_argument_json_is_rejected():
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="one_way_anova",
            arguments="{broken",
        ),
    )
    planner = MetaModelPlanner(
        client=_FakeClient(_response(tool_calls=[call]))
    )

    with pytest.raises(MetaModelError, match="invalid JSON"):
        planner.plan(
            "Compare sales",
            dataset_context={"columns": ["sales", "region"]},
            tool_catalog=_catalog(),
        )


def test_raw_numeric_arrays_cannot_be_returned_as_bindings():
    call = _tool_call(
        arguments={
            "objective": "Compare sales",
            "reason": "Group comparison",
            "parameters": {},
            "bindings": {"measure": [1.0, 2.0, 3.0]},
        }
    )
    planner = MetaModelPlanner(
        client=_FakeClient(_response(tool_calls=[call]))
    )

    with pytest.raises(MetaModelError, match="Planner bindings"):
        planner.plan(
            "Compare sales",
            dataset_context={"columns": ["sales", "region"]},
            tool_catalog=_catalog(),
        )


def test_request_uses_meta_supported_auto_tool_choice_and_developer_message():
    client = _FakeClient(
        _response(content="No analysis", finish_reason="stop")
    )
    planner = MetaModelPlanner(client=client)

    planner.plan(
        "Inspect the dataset",
        dataset_context={
            "dataset_id": "d1",
            "row_count": 20,
            "columns": ["sales"],
        },
        tool_catalog=_catalog(),
    )

    request = client.chat.completions.calls[0]
    assert request["model"] == "muse-spark-1.3"
    assert request["tool_choice"] == "auto"
    assert request["parallel_tool_calls"] is True
    assert request["messages"][0]["role"] == "developer"
    assert request["messages"][1]["role"] == "user"
    assert len(request["tools"]) == 2


def test_usage_and_result_are_serializable():
    planner = MetaModelPlanner(
        client=_FakeClient(_response(tool_calls=[_tool_call()]))
    )
    result = planner.plan(
        "Compare regions",
        dataset_context={
            "dataset_id": "d1",
            "columns": ["sales", "region"],
        },
        tool_catalog=_catalog(),
    )

    payload = result.as_dict()

    assert payload["model"] == "muse-spark-1.3"
    assert payload["usage"]["total_tokens"] == 120
    assert payload["traces"][0]["tool"] == "one_way_anova"
