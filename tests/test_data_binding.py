import pandas as pd
import pytest

from agent.data_binding import BoundToolCall, DataBindingError, DatasetBinder
from agent.tool_call_trace import ToolCallTrace


def _trace(tool, bindings, parameters=None):
    return ToolCallTrace.planned(
        call_id="call-1",
        tool=tool,
        objective="Test local binding",
        reason="Exercise deterministic binder",
        parameters=parameters or {},
        bindings=bindings,
    )


def test_anova_semantic_columns_are_bound_to_grouped_local_values():
    frame = pd.DataFrame(
        {
            "region": ["North", "North", "South", "South", "West", "West"],
            "sales": [10, 12, 20, 22, 30, 32],
        }
    )
    trace = _trace(
        "one_way_anova",
        {"grouping_column": "region", "value_column": "sales"},
        {"alpha": 0.05},
    )

    bound = DatasetBinder().bind(trace, frame)

    assert bound.tool == "one_way_anova"
    assert bound.bindings == {
        "data": {
            "North": [10, 12],
            "South": [20, 22],
            "West": [30, 32],
        }
    }
    assert bound.parameters == {"alpha": 0.05}


def test_binding_does_not_mutate_or_expose_raw_values_in_trace():
    frame = pd.DataFrame({"region": ["A", "A", "B", "B"], "sales": [1, 2, 3, 4]})
    trace = _trace(
        "one_way_anova",
        {"grouping_column": "region", "value_column": "sales"},
    )

    DatasetBinder().bind(trace, frame)

    assert trace.bindings == {
        "grouping_column": "region",
        "value_column": "sales",
    }
    assert "data" not in trace.bindings


def test_missing_measure_values_become_none_for_downstream_method():
    frame = pd.DataFrame(
        {"region": ["A", "A", "B", "B"], "sales": [1.0, None, 3.0, 4.0]}
    )
    bound = DatasetBinder().bind(
        _trace(
            "kruskal_wallis_test",
            {"grouping_column": "region", "value_column": "sales"},
        ),
        frame,
    )

    assert bound.bindings["data"]["A"] == [1.0, None]


def test_rows_with_missing_group_are_not_turned_into_fake_group():
    frame = pd.DataFrame(
        {"region": ["A", None, "B", "B"], "sales": [1, 99, 3, 4]}
    )
    bound = DatasetBinder().bind(
        _trace(
            "one_way_anova",
            {"grouping_column": "region", "value_column": "sales"},
        ),
        frame,
    )

    assert set(bound.bindings["data"]) == {"A", "B"}
    assert 99 not in bound.bindings["data"].get("A", [])


def test_descriptive_statistics_maps_value_column_to_values():
    frame = pd.DataFrame({"sales": [1.0, 2.0, 3.0]})
    bound = DatasetBinder().bind(
        _trace("descriptive_statistics", {"value_column": "sales"}),
        frame,
    )

    assert bound.bindings == {"values": [1.0, 2.0, 3.0]}


def test_mann_whitney_infers_two_groups():
    frame = pd.DataFrame(
        {"segment": ["A", "A", "B", "B"], "sales": [1, 2, 8, 9]}
    )
    bound = DatasetBinder().bind(
        _trace(
            "mann_whitney_u_test",
            {"grouping_column": "segment", "value_column": "sales"},
        ),
        frame,
    )

    assert bound.bindings == {"sample_a": [1, 2], "sample_b": [8, 9]}


def test_mann_whitney_requires_explicit_groups_when_more_than_two():
    frame = pd.DataFrame(
        {"segment": ["A", "B", "C"], "sales": [1, 2, 3]}
    )

    with pytest.raises(DataBindingError):
        DatasetBinder().bind(
            _trace(
                "mann_whitney_u_test",
                {"grouping_column": "segment", "value_column": "sales"},
            ),
            frame,
        )


def test_unknown_column_is_rejected():
    frame = pd.DataFrame({"sales": [1, 2, 3]})

    with pytest.raises(DataBindingError, match="Unknown dataset column"):
        DatasetBinder().bind(
            _trace("descriptive_statistics", {"value_column": "revenue"}),
            frame,
        )


def test_grouped_analysis_requires_two_groups():
    frame = pd.DataFrame({"region": ["A", "A"], "sales": [1, 2]})

    with pytest.raises(DataBindingError, match="at least two"):
        DatasetBinder().bind(
            _trace(
                "one_way_anova",
                {"grouping_column": "region", "value_column": "sales"},
            ),
            frame,
        )


def test_executor_kwargs_merge_parameters_and_bound_values():
    call = BoundToolCall(
        call_id="call-1",
        tool="x",
        parameters={"alpha": 0.05},
        bindings={"values": [1, 2, 3]},
        source_bindings={"value_column": "sales"},
    )

    assert call.executor_kwargs() == {"alpha": 0.05, "values": [1, 2, 3]}


def test_executor_kwargs_reject_parameter_binding_overlap():
    call = BoundToolCall(
        call_id="call-1",
        tool="x",
        parameters={"values": [9]},
        bindings={"values": [1, 2, 3]},
        source_bindings={"value_column": "sales"},
    )

    with pytest.raises(DataBindingError, match="overlap"):
        call.executor_kwargs()



def test_anova_accepts_model_column_name_bindings():
    frame = pd.DataFrame(
        {
            "region": ["North", "North", "South", "South", "West", "West"],
            "sales": [10, 12, 20, 22, 30, 32],
        }
    )
    bound = DatasetBinder().bind(
        _trace("one_way_anova", {"region": "region", "sales": "sales"}),
        frame,
    )

    assert bound.bindings == {
        "data": {
            "North": [10, 12],
            "South": [20, 22],
            "West": [30, 32],
        }
    }


def test_grouped_binding_infers_one_categorical_and_one_numeric_column():
    frame = pd.DataFrame(
        {
            "market": ["A", "A", "B", "B"],
            "profit": [1.0, 2.0, 8.0, 9.0],
        }
    )
    bound = DatasetBinder().bind(
        _trace("kruskal_wallis_test", {"x": "market", "y": "profit"}),
        frame,
    )

    assert bound.bindings["data"] == {"A": [1.0, 2.0], "B": [8.0, 9.0]}


def test_descriptive_statistics_accepts_model_column_name_binding():
    frame = pd.DataFrame({"sales": [1.0, 2.0, 3.0]})
    bound = DatasetBinder().bind(
        _trace("descriptive_statistics", {"sales": "sales"}),
        frame,
    )

    assert bound.bindings == {"values": [1.0, 2.0, 3.0]}


def test_grouped_binding_rejects_ambiguous_numeric_columns_without_hints():
    frame = pd.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [10, 20, 30, 40],
        }
    )

    with pytest.raises(DataBindingError, match="Could not deterministically resolve"):
        DatasetBinder().bind(
            _trace("one_way_anova", {"x": "a", "y": "b"}),
            frame,
        )
