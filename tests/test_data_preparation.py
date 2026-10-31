from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import _safe_dataset_profile, create_app
from backend.data_preparation import (
    DatasetPreparationError,
    build_time_series,
    prepare_dataset,
    resolve_intervention_index,
)


def _frame():
    return pd.DataFrame({
        "date": ["2026-08-02", "2026-08-01", "2026-08-01", "2026-08-03"],
        "region": ["North", "North", "South", "West"],
        "sales": [110, 120, 150, 130],
    })


def test_iso_date_converted_to_datetime_and_safe_time_role():
    prepared = prepare_dataset(_frame())
    assert prepared.temporal_columns == ("date",)
    assert pd.api.types.is_datetime64_any_dtype(prepared.frame["date"])
    profile = _safe_dataset_profile(prepared.frame)
    assert [(x["name"], x["role"]) for x in profile["columns"]] == [
        ("date", "time"), ("region", "dimension"), ("sales", "measure")
    ]
    assert set(profile) == {"row_count", "column_count", "columns"}
    assert "North" not in str(profile)  # no raw region values leaked


def test_unrelated_text_is_never_converted():
    frame = pd.DataFrame({"region": ["2026-08-01", "2026-08-02"], "sales": [1, 2]})
    out = prepare_dataset(frame).frame
    assert not pd.api.types.is_datetime64_any_dtype(out["region"])
    assert out["region"].tolist() == ["2026-08-01", "2026-08-02"]


def test_invalid_explicit_date_rejected_not_silently_coerced():
    with pytest.raises(DatasetPreparationError, match="Invalid date"):
        prepare_dataset(pd.DataFrame({"date": ["2026-08-01", "NOT_A_DATE"]}))


def test_missing_explicit_dates_remain_missing():
    frame = pd.DataFrame({"date": ["2026-08-01", None], "sales": [10, 20]})
    prepared = prepare_dataset(frame)
    assert pd.isna(prepared.frame.loc[1, "date"])


def test_other_semantic_date_aliases():
    frame = pd.DataFrame({"Timestamp": ["2026-08-01", "2026-08-02"]})
    assert prepare_dataset(frame).temporal_columns == ("Timestamp",)


def test_already_sorted_and_unsorted_series():
    prepared = prepare_dataset(_frame()).frame
    series = build_time_series(prepared, time_column="date", value_column="sales", aggregation="sum")
    assert [t.strftime("%Y-%m-%d") for t in series.timestamps] == [
        "2026-08-01", "2026-08-02", "2026-08-03"
    ]
    assert series.values == (270.0, 110.0, 130.0)
    assert series.source_row_count == 4
    assert series.period_count == 3
    assert series.aggregation == "sum"
    sorted_frame = prepared.sort_values("date")
    assert build_time_series(sorted_frame, time_column="date", value_column="sales", aggregation="sum").values == series.values


def test_grouped_mean_is_arithmetic_mean():
    view = build_time_series(prepare_dataset(_frame()).frame, time_column="date", value_column="sales", aggregation="mean")
    assert view.values == (135.0, 110.0, 130.0)


@pytest.mark.parametrize("name,expected", [
    ("missing_time", "Unknown time column"),
    ("missing_value", "Unknown value column"),
    ("not_numeric", "must be numeric"),
    ("one_period", "at least two distinct periods"),
    ("null_values", "missing timestamps or values"),
    ("invalid_aggregation", "Aggregation must"),
])
def test_invalid_time_series_inputs(name, expected):
    frame = prepare_dataset(_frame()).frame
    kwargs = dict(time_column="date", value_column="sales", aggregation="sum")
    if name == "missing_time": kwargs["time_column"] = "unknown"
    if name == "missing_value": kwargs["value_column"] = "unknown"
    if name == "not_numeric": kwargs["value_column"] = "region"
    if name == "one_period": frame = frame.iloc[[1, 2]]
    if name == "null_values": frame.loc[0, "sales"] = None
    if name == "invalid_aggregation": kwargs["aggregation"] = "max"
    with pytest.raises(DatasetPreparationError, match=expected):
        build_time_series(frame, **kwargs)


def test_frequency_is_only_asserted_when_inferable():
    view = build_time_series(prepare_dataset(_frame()).frame, time_column="date", value_column="sales", aggregation="sum")
    assert view.is_regular is True
    assert view.inferred_frequency == "D"
    assert view.has_temporal_gaps is False


def test_irregular_frequency_and_no_filling():
    frame = prepare_dataset(pd.DataFrame({"date": ["2026-08-01", "2026-08-02", "2026-08-05"], "sales": [1, 2, 3]})).frame
    view = build_time_series(frame, time_column="date", value_column="sales", aggregation="sum")
    assert view.period_count == 3
    assert view.is_regular is False
    assert view.inferred_frequency is None
    assert view.has_temporal_gaps is None  # frequency unknown, cannot assert absence of gaps
    assert view.values == (1.0, 2.0, 3.0)


def test_two_periods_cannot_infer_frequency():
    frame = prepare_dataset(pd.DataFrame({"date": ["2026-08-01", "2026-08-02"], "sales": [1, 2]})).frame
    view = build_time_series(frame, time_column="date", value_column="sales", aggregation="sum")
    assert view.inferred_frequency is None
    assert view.is_regular is False


def test_source_dataset_is_never_mutated_or_aggregated_implicitly():
    frame = _frame()
    before = frame.copy(deep=True)
    prepared = prepare_dataset(frame)
    build_time_series(prepared.frame, time_column="date", value_column="sales", aggregation="sum")
    pd.testing.assert_frame_equal(frame, before)
    assert len(prepared.frame) == 4
    assert prepared.frame["sales"].tolist() == [110, 120, 150, 130]


def test_intervention_index_requires_exact_timestamp():
    view = build_time_series(prepare_dataset(_frame()).frame, time_column="date", value_column="sales", aggregation="sum")
    assert resolve_intervention_index(view, "2026-08-02") == 1
    with pytest.raises(DatasetPreparationError, match="exactly match"):
        resolve_intervention_index(view, "2026-08-04")


def test_end_to_end_api_upload_preserves_grain_and_profiles_date():
    app = create_app(planner_factory=lambda: object())
    client = TestClient(app)
    csv = b"date,region,sales\n2026-08-01,North,100\n2026-08-01,South,120\n2026-08-02,North,110\n2026-08-02,South,125\n"
    response = client.post("/datasets", files={"file": ("sales.csv", csv, "text/csv")})
    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"dataset_id", "filename", "profile"}
    assert [(x["name"], x["role"]) for x in body["profile"]["columns"]] == [
        ("date", "time"), ("region", "dimension"), ("sales", "measure")
    ]
    assert "North" not in str(body["profile"])
    stored = app.state.datasets[body["dataset_id"]]["frame"]
    assert len(stored) == 4
    view = build_time_series(stored, time_column="date", value_column="sales", aggregation="sum")
    assert view.values == (220.0, 235.0)


def test_api_bad_date_and_empty_csv_clean_error():
    client = TestClient(create_app(planner_factory=lambda: object()))
    bad = client.post("/datasets", files={"file": ("bad.csv", b"date,sales\n2026-08-01,100\nNOT_A_DATE,120\n", "text/csv")})
    assert bad.status_code == 400
    assert "Invalid date" in bad.json()["detail"]
    empty = client.post("/datasets", files={"file": ("empty.csv", b"", "text/csv")})
    assert empty.status_code == 400
    malformed = client.post("/datasets", files={"file": ("broken.csv", b"date,sales\n2026-08-01,10,unexpected\n", "text/csv")})
    assert malformed.status_code == 400
