"""Regressions motivated by real provider responses, independent of live keys."""
import json

import pandas as pd
import pytest

from agent.analytics_executor import AnalyticsToolExecutor
from agent.meta_planner import MetaModelError, MetaModelPlanner, _safe_dataset_context, build_meta_tools
from backend.app import _safe_dataset_profile
from scripts.run_demo_scenario import load_demo_dataset
from tests.test_meta_planner import _FakeClient, _response, _tool_call


def test_provider_instruction_roles_remain_explicit_and_portable():
    for expected in ['developer', 'system']:
        client = _FakeClient(_response(content='No analysis', finish_reason='stop'))
        MetaModelPlanner(client=client, instruction_role=expected).plan('Inspect', dataset_context={}, tool_catalog=[{'name': 'linear_trend'}])
        assert client.chat.completions.calls[0]['messages'][0]['role'] == expected
    with pytest.raises(MetaModelError):
        MetaModelPlanner(client=object(), instruction_role='user')


def test_temporal_profile_bounds_are_deterministic_and_not_row_examples():
    frame = load_demo_dataset()
    profile = _safe_dataset_profile(frame)
    date = profile['columns'][0]
    assert date['min_timestamp'] == '2026-08-01T00:00:00'
    assert date['max_timestamp'] == '2026-09-30T00:00:00'
    assert _safe_dataset_context(profile) == profile
    assert profile == _safe_dataset_profile(frame.sample(frac=1, random_state=7))
    assert 'North' not in json.dumps(profile)
    assert all('min_timestamp' not in col for col in profile['columns'] if col['role'] != 'time')


def test_empty_temporal_column_has_no_invented_bounds():
    frame = pd.DataFrame({'date': pd.to_datetime([None, None]), 'sales': [1, 2]})
    assert 'min_timestamp' not in _safe_dataset_profile(frame)['columns'][0]


@pytest.mark.parametrize('changes', [
    {'min_timestamp': 'NOT_A_DATE'}, {'max_timestamp': ['raw', 'rows']},
    {'min_timestamp': '2027-01-01'}, {'role': 'dimension'},
    {'max_timestamp': '2026-09-30T00:00:00+01:00'},
])
def test_outbound_temporal_metadata_stays_typed_and_ordered(changes):
    profile = _safe_dataset_profile(load_demo_dataset())
    profile['columns'][0].update(changes)
    with pytest.raises(MetaModelError):
        _safe_dataset_context(profile)


def test_half_temporal_range_is_rejected():
    profile = _safe_dataset_profile(load_demo_dataset())
    del profile['columns'][0]['max_timestamp']
    with pytest.raises(MetaModelError):
        _safe_dataset_context(profile)


def test_duplicate_time_requires_model_to_choose_aggregation_in_schema_and_parser():
    frame = load_demo_dataset()
    profile = _safe_dataset_profile(frame)
    catalog = AnalyticsToolExecutor().available_tools()
    for tool in ('linear_trend', 'mean_shift_change_point', 'variance_shift_change_point', 'bayesian_structural_causal_impact'):
        function = next(t['function'] for t in build_meta_tools(catalog, dataset_context=profile) if t['function']['name'] == tool)
        assert 'aggregation' in function['parameters']['properties']['bindings']['required']
    client = _FakeClient(_response(tool_calls=[_tool_call(name='linear_trend', arguments={
        'objective': 'Trend', 'reason': 'Time trend', 'parameters': {},
        'bindings': {'time_column': 'date', 'value_column': 'sales'},
    })]))
    with pytest.raises(MetaModelError, match='explicit aggregation'):
        MetaModelPlanner(client=client).plan('Trend', dataset_context=profile, tool_catalog=catalog)
    # No default sum/mean was injected on behalf of the model.
    assert len(client.chat.completions.calls) == 1


def test_unique_timestamps_keep_aggregation_optional():
    profile = _safe_dataset_profile(load_demo_dataset().query("region == 'North'"))
    tool = build_meta_tools([{'name': 'linear_trend'}], dataset_context=profile)[0]
    assert tool['function']['parameters']['properties']['bindings']['required'] == ['time_column', 'value_column']


def test_causal_tool_guidance_does_not_supply_a_fabricated_intervention_year():
    definition = json.dumps(build_meta_tools([{'name': 'bayesian_structural_causal_impact'}]))
    assert '2026-09-01' not in definition
    assert 'temporal metadata' in definition


def test_sampling_temperature_is_optional_and_explicit():
    for expected in [None, 0.0]:
        client = _FakeClient(_response(content='No analysis', finish_reason='stop'))
        MetaModelPlanner(client=client, temperature=expected).plan('Inspect', dataset_context={}, tool_catalog=[{'name': 'linear_trend'}])
        assert client.chat.completions.calls[0].get('temperature') == expected


@pytest.mark.parametrize('temperature', [-1, 3, True, float('nan'), float('inf')])
def test_invalid_sampling_temperature_is_rejected(temperature):
    with pytest.raises(MetaModelError):
        MetaModelPlanner(client=object(), temperature=temperature)
