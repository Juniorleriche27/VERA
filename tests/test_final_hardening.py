"""Adversarial regressions at external planning and verification boundaries."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.data_binding import DatasetBinder, DataBindingError
from agent.meta_planner import MetaModelError, MetaModelPlanner
from agent.pipeline import PipelineError, VerifiedAnalyticsPipeline
from agent.tool_call_trace import ToolCallTrace, ToolCallTraceError, VerificationStatus
from backend.app import create_app
from tests.test_meta_planner import _FakeClient, _response, _tool_call, _catalog
from tests.test_replanning_pipeline_resilience import planned, temporal, run
from verification.verifier import DeterministicVerifier


@pytest.mark.parametrize('context', [
    {'preview': [{'customer': 'PRIVATE', 'sales': 123}]},
    {'columns': [{'name': 'sales', 'preview': ['PRIVATE']}]},
    {'row_count': ['PRIVATE']},
])
def test_context_allowlist_blocks_raw_payload_aliases_before_network(context):
    client = _FakeClient(_response())
    with pytest.raises(MetaModelError):
        MetaModelPlanner(client=client).plan('Analyze sales', dataset_context=context, tool_catalog=_catalog())
    assert client.chat.completions.calls == []


def test_pipeline_checks_context_even_for_an_injected_provider():
    class Provider:
        def plan(self, *args, **kwargs):
            pytest.fail('unsafe context reached provider')
    with pytest.raises(MetaModelError):
        VerifiedAnalyticsPipeline(Provider()).run('Analyze', dataset={}, dataset_context={'preview': ['PRIVATE']})


def test_planner_cannot_forge_resolution_lineage():
    call = _tool_call()
    args = json.loads(call.function.arguments)
    args['source_call_ids'] = ['nonexistent']
    call.function.arguments = json.dumps(args)
    with pytest.raises(MetaModelError):
        MetaModelPlanner(client=_FakeClient(_response(tool_calls=[call]))).plan(
            'Compare', dataset_context={}, tool_catalog=_catalog())


@pytest.mark.parametrize('traces', [
    [temporal('same'), temporal('same')],
    [planned(call_id='root'), planned(call_id='child', sources=('root',))],
])
def test_pipeline_rejects_duplicate_or_untrusted_initial_lineage(traces):
    with pytest.raises(PipelineError):
        run(traces)


def test_replan_ids_do_not_collide_with_remaining_initial_calls():
    result = run([planned(call_id='root'), temporal('root-replan-1')], statuses={
        'root': (VerificationStatus.NEEDS_MORE_ANALYSIS, 'run_robustness_check'),
    })
    ids = [step.trace.call_id for step in result.steps]
    assert len(ids) == len(set(ids))
    assert result.succeeded


@pytest.mark.parametrize('key', ['grouping_column', 'value_column'])
def test_explicit_unknown_grouped_column_is_not_silently_inferred(key):
    bindings = {'grouping_column': 'region', 'value_column': 'sales', key: 'invented'}
    frame = pd.DataFrame({'region': ['A', 'B'], 'sales': [1, 2]})
    with pytest.raises(DataBindingError):
        DatasetBinder().bind(planned(bindings=bindings), frame)


def test_mann_whitney_preserves_single_explicit_group_and_direction():
    frame = pd.DataFrame({'region': ['A', 'A', 'B', 'B'], 'sales': [1, 2, 10, 11]})
    trace = planned('mann_whitney_u_test', bindings={
        'grouping_column': 'region', 'value_column': 'sales', 'group_a': 'B',
    })
    bound = DatasetBinder().bind(trace, frame)
    assert list(bound.bindings['data']) == ['B', 'A']


@pytest.mark.parametrize('payload', [
    {'sample_size': -1}, {'sample_size': 3.5}, {'sample_size': '20'},
    {'p_values': 'invalid'}, {'ci_lower': 2},
])
def test_malformed_numerical_evidence_is_rejected(payload):
    trace = planned().complete(payload)
    _, report = DeterministicVerifier().verify(trace)
    assert report.status is VerificationStatus.REJECTED


def test_causal_array_length_must_match_declared_post_count():
    from tests.test_normalized_verification import causal_payload
    result = causal_payload(counterfactual_post_interval=[[1, 2]])
    _, report = DeterministicVerifier().verify(planned('bayesian_structural_causal_impact').complete(result))
    assert report.status is VerificationStatus.REJECTED


@pytest.mark.parametrize('field,value', [('_parameters_json', '{"x":NaN}'), ('_result_json', '[Infinity]')])
def test_direct_trace_construction_cannot_bypass_finite_json(field, value):
    with pytest.raises(ToolCallTraceError):
        replace(planned(), **{field: value})


def test_provider_failure_returns_safe_json_and_no_provider_details():
    class Provider:
        def plan(self, *args, **kwargs):
            raise MetaModelError('provider leaked PRIVATE_TOKEN')
    client = TestClient(create_app(planner_factory=Provider), raise_server_exceptions=False)
    dataset = client.post('/datasets', files={'file': ('x.csv', b'region,sales\nA,1\nB,2\n')}).json()
    response = client.post('/analyses', json={'dataset_id': dataset['dataset_id'], 'question': 'Compare'})
    assert response.status_code == 502
    assert response.headers['content-type'].startswith('application/json')
    assert 'PRIVATE_TOKEN' not in response.text


def test_api_includes_unresolved_call_ids():
    class Provider:
        def plan(self, *args, **kwargs):
            return SimpleNamespace(traces=(temporal(),))
    client = TestClient(create_app(planner_factory=Provider))
    dataset = client.post('/datasets', files={'file': ('x.csv', b'date,sales\n2026-01-01,1\n2026-01-02,2\n')}).json()
    response = client.post('/analyses', json={'dataset_id': dataset['dataset_id'], 'question': 'Trend'})
    assert response.status_code == 201
    assert response.json()['unresolved_call_ids'] == ['trend']


def test_upload_read_is_bounded_before_size_rejection(monkeypatch):
    from starlette.datastructures import UploadFile
    from backend.config import AppSettings
    calls = []
    original = UploadFile.read

    async def read(self, size=-1):
        calls.append(size)
        return await original(self, size)

    monkeypatch.setattr(UploadFile, 'read', read)
    client = TestClient(create_app(settings=AppSettings(max_upload_bytes=10)))
    response = client.post('/datasets', files={'file': ('x.csv', b'x\n' + b'1\n' * 100)})
    assert response.status_code == 413
    assert calls == [11]
    assert not client.app.state.datasets


@pytest.mark.parametrize('binding,value', [('group_a', None), ('group_b', ['A']), ('control_columns', None)])
def test_optional_bindings_follow_their_declared_types(binding, value):
    from agent.meta_planner import _validate_semantic_tool_contract
    if binding == 'control_columns':
        tool = 'bayesian_structural_causal_impact'
        bindings = {'time_column': 'date', 'value_column': 'sales', 'intervention_date': '2026-09-01'}
    else:
        tool = 'mann_whitney_u_test'
        bindings = {'grouping_column': 'region', 'value_column': 'sales'}
    with pytest.raises(MetaModelError):
        _validate_semantic_tool_contract(tool, {}, {**bindings, binding: value}, 'call')


@pytest.mark.parametrize('field', ['parameters', 'bindings'])
def test_required_planning_fields_are_not_silently_defaulted(field):
    call = _tool_call()
    args = json.loads(call.function.arguments)
    del args[field]
    call.function.arguments = json.dumps(args)
    with pytest.raises(MetaModelError):
        MetaModelPlanner(client=_FakeClient(_response(tool_calls=[call]))).plan(
            'Compare', dataset_context={}, tool_catalog=_catalog())


def test_api_serializes_dataclass_brief_with_derived_evidence_metadata():
    from backend.app import _brief_payload
    from scripts.run_demo_scenario import run_demo
    *_, brief, visuals = run_demo()
    payload = _brief_payload(brief)
    assert payload == brief.as_dict()
    assert payload['evidence_count'] == 4
    assert len(payload['tools_used']) == 4
    assert [item['trace_hash'] for item in payload['evidence']['items']] == [spec.trace_hash for spec in visuals]


def test_provider_configuration_error_is_safe_and_actionable():
    from agent.meta_planner import MetaModelConfigurationError
    def factory():
        raise MetaModelConfigurationError('PRIVATE_CONFIG')
    client = TestClient(create_app(planner_factory=factory))
    dataset = client.post('/datasets', files={'file': ('x.csv', b'region,sales\nA,1\nB,2\n')}).json()
    response = client.post('/analyses', json={'dataset_id': dataset['dataset_id'], 'question': 'Compare'})
    assert response.status_code == 503
    assert 'PRIVATE_CONFIG' not in response.text
    assert 'configured' in response.json()['detail']


@pytest.mark.parametrize('tool,parameters,bindings', [
    ('linear_trend', {}, {'time_column': 'date', 'value_column': 'sales', 'aggregation': 'sum'}),
    ('mean_shift_change_point', {'min_segment_size': 4, 'threshold': 1.2}, {'time_column': 'date', 'value_column': 'sales', 'aggregation': 'sum'}),
    ('variance_shift_change_point', {'min_segment_size': 4, 'variance_ratio_threshold': 2.5}, {'time_column': 'date', 'value_column': 'sales', 'aggregation': 'sum'}),
    ('one_way_anova', {}, {'grouping_column': 'region', 'value_column': 'sales'}),
    ('kruskal_wallis_test', {}, {'grouping_column': 'region', 'value_column': 'sales'}),
    ('mann_whitney_u_test', {'alternative': 'greater'}, {'grouping_column': 'region', 'value_column': 'sales', 'group_a': 'North', 'group_b': 'South'}),
    ('bayesian_structural_causal_impact', {'credibility': .9, 'prior_scale': 5.}, {'time_column': 'date', 'value_column': 'sales', 'aggregation': 'sum', 'intervention_date': '2026-09-01', 'control_columns': ['control_sales']}),
])
def test_seven_strategic_contracts_through_real_verification(tool, parameters, bindings):
    from agent.analytics_executor import AnalyticsToolExecutor
    from backend.app import _safe_dataset_profile
    from scripts.run_demo_scenario import load_demo_dataset
    frame = load_demo_dataset()
    executor = AnalyticsToolExecutor()
    client = _FakeClient(_response(tool_calls=[_tool_call(name=tool, arguments={
        'objective': 'Audit contract', 'reason': 'Verify real execution',
        'parameters': parameters, 'bindings': bindings,
    })]))
    plan = MetaModelPlanner(client=client).plan(
        'Audit contract', dataset_context=_safe_dataset_profile(frame), tool_catalog=executor.available_tools())
    trace = plan.traces[0]
    bound = DatasetBinder().bind(trace, frame)
    result = executor.execute(tool, bound.parameters, bound.bindings)
    if 'status' in result:
        assert result['status'] == 'success'
    payload = result.get('result', result)
    for key, value in parameters.items():
        assert payload[key] == value
    completed = trace.complete(result)
    verified, report = DeterministicVerifier().verify(completed)
    assert report.status is VerificationStatus.VERIFIED
    assert verified.trace_hash == report.trace_hash
    assert verified.bindings == bindings
    if tool == 'mann_whitney_u_test':
        assert payload['first_sample'] == 'North'
        assert payload['second_sample'] == 'South'
