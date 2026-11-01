"""Official Meta runtime configuration, lazy startup and injected end-to-end contract."""
import json

import pytest
from fastapi.testclient import TestClient

from agent.meta_planner import MetaModelConfigurationError, MetaModelError, MetaModelPlanner
from backend.app import create_app
from scripts.run_demo_scenario import CSV_PATH, QUESTION
from tests.test_bounded_planning import Client, call


def test_official_runtime_defaults():
    planner = MetaModelPlanner(client=object())
    assert planner.provider == 'meta_model_api'
    assert planner.base_url == 'https://api.meta.ai/v1'
    assert planner.model == 'muse-spark-1.3'
    assert planner.instruction_role == 'developer'
    assert planner.parallel_tool_calls is True
    assert planner.max_planning_turns == planner.max_initial_calls == 4
    assert planner.temperature is None
    assert create_app().state.planner_factory is MetaModelPlanner


def test_missing_key_is_a_configuration_failure(monkeypatch):
    monkeypatch.delenv('MODEL_API_KEY', raising=False)
    with pytest.raises(MetaModelConfigurationError, match='MODEL_API_KEY'):
        MetaModelPlanner()


def test_sdk_uses_only_official_endpoint_and_environment_key(monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY', 'test-private-credential')
    captured = {}
    def sdk(**kwargs):
        captured.update(kwargs)
        return object()
    monkeypatch.setattr('openai.OpenAI', sdk)
    MetaModelPlanner()
    assert captured == {'base_url': MetaModelPlanner.DEFAULT_BASE_URL, 'api_key': 'test-private-credential'}


@pytest.mark.parametrize('field', ['model', 'base_url'])
def test_blank_provider_configuration_is_rejected(field):
    with pytest.raises(MetaModelError):
        MetaModelPlanner(client=object(), **{field: '   '})


def test_import_health_upload_work_without_key_and_analysis_errors_are_safe(monkeypatch):
    monkeypatch.delenv('MODEL_API_KEY', raising=False)
    import importlib
    module = importlib.reload(importlib.import_module('backend.app'))
    with TestClient(module.create_app()) as api:
        assert api.get('/health').status_code == 200
        upload = api.post('/datasets', files={'file': ('x.csv', b'region,sales\nA,1\nB,2\n', 'text/csv')})
        assert upload.status_code == 201
        payload = {'dataset_id': upload.json()['dataset_id'], 'question': 'Compare regions'}
        response = api.post('/analyses', json=payload)
        assert response.status_code == 503
        assert response.json() == {'detail': 'Planning provider is not configured.'}
        stream = api.post('/analyses/stream', json=payload)
        assert 'configuration_error' in stream.text
        assert 'MODEL_API_KEY' not in stream.text and 'Traceback' not in stream.text
        assert not api.app.state.analyses


def test_injected_meta_client_full_evidence_path():
    client = Client([[call(i, call_id=f'id-{i}') for i in range(4)]])
    with TestClient(create_app(planner_factory=lambda: MetaModelPlanner(client=client))) as api:
        upload = api.post('/datasets', files={'file': (CSV_PATH.name, CSV_PATH.read_bytes(), 'text/csv')})
        response = api.post('/analyses', json={'dataset_id': upload.json()['dataset_id'], 'question': QUESTION})
        assert response.status_code == 201
        body = response.json()
        assert body['succeeded'] is True
        assert len(body['steps']) == len(body['visualizations']) == 4
        for step in body['steps']:
            trace = step['trace']
            assert trace['verification_status'] == 'verified'
            assert trace['trace_hash'] == step['report']['trace_hash']
            assert trace['deterministic_result']  # Execution occurred locally, not in fake response.
        assert body['decision_brief']['evidence_count'] == 4
        hashes = {step['trace']['trace_hash'] for step in body['steps']}
        assert {chart['evidence']['trace_hash'] for chart in body['visualizations']} == hashes
        request = client.calls[0]
        assert QUESTION in request['messages'][1]['content']
        assert request['model'] == MetaModelPlanner.DEFAULT_MODEL
        assert request['tool_choice'] == 'auto' and request['parallel_tool_calls'] is True
        assert request['messages'][0]['role'] == 'developer'
        assert 'temperature' not in request
        names = {tool['function']['name'] for tool in request['tools']}
        assert len(names) < 102 and {'linear_trend', 'one_way_anova'} <= names
        prompt = json.dumps(request['messages'])
        assert 'North' not in prompt and 'observed_post' not in prompt
        assert 'control_sales' in prompt
