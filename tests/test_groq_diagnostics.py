"""Safety and real-path checks for the diagnostic harness; no live credentials."""
import json
from types import SimpleNamespace

from scripts import diagnose_groq_planner as diagnostics
from scripts.run_demo_scenario import load_demo_dataset
from backend.app import _safe_dataset_profile
from tests.test_meta_planner import _FakeClient, _response, _tool_call


def events(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_missing_credential_blocks_before_any_network(monkeypatch, capsys):
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    monkeypatch.setattr('sys.argv', ['diagnose_groq_planner.py'])
    assert diagnostics.main() == 2
    assert events(capsys) == [{'event': 'blocked', 'reason': 'Provider credential is unavailable in this process.'}]


def test_provider_message_sanitization_never_emits_credential(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'test-private-credential')
    exc = RuntimeError('Do not print request or arbitrary repr')
    exc.status_code = 400
    exc.body = {'error': {'message': 'Bad request test-private-credential Bearer other-token'}}
    result = diagnostics.error_metadata(exc, 'provider')
    assert result['exception_class'] == 'RuntimeError'
    assert result['provider_status_code'] == 400
    assert 'test-private-credential' not in json.dumps(result)
    assert 'other-token' not in json.dumps(result)
    assert 'Do not print' not in json.dumps(result)
    assert 'Bad request' in result['message']


def test_real_selector_prompt_parser_binding_execution_and_verification(capsys):
    call = _tool_call(name='linear_trend', arguments={
        'objective': 'Quantify trend', 'reason': 'Time trend', 'parameters': {},
        'bindings': {'time_column': 'date', 'value_column': 'sales', 'aggregation': 'sum'},
    })
    client = _FakeClient(_response(tool_calls=[call]))
    frame = load_demo_dataset()
    diagnostics.run_case(client, 'openai/gpt-oss-20b', False, frame, _safe_dataset_profile(frame), max_planning_turns=1)
    request, response, pipeline = events(capsys)
    assert {'linear_trend', 'mean_shift_change_point', 'one_way_anova', 'bayesian_structural_causal_impact'} <= set(request['selected_tools'])
    assert response['tool_call_count'] == 1
    assert pipeline['steps'][0]['verification_status'] == 'verified'
    assert pipeline['visualization_count'] == 1
    # Observe exactly the production request; the harness does not substitute
    # a diagnostic-only prompt or simplify the outbound schema.
    actual = client.chat.completions.calls[0]
    assert actual['parallel_tool_calls'] is False
    assert actual['messages'][0]['role'] == 'system'
    assert 'Never guess or invent dates' in actual['messages'][0]['content']
    assert 'North' not in actual['messages'][1]['content']
    assert not any('messages' in entry for entry in (request, response, pipeline))


def test_provider_and_parser_failures_are_distinguished(capsys):
    call = _tool_call(arguments={'objective': 'Compare', 'reason': 'Test', 'parameters': {}, 'bindings': {}})
    client = _FakeClient(_response(tool_calls=[call]))
    frame = load_demo_dataset()
    diagnostics.run_case(client, 'qwen/qwen3.8-27b', True, frame, _safe_dataset_profile(frame))
    request, response, error = events(capsys)
    assert request['parallel_tool_calls'] is True
    assert response['tool_call_count'] == 1
    assert error['stage'] == 'vera'
    assert error['exception_class'] == 'MetaModelError'
    assert error['provider_status_code'] is None


def test_temporal_year_requires_safe_bounds_to_be_identifiable():
    import pandas as pd
    frame = load_demo_dataset()
    other_year = frame.copy()
    other_year['date'] = frame['date'] + pd.DateOffset(years=1)
    # Different intervention years produce identical model-visible inputs.
    # A date cannot be inferred uniquely from a month name and this profile.
    first, second = _safe_dataset_profile(frame), _safe_dataset_profile(other_year)
    assert first != second
    for profile in (first, second):
        for column in profile['columns']:
            column.pop('min_timestamp', None)
            column.pop('max_timestamp', None)
    assert first == second


def test_provider_failure_keeps_safe_exact_message_without_duplicate_parser_error(monkeypatch, capsys):
    monkeypatch.setenv('GROQ_API_KEY', 'test-private-credential')
    def fail(**kwargs):
        exc = RuntimeError('An arbitrary request repr must not be printed')
        exc.status_code = 400
        exc.body = {'error': {'message': 'Unsupported schema test-private-credential'}}
        raise exc
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    frame = load_demo_dataset()
    diagnostics.run_case(client, 'qwen/qwen3.8-27b', True, frame, _safe_dataset_profile(frame))
    request, error = events(capsys)
    assert request['event'] == 'request'
    assert error['stage'] == 'provider'
    assert error['provider_status_code'] == 400
    assert error['message'] == 'Unsupported schema [REDACTED]'


def test_diagnostic_cli_does_not_exit_successfully_after_a_failed_case(monkeypatch, capsys):
    monkeypatch.setenv('GROQ_API_KEY', 'test-private-credential')
    monkeypatch.setattr('sys.argv', ['diagnose_groq_planner.py', '--models', 'qwen/qwen3.8-27b', '--parallel', 'true'])
    client = SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id='qwen/qwen3.8-27b')])))
    monkeypatch.setattr('openai.OpenAI', lambda **kwargs: client)
    monkeypatch.setattr(diagnostics, 'run_case', lambda *args, **kwargs: False)
    assert diagnostics.main() == 1
    assert 'test-private-credential' not in capsys.readouterr().out
