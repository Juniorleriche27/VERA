"""Real pipeline event ordering, metadata privacy and SSE lifecycle regressions."""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from agent.events import AnalysisEventEmitter
from agent.meta_planner import MetaModelError
from agent.pipeline import VerifiedAnalyticsPipeline
from backend.app import create_app
from backend.streaming import analysis_stream
from scripts.run_demo_scenario import CSV_PATH, QUESTION, DemoPlanner, load_demo_dataset
from backend.app import _safe_dataset_profile


def read_events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]


def uploaded(client):
    return client.post('/datasets', files={'file': ('demo.csv', CSV_PATH.read_bytes(), 'text/csv')}).json()['dataset_id']


def test_real_sse_path_matches_existing_result_and_emits_causal_order():
    client = TestClient(create_app(planner_factory=DemoPlanner))
    dataset_id = uploaded(client)
    stream = client.post('/analyses/stream', json={'dataset_id': dataset_id, 'question': QUESTION})
    assert stream.status_code == 200
    assert stream.headers['content-type'].startswith('text/event-stream')
    assert stream.headers['x-accel-buffering'] == 'no'
    assert 'no-store' in stream.headers['cache-control']
    events = read_events(stream)
    names = [e['event'] for e in events]
    assert names[:4] == ['analysis_started', 'planning_started', 'planning_completed', 'tools_selected']
    assert names[-5:] == ['decision_brief_started', 'decision_brief_built', 'visualizations_started', 'visualizations_built', 'analysis_completed']
    assert [e['sequence'] for e in events] == list(range(1, len(events) + 1))
    assert [e['elapsed_ms'] for e in events] == sorted(e['elapsed_ms'] for e in events)
    calls = events[3]['data']['calls']
    assert len(calls) == 4
    for call in calls:
        tool_events = [e['event'] for e in events if e['data'].get('call_id') == call['call_id']]
        assert tool_events == ['tool_started', 'tool_completed', 'verification_started', 'verification_completed', 'tool_verified']
    assert events[-1]['data'] == {'analyses': 4, 'verified': 4, 'visualizations': 4, 'unresolved': 0, 'succeeded': True, 'exhausted': False}
    final = client.get('/analyses/' + events[-1]['analysis_id']).json()
    sync = client.post('/analyses', json={'dataset_id': dataset_id, 'question': QUESTION}).json()
    final.pop('analysis_id'); sync.pop('analysis_id')
    assert final == sync
    # The SSE payload is metadata only, including the terminal event.
    for forbidden in ('deterministic_result', 'observed_post', 'parameters', 'bindings', 'North', 'control_sales', 'messages', 'api_key'):
        assert forbidden not in stream.text


def test_observer_does_not_change_trace_hashes_and_sees_events_before_tool_execution():
    from agent.analytics_executor import AnalyticsToolExecutor
    recorded = []
    class Analytics(AnalyticsToolExecutor):
        def execute(self, name, parameters, bindings):
            assert recorded[-1].event == 'tool_started'
            assert recorded[-1].as_dict()['data']['tool'] == name
            return super().execute(name, parameters, bindings)
    frame = load_demo_dataset()
    args = dict(question=QUESTION, dataset=frame, dataset_context=_safe_dataset_profile(frame))
    baseline = VerifiedAnalyticsPipeline(DemoPlanner()).run(**args)
    observed = VerifiedAnalyticsPipeline(DemoPlanner(), analytics=Analytics()).run(
        **args, events=AnalysisEventEmitter('test', recorded.append))
    assert [s.trace.as_dict() for s in baseline.steps] == [s.trace.as_dict() for s in observed.steps]


def test_planner_failure_is_safe_terminal_sse():
    class Failure:
        def plan(self, *args, **kwargs):
            raise MetaModelError('PRIVATE_PROVIDER_PAYLOAD')
    client = TestClient(create_app(planner_factory=Failure))
    response = client.post('/analyses/stream', json={'dataset_id': uploaded(client), 'question': QUESTION})
    events = read_events(response)
    assert [e['event'] for e in events] == ['analysis_started', 'planning_started', 'analysis_failed']
    assert events[-1]['data'] == {'code': 'planning_failed'}
    assert 'PRIVATE_PROVIDER_PAYLOAD' not in response.text
    assert client.app.state.analyses == {}


def test_failed_binding_is_not_called_executed_or_verified():
    from types import SimpleNamespace
    from agent.tool_call_trace import ToolCallTrace
    class Failure:
        def plan(self, *args, **kwargs):
            return SimpleNamespace(traces=(ToolCallTrace.planned(call_id='bad', tool='linear_trend',
                objective='Trend', reason='Check', bindings={'time_column': 'PRIVATE_UNKNOWN', 'value_column': 'sales'}),))
    client = TestClient(create_app(planner_factory=Failure))
    response = client.post('/analyses/stream', json={'dataset_id': uploaded(client), 'question': QUESTION})
    names = [e['event'] for e in read_events(response)]
    assert 'tool_failed' in names
    assert not {'tool_started', 'tool_completed', 'tool_verified', 'decision_brief_built'} & set(names)
    assert 'PRIVATE_UNKNOWN' not in response.text
    assert read_events(response)[-1]['data']['succeeded'] is False


def test_empty_plan_completes_without_invented_execution_events():
    from types import SimpleNamespace
    class Empty:
        def plan(self, *args, **kwargs): return SimpleNamespace(traces=())
    client = TestClient(create_app(planner_factory=Empty))
    events = read_events(client.post('/analyses/stream', json={'dataset_id': uploaded(client), 'question': 'Hello'}))
    assert events[3]['data']['calls'] == []
    assert not any(e['event'] == 'tool_started' for e in events)
    assert not any(e['event'] in {'execution_started', 'execution_completed'} for e in events)
    assert events[-1]['data']['analyses'] == 0 and events[-1]['data']['succeeded'] is False


@pytest.mark.parametrize('event,data', [
    ('tool_started', {'tool': 'linear_trend', 'call_id': 'x', 'bindings': {'values': [1, 2]}}),
    ('tools_selected', {'calls': [{'tool': 'linear_trend', 'call_id': 'x', 'source_call_ids': [], 'selection': 'ai', 'rows': [1]}]}),
    ('analysis_failed', {'code': 'PRIVATE_PROVIDER_PAYLOAD'}),
])
def test_event_contract_rejects_unapproved_payloads(event, data):
    events = []
    with pytest.raises(ValueError):
        AnalysisEventEmitter('x', events.append).emit(event, **data)
    assert events == []


def test_stream_validation_happens_before_sse_headers():
    client = TestClient(create_app(planner_factory=DemoPlanner))
    assert client.post('/analyses/stream', json={'dataset_id': 'missing', 'question': QUESTION}).status_code == 404
    assert client.post('/analyses/stream', json={'dataset_id': uploaded(client), 'question': ' '}).status_code == 422


def test_disconnect_stops_future_events_and_prevents_binding_or_persistence():
    from threading import Event as ThreadEvent
    started, release, stopped = ThreadEvent(), ThreadEvent(), ThreadEvent()
    class WaitingPlanner(DemoPlanner):
        def plan(self, *args, **kwargs):
            started.set()
            assert release.wait(5)  # Test synchronization, never a simulated product delay.
            try:
                return super().plan(*args, **kwargs)
            finally:
                stopped.set()
    async def check():
        frame = load_demo_dataset()
        analyses = {}
        stream = analysis_stream(analysis_id='cancelled', dataset_id='dataset', question=QUESTION,
            dataset={'frame': frame, 'profile': _safe_dataset_profile(frame)}, planner_factory=WaitingPlanner, analyses=analyses)
        assert 'analysis_started' in await anext(stream)
        assert 'planning_started' in await anext(stream)
        assert await asyncio.to_thread(started.wait, 5)
        await stream.aclose()
        release.set()
        assert await asyncio.to_thread(stopped.wait, 5)
        # A second event-loop turn lets the worker observe cancellation after planning.
        await asyncio.to_thread(lambda: None)
        assert analyses == {}
    asyncio.run(check())


def test_executor_failure_has_started_but_never_completed_or_verified():
    from agent.analytics_executor import AnalyticsToolExecutor
    class Failure(AnalyticsToolExecutor):
        def execute(self, *args, **kwargs): raise ValueError('PRIVATE_EXECUTION_VALUES')
    records = []
    frame = load_demo_dataset()
    result = VerifiedAnalyticsPipeline(DemoPlanner(), analytics=Failure()).run(
        QUESTION, dataset=frame, dataset_context=_safe_dataset_profile(frame),
        events=AnalysisEventEmitter('execution-failure', records.append))
    assert not result.succeeded
    assert sum(e.event == 'tool_started' for e in records) == 4
    assert sum(e.event == 'tool_failed' for e in records) == 4
    assert not any(e.event in {'tool_completed', 'tool_verified'} for e in records)
    assert 'PRIVATE_EXECUTION_VALUES' not in json.dumps([e.as_dict() for e in records])


def test_replanning_event_retains_origin_and_actual_verification_decision():
    from tests.test_replanning_pipeline_resilience import ControlledVerifier
    from agent.tool_call_trace import VerificationStatus
    records = []
    frame = load_demo_dataset()
    result = VerifiedAnalyticsPipeline(DemoPlanner(), verifier=ControlledVerifier({
        'demo-regions': (VerificationStatus.NEEDS_MORE_ANALYSIS, 'run_robustness_check'),
    })).run(QUESTION, dataset=frame, dataset_context=_safe_dataset_profile(frame), events=AnalysisEventEmitter('replan', records.append))
    assert result.succeeded and len(result.steps) == 5
    planned = next(e.as_dict()['data']['call'] for e in records if e.event == 'tool_planned')
    assert planned['selection'] == 'deterministic'
    assert planned['source_call_ids'] == ['demo-regions']
    assert planned['tool'] == 'kruskal_wallis_test'
    original = [e.event for e in records if e.as_dict()['data'].get('call_id') == 'demo-regions']
    assert 'verification_completed' in original and 'tool_verified' not in original
