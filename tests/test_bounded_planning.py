"""Bounded completion of analytic objectives without deterministic tool injection."""
from types import SimpleNamespace

import pytest

from agent.meta_planner import MetaModelError, MetaModelPlanner
from agent.pipeline import VerifiedAnalyticsPipeline
from backend.app import _safe_dataset_profile
from scripts.run_demo_scenario import QUESTION, load_demo_dataset
from tests.test_meta_planner import _response, _tool_call


SPECS = [
    ('linear_trend', {'time_column': 'date', 'value_column': 'sales', 'aggregation': 'sum'}),
    ('mean_shift_change_point', {'time_column': 'date', 'value_column': 'sales', 'aggregation': 'sum'}),
    ('one_way_anova', {'grouping_column': 'region', 'value_column': 'sales'}),
    ('bayesian_structural_causal_impact', {'time_column': 'date', 'value_column': 'sales', 'aggregation': 'sum', 'intervention_date': '2026-09-01', 'control_columns': ['control_sales']}),
]


def call(index, *, call_id='provider-id'):
    name, bindings = SPECS[index]
    return _tool_call(name=name, call_id=call_id, arguments={
        'objective': name, 'reason': 'Answer requested objective',
        'parameters': {}, 'bindings': bindings,
    })


class Client:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _response(tool_calls=next(self.batches))


def run(client, question=QUESTION, **kwargs):
    frame = load_demo_dataset()
    return VerifiedAnalyticsPipeline(MetaModelPlanner(client=client, **kwargs)).run(
        question, dataset=frame, dataset_context=_safe_dataset_profile(frame))


def test_four_single_call_responses_cover_objectives_without_raw_tool_results():
    client = Client([[call(i)] for i in range(4)])
    result = run(client, parallel_tool_calls=False)
    assert result.succeeded
    assert len(client.calls) == result.planning.planning_turns == 4
    assert len(result.steps) == 4
    assert len(result.build_visualizations()) == 4
    assert len({step.trace.call_id for step in result.steps}) == 4  # IDs restarted at provider each turn
    assert all(step.trace.source_call_ids == () for step in result.steps)
    assert all(step.trace.trace_hash == step.report.trace_hash for step in result.steps)
    assert result.planning.usage['total_tokens'] == 480
    assert result.planning.as_dict()['planning_turns'] == 4
    for request in client.calls:
        assert len(request['messages']) == 2
        assert request['messages'][0]['role'] == 'developer'
        text = request['messages'][1]['content']
        assert 'North' not in text and 'observed_post' not in text
        assert QUESTION in text
    second_tools = {tool['function']['name'] for tool in client.calls[1]['tools']}
    assert 'linear_trend' not in second_tools


def test_native_complete_batch_needs_no_continuation():
    client = Client([[call(i, call_id=f'id-{i}') for i in range(4)]])
    assert run(client).succeeded
    assert len(client.calls) == 1


def test_partial_batch_only_prompts_for_uncovered_scopes():
    client = Client([[call(1, call_id='change'), call(3, call_id='causal')], [call(0, call_id='trend'), call(2, call_id='groups')]])
    result = run(client)
    assert result.succeeded and result.planning.planning_turns == 2
    names = {tool['function']['name'] for tool in client.calls[1]['tools']}
    assert names == {'linear_trend', 'one_way_anova', 'kruskal_wallis_test'}


@pytest.mark.parametrize('question,index', [
    ('How are sales evolving over time?', 0),
    ('Are sales significantly different across regions?', 2),
    ('Did the September campaign improve sales?', 3),
])
def test_targeted_question_does_not_expand_to_four_tools(question, index):
    client = Client([[call(index)]])
    assert run(client, question).succeeded
    assert len(client.calls) == 1


def test_clarification_after_partial_plan_does_not_silently_claim_complete_coverage():
    client = Client([[call(0)], []])
    with pytest.raises(MetaModelError, match='did not cover'):
        run(client)
    assert len(client.calls) == 2


def test_planning_turn_budget_is_strict_and_never_injects_missing_tools():
    client = Client([[call(0)], [call(1)]])
    with pytest.raises(MetaModelError, match='planning budget'):
        run(client, max_planning_turns=2)
    assert len(client.calls) == 2


def test_semantically_duplicate_calls_are_rejected():
    client = Client([[call(0, call_id='a'), call(0, call_id='b')]])
    with pytest.raises(MetaModelError, match='duplicate analytical'):
        run(client)


def test_followup_cannot_exceed_remaining_total_call_budget():
    client = Client([[call(0)], [call(i, call_id=f'i-{i}') for i in range(4)]])
    with pytest.raises(MetaModelError, match='maximum is 3'):
        run(client)
    assert len(client.calls) == 2


@pytest.mark.parametrize('value', [0, 5, True, 1.5])
def test_invalid_turn_budget_is_rejected(value):
    with pytest.raises(MetaModelError):
        MetaModelPlanner(client=object(), max_planning_turns=value)


@pytest.mark.parametrize('turns', [1, 4])
def test_total_call_budget_cannot_be_increased_past_four_in_bounded_mode(turns):
    with pytest.raises(MetaModelError, match='at most four'):
        MetaModelPlanner(client=object(), max_planning_turns=turns, max_initial_calls=5)


def test_complete_plan_through_api_has_four_verified_charts_and_matching_hashes(capsys):
    from scripts.diagnose_meta_planner import run_api_case
    client = Client([[call(i, call_id=f'id-{i}') for i in range(4)]])
    # This uses actual CSV upload, API serialization, executor and verifier.
    run_api_case(client, 'muse-spark-1.3', True)
    assert len(client.calls) == 1
    import json
    api = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert api['status_code'] == 201 and api['succeeded']
    assert api['verification_statuses'] == ['verified'] * 4
    assert api['visualization_count'] == 4 and api['hashes_match']
