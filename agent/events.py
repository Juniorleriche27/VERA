"""Transport-independent, metadata-only observations of actual execution boundaries."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from threading import Event
from time import monotonic
from typing import Any, Callable


class AnalysisCancelled(BaseException):
    """Cooperative cancellation; must not become a failed statistical tool."""


_FIELDS = {
    'analysis_started': set(), 'planning_started': set(),
    'planning_completed': {'planning_turns'}, 'tools_selected': {'calls'},
    'execution_started': set(), 'execution_completed': {'exhausted'},
    'tool_planned': {'call'}, 'tool_started': {'tool', 'call_id'},
    'tool_completed': {'tool', 'call_id'}, 'tool_failed': {'tool', 'call_id'},
    'verification_started': {'tool', 'call_id'},
    'verification_completed': {'tool', 'call_id', 'status'},
    'tool_verified': {'tool', 'call_id'},
    'replanning_completed': {'call_id', 'status'},
    'decision_brief_started': set(), 'decision_brief_built': set(),
    'visualizations_started': set(), 'visualizations_built': {'count'},
    'analysis_completed': {'succeeded', 'analyses', 'verified', 'visualizations', 'unresolved', 'exhausted'},
    'analysis_failed': {'code'},
}
_REF_FIELDS = {'tool', 'call_id', 'source_call_ids', 'selection'}
_TOKEN = re.compile(r'^[A-Za-z0-9_-]{1,256}$')
_STATUSES = {'verified', 'rejected', 'needs_more_analysis', 'not_verified',
             'scheduled', 'no_fallback', 'unsupported_decision', 'max_steps_reached'}


def tool_reference(trace: Any, selection: str = 'ai') -> dict[str, Any]:
    return {'tool': trace.tool, 'call_id': trace.call_id,
            'source_call_ids': list(trace.source_call_ids), 'selection': selection}


def _validate(data: dict[str, Any]) -> None:
    for key, value in data.items():
        if key in {'tool', 'call_id'}:
            if not isinstance(value, str) or not _TOKEN.fullmatch(value):
                raise ValueError('Invalid event reference.')
        elif key in {'analyses', 'verified', 'visualizations', 'unresolved', 'count', 'planning_turns'}:
            if type(value) is not int or value < 0:
                raise ValueError('Invalid event count.')
        elif key in {'succeeded', 'exhausted'}:
            if type(value) is not bool:
                raise ValueError('Invalid event flag.')
        elif key == 'status' and value not in _STATUSES:
            raise ValueError('Invalid event status.')
        elif key == 'code' and value not in {'planning_failed', 'configuration_error', 'analysis_error'}:
            raise ValueError('Invalid safe error code.')
        elif key == 'selection' and value not in {'ai', 'deterministic'}:
            raise ValueError('Invalid selection origin.')
        elif key == 'source_call_ids':
            if not isinstance(value, list):
                raise ValueError('Invalid lineage.')
            for source in value:
                _validate({'call_id': source})
        elif key in {'calls', 'call'}:
            refs = value if key == 'calls' else [value]
            if not isinstance(refs, list):
                raise ValueError('Invalid event tool list.')
            for ref in refs:
                if not isinstance(ref, dict) or set(ref) != _REF_FIELDS:
                    raise ValueError('Only semantic tool references may be streamed.')
                _validate(ref)


@dataclass(frozen=True, slots=True)
class AnalysisEvent:
    analysis_id: str
    sequence: int
    event: str
    elapsed_ms: float
    _data_json: str

    def as_dict(self) -> dict[str, Any]:
        return {'version': 1, 'analysis_id': self.analysis_id, 'sequence': self.sequence,
                'event': self.event, 'elapsed_ms': self.elapsed_ms,
                'data': json.loads(self._data_json)}

    def as_sse(self) -> str:
        return f'id: {self.sequence}\nevent: {self.event}\ndata: {json.dumps(self.as_dict(), allow_nan=False)}\n\n'


class AnalysisEventEmitter:
    def __init__(self, analysis_id: str, sink: Callable[[AnalysisEvent], None], cancelled: Event | None = None):
        self.analysis_id = analysis_id
        self.sink = sink
        self.cancelled = cancelled or Event()
        self.started = monotonic()
        self.sequence = 0

    def emit(self, event: str, **data: Any) -> None:
        if self.cancelled.is_set():
            raise AnalysisCancelled()
        if event not in _FIELDS or set(data) != _FIELDS[event]:
            raise ValueError('Unsupported event payload fields.')
        _validate(data)
        encoded = json.dumps(data, sort_keys=True, allow_nan=False)
        self.sequence += 1
        self.sink(AnalysisEvent(self.analysis_id, self.sequence, event,
                                round((monotonic() - self.started) * 1000, 3), encoded))
