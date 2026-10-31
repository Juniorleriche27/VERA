"""One analysis service shared by synchronous HTTP and event streaming."""
from typing import Any, Callable

from fastapi.encoders import jsonable_encoder

from agent.events import AnalysisCancelled, AnalysisEventEmitter
from agent.pipeline import VerifiedAnalyticsPipeline


def _brief_payload(brief: Any) -> Any:
    if hasattr(brief, 'as_dict') and callable(brief.as_dict):
        return jsonable_encoder(brief.as_dict())
    return jsonable_encoder(brief)


def run_analysis(*, analysis_id: str, dataset_id: str, question: str,
                 dataset: dict[str, Any], planner_factory: Callable[[], Any],
                 analyses: dict[str, Any], events: AnalysisEventEmitter | None = None) -> dict[str, Any]:
    emit = events.emit if events is not None else lambda *args, **kwargs: None
    emit('analysis_started')
    pipeline = VerifiedAnalyticsPipeline(planner_factory())
    result = pipeline.run(question, dataset=dataset['frame'], dataset_context=dataset['profile'], events=events)
    payload: dict[str, Any] = {
        'analysis_id': analysis_id, 'dataset_id': dataset_id, 'question': question,
        'succeeded': result.succeeded, 'exhausted': result.exhausted,
        'unresolved_call_ids': list(result.unresolved_call_ids),
        'steps': [step.as_dict() for step in result.steps],
        'final_status': result.final_trace.verification_status.value if result.final_trace else None,
        'decision_brief': None, 'visualizations': [],
    }
    if result.succeeded:
        emit('decision_brief_started')
        brief = result.build_decision_brief()
        payload['decision_brief'] = _brief_payload(brief)
        emit('decision_brief_built')
        emit('visualizations_started')
        payload['visualizations'] = [spec.as_dict() for spec in result.build_visualizations()]
        emit('visualizations_built', count=len(payload['visualizations']))
    encoded = jsonable_encoder(payload)
    if events is not None and events.cancelled.is_set():
        raise AnalysisCancelled()
    analyses[analysis_id] = encoded
    emit('analysis_completed', succeeded=result.succeeded, analyses=len(result.steps),
         verified=sum(step.trace.verification_status.value == 'verified' for step in result.steps),
         visualizations=len(payload['visualizations']), unresolved=len(result.unresolved_call_ids),
         exhausted=result.exhausted)
    return encoded
