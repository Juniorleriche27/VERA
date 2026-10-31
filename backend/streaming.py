"""SSE transport adapter. The agent knows only its metadata event emitter."""
import asyncio
from threading import Event
from typing import Any, AsyncIterator, Callable

from agent.events import AnalysisCancelled, AnalysisEvent, AnalysisEventEmitter
from agent.groq_planner import GroqModelError
from agent.meta_planner import MetaModelError
from agent.pipeline import PipelineError
from backend.analysis import run_analysis


async def analysis_stream(*, analysis_id: str, dataset_id: str, question: str,
                          dataset: dict[str, Any], planner_factory: Callable[[], Any],
                          analyses: dict[str, Any]) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    # Event count is bounded by four planned calls and pipeline max_steps=5.
    # No analytical arrays, model payloads or arbitrary error strings enter this queue.
    queue: asyncio.Queue[AnalysisEvent] = asyncio.Queue()
    cancelled = Event()

    def publish(event: AnalysisEvent) -> None:
        if cancelled.is_set():
            raise AnalysisCancelled()
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except RuntimeError:
            raise AnalysisCancelled() from None

    events = AnalysisEventEmitter(analysis_id, publish, cancelled)

    def work() -> None:
        try:
            run_analysis(analysis_id=analysis_id, dataset_id=dataset_id, question=question,
                         dataset=dataset, planner_factory=planner_factory, analyses=analyses, events=events)
        except AnalysisCancelled:
            return
        except Exception as exc:
            code = ('configuration_error' if isinstance(exc, GroqModelError) else
                    'planning_failed' if isinstance(exc, (MetaModelError, PipelineError)) else 'analysis_error')
            try:
                events.emit('analysis_failed', code=code)
            except AnalysisCancelled:
                pass

    worker = asyncio.create_task(asyncio.to_thread(work))
    try:
        while True:
            event = await queue.get()
            yield event.as_sse()
            if event.event in {'analysis_completed', 'analysis_failed'}:
                break
    finally:
        # Cancels pending work cooperatively at the next event boundary. A provider
        # request already in flight cannot be forcibly interrupted by Python threads.
        cancelled.set()
        worker.cancel()
