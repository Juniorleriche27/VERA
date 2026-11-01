"""Bounded live Meta diagnostics using VERA's real planning/execution path.

Only synthetic demo input is used. Output contains tool names, safe execution
metadata and fixed error messages, never provider credentials or raw requests.
Run: PYTHONPATH=. python scripts/diagnose_meta_planner.py
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

from agent.meta_planner import MetaModelPlanner
from agent.pipeline import VerifiedAnalyticsPipeline
from scripts.run_demo_scenario import QUESTION


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


def error_metadata(exc: Exception, stage: str) -> dict[str, Any]:
    # Never print exception text, provider bodies, headers or arbitrary reprs.
    return {
        "event": "error", "stage": stage,
        "exception_class": type(exc).__name__,
        "cause_class": type(exc.__cause__).__name__ if exc.__cause__ else None,
        "provider_status_code": getattr(exc, "status_code", None),
        "message": "Meta request failed." if stage == "provider" else "VERA planning or verification failed.",
    }


class ObservedCompletions:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.provider_failed = False

    def create(self, **kwargs: Any) -> Any:
        self.provider_failed = False
        emit({
            "event": "request", "model": kwargs["model"],
            "parallel_tool_calls": kwargs["parallel_tool_calls"],
            "selected_tools": [tool["function"]["name"] for tool in kwargs["tools"]],
        })
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            self.provider_failed = True
            emit(error_metadata(exc, "provider"))
            raise
        choices = getattr(response, "choices", ()) or ()
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None)
        calls = getattr(message, "tool_calls", ()) or ()
        emit({
            "event": "response", "model": kwargs["model"],
            "finish_reason": getattr(choice, "finish_reason", None),
            "tool_call_count": len(calls),
        })
        return response


def run_case(client: Any, model: str, parallel: bool, frame: Any, profile: dict[str, Any], *, max_planning_turns: int = 4, question: str = QUESTION) -> bool:
    observed = ObservedCompletions(client)
    proxy = SimpleNamespace(chat=SimpleNamespace(completions=observed))
    planner = MetaModelPlanner(client=proxy, model=model, parallel_tool_calls=parallel, max_planning_turns=max_planning_turns)
    try:
        result = VerifiedAnalyticsPipeline(planner).run(
            question, dataset=frame, dataset_context=profile,
        )
    except Exception as exc:
        if not observed.provider_failed:
            emit(error_metadata(exc, "vera"))
        return False
    emit({
        "event": "pipeline", "succeeded": result.succeeded,
        "planning_turns": result.planning.planning_turns,
        "exhausted": result.exhausted,
        "unresolved_call_ids": list(result.unresolved_call_ids),
        "steps": [{
            "tool": step.trace.tool, "status": step.trace.status.value,
            "verification_status": step.trace.verification_status.value,
            "trace_hash": step.trace.trace_hash,
        } for step in result.steps],
        "visualization_count": len(result.build_visualizations()) if result.succeeded else 0,
    })
    return result.succeeded


def run_api_case(client: Any, model: str, parallel: bool, *, max_planning_turns: int = 4) -> bool:
    from fastapi.testclient import TestClient
    from backend.app import create_app
    from scripts.run_demo_scenario import CSV_PATH

    observed = ObservedCompletions(client)
    proxy = SimpleNamespace(chat=SimpleNamespace(completions=observed))
    app = create_app(planner_factory=lambda: MetaModelPlanner(
        client=proxy, model=model, parallel_tool_calls=parallel,
        max_planning_turns=max_planning_turns,
    ))
    with TestClient(app) as api:
        upload = api.post("/datasets", files={"file": (CSV_PATH.name, CSV_PATH.read_bytes(), "text/csv")})
        if upload.status_code != 201:
            emit({"event": "api", "upload_status": upload.status_code})
            return False
        response = api.post("/analyses", json={"dataset_id": upload.json()["dataset_id"], "question": QUESTION})
        body = response.json()
        steps = body.get("steps", [])
        emit({
            "event": "api", "status_code": response.status_code,
            "succeeded": body.get("succeeded"), "final_status": body.get("final_status"),
            "tools": [step["trace"]["tool"] for step in steps],
            "verification_statuses": [step["trace"]["verification_status"] for step in steps],
            "visualization_count": len(body.get("visualizations", [])),
            "brief_produced": bool(body.get("decision_brief")),
            "hashes_match": bool(steps) and all(
                step.get("report") and step["trace"]["trace_hash"] == step["report"]["trace_hash"]
                for step in steps
            ),
        })
        return (
            response.status_code == 201 and body.get("succeeded") is True
            and bool(steps) and bool(body.get("decision_brief"))
            and len(body.get("visualizations", [])) == len(steps)
            and all(step["trace"]["verification_status"] == "verified"
                    and step.get("report")
                    and step["trace"]["trace_hash"] == step["report"]["trace_hash"]
                    for step in steps)
        )


def main() -> int:
    if not os.getenv("MODEL_API_KEY"):
        emit({"status": "FAILED", "stage": "configuration", "reason": "MODEL_API_KEY is unavailable."})
        return 2
    try:
        from openai import OpenAI
        client = OpenAI(base_url=MetaModelPlanner.DEFAULT_BASE_URL,
                        api_key=os.environ["MODEL_API_KEY"], timeout=45.0, max_retries=0)
        response = client.chat.completions.create(
            model=MetaModelPlanner.DEFAULT_MODEL,
            messages=[{"role": "user", "content": "Reply with OK."}],
        )
        if not response.choices or not response.choices[0].message.content:
            emit({"status": "FAILED", "stage": "minimal_request"})
            return 1
        emit({"status": "SUCCESS", "stage": "minimal_request"})
        # Actual filtered tools, safe metadata, local execution and verification.
        success = run_api_case(client, MetaModelPlanner.DEFAULT_MODEL, True)
    except Exception as exc:
        emit(error_metadata(exc, "provider"))
        emit({"status": "FAILED", "stage": "diagnostic"})
        return 1
    emit({"status": "SUCCESS" if success else "FAILED", "stage": "tool_calling_pipeline"})
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
