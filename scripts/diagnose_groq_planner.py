"""Bounded live Groq diagnostics using VERA's real planning/execution path.

Only synthetic demo input is used. Output contains tool names, safe execution
metadata and sanitized errors, never provider credentials or raw requests.
Run: PYTHONPATH=. python scripts/diagnose_groq_planner.py --repeats 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from types import SimpleNamespace
from typing import Any

from agent.groq_planner import GroqModelPlanner
from agent.pipeline import VerifiedAnalyticsPipeline
from backend.app import _safe_dataset_profile
from scripts.run_demo_scenario import QUESTION, load_demo_dataset


MODELS = ("qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b")


def sanitized_message(message: Any) -> str:
    text = message if isinstance(message, str) else "No safe provider message available."
    secret = os.getenv("GROQ_API_KEY")
    if secret:
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"\b(?:gsk_|sk-proj-)[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [REDACTED]", text)
    text = re.sub(r"\borg_[A-Za-z0-9_-]+", "[REDACTED_ORG]", text)
    return text[:1600]


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


def error_metadata(exc: Exception, stage: str) -> dict[str, Any]:
    # Inspect only the provider's message field, not request/headers or an
    # arbitrary exception repr, which can contain authorization material.
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        message = error.get("message") if isinstance(error, dict) else None
    else:
        message = None
    if message is None and stage == "vera":
        message = str(exc)
    return {
        "event": "error", "stage": stage,
        "exception_class": type(exc).__name__,
        "cause_class": type(exc.__cause__).__name__ if exc.__cause__ is not None else None,
        "provider_status_code": getattr(exc, "status_code", None),
        "message": sanitized_message(message),
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
            "event": "response", "model": getattr(response, "model", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "tool_call_count": len(calls),
            "tools": [getattr(getattr(call, "function", None), "name", None) for call in calls],
        })
        return response


def run_case(client: Any, model: str, parallel: bool, frame: Any, profile: dict[str, Any], *, max_planning_turns: int = 4, question: str = QUESTION) -> bool:
    observed = ObservedCompletions(client)
    proxy = SimpleNamespace(chat=SimpleNamespace(completions=observed))
    planner = GroqModelPlanner(client=proxy, model=model, parallel_tool_calls=parallel, max_planning_turns=max_planning_turns)
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
            "bindings": step.trace.bindings,
            "execution_error": sanitized_message(step.trace.execution_error),
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
    app = create_app(planner_factory=lambda: GroqModelPlanner(
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
            "hashes_match": bool(steps) and all(
                step.get("report") and step["trace"]["trace_hash"] == step["report"]["trace_hash"]
                for step in steps
            ),
        })
        return response.status_code == 201 and body.get("succeeded") is True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=1)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--parallel", choices=("false", "true", "both"), default="both")
    parser.add_argument("--pause-seconds", type=int, choices=range(61), default=35)
    parser.add_argument("--planning-turns", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--via-api", action="store_true")
    parser.add_argument("--transport-retries", type=int, choices=range(3), default=0)
    args = parser.parse_args()
    if not os.getenv("GROQ_API_KEY"):
        emit({"event": "blocked", "reason": "Provider credential is unavailable in this process."})
        return 2
    from openai import OpenAI
    client = OpenAI(
        base_url=GroqModelPlanner.default_base_url,
        api_key=os.environ["GROQ_API_KEY"], timeout=45.0, max_retries=args.transport_retries,
    )
    try:
        available = sorted(model.id for model in client.models.list().data)
    except Exception as exc:
        emit(error_metadata(exc, "provider"))
        return 1
    emit({"event": "account_models", "models": available})
    frame = load_demo_dataset()
    profile = _safe_dataset_profile(frame)
    parallel_values = (False, True) if args.parallel == "both" else (args.parallel == "true",)
    failed = False
    for model in dict.fromkeys(args.models):
        if model not in available:
            emit({"event": "skipped", "model": model, "reason": "Not listed by account endpoint."})
            failed = True
            continue
        first = True
        for parallel in parallel_values:
            for repeat in range(1, args.repeats + 1):
                if not first:
                    time.sleep(args.pause_seconds)
                first = False
                emit({"event": "case", "model": model, "parallel_tool_calls": parallel, "repeat": repeat})
                try:
                    if args.via_api:
                        success = run_api_case(client, model, parallel, max_planning_turns=args.planning_turns)
                    else:
                        success = run_case(client, model, parallel, frame, profile, max_planning_turns=args.planning_turns)
                    failed = failed or not success
                except Exception as exc:
                    emit(error_metadata(exc, "vera"))
                    failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
