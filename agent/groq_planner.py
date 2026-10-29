from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from agent.meta_planner import MetaModelPlanner, MetaPlanningResult


class GroqModelError(ValueError):
    """Raised when the temporary Groq planning provider is misconfigured."""


class GroqModelPlanner:
    """OpenAI-compatible Groq adapter for VERA's existing planning contract."""

    provider = "groq"
    default_base_url = "https://api.groq.com/openai/v1"
    default_model = "openai/gpt-oss-20b"

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str = default_base_url,
        model: str = default_model,
        parallel_tool_calls: bool = False,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise GroqModelError("'base_url' must be a non-empty string.")
        if not isinstance(model, str) or not model.strip():
            raise GroqModelError("'model' must be a non-empty string.")

        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        if client is None and not resolved_key:
            raise GroqModelError(
                "GROQ_API_KEY is required when no client is supplied."
            )

        self.base_url = base_url.strip()
        self.model = model.strip()
        self.parallel_tool_calls = bool(parallel_tool_calls)

        self._planner = MetaModelPlanner(
            client=client,
            api_key=resolved_key,
            base_url=self.base_url,
            model=self.model,
            parallel_tool_calls=self.parallel_tool_calls,
        )

    def plan(
        self,
        question: str,
        *,
        dataset_context: Mapping[str, Any],
        tool_catalog: Sequence[Mapping[str, Any]],
    ) -> MetaPlanningResult:
        return self._planner.plan(
            question,
            dataset_context=dataset_context,
            tool_catalog=tool_catalog,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "parallel_tool_calls": self.parallel_tool_calls,
            "fallback_only": True,
            "primary_provider": "meta_model_api",
        }
