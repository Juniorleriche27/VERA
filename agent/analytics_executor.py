
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from analytics_extensions.registry import HACKATHON_ANALYTICS
from data_analysis_engine import DataAnalysisEngine


class AnalyticsExecutorError(ValueError):
    """Raised when an analytics tool cannot be exposed or executed safely."""


def _merged_inputs(
    parameters: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(parameters, Mapping):
        raise AnalyticsExecutorError("'parameters' must be a mapping.")
    if not isinstance(bindings, Mapping):
        raise AnalyticsExecutorError("'bindings' must be a mapping.")

    merged = dict(bindings)
    for key, value in parameters.items():
        if key in merged and merged[key] != value:
            raise AnalyticsExecutorError(
                f"Conflicting values for analytics input {key!r}."
            )
        merged[key] = value
    return merged


@dataclass(frozen=True, slots=True)
class AnalyticsToolDescriptor:
    name: str
    source: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload["name"] = self.name
        payload["source"] = self.source
        return payload


class AnalyticsToolExecutor:
    """
    Bridge between VERA's agent runtime and the deterministic analytics layer.

    VERA extension implementations take precedence whenever a tool name also
    exists in the pre-existing Data Analysis Engine.
    """

    def __init__(self, engine: DataAnalysisEngine | None = None) -> None:
        self._engine = engine or DataAnalysisEngine()
        self._extensions = {
            method_class().name: method_class
            for method_class in HACKATHON_ANALYTICS
        }

        core_methods = self._engine.available_methods()
        self._core_metadata = {
            item["name"]: dict(item)
            for item in core_methods
        }

        if len(self._core_metadata) != len(core_methods):
            raise AnalyticsExecutorError(
                "The core analytics engine exposes duplicate method names."
            )

        self._catalog = self._build_catalog()

    def _build_catalog(self) -> dict[str, AnalyticsToolDescriptor]:
        catalog: dict[str, AnalyticsToolDescriptor] = {}

        for name, metadata in self._core_metadata.items():
            catalog[name] = AnalyticsToolDescriptor(
                name=name,
                source="core_engine",
                metadata=dict(metadata),
            )

        for name, method_class in self._extensions.items():
            catalog[name] = AnalyticsToolDescriptor(
                name=name,
                source="vera_extension",
                metadata=dict(method_class().metadata()),
            )

        return catalog

    @property
    def tool_count(self) -> int:
        return len(self._catalog)

    def available_tools(self) -> list[dict[str, Any]]:
        return [
            self._catalog[name].as_dict()
            for name in sorted(self._catalog)
        ]

    def tool_source(self, name: str) -> str:
        try:
            return self._catalog[name].source
        except KeyError as exc:
            raise AnalyticsExecutorError(
                f"Unknown analytics tool: {name!r}."
            ) from exc

    def execute(
        self,
        name: str,
        parameters: Mapping[str, Any],
        bindings: Mapping[str, Any],
    ) -> Any:
        if name not in self._catalog:
            raise AnalyticsExecutorError(
                f"Unknown analytics tool: {name!r}."
            )

        kwargs = _merged_inputs(parameters, bindings)

        if name in self._extensions:
            return self._extensions[name]().execute(**kwargs)

        result = self._engine.analyze(name, **kwargs)
        if hasattr(result, "to_dict") and callable(result.to_dict):
            return result.to_dict()
        return result

    def executor_for(self, name: str):
        if name not in self._catalog:
            raise AnalyticsExecutorError(
                f"Unknown analytics tool: {name!r}."
            )

        def _executor(
            parameters: Mapping[str, Any],
            bindings: Mapping[str, Any],
        ) -> Any:
            return self.execute(name, parameters, bindings)

        return _executor

    def executor_map(self) -> dict[str, Any]:
        return {
            name: self.executor_for(name)
            for name in self._catalog
        }
