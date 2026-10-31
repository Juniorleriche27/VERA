from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Mapping, Sequence

from agent.tool_call_trace import ToolCallTrace


class MetaModelError(RuntimeError):
    """Raised when Meta Model API planning violates VERA's planning contract."""


_FORBIDDEN_CONTEXT_KEYS = {
    "data",
    "dataframe",
    "observations",
    "raw_data",
    "records",
    "rows",
}
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def _json_clone(value: Any, field: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise MetaModelError(
            f"'{field}' must contain only finite JSON-compatible values."
        ) from exc


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MetaModelError(f"'{field}' must be a non-empty string.")
    return value.strip()


def _safe_dataset_context(context: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(context, Mapping):
        raise MetaModelError("'dataset_context' must be a mapping.")

    cloned = _json_clone(dict(context), "dataset_context")

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).strip().lower()
                if normalized in _FORBIDDEN_CONTEXT_KEYS:
                    raise MetaModelError(
                        f"Raw dataset payload is forbidden in planner context: "
                        f"{path}.{key}."
                    )
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(cloned, "dataset_context")
    return cloned


def _normalized_catalog(
    catalog: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(catalog, (str, bytes)) or not isinstance(catalog, Sequence):
        raise MetaModelError("'tool_catalog' must be a sequence of mappings.")

    normalized: list[dict[str, Any]] = []
    names: set[str] = set()

    for index, item in enumerate(catalog):
        if not isinstance(item, Mapping):
            raise MetaModelError(
                f"tool_catalog[{index}] must be a mapping."
            )
        metadata = _json_clone(dict(item), f"tool_catalog[{index}]")
        name = _required_text(metadata.get("name"), "tool name")
        if not _TOOL_NAME.fullmatch(name):
            raise MetaModelError(
                f"Tool name {name!r} contains unsupported characters."
            )
        if name in names:
            raise MetaModelError(f"Duplicate tool name: {name!r}.")
        names.add(name)

        description = metadata.get("description", "")
        if description is None:
            description = ""
        if not isinstance(description, str):
            raise MetaModelError(
                f"Tool {name!r} has a non-string description."
            )

        normalized.append(
            {
                "name": name,
                "description": description.strip(),
                "category": metadata.get("category", ""),
                "selection_mode": metadata.get("selection_mode", ""),
                "analysis_unit": metadata.get("analysis_unit", ""),
                "supported_roles": metadata.get("supported_roles", []),
                "role_requirements": metadata.get("role_requirements", {}),
            }
        )

    if not normalized:
        raise MetaModelError("'tool_catalog' cannot be empty.")

    return tuple(normalized)


def _binding_guidance(name: str) -> str:
    if name in {"linear_trend", "mean_shift_change_point", "variance_shift_change_point", "moving_average_trend", "additive_seasonal_decomposition"}:
        return (
            'Use bindings {"time_column":"date","value_column":"sales","aggregation":"sum"} '
            "when the measure is additive; sum/mean must be deliberate. "
            "Place window, period, threshold or min_segment_size in parameters, not bindings."
        )
    if name == "bayesian_structural_causal_impact":
        return (
            'Use bindings {"time_column":"date","value_column":"sales",'
            '"aggregation":"sum","intervention_date":"2026-09-01",'
            '"control_columns":["control_sales"]} if these columns and date exist. '
            "Use value_column (the current Binder contract), not outcome_column. "
            "Provide a date from the available context; never invent dates or controls."
        )
    if name in {"one_way_anova", "kruskal_wallis_test"}:
        return 'Use bindings {"grouping_column":"region","value_column":"sales"} when roles match.'
    if name == "mann_whitney_u_test":
        return 'Use grouping_column, value_column and existing group_a/group_b names; never invent groups.'
    return ""


def build_meta_tools(
    catalog: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert VERA analytics metadata into Meta function-tool definitions."""

    normalized = _normalized_catalog(catalog)
    tools: list[dict[str, Any]] = []

    for item in normalized:
        description_parts = [
            item["description"] or f"Run the {item['name']} analytics method.",
            f"Category: {item['category'] or 'unspecified'}.",
            f"Selection mode: {item['selection_mode'] or 'unspecified'}.",
            f"Analysis unit: {item['analysis_unit'] or 'unspecified'}.",
            "Bindings must contain column names or semantic references only; "
            "never raw dataset rows or raw value arrays.",
            _binding_guidance(item["name"]),
        ]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": item["name"],
                    "description": " ".join(description_parts),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "objective": {
                                "type": "string",
                                "description": (
                                    "The concrete analytical objective this call answers."
                                ),
                            },
                            "reason": {
                                "type": "string",
                                "description": (
                                    "Why this analytical method is appropriate now."
                                ),
                            },
                            "parameters": {
                                "type": "object",
                                "description": (
                                    "Method configuration only, such as alpha, lag, "
                                    "threshold, alternative, or horizon."
                                ),
                                "additionalProperties": True,
                            },
                            "bindings": {
                                "type": "object",
                                "description": (
                                    "Logical references to dataset columns or prior "
                                    "evidence. Do not include raw observations."
                                ),
                                "additionalProperties": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        {"type": "null"},
                                    ]
                                },
                            },
                            "source_call_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Earlier VERA call IDs this analysis depends on."
                                ),
                            },
                        },
                        "required": [
                            "objective",
                            "reason",
                            "parameters",
                            "bindings",
                        ],
                        "additionalProperties": False,
                    },
                },
            }
        )

    return tools


def _validate_bindings(bindings: Any) -> dict[str, Any]:
    if not isinstance(bindings, Mapping):
        raise MetaModelError("'bindings' returned by Meta must be an object.")

    cloned = _json_clone(dict(bindings), "bindings")
    for key, value in cloned.items():
        if value is None or isinstance(value, str):
            continue
        if isinstance(value, list) and all(
            isinstance(item, str) for item in value
        ):
            continue
        raise MetaModelError(
            "Planner bindings may contain only strings, lists of strings, "
            f"or null; invalid binding {key!r}."
        )
    return cloned


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, Mapping):
        return _json_clone(dict(usage), "usage")
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return _json_clone(dict(dumped), "usage")

    result: dict[str, Any] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, field, None)
        if value is not None:
            result[field] = value
    return _json_clone(result, "usage")


@dataclass(frozen=True, slots=True)
class MetaPlanningResult:
    traces: tuple[ToolCallTrace, ...]
    assistant_content: str
    finish_reason: str
    model: str
    usage: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "traces": [trace.as_dict() for trace in self.traces],
            "assistant_content": self.assistant_content,
            "finish_reason": self.finish_reason,
            "model": self.model,
            "usage": _json_clone(self.usage, "usage"),
        }


class MetaModelPlanner:
    """Privacy-preserving Meta Model API planner for VERA analytics tools."""

    DEFAULT_BASE_URL = "https://api.meta.ai/v1"
    DEFAULT_MODEL = "muse-spark-1.3"

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        parallel_tool_calls: bool = True,
        max_initial_calls: int = 4,
    ) -> None:
        if isinstance(max_initial_calls, bool) or not isinstance(max_initial_calls, int) or max_initial_calls < 1:
            raise MetaModelError("'max_initial_calls' must be a positive integer.")
        self.max_initial_calls = max_initial_calls
        self.model = _required_text(model, "model")
        self.base_url = _required_text(base_url, "base_url")
        self.parallel_tool_calls = bool(parallel_tool_calls)

        if client is None:
            key = api_key or os.environ.get("MODEL_API_KEY")
            if not key:
                raise MetaModelError(
                    "MODEL_API_KEY is required when no client is injected."
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise MetaModelError(
                    "The 'openai' package is required for Meta Model API access."
                ) from exc
            client = OpenAI(base_url=self.base_url, api_key=key)

        self._client = client

    def plan(
        self,
        question: str,
        *,
        dataset_context: Mapping[str, Any],
        tool_catalog: Sequence[Mapping[str, Any]],
    ) -> MetaPlanningResult:
        question_text = _required_text(question, "question")
        safe_context = _safe_dataset_context(dataset_context)
        normalized_catalog = _normalized_catalog(tool_catalog)
        allowed_names = {item["name"] for item in normalized_catalog}
        tools = build_meta_tools(normalized_catalog)

        developer_message = (
            "You are VERA's analytics planner. Decide which deterministic "
            "analytics tools are needed to answer the user's business question. "
            "Use only the supplied function tools. The dataset context contains "
            "metadata and safe aggregates, not raw rows. Never request or invent "
            "raw dataset values. Put only column names or semantic references in "
            "bindings. Put method settings in parameters. Explain each tool call "
            "with a concrete objective and reason. Decompose multi-part business "
            "questions into complementary analytical evidence (trend, timing of "
            "change, group comparison, and intervention when relevant). Diagnosis "
            "of why sales declined is not the same as causal effect of a campaign. "
            "Choose only tools necessary to cover the question; avoid redundant "
            "tests. Prefer descriptive diagnostic, change localization, group "
            "comparison, then intervention when the dataset supports these. "
            "Request at most four independent initial tool calls to leave a fifth "
            "execution slot for deterministic robustness replanning. Independent "
            "calls should have empty source_call_ids: declare dependencies only "
            "for genuine prior evidence, not artificial ordering (the runtime "
            "cannot generally inject previous results into later tool inputs). "
            "For additive sales or revenue sum may be appropriate, but never "
            "guess aggregation for other measures. Never invent analytical results "
            "or dates; local deterministic tools calculate actual values. "
            "If no analysis is needed, answer briefly instead of inventing a call."
        )
        user_message = (
            "Business question:\n"
            f"{question_text}\n\n"
            "Safe dataset context:\n"
            + json.dumps(
                safe_context,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "developer", "content": developer_message},
                    {"role": "user", "content": user_message},
                ],
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=self.parallel_tool_calls,
            )
        except Exception as exc:
            raise MetaModelError(
                f"Meta Model API planning request failed: {exc}"
            ) from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise MetaModelError("Meta Model API returned no choices.")

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise MetaModelError("Meta Model API returned no assistant message.")

        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        assistant_content = str(getattr(message, "content", "") or "")
        tool_calls = getattr(message, "tool_calls", None) or []
        if len(tool_calls) > self.max_initial_calls:
            raise MetaModelError(
                f"Planner returned {len(tool_calls)} initial calls; maximum is {self.max_initial_calls}."
            )

        traces: list[ToolCallTrace] = []
        seen_call_ids: set[str] = set()

        for call in tool_calls:
            call_id = _required_text(getattr(call, "id", None), "tool_call.id")
            if call_id in seen_call_ids:
                raise MetaModelError(
                    f"Meta returned duplicate tool call id: {call_id!r}."
                )
            seen_call_ids.add(call_id)

            function = getattr(call, "function", None)
            if function is None:
                raise MetaModelError(
                    f"Tool call {call_id!r} has no function payload."
                )
            tool_name = _required_text(
                getattr(function, "name", None),
                "tool_call.function.name",
            )
            if tool_name not in allowed_names:
                raise MetaModelError(
                    f"Meta requested unknown analytics tool: {tool_name!r}."
                )

            raw_arguments = getattr(function, "arguments", None)
            if not isinstance(raw_arguments, str):
                raise MetaModelError(
                    f"Tool call {call_id!r} arguments must be JSON text."
                )
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise MetaModelError(
                    f"Tool call {call_id!r} returned invalid JSON arguments."
                ) from exc
            if not isinstance(arguments, dict):
                raise MetaModelError(
                    f"Tool call {call_id!r} arguments must decode to an object."
                )

            allowed_argument_keys = {
                "objective",
                "reason",
                "parameters",
                "bindings",
                "source_call_ids",
            }
            unknown_keys = set(arguments) - allowed_argument_keys
            if unknown_keys:
                raise MetaModelError(
                    f"Tool call {call_id!r} returned unsupported argument keys: "
                    f"{sorted(unknown_keys)}."
                )

            objective = _required_text(
                arguments.get("objective"),
                "objective",
            )
            reason = _required_text(arguments.get("reason"), "reason")

            parameters = arguments.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise MetaModelError(
                    f"Tool call {call_id!r} 'parameters' must be an object."
                )
            clean_parameters = _json_clone(
                dict(parameters),
                "parameters",
            )
            bindings = _validate_bindings(arguments.get("bindings", {}))

            source_ids = arguments.get("source_call_ids", [])
            if not isinstance(source_ids, list) or not all(
                isinstance(value, str) and value.strip()
                for value in source_ids
            ):
                raise MetaModelError(
                    f"Tool call {call_id!r} 'source_call_ids' must be "
                    "a list of non-empty strings."
                )

            traces.append(
                ToolCallTrace.planned(
                    call_id=call_id,
                    tool=tool_name,
                    objective=objective,
                    reason=reason,
                    parameters=clean_parameters,
                    bindings=bindings,
                    source_call_ids=source_ids,
                )
            )

        response_model = str(
            getattr(response, "model", self.model) or self.model
        )

        return MetaPlanningResult(
            traces=tuple(traces),
            assistant_content=assistant_content,
            finish_reason=finish_reason,
            model=response_model,
            usage=_usage_dict(getattr(response, "usage", None)),
        )
