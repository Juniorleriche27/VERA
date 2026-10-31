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
    "df",
    "observations",
    "raw_data",
    "records",
    "rows",
    "values",
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
            "when the measure is additive; aggregation ('sum' or 'mean') belongs in bindings, not parameters. "
            "For linear_trend, parameters must be an empty object {}. "
            "Place method settings such as min_segment_size, threshold, or variance_ratio_threshold in parameters."
        )
    if name == "bayesian_structural_causal_impact":
        return (
            'Use bindings {"time_column":"date","value_column":"sales",'
            '"aggregation":"sum","intervention_date":"2026-09-01",'
            '"control_columns":["control_sales"]} if these columns and date exist. '
            "Notice: intervention_date belongs strictly in bindings, not parameters. "
            "Place method settings such as credibility or prior_scale in parameters."
        )
    if name in {"one_way_anova", "kruskal_wallis_test"}:
        return 'Use bindings {"grouping_column":"region","value_column":"sales"}. ANOVA and Kruskal take no execution parameters (parameters must be empty {}).'
    if name == "mann_whitney_u_test":
        return 'Use bindings {"grouping_column":"region","value_column":"sales"} with optional group_a/group_b names in bindings; place alternative ("two-sided", "greater", "less") in parameters.'
    return ""


_TOOL_SPECIFIC_SCHEMAS: dict[str, dict[str, Any]] = {
    "linear_trend": {
        "bindings_properties": {
            "time_column": {
                "type": "string",
                "description": "Chronological date or timestamp column name.",
            },
            "value_column": {
                "type": "string",
                "description": "Continuous numeric measure column name (e.g. 'sales').",
            },
            "aggregation": {
                "type": "string",
                "enum": ["sum", "mean"],
                "description": "Aggregation method for repeated timestamps ('sum' or 'mean').",
            },
        },
        "required_bindings": ["time_column", "value_column"],
        "parameters_properties": {},
    },
    "mean_shift_change_point": {
        "bindings_properties": {
            "time_column": {
                "type": "string",
                "description": "Chronological date or timestamp column name.",
            },
            "value_column": {
                "type": "string",
                "description": "Continuous numeric measure column name.",
            },
            "aggregation": {
                "type": "string",
                "enum": ["sum", "mean"],
                "description": "Aggregation method for repeated timestamps ('sum' or 'mean').",
            },
        },
        "required_bindings": ["time_column", "value_column"],
        "parameters_properties": {
            "min_segment_size": {
                "type": "integer",
                "description": "Minimum observations in each segment before/after change point (integer >= 2, default 2).",
            },
            "threshold": {
                "type": "number",
                "description": "Change detection standardized mean-shift sensitivity threshold (finite number > 0, default 1.0).",
            },
        },
    },
    "variance_shift_change_point": {
        "bindings_properties": {
            "time_column": {
                "type": "string",
                "description": "Chronological date or timestamp column name.",
            },
            "value_column": {
                "type": "string",
                "description": "Continuous numeric measure column name.",
            },
            "aggregation": {
                "type": "string",
                "enum": ["sum", "mean"],
                "description": "Aggregation method ('sum' or 'mean').",
            },
        },
        "required_bindings": ["time_column", "value_column"],
        "parameters_properties": {
            "min_segment_size": {
                "type": "integer",
                "description": "Minimum observations per segment (integer >= 2, default 3).",
            },
            "variance_ratio_threshold": {
                "type": "number",
                "description": "Variance ratio threshold strictly greater than 1 (finite number > 1.0, default 4.0).",
            },
        },
    },
    "one_way_anova": {
        "bindings_properties": {
            "grouping_column": {
                "type": "string",
                "description": "Categorical column defining independent groups (e.g. 'region').",
            },
            "value_column": {
                "type": "string",
                "description": "Continuous numeric measure to compare across groups (e.g. 'sales').",
            },
        },
        "required_bindings": ["grouping_column", "value_column"],
        "parameters_properties": {},
    },
    "kruskal_wallis_test": {
        "bindings_properties": {
            "grouping_column": {
                "type": "string",
                "description": "Categorical column defining independent groups.",
            },
            "value_column": {
                "type": "string",
                "description": "Continuous numeric measure to compare ranks across groups.",
            },
        },
        "required_bindings": ["grouping_column", "value_column"],
        "parameters_properties": {},
    },
    "mann_whitney_u_test": {
        "bindings_properties": {
            "grouping_column": {
                "type": "string",
                "description": "Categorical column containing the groups to compare.",
            },
            "value_column": {
                "type": "string",
                "description": "Continuous numeric measure to compare.",
            },
            "group_a": {
                "type": "string",
                "description": "First group name (optional if exactly 2 groups).",
            },
            "group_b": {
                "type": "string",
                "description": "Second group name (optional if exactly 2 groups).",
            },
        },
        "required_bindings": ["grouping_column", "value_column"],
        "parameters_properties": {
            "alternative": {
                "type": "string",
                "enum": ["two-sided", "greater", "less"],
                "description": "Alternative hypothesis direction.",
            },
        },
    },
    "bayesian_structural_causal_impact": {
        "bindings_properties": {
            "time_column": {
                "type": "string",
                "description": "Chronological date or timestamp column name.",
            },
            "value_column": {
                "type": "string",
                "description": "Outcome measure column name (e.g. 'sales').",
            },
            "intervention_date": {
                "type": "string",
                "description": "Exact ISO date where intervention begins (e.g. '2026-09-01').",
            },
            "control_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Control column names sharing identical timestamps.",
            },
            "aggregation": {
                "type": "string",
                "enum": ["sum", "mean"],
                "description": "Temporal aggregation method ('sum' or 'mean').",
            },
        },
        "required_bindings": ["time_column", "value_column", "intervention_date"],
        "parameters_properties": {
            "credibility": {
                "type": "number",
                "description": "Credible interval probability strictly between 0 and 1 (default 0.95).",
            },
            "prior_scale": {
                "type": "number",
                "description": "Prior standard deviation scale, finite positive number (default 10.0).",
            },
        },
    },
}


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

        spec = _TOOL_SPECIFIC_SCHEMAS.get(item["name"])
        bindings_props: dict[str, Any] = dict(spec["bindings_properties"]) if spec else {}
        params_props: dict[str, Any] = dict(spec["parameters_properties"]) if spec else {}
        required_bindings: list[str] = list(spec.get("required_bindings", [])) if spec else []

        parameters_schema: dict[str, Any] = {
            "type": "object",
            "description": (
                "Method configuration only. For tools without execution settings, "
                "this must be an empty object {}."
            ),
            "properties": params_props,
            "additionalProperties": False if spec is not None else True,
        }

        bindings_schema: dict[str, Any] = {
            "type": "object",
            "description": (
                "Logical references to dataset columns or prior "
                "evidence. Do not include raw observations."
            ),
            "properties": bindings_props,
            "additionalProperties": False if spec is not None else {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    {"type": "null"},
                ]
            },
        }
        if required_bindings:
            bindings_schema["required"] = required_bindings

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
                            "parameters": parameters_schema,
                            "bindings": bindings_schema,
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


def _validate_semantic_tool_contract(
    tool: str,
    parameters: Mapping[str, Any],
    bindings: Mapping[str, Any],
    call_id: str,
) -> None:
    for key in parameters:
        if str(key).strip().lower() in _FORBIDDEN_CONTEXT_KEYS:
            raise MetaModelError(
                f"Tool call {call_id!r} 'parameters' contains forbidden raw data key: {key!r}."
            )
    for key in bindings:
        if str(key).strip().lower() in _FORBIDDEN_CONTEXT_KEYS:
            raise MetaModelError(
                f"Tool call {call_id!r} 'bindings' contains forbidden raw data key: {key!r}."
            )

    # Aggregation rule: aggregation belongs to bindings, NOT parameters
    if "aggregation" in parameters:
        raise MetaModelError(
            f"Tool call {call_id!r} placed 'aggregation' in 'parameters'. "
            "For temporal tools, 'aggregation' is a dataset preparation instruction "
            "that must belong to 'bindings', not 'parameters'."
        )
    if "aggregation" in bindings:
        agg = bindings["aggregation"]
        if agg not in ("sum", "mean"):
            raise MetaModelError(
                f"Tool call {call_id!r} specifies invalid aggregation {agg!r}; must be 'sum' or 'mean'."
            )

    # Causal impact rules:
    if tool == "bayesian_structural_causal_impact":
        if "intervention_date" in parameters:
            raise MetaModelError(
                f"Tool call {call_id!r} placed 'intervention_date' in 'parameters'. "
                "Causal impact requires 'intervention_date' in 'bindings', not 'parameters'."
            )
        if "intervention_date" not in bindings:
            raise MetaModelError(
                f"Tool call {call_id!r} for 'bayesian_structural_causal_impact' requires an 'intervention_date' binding."
            )
        date_val = bindings.get("intervention_date")
        if not isinstance(date_val, str) or not date_val.strip():
            raise MetaModelError(
                f"Tool call {call_id!r} 'intervention_date' must be a non-empty string."
            )

    spec = _TOOL_SPECIFIC_SCHEMAS.get(tool)
    if spec is not None:
        # Check required bindings
        for req in spec.get("required_bindings", []):
            if req not in bindings:
                raise MetaModelError(
                    f"Tool call {call_id!r} for {tool!r} is missing required binding {req!r}."
                )
            val = bindings[req]
            if not isinstance(val, str) or not val.strip():
                raise MetaModelError(
                    f"Tool call {call_id!r} binding {req!r} must be a non-empty string."
                )

        # Check unknown parameters (additionalProperties: False)
        allowed_params = set(spec.get("parameters_properties", {}).keys())
        unknown_params = set(parameters.keys()) - allowed_params
        if unknown_params:
            raise MetaModelError(
                f"Tool call {call_id!r} for {tool!r} returned unsupported parameter keys: {sorted(unknown_params)}."
            )

        # Check unknown bindings (additionalProperties: False)
        allowed_bindings = set(spec.get("bindings_properties", {}).keys())
        unknown_bindings = set(bindings.keys()) - allowed_bindings
        if unknown_bindings:
            raise MetaModelError(
                f"Tool call {call_id!r} for {tool!r} returned unsupported binding keys: {sorted(unknown_bindings)}."
            )

        # Validate parameter bounds:
        if "min_segment_size" in parameters:
            val = parameters["min_segment_size"]
            if isinstance(val, bool) or not isinstance(val, int) or val < 2:
                raise MetaModelError(
                    f"Tool call {call_id!r} parameter 'min_segment_size' must be an integer >= 2."
                )

        if "threshold" in parameters:
            val = parameters["threshold"]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not (val > 0) or val == float("inf"):
                raise MetaModelError(
                    f"Tool call {call_id!r} parameter 'threshold' must be a finite number > 0."
                )

        if "variance_ratio_threshold" in parameters:
            val = parameters["variance_ratio_threshold"]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not (val > 1.0) or val == float("inf"):
                raise MetaModelError(
                    f"Tool call {call_id!r} parameter 'variance_ratio_threshold' must be a finite number > 1."
                )

        if "alternative" in parameters:
            val = parameters["alternative"]
            if val not in ("two-sided", "greater", "less"):
                raise MetaModelError(
                    f"Tool call {call_id!r} parameter 'alternative' must be 'two-sided', 'greater', or 'less'."
                )

        if "credibility" in parameters:
            val = parameters["credibility"]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or val <= 0.0 or val >= 1.0:
                raise MetaModelError(
                    f"Tool call {call_id!r} parameter 'credibility' must be a number strictly between 0 and 1."
                )

        if "prior_scale" in parameters:
            val = parameters["prior_scale"]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not (val > 0) or val == float("inf"):
                raise MetaModelError(
                    f"Tool call {call_id!r} parameter 'prior_scale' must be a finite positive number."
                )


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
            "raw dataset values. Put column names, semantic references, and dataset "
            "preparation instructions (such as 'aggregation' and 'intervention_date') in bindings. "
            "Put method configuration settings in parameters. For tools without execution "
            "parameters (such as linear_trend, one_way_anova, kruskal_wallis_test), parameters must be an empty object {}. "
            "Explain each tool call with a concrete objective and reason.\n\n"
            "Query Scope Guidelines:\n"
            "- Multi-part diagnostic queries (e.g. asking why a metric declined, "
            "when it started, whether regions differ, and if an intervention helped): "
            "Decompose into complementary analytical evidence (trend, change point, "
            "group comparison, and causal impact). Choose at most four independent initial calls.\n"
            "- Targeted single-aspect queries (e.g. only asking whether regions differ, "
            "or only asking what the trend is): Call ONLY the 1 or 2 tools directly "
            "answering that aspect (e.g. one_way_anova for group differences). Do NOT "
            "call unrelated temporal, change-point, or causal tools.\n"
            "- Never guess or invent dates, columns, or values. Independent calls should "
            "have empty source_call_ids. For additive measures (sales, revenue) sum is "
            "appropriate; sum/mean must be deliberate. If no analysis is needed, "
            "answer briefly instead of inventing a tool call."
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
            _validate_semantic_tool_contract(
                tool_name, clean_parameters, bindings, call_id
            )

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
