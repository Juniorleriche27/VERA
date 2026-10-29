from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Mapping, Sequence


class ToolSelectionError(ValueError):
    """Raised when the local analytics-tool preselection cannot be performed safely."""


@dataclass(frozen=True, slots=True)
class ToolSelection:
    """Inspectable result of deterministic local tool preselection."""

    tools: tuple[dict[str, Any], ...]
    total_available: int
    max_tools: int
    ranked_names: tuple[str, ...]


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value).strip()


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalize(value))
        if len(token) > 1
    }


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(
            f"{key} {_flatten_text(item)}"
            for key, item in value.items()
            if key not in {"examples", "example", "schema"}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(_flatten_text(item) for item in value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


class ToolCatalogSelector:
    """
    Deterministically reduce the analytics catalog before any LLM call.

    The selector is deliberately local and data-value blind. It uses only the
    business question, safe dataset metadata, and analytics-tool metadata.
    """

    def __init__(self, max_tools: int = 10) -> None:
        if not isinstance(max_tools, int) or isinstance(max_tools, bool) or max_tools < 1:
            raise ToolSelectionError("'max_tools' must be a positive integer.")
        self.max_tools = max_tools

    def select(
        self,
        question: str,
        *,
        dataset_context: Mapping[str, Any],
        tool_catalog: Sequence[Mapping[str, Any]],
    ) -> ToolSelection:
        if not isinstance(question, str) or not question.strip():
            raise ToolSelectionError("'question' must be a non-empty string.")
        if not isinstance(dataset_context, Mapping):
            raise ToolSelectionError("'dataset_context' must be a mapping.")
        if not isinstance(tool_catalog, Sequence) or isinstance(
            tool_catalog, (str, bytes, bytearray)
        ):
            raise ToolSelectionError("'tool_catalog' must be a sequence of mappings.")

        catalog: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in tool_catalog:
            if not isinstance(item, Mapping):
                raise ToolSelectionError("Every tool catalog entry must be a mapping.")
            payload = dict(item)
            name = payload.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ToolSelectionError("Every tool catalog entry must have a non-empty name.")
            if name in seen:
                raise ToolSelectionError(f"Duplicate analytics tool name: {name!r}.")
            seen.add(name)
            catalog.append(payload)

        if not catalog:
            return ToolSelection((), 0, self.max_tools, ())

        question_norm = _normalize(question)
        question_tokens = _tokens(question)
        context_text = _flatten_text(dataset_context)
        context_tokens = _tokens(context_text)

        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for tool in catalog:
            name = str(tool["name"])
            tool_text = _flatten_text(tool)
            tool_tokens = _tokens(tool_text)
            name_tokens = _tokens(name)

            score = 0.0
            score += 4.0 * len(question_tokens & name_tokens)
            score += 1.5 * len(question_tokens & tool_tokens)
            score += 0.20 * len(context_tokens & tool_tokens)
            score += self._intent_bonus(name, question_norm)

            ranked.append((score, name, tool))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected = ranked[: min(self.max_tools, len(ranked))]
        tools = tuple(dict(item[2]) for item in selected)
        names = tuple(item[1] for item in selected)

        return ToolSelection(
            tools=tools,
            total_available=len(catalog),
            max_tools=self.max_tools,
            ranked_names=names,
        )

    @staticmethod
    def _intent_bonus(name: str, question: str) -> float:
        name_norm = _normalize(name)
        score = 0.0

        compare = any(word in question for word in ("compare", "difference", "different", "across", "between"))
        means = any(word in question for word in ("average", "mean", "means"))
        significance = any(word in question for word in ("significant", "significance", "statistical"))
        nonparametric = any(word in question for word in ("nonparametric", "non parametric", "rank"))
        two_groups = bool(re.search(r"\b(two|2)\b", question))
        many_groups = bool(re.search(r"\b(three|3|multiple|several)\b", question))
        causal = any(word in question for word in ("causal", "impact", "campaign", "intervention", "treatment", "effect"))
        temporal = any(word in question for word in ("time", "trend", "over time", "date", "month", "week", "year"))
        breakpoint = any(word in question for word in ("change point", "changepoint", "break", "shift", "structural change"))
        relation = any(word in question for word in ("correlation", "relationship", "association"))
        predict = any(word in question for word in ("predict", "prediction", "forecast", "driver", "explain", "regression"))
        summary = any(word in question for word in ("summary", "summarize", "describe", "descriptive"))
        tail = any(word in question for word in ("tail", "extreme", "var", "expected shortfall", "risk"))
        direction = any(word in question for word in ("direction", "information flow", "lead lag", "lead-lag"))

        if name_norm == "one way anova":
            if compare:
                score += 8.0
            if means:
                score += 10.0
            if significance:
                score += 8.0
            if many_groups:
                score += 8.0
            if nonparametric:
                score -= 6.0

        if name_norm == "mann whitney u test":
            if compare:
                score += 6.0
            if two_groups:
                score += 8.0
            if nonparametric:
                score += 12.0

        if name_norm == "kruskal wallis test":
            if compare:
                score += 6.0
            if many_groups:
                score += 8.0
            if nonparametric:
                score += 12.0

        if name_norm == "bayesian structural causal impact" and causal:
            score += 24.0
            if temporal:
                score += 6.0

        if name_norm == "directed transfer entropy" and direction:
            score += 24.0

        if name_norm == "extreme value tail risk" and tail:
            score += 24.0

        if "change" in name_norm and "point" in name_norm and breakpoint:
            score += 20.0

        if "correlation" in name_norm and relation:
            score += 18.0

        if "regression" in name_norm and predict:
            score += 16.0

        if "forecast" in name_norm and predict:
            score += 14.0

        if "time series" in name_norm and temporal:
            score += 10.0

        if name_norm == "descriptive statistics" and summary:
            score += 18.0

        if name_norm == "multiple testing correction" and any(
            phrase in question
            for phrase in ("multiple test", "multiple comparison", "p value", "p-value", "false discovery")
        ):
            score += 22.0

        return score
