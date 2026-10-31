"""Presentation-only visualizations assembled from verified deterministic evidence."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from agent.decision_brief import EvidencePackage


class VisualizationSpecError(ValueError):
    """Raised when verified evidence cannot be converted into a safe chart spec."""


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise VisualizationSpecError(f"{label} must be a finite number.")
    return float(value)


def _payload(tool: str, original: Mapping[str, Any]) -> Mapping[str, Any]:
    if original.get("status") == "success" and original.get("method") == tool and isinstance(original.get("result"), Mapping):
        return original["result"]
    return original


def _priority(tool: str) -> int:
    return {"linear_trend": 0, "mean_shift_change_point": 1,
            "variance_shift_change_point": 1, "one_way_anova": 2,
            "kruskal_wallis_test": 2, "bayesian_structural_causal_impact": 3}.get(tool, 4)


@dataclass(frozen=True, slots=True)
class VisualizationSeries:
    data_key: str
    label: str
    axis_label: str = ""
    role: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"dataKey": self.data_key, "label": self.label}
        if self.axis_label:
            payload["axisLabel"] = self.axis_label
        if self.role:
            payload["role"] = self.role
        return payload


@dataclass(frozen=True, slots=True)
class VisualizationSpec:
    chart_type: str
    title: str
    description: str
    x_key: str
    series: tuple[VisualizationSeries, ...]
    data: tuple[dict[str, Any], ...]
    call_id: str
    trace_hash: str
    evidence_tool: str
    annotations: tuple[dict[str, Any], ...] = ()
    bands: tuple[dict[str, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "chartType": self.chart_type,
            "meta": {
                "title": self.title,
                "description": self.description,
                "footer": f"Verified evidence • {self.evidence_tool} • {self.trace_hash[:12]}",
            },
            "xKey": self.x_key,
            "series": [item.as_dict() for item in self.series],
            "data": [dict(row) for row in self.data],
            "evidence": {"call_id": self.call_id, "trace_hash": self.trace_hash, "tool": self.evidence_tool},
            "annotations": [dict(item) for item in self.annotations],
            "bands": [dict(item) for item in self.bands],
        }


class VerifiedVisualizationBuilder:
    """Only render analytical series already present in a verified EvidencePackage."""

    def build(self, evidence: EvidencePackage) -> tuple[VisualizationSpec, ...]:
        if not isinstance(evidence, EvidencePackage):
            raise VisualizationSpecError("'evidence' must be an EvidencePackage.")
        builders = {
            "linear_trend": self._trend,
            "mean_shift_change_point": self._mean_shift,
            "variance_shift_change_point": self._variance_shift,
            "one_way_anova": self._anova,
            "kruskal_wallis_test": self._kruskal,
            "bayesian_structural_causal_impact": self._causal,
        }
        indexed = sorted(enumerate(evidence.items), key=lambda pair: (_priority(pair[1].tool), pair[0]))
        specs: list[VisualizationSpec] = []
        for _, item in indexed:
            if item.tool not in builders:
                continue
            if not item.call_id or not item.trace_hash:
                raise VisualizationSpecError("Evidence provenance is incomplete.")
            if not isinstance(item.result, Mapping):
                raise VisualizationSpecError("Evidence result must be a mapping.")
            result = _payload(item.tool, item.result)
            built = builders[item.tool](item, result)
            if built is not None:
                specs.append(built)
        return tuple(specs)

    @staticmethod
    def _spec(item: Any, chart: str, title: str, description: str, x: str,
              series: tuple[VisualizationSeries, ...], rows: list[dict[str, Any]],
              *, annotations: tuple[dict[str, Any], ...] = (),
              bands: tuple[dict[str, str], ...] = ()) -> VisualizationSpec:
        return VisualizationSpec(
            chart_type=chart, title=title, description=description, x_key=x,
            series=series, data=tuple(rows), call_id=item.call_id,
            trace_hash=item.trace_hash, evidence_tool=item.tool,
            annotations=annotations, bands=bands,
        )

    def _trend(self, item: Any, result: Mapping[str, Any]) -> VisualizationSpec:
        times, fitted = result.get("timestamps"), result.get("fitted_values")
        if not isinstance(times, (list, tuple)) or not isinstance(fitted, (list, tuple)) or not times or len(times) != len(fitted):
            raise VisualizationSpecError("Trend timestamps and fitted_values require equal non-zero lengths.")
        rows = []
        for index, (time, value) in enumerate(zip(times, fitted)):
            if not isinstance(time, str) or not time.strip():
                raise VisualizationSpecError(f"Trend timestamp at position {index} must be a non-empty string.")
            rows.append({"timestamp": time, "fitted": _finite_number(value, "fitted")})
        return self._spec(item, "line", "Verified sales trend",
                          "Fitted values from the verified linear-trend analysis; not observed sales.",
                          "timestamp", (VisualizationSeries("fitted", "Verified fitted trend", role="fitted"),), rows)

    def _shift(self, item: Any, result: Mapping[str, Any], *, kind: str) -> VisualizationSpec | None:
        if result.get("detected") is not True:
            return None
        candidate = result.get("strongest_candidate")
        if not isinstance(candidate, Mapping):
            raise VisualizationSpecError(f"{kind} shift requires strongest_candidate mapping.")
        key = "mean" if kind == "mean" else "variance"
        left = _finite_number(candidate.get(f"left_{key}"), f"left_{key}")
        right = _finite_number(candidate.get(f"right_{key}"), f"right_{key}")
        time = result.get("change_timestamp", candidate.get("timestamp"))
        if time is not None and (not isinstance(time, str) or not time.strip()):
            raise VisualizationSpecError("Change timestamp must be a non-empty string when present.")
        description = f"Verified {key} before and after the detected shift"
        annotations: tuple[dict[str, Any], ...] = ()
        if time:
            description += f" around {time}"
            annotations = ({"type": "change_point", "label": "Detected shift", "value": time},)
        rows = [{"segment": "Before shift", key: left}, {"segment": "After shift", key: right}]
        return self._spec(item, "bar", f"Verified {key} shift", description + ".", "segment",
                          (VisualizationSeries(key, key.title(), role=key),), rows, annotations=annotations)

    def _mean_shift(self, item: Any, result: Mapping[str, Any]) -> VisualizationSpec | None:
        return self._shift(item, result, kind="mean")

    def _variance_shift(self, item: Any, result: Mapping[str, Any]) -> VisualizationSpec | None:
        return self._shift(item, result, kind="variance")

    def _groups(self, item: Any, result: Mapping[str, Any], *, stat: str) -> VisualizationSpec:
        groups = result.get("group_statistics")
        if not isinstance(groups, Mapping) or not groups:
            raise VisualizationSpecError(f"{item.tool} evidence must contain non-empty 'group_statistics'.")
        rows: list[dict[str, Any]] = []
        for name, group in groups.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(group, Mapping):
                raise VisualizationSpecError(f"{item.tool} group names and statistics must be valid.")
            rows.append({"group": name, stat: _finite_number(group.get(stat), f"{item.tool} group {name!r} {stat}")})
        description = ("Group means derived only from the verified ANOVA result." if stat == "mean"
                       else "Group medians from the verified Kruskal-Wallis evidence.")
        return self._spec(item, "bar", f"Verified group {stat}s", description, "group",
                          (VisualizationSeries(stat, stat.title(), axis_label=stat.title(), role=stat),), rows)

    def _anova(self, item: Any, result: Mapping[str, Any]) -> VisualizationSpec:
        return self._groups(item, result, stat="mean")

    def _kruskal(self, item: Any, result: Mapping[str, Any]) -> VisualizationSpec:
        return self._groups(item, result, stat="median")

    def _causal(self, item: Any, result: Mapping[str, Any]) -> VisualizationSpec:
        observed = result.get("observed_post")
        counterfactual = result.get("counterfactual_post_mean")
        intervals = result.get("counterfactual_post_interval")
        if not all(isinstance(value, (list, tuple)) for value in (observed, counterfactual, intervals)):
            raise VisualizationSpecError("Causal chart requires observed, counterfactual and interval arrays.")
        if not observed or len(observed) != len(counterfactual) or len(observed) != len(intervals):
            raise VisualizationSpecError("Causal observed, counterfactual and interval lengths must match and be non-empty.")
        rows: list[dict[str, Any]] = []
        for index, (obs, predicted, interval) in enumerate(zip(observed, counterfactual, intervals), 1):
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise VisualizationSpecError(f"Causal interval at post period {index} requires two bounds.")
            lower = _finite_number(interval[0], "counterfactual_lower")
            upper = _finite_number(interval[1], "counterfactual_upper")
            if lower > upper:
                raise VisualizationSpecError(f"Causal interval at post period {index} is inverted.")
            rows.append({
                "post_period": index,
                "observed": _finite_number(obs, "observed"),
                "counterfactual": _finite_number(predicted, "counterfactual"),
                "counterfactual_lower": lower,
                "counterfactual_upper": upper,
            })
        credibility = result.get("credibility")
        if credibility is not None:
            rate = _finite_number(credibility, "credibility")
            if not 0 < rate < 1:
                raise VisualizationSpecError("Causal credibility must lie strictly between zero and one.")
            description = f"Observed post-intervention values and model counterfactual with {rate * 100:g}% credible interval."
        else:
            description = "Observed post-intervention values and model counterfactual with credible interval (level unspecified)."
        return self._spec(item, "line", "Verified campaign impact", description,
                          "post_period", (
                              VisualizationSeries("observed", "Observed post-intervention", role="observed"),
                              VisualizationSeries("counterfactual", "Model counterfactual", role="counterfactual"),
                              VisualizationSeries("counterfactual_lower", "Credible lower bound", role="interval_lower"),
                              VisualizationSeries("counterfactual_upper", "Credible upper bound", role="interval_upper"),
                          ), rows, bands=({"lowerKey": "counterfactual_lower",
                                          "upperKey": "counterfactual_upper",
                                          "label": "Counterfactual credible interval"},))
