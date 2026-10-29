from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from agent.decision_brief import EvidencePackage


class VisualizationSpecError(ValueError):
    """Raised when verified evidence cannot be converted into a safe chart spec."""


@dataclass(frozen=True, slots=True)
class VisualizationSeries:
    data_key: str
    label: str
    axis_label: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dataKey": self.data_key,
            "label": self.label,
        }
        if self.axis_label:
            payload["axisLabel"] = self.axis_label
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
            "evidence": {
                "call_id": self.call_id,
                "trace_hash": self.trace_hash,
                "tool": self.evidence_tool,
            },
        }


class VerifiedVisualizationBuilder:
    """Build chart specifications strictly from verified evidence values."""

    def build(self, evidence: EvidencePackage) -> tuple[VisualizationSpec, ...]:
        if not isinstance(evidence, EvidencePackage):
            raise VisualizationSpecError("'evidence' must be an EvidencePackage.")

        specs: list[VisualizationSpec] = []
        for item in evidence.items:
            if item.tool == "one_way_anova":
                specs.append(self._anova(item))

        return tuple(specs)

    def _anova(self, item: Any) -> VisualizationSpec:
        result = item.result
        if not isinstance(result, Mapping):
            raise VisualizationSpecError("ANOVA evidence result must be a mapping.")

        groups = result.get("group_statistics")
        if not isinstance(groups, Mapping) or not groups:
            raise VisualizationSpecError(
                "ANOVA evidence must contain non-empty 'group_statistics'."
            )

        rows: list[dict[str, Any]] = []
        for group_name, stats in groups.items():
            if not isinstance(group_name, str) or not group_name:
                raise VisualizationSpecError("ANOVA group names must be non-empty strings.")
            if not isinstance(stats, Mapping):
                raise VisualizationSpecError("ANOVA group statistics must be mappings.")

            mean = stats.get("mean")
            if not isinstance(mean, (int, float)) or isinstance(mean, bool):
                raise VisualizationSpecError(
                    f"ANOVA group {group_name!r} has no finite numeric mean."
                )
            if mean != mean or mean in (float("inf"), float("-inf")):
                raise VisualizationSpecError(
                    f"ANOVA group {group_name!r} has no finite numeric mean."
                )

            rows.append({"group": group_name, "mean": float(mean)})

        if not item.call_id or not item.trace_hash:
            raise VisualizationSpecError("Evidence provenance is incomplete.")

        return VisualizationSpec(
            chart_type="bar",
            title="Verified group means",
            description="Group means derived only from the verified ANOVA result.",
            x_key="group",
            series=(
                VisualizationSeries(
                    data_key="mean",
                    label="Mean",
                    axis_label="Mean",
                 ),
            ),
            data=tuple(rows),
            call_id=item.call_id,
            trace_hash=item.trace_hash,
            evidence_tool=item.tool,
        )
