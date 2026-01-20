from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, ValidationError

from sgr_agent_core.base_tool import BaseTool

from examples.call_center_mock.contracts import (
    AggregateSummary,
    CallCenterFilters,
    CallCenterRecord,
    DateRange,
    GroupCount,
    GroupKpi,
    LoadSummary,
    SampleRecord,
    SCHEMA_VERSION,
    TrendSummary,
)
from examples.call_center_mock.metrics import compute_kpis, filter_records, group_kpis, trend_kpis

DATASET_PATH = Path(__file__).parent / "mock_data" / "call_center_records.json"


def _get_nested(data: object, *keys: str, default=None):
    current = data
    for key in keys:
        if isinstance(current, BaseModel):
            current = getattr(current, key, None)
        elif isinstance(current, dict):
            current = current.get(key)
        else:
            return default
    return default if current is None else current


class CallCenterLoadDatasetTool(BaseTool):
    """Load and filter call center records for analysis."""

    tool_name: ClassVar[str] = "call_center_load_dataset"

    reasoning: str = Field(description="Why this dataset slice is needed")
    date_from: str = Field(description="Start date (YYYY-MM-DD)")
    date_to: str = Field(description="End date (YYYY-MM-DD)")
    filters: CallCenterFilters | None = Field(default=None, description="Optional filters")
    sample_limit: int = Field(default=200, ge=1, le=1000, description="Maximum records to load")
    include_transcripts: bool = Field(
        default=False, description="Whether to keep transcripts in the stored dataset"
    )

    async def __call__(self, context, config, **kwargs) -> str:
        if not DATASET_PATH.exists():
            return json.dumps({"error": f"Dataset not found at {DATASET_PATH}"}, indent=2)

        raw_payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        raw_records = raw_payload.get("records") if isinstance(raw_payload, dict) else raw_payload
        filters = self.filters or CallCenterFilters()
        validated_records: list[CallCenterRecord] = []
        invalid_records = 0
        for record in raw_records:
            try:
                validated_records.append(CallCenterRecord.model_validate(record))
            except ValidationError:
                invalid_records += 1
        filtered = filter_records(validated_records, self.date_from, self.date_to, filters)
        if not self.include_transcripts:
            filtered = [
                record.model_copy(update={"nlp": record.nlp.model_copy(update={"transcript": ""})})
                for record in filtered
            ]
        filtered = filtered[: self.sample_limit]

        context.custom_context = context.custom_context or {}
        context.custom_context["call_center_dataset"] = {
            "date_from": self.date_from,
            "date_to": self.date_to,
            "filters": filters.model_dump(),
            "records": filtered,
            "schema_version": SCHEMA_VERSION,
        }

        segments = Counter(segment for segment in (_get_nested(record, "caller", "segment") for record in filtered) if segment)
        tags = Counter(
            tag
            for record in filtered
            for tag in _get_nested(record, "nlp", "tags", default=[])
            if tag
        )
        sentiments = Counter(
            label for label in (_get_nested(record, "nlp", "sentiment_label") for record in filtered) if label
        )
        kpis = compute_kpis(filtered)

        summary = LoadSummary(
            total_records=len(filtered),
            date_range=DateRange(start=self.date_from, end=self.date_to),
            segment_breakdown=[GroupCount(group=key, count=value) for key, value in segments.most_common()],
            top_tags=[GroupCount(group=key, count=value) for key, value in tags.most_common(8)],
            sentiment_distribution=[
                GroupCount(group=key, count=value) for key, value in sentiments.most_common()
            ],
            avg_sentiment_score=kpis.avg_sentiment_score,
            avg_wait_time_sec=kpis.avg_wait_sec,
            escalation_rate=kpis.escalation_rate,
            sample_records=[
                SampleRecord(
                    call_id=record.call_id,
                    queue=record.queue,
                    segment=record.caller.segment,
                    issue=record.case.issue_category,
                    sentiment=record.nlp.sentiment_label,
                )
                for record in filtered[:5]
            ],
            invalid_records=invalid_records,
        )
        return json.dumps(summary.model_dump(by_alias=True), indent=2)


class CallCenterAggregateTool(BaseTool):
    """Aggregate KPIs and trend signals from the loaded dataset."""

    tool_name: ClassVar[str] = "call_center_aggregate"

    reasoning: str = Field(description="Why this aggregation is needed")
    group_by: Literal["tag", "agent", "segment", "product", "queue", "channel"] = Field(
        default="tag", description="Dimension to group by"
    )
    focus_metric: Literal["volume", "sentiment", "escalation", "wait_time", "duration"] = Field(
        default="sentiment", description="Metric used to rank groups"
    )
    top_n: int = Field(default=5, ge=1, le=15, description="How many top groups to return")
    min_calls: int = Field(default=2, ge=1, description="Minimum calls per group")

    async def __call__(self, context, config, **kwargs) -> str:
        dataset = (context.custom_context or {}).get("call_center_dataset", {})
        records = dataset.get("records", [])
        if not records:
            return json.dumps({"error": "No dataset loaded. Run call_center_load_dataset first."}, indent=2)

        overall = compute_kpis(records)
        group_metrics = group_kpis(records, self.group_by, min_calls=self.min_calls)

        def _sort_key(item: GroupKpi):
            if self.focus_metric == "volume":
                return item.call_count
            if self.focus_metric == "sentiment":
                return item.avg_sentiment_score if item.avg_sentiment_score is not None else 0
            if self.focus_metric == "escalation":
                return item.escalation_rate
            if self.focus_metric == "wait_time":
                return item.avg_wait_sec if item.avg_wait_sec is not None else 0
            if self.focus_metric == "duration":
                return item.avg_duration_sec if item.avg_duration_sec is not None else 0
            return 0

        reverse = self.focus_metric in {"volume", "escalation", "wait_time", "duration"}
        sorted_groups = sorted(group_metrics, key=_sort_key, reverse=reverse)

        insights = []
        if overall.avg_sentiment_score is not None and overall.avg_sentiment_score < -0.15:
            insights.append("Overall sentiment skews negative; prioritize root-cause fixes.")
        if overall.escalation_rate > 0.25:
            insights.append("Escalation rate is elevated; review escalation triggers and routing.")
        if overall.avg_wait_sec and overall.avg_wait_sec > 90:
            insights.append("Average wait time is high; staffing or IVR deflection may be needed.")
        if sorted_groups:
            worst = sorted_groups[0]
            insights.append(f"Most challenged group by {self.focus_metric}: {worst.group}.")

        summary = AggregateSummary(
            overall_kpis=overall,
            group_by=self.group_by,
            focus_metric=self.focus_metric,
            top_groups=sorted_groups[: self.top_n],
            insights=insights,
        )
        return json.dumps(summary.model_dump(), indent=2)


class CallCenterTrendTool(BaseTool):
    """Compute KPI trends over time for the loaded dataset."""

    tool_name: ClassVar[str] = "call_center_trend"

    reasoning: str = Field(description="Why trend analysis is needed")
    period: Literal["month"] = Field(default="month", description="Trend granularity")

    async def __call__(self, context, config, **kwargs) -> str:
        dataset = (context.custom_context or {}).get("call_center_dataset", {})
        records = dataset.get("records", [])
        if not records:
            return json.dumps({"error": "No dataset loaded. Run call_center_load_dataset first."}, indent=2)

        trend = trend_kpis(records, period=self.period)
        summary = TrendSummary(period=trend.period, points=trend.points)
        return json.dumps(summary.model_dump(), indent=2)
