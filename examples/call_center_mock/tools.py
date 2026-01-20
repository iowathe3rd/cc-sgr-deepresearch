from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import ClassVar, Iterable, Literal

from pydantic import BaseModel, Field, ValidationError

from sgr_agent_core.base_tool import BaseTool

from examples.call_center_mock.contracts import (
    AggregateSummary,
    CallCenterRecord,
    DateRange,
    GroupCount,
    GroupKpi,
    KpiSummary,
    LoadSummary,
    SampleRecord,
    SCHEMA_VERSION,
)

DATASET_PATH = Path(__file__).parent / "mock_data" / "call_center_records.json"


class CallCenterFilters(BaseModel):
    segments: list[str] | None = None
    tags: list[str] | None = None
    products: list[str] | None = None
    queues: list[str] | None = None
    channels: list[str] | None = None
    sentiment_labels: list[str] | None = None
    escalation_only: bool = False
    min_sentiment: float | None = None
    max_sentiment: float | None = None


def _parse_datetime(value: str) -> datetime:
    cleaned = value.replace("Z", "+00:00")
    if len(cleaned) == 10:
        cleaned = f"{cleaned}T00:00:00"
    parsed = datetime.fromisoformat(cleaned)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


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


def _safe_mean(values: Iterable[float]) -> float | None:
    values = [value for value in values if value is not None]
    return mean(values) if values else None


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
        start_dt = _parse_datetime(self.date_from)
        end_dt = _parse_datetime(self.date_to)
        filters = self.filters or CallCenterFilters()

        validated_records: list[CallCenterRecord] = []
        invalid_records = 0
        for record in raw_records:
            try:
                validated_records.append(CallCenterRecord.model_validate(record))
            except ValidationError:
                invalid_records += 1

        filtered: list[CallCenterRecord] = []
        for record in validated_records:
            call_start = record.call_start
            if call_start < start_dt or call_start > end_dt:
                continue

            segment = _get_nested(record, "caller", "segment")
            if filters.segments and segment not in filters.segments:
                continue

            tags = _get_nested(record, "nlp", "tags", default=[])
            if filters.tags and not set(tags).intersection(filters.tags):
                continue

            product = _get_nested(record, "case", "product")
            if filters.products and product not in filters.products:
                continue

            queue = _get_nested(record, "queue")
            if filters.queues and queue not in filters.queues:
                continue

            channel = _get_nested(record, "channel")
            if filters.channels and channel not in filters.channels:
                continue

            sentiment_label = _get_nested(record, "nlp", "sentiment_label")
            if filters.sentiment_labels and sentiment_label not in filters.sentiment_labels:
                continue

            sentiment_score = _get_nested(record, "nlp", "sentiment_score")
            if filters.min_sentiment is not None and sentiment_score is not None:
                if sentiment_score < filters.min_sentiment:
                    continue
            if filters.max_sentiment is not None and sentiment_score is not None:
                if sentiment_score > filters.max_sentiment:
                    continue

            escalated = _get_nested(record, "interaction", "escalated", default=False)
            if filters.escalation_only and not escalated:
                continue

            if not self.include_transcripts:
                record = record.model_copy(
                    update={"nlp": record.nlp.model_copy(update={"transcript": ""})}
                )

            filtered.append(record)
            if len(filtered) >= self.sample_limit:
                break

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
        avg_sentiment = _safe_mean(_get_nested(record, "nlp", "sentiment_score") for record in filtered)
        avg_wait = _safe_mean(_get_nested(record, "interaction", "wait_time_sec") for record in filtered)
        escalation_rate = (
            sum(1 for record in filtered if _get_nested(record, "interaction", "escalated")) / len(filtered)
            if filtered
            else 0
        )

        summary = LoadSummary(
            total_records=len(filtered),
            date_range=DateRange(start=self.date_from, end=self.date_to),
            segment_breakdown=[GroupCount(group=key, count=value) for key, value in segments.most_common()],
            top_tags=[GroupCount(group=key, count=value) for key, value in tags.most_common(8)],
            sentiment_distribution=[
                GroupCount(group=key, count=value) for key, value in sentiments.most_common()
            ],
            avg_sentiment_score=avg_sentiment,
            avg_wait_time_sec=avg_wait,
            escalation_rate=round(escalation_rate, 3) if filtered else 0,
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

        overall_sentiments = [
            _get_nested(record, "nlp", "sentiment_score")
            for record in records
            if _get_nested(record, "nlp", "sentiment_score") is not None
        ]
        overall = {
            "call_count": len(records),
            "avg_duration_sec": _safe_mean(_get_nested(record, "duration_sec") for record in records),
            "avg_wait_sec": _safe_mean(_get_nested(record, "interaction", "wait_time_sec") for record in records),
            "avg_sentiment_score": _safe_mean(overall_sentiments),
            "escalation_rate": round(
                sum(1 for record in records if _get_nested(record, "interaction", "escalated")) / len(records), 3
            )
            if records
            else 0,
            "fcr_rate": round(
                sum(1 for record in records if _get_nested(record, "interaction", "first_call_resolution")) / len(records),
                3,
            )
            if records
            else 0,
            "avg_nps": _safe_mean(_get_nested(record, "caller", "nps_last") for record in records),
        }

        groups: dict[str, dict] = defaultdict(lambda: {"records": []})

        for record in records:
            if self.group_by == "tag":
                group_keys = _get_nested(record, "nlp", "tags", default=[]) or ["(untagged)"]
            elif self.group_by == "agent":
                group_keys = [_get_nested(record, "agent", "agent_id", default="unknown")]
            elif self.group_by == "segment":
                group_keys = [_get_nested(record, "caller", "segment", default="unknown")]
            elif self.group_by == "product":
                group_keys = [_get_nested(record, "case", "product", default="unknown")]
            elif self.group_by == "queue":
                group_keys = [record.get("queue", "unknown")]
            elif self.group_by == "channel":
                group_keys = [record.get("channel", "unknown")]
            else:
                group_keys = ["unknown"]

            for key in group_keys:
                groups[key]["records"].append(record)

        group_metrics: list[GroupKpi] = []
        for key, data in groups.items():
            group_records = data["records"]
            if len(group_records) < self.min_calls:
                continue
            sentiments = [
                _get_nested(record, "nlp", "sentiment_score")
                for record in group_records
                if _get_nested(record, "nlp", "sentiment_score") is not None
            ]
            esc_rate = sum(1 for record in group_records if _get_nested(record, "interaction", "escalated")) / len(
                group_records
            )
            fcr_rate = sum(
                1 for record in group_records if _get_nested(record, "interaction", "first_call_resolution")
            ) / len(group_records)
            group_metrics.append(
                GroupKpi(
                    group=key,
                    call_count=len(group_records),
                    avg_duration_sec=_safe_mean(_get_nested(record, "duration_sec") for record in group_records),
                    avg_wait_sec=_safe_mean(
                        _get_nested(record, "interaction", "wait_time_sec") for record in group_records
                    ),
                    avg_sentiment_score=_safe_mean(sentiments),
                    escalation_rate=round(esc_rate, 3),
                    fcr_rate=round(fcr_rate, 3),
                    avg_nps=_safe_mean(_get_nested(record, "caller", "nps_last") for record in group_records),
                )
            )

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
        if overall["avg_sentiment_score"] is not None and overall["avg_sentiment_score"] < -0.15:
            insights.append("Overall sentiment skews negative; prioritize root-cause fixes.")
        if overall["escalation_rate"] > 0.25:
            insights.append("Escalation rate is elevated; review escalation triggers and routing.")
        if overall["avg_wait_sec"] and overall["avg_wait_sec"] > 90:
            insights.append("Average wait time is high; staffing or IVR deflection may be needed.")
        if sorted_groups:
            worst = sorted_groups[0]
            insights.append(f"Most challenged group by {self.focus_metric}: {worst.group}.")

        summary = AggregateSummary(
            overall_kpis=KpiSummary(**overall),
            group_by=self.group_by,
            focus_metric=self.focus_metric,
            top_groups=sorted_groups[: self.top_n],
            insights=insights,
        )
        return json.dumps(summary.model_dump(), indent=2)
