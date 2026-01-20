from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean
from typing import Iterable, Literal

from examples.call_center_mock.contracts import (
    CallCenterFilters,
    CallCenterRecord,
    GroupKpi,
    KpiSummary,
    TrendDelta,
    TrendPoint,
    TrendSummary,
)


def _safe_mean(values: Iterable[float | int | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    return mean(cleaned) if cleaned else None


def _parse_date(value: str) -> datetime:
    cleaned = value.replace("Z", "+00:00")
    if len(cleaned) == 10:
        cleaned = f"{cleaned}T00:00:00"
    parsed = datetime.fromisoformat(cleaned)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def filter_records(
    records: list[CallCenterRecord],
    date_from: str,
    date_to: str,
    filters: CallCenterFilters | None = None,
) -> list[CallCenterRecord]:
    filters = filters or CallCenterFilters()
    start_dt = _parse_date(date_from)
    end_dt = _parse_date(date_to)
    filtered: list[CallCenterRecord] = []

    for record in records:
        call_start = record.call_start
        if call_start < start_dt or call_start > end_dt:
            continue

        if filters.segments and record.caller.segment not in filters.segments:
            continue
        if filters.tags and not set(record.nlp.tags).intersection(filters.tags):
            continue
        if filters.products and record.case.product not in filters.products:
            continue
        if filters.queues and record.queue not in filters.queues:
            continue
        if filters.channels and record.channel not in filters.channels:
            continue
        if filters.sentiment_labels and record.nlp.sentiment_label not in filters.sentiment_labels:
            continue

        if filters.min_sentiment is not None and record.nlp.sentiment_score < filters.min_sentiment:
            continue
        if filters.max_sentiment is not None and record.nlp.sentiment_score > filters.max_sentiment:
            continue

        if filters.escalation_only and not record.interaction.escalated:
            continue

        filtered.append(record)

    return filtered


def compute_kpis(records: list[CallCenterRecord]) -> KpiSummary:
    call_count = len(records)
    if call_count == 0:
        return KpiSummary(
            call_count=0,
            avg_duration_sec=None,
            avg_wait_sec=None,
            avg_sentiment_score=None,
            escalation_rate=0.0,
            fcr_rate=0.0,
            avg_nps=None,
        )

    avg_duration = _safe_mean(record.duration_sec for record in records)
    avg_wait = _safe_mean(record.interaction.wait_time_sec for record in records)
    avg_sentiment = _safe_mean(record.nlp.sentiment_score for record in records)
    escalation_rate = sum(1 for record in records if record.interaction.escalated) / call_count
    fcr_rate = sum(1 for record in records if record.interaction.first_call_resolution) / call_count
    avg_nps = _safe_mean(record.caller.nps_last for record in records)

    return KpiSummary(
        call_count=call_count,
        avg_duration_sec=avg_duration,
        avg_wait_sec=avg_wait,
        avg_sentiment_score=avg_sentiment,
        escalation_rate=round(escalation_rate, 3),
        fcr_rate=round(fcr_rate, 3),
        avg_nps=avg_nps,
    )


def _group_keys(record: CallCenterRecord, group_by: str) -> list[str]:
    if group_by == "tag":
        return record.nlp.tags or ["(untagged)"]
    if group_by == "agent":
        return [record.agent.agent_id]
    if group_by == "segment":
        return [record.caller.segment]
    if group_by == "product":
        return [record.case.product]
    if group_by == "queue":
        return [record.queue]
    if group_by == "channel":
        return [record.channel]
    return ["unknown"]


def group_kpis(
    records: list[CallCenterRecord],
    group_by: Literal["tag", "agent", "segment", "product", "queue", "channel"],
    min_calls: int = 2,
) -> list[GroupKpi]:
    groups: dict[str, list[CallCenterRecord]] = defaultdict(list)
    for record in records:
        for key in _group_keys(record, group_by):
            groups[key].append(record)

    group_metrics: list[GroupKpi] = []
    for key, group_records in groups.items():
        if len(group_records) < min_calls:
            continue
        kpis = compute_kpis(group_records)
        group_metrics.append(
            GroupKpi(
                group=key,
                call_count=kpis.call_count,
                avg_duration_sec=kpis.avg_duration_sec,
                avg_wait_sec=kpis.avg_wait_sec,
                avg_sentiment_score=kpis.avg_sentiment_score,
                escalation_rate=kpis.escalation_rate,
                fcr_rate=kpis.fcr_rate,
                avg_nps=kpis.avg_nps,
            )
        )

    return group_metrics


def trend_kpis(
    records: list[CallCenterRecord],
    period: Literal["month"] = "month",
) -> TrendSummary:
    grouped: dict[str, list[CallCenterRecord]] = defaultdict(list)
    for record in records:
        if period == "month":
            key = record.call_start.strftime("%Y-%m")
        else:
            key = record.call_start.strftime("%Y-%m")
        grouped[key].append(record)

    points: list[TrendPoint] = []
    previous: KpiSummary | None = None
    for period_key in sorted(grouped.keys()):
        current_kpis = compute_kpis(grouped[period_key])
        deltas: dict[str, TrendDelta] = {}
        if previous:
            deltas = {
                "call_count": _delta(current_kpis.call_count, previous.call_count),
                "avg_duration_sec": _delta(current_kpis.avg_duration_sec, previous.avg_duration_sec),
                "avg_wait_sec": _delta(current_kpis.avg_wait_sec, previous.avg_wait_sec),
                "avg_sentiment_score": _delta(current_kpis.avg_sentiment_score, previous.avg_sentiment_score),
                "escalation_rate": _delta(current_kpis.escalation_rate, previous.escalation_rate),
                "fcr_rate": _delta(current_kpis.fcr_rate, previous.fcr_rate),
                "avg_nps": _delta(current_kpis.avg_nps, previous.avg_nps),
            }
        points.append(TrendPoint(period=period_key, kpis=current_kpis, deltas=deltas))
        previous = current_kpis

    return TrendSummary(period=period, points=points)


def _delta(current: float | int | None, previous: float | int | None) -> TrendDelta:
    if current is None or previous is None:
        return TrendDelta(value=None, pct=None)
    value = float(current) - float(previous)
    if previous == 0:
        pct = None
    else:
        pct = value / float(previous)
    return TrendDelta(value=round(value, 4), pct=round(pct, 4) if pct is not None else None)
