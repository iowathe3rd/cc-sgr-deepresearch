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


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 3)


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
        if filters.languages and record.language not in filters.languages:
            continue
        if filters.regions and record.region not in filters.regions:
            continue
        if filters.branches and record.branch not in filters.branches:
            continue
        if filters.issue_categories and record.case.issue_category not in filters.issue_categories:
            continue
        if filters.issue_subcategories and record.case.issue_subcategory not in filters.issue_subcategories:
            continue
        if filters.churn_risks and record.caller.churn_risk not in filters.churn_risks:
            continue
        if filters.customer_tiers and record.caller.customer_tier not in filters.customer_tiers:
            continue
        if filters.sentiment_labels and record.nlp.sentiment_label not in filters.sentiment_labels:
            continue

        if filters.min_sentiment is not None and record.nlp.sentiment_score < filters.min_sentiment:
            continue
        if filters.max_sentiment is not None and record.nlp.sentiment_score > filters.max_sentiment:
            continue

        if filters.escalation_only and not record.interaction.escalated:
            continue
        if filters.abandoned_only and not record.interaction.abandoned:
            continue
        if filters.min_wait_sec is not None:
            if record.interaction.wait_time_sec is None or record.interaction.wait_time_sec < filters.min_wait_sec:
                continue
        if filters.max_wait_sec is not None:
            if record.interaction.wait_time_sec is None or record.interaction.wait_time_sec > filters.max_wait_sec:
                continue
        if filters.min_duration_sec is not None and record.duration_sec < filters.min_duration_sec:
            continue
        if filters.max_duration_sec is not None and record.duration_sec > filters.max_duration_sec:
            continue
        if filters.min_quality_score is not None and record.agent.quality_score < filters.min_quality_score:
            continue
        if filters.min_script_adherence is not None and record.quality.script_adherence < filters.min_script_adherence:
            continue

        filtered.append(record)

    return filtered


def compute_kpis(records: list[CallCenterRecord]) -> KpiSummary:
    call_count = len(records)
    if call_count == 0:
        return KpiSummary(
            call_count=0,
            unique_customers=0,
            repeat_caller_rate=0.0,
            abandon_rate=0.0,
            callback_rate=0.0,
            avg_duration_sec=None,
            avg_wait_sec=None,
            avg_hold_sec=None,
            avg_after_call_work_sec=None,
            transfer_rate=0.0,
            escalation_rate=0.0,
            fcr_rate=0.0,
            resolution_rate=0.0,
            avg_resolution_days=None,
            sla_20s_rate=0.0,
            sla_60s_rate=0.0,
            avg_sentiment_score=None,
            negative_sentiment_rate=0.0,
            complaint_rate=0.0,
            fraud_risk_rate=0.0,
            regulatory_risk_rate=0.0,
            auth_fail_rate=0.0,
            compliance_flag_rate=0.0,
            avg_quality_score=None,
            avg_script_adherence=None,
            avg_empathy_score=None,
            avg_speech_rate_wpm=None,
            avg_nps=None,
            total_refunds_usd=0.0,
            total_fraud_loss_usd=0.0,
            total_chargeback_usd=0.0,
            total_revenue_impact_usd=0.0,
            avg_revenue_impact_usd=None,
        )

    unique_customers = {record.caller.customer_id for record in records}
    repeat_flags = [
        record.caller.recent_calls_30d > 1 or "repeat_caller" in record.nlp.tags for record in records
    ]
    abandoned = [record.interaction.abandoned for record in records]
    callbacks = [record.interaction.callback_requested for record in records]
    transfers = [record.interaction.transfer_count > 0 for record in records]
    escalations = [record.interaction.escalated for record in records]
    fcrs = [record.interaction.first_call_resolution for record in records]
    resolutions = [record.case.resolution == "resolved" for record in records]
    negative_sentiment = [record.nlp.sentiment_label == "negative" for record in records]
    complaint_tags = ["complaint" in record.nlp.tags for record in records]
    fraud_tags = ["fraud_risk" in record.nlp.tags for record in records]
    regulatory_tags = ["regulatory_risk" in record.nlp.tags for record in records]
    auth_fails = [not record.interaction.auth_passed for record in records]
    compliance_flags = [bool(record.nlp.compliance_flags) for record in records]
    wait_times = [record.interaction.wait_time_sec for record in records if record.interaction.wait_time_sec is not None]

    avg_duration = _safe_mean(record.duration_sec for record in records)
    avg_wait = _safe_mean(record.interaction.wait_time_sec for record in records)
    avg_hold = _safe_mean(record.interaction.hold_time_sec for record in records)
    avg_after_call_work = _safe_mean(record.interaction.after_call_work_sec for record in records)
    avg_sentiment = _safe_mean(record.nlp.sentiment_score for record in records)
    escalation_rate = _rate(sum(escalations), call_count)
    fcr_rate = _rate(sum(fcrs), call_count)
    resolution_rate = _rate(sum(resolutions), call_count)
    avg_resolution_days = _safe_mean(record.case.resolution_time_days for record in records)
    sla_20s_rate = _rate(sum(1 for value in wait_times if value is not None and value <= 20), len(wait_times))
    sla_60s_rate = _rate(sum(1 for value in wait_times if value is not None and value <= 60), len(wait_times))
    negative_sentiment_rate = _rate(sum(negative_sentiment), call_count)
    complaint_rate = _rate(sum(complaint_tags), call_count)
    fraud_risk_rate = _rate(sum(fraud_tags), call_count)
    regulatory_risk_rate = _rate(sum(regulatory_tags), call_count)
    auth_fail_rate = _rate(sum(auth_fails), call_count)
    compliance_flag_rate = _rate(sum(compliance_flags), call_count)
    avg_quality_score = _safe_mean(record.agent.quality_score for record in records)
    avg_script_adherence = _safe_mean(record.quality.script_adherence for record in records)
    avg_empathy_score = _safe_mean(record.quality.empathy_score for record in records)
    avg_speech_rate = _safe_mean(record.quality.speech_rate_wpm for record in records)
    avg_nps = _safe_mean(record.caller.nps_last for record in records)
    total_refunds = sum(record.financials.fee_refund_usd for record in records)
    total_fraud_loss = sum(record.financials.fraud_loss_usd for record in records)
    total_chargeback = sum(record.financials.chargeback_amount_usd for record in records)
    total_revenue_impact = sum(record.financials.revenue_impact_usd for record in records)
    avg_revenue_impact = _safe_mean(record.financials.revenue_impact_usd for record in records)

    return KpiSummary(
        call_count=call_count,
        unique_customers=len(unique_customers),
        repeat_caller_rate=_rate(sum(repeat_flags), call_count),
        abandon_rate=_rate(sum(abandoned), call_count),
        callback_rate=_rate(sum(callbacks), call_count),
        avg_duration_sec=avg_duration,
        avg_wait_sec=avg_wait,
        avg_hold_sec=avg_hold,
        avg_after_call_work_sec=avg_after_call_work,
        transfer_rate=_rate(sum(transfers), call_count),
        avg_sentiment_score=avg_sentiment,
        escalation_rate=escalation_rate,
        fcr_rate=fcr_rate,
        resolution_rate=resolution_rate,
        avg_resolution_days=avg_resolution_days,
        sla_20s_rate=sla_20s_rate,
        sla_60s_rate=sla_60s_rate,
        negative_sentiment_rate=negative_sentiment_rate,
        complaint_rate=complaint_rate,
        fraud_risk_rate=fraud_risk_rate,
        regulatory_risk_rate=regulatory_risk_rate,
        auth_fail_rate=auth_fail_rate,
        compliance_flag_rate=compliance_flag_rate,
        avg_quality_score=avg_quality_score,
        avg_script_adherence=avg_script_adherence,
        avg_empathy_score=avg_empathy_score,
        avg_speech_rate_wpm=avg_speech_rate,
        avg_nps=avg_nps,
        total_refunds_usd=round(total_refunds, 2),
        total_fraud_loss_usd=round(total_fraud_loss, 2),
        total_chargeback_usd=round(total_chargeback, 2),
        total_revenue_impact_usd=round(total_revenue_impact, 2),
        avg_revenue_impact_usd=avg_revenue_impact,
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
    if group_by == "issue_category":
        return [record.case.issue_category]
    if group_by == "issue_subcategory":
        return [record.case.issue_subcategory]
    if group_by == "queue":
        return [record.queue]
    if group_by == "channel":
        return [record.channel]
    if group_by == "region":
        return [record.region]
    if group_by == "language":
        return [record.language]
    return ["unknown"]


def group_kpis(
    records: list[CallCenterRecord],
    group_by: Literal[
        "tag",
        "agent",
        "segment",
        "product",
        "issue_category",
        "issue_subcategory",
        "queue",
        "channel",
        "region",
        "language",
    ],
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
                unique_customers=kpis.unique_customers,
                repeat_caller_rate=kpis.repeat_caller_rate,
                abandon_rate=kpis.abandon_rate,
                callback_rate=kpis.callback_rate,
                avg_duration_sec=kpis.avg_duration_sec,
                avg_wait_sec=kpis.avg_wait_sec,
                avg_hold_sec=kpis.avg_hold_sec,
                avg_after_call_work_sec=kpis.avg_after_call_work_sec,
                transfer_rate=kpis.transfer_rate,
                avg_sentiment_score=kpis.avg_sentiment_score,
                escalation_rate=kpis.escalation_rate,
                fcr_rate=kpis.fcr_rate,
                resolution_rate=kpis.resolution_rate,
                avg_resolution_days=kpis.avg_resolution_days,
                sla_20s_rate=kpis.sla_20s_rate,
                sla_60s_rate=kpis.sla_60s_rate,
                negative_sentiment_rate=kpis.negative_sentiment_rate,
                complaint_rate=kpis.complaint_rate,
                fraud_risk_rate=kpis.fraud_risk_rate,
                regulatory_risk_rate=kpis.regulatory_risk_rate,
                auth_fail_rate=kpis.auth_fail_rate,
                compliance_flag_rate=kpis.compliance_flag_rate,
                avg_quality_score=kpis.avg_quality_score,
                avg_script_adherence=kpis.avg_script_adherence,
                avg_empathy_score=kpis.avg_empathy_score,
                avg_speech_rate_wpm=kpis.avg_speech_rate_wpm,
                avg_nps=kpis.avg_nps,
                total_refunds_usd=kpis.total_refunds_usd,
                total_fraud_loss_usd=kpis.total_fraud_loss_usd,
                total_chargeback_usd=kpis.total_chargeback_usd,
                total_revenue_impact_usd=kpis.total_revenue_impact_usd,
                avg_revenue_impact_usd=kpis.avg_revenue_impact_usd,
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
            deltas = _kpi_delta_map(current_kpis, previous)
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


def _kpi_delta_map(current: KpiSummary, previous: KpiSummary) -> dict[str, TrendDelta]:
    deltas: dict[str, TrendDelta] = {}
    current_values = current.model_dump()
    for key, value in current_values.items():
        prev_value = getattr(previous, key, None)
        if isinstance(value, (int, float)) or value is None:
            deltas[key] = _delta(value, prev_value)
    return deltas
