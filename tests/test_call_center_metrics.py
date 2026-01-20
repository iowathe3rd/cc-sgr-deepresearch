from datetime import datetime, timedelta, timezone

import pytest

from examples.call_center_mock.contracts import (
    AgentProfile,
    CallCenterRecord,
    CallerProfile,
    CaseInfo,
    InteractionMeta,
    NlpSignals,
    QualitySignals,
)
from examples.call_center_mock.metrics import compute_kpis, group_kpis, trend_kpis


def _make_record(
    call_id: str,
    call_start: datetime,
    duration_sec: int,
    wait_time_sec: int,
    sentiment_score: float,
    escalated: bool,
    fcr: bool,
    nps_last: int | None,
    tags: list[str],
    segment: str = "mass_market",
    product: str = "checking",
    queue: str = "retail_support",
    channel: str = "voice",
) -> CallCenterRecord:
    call_end = call_start + timedelta(seconds=duration_sec)
    return CallCenterRecord(
        call_id=call_id,
        call_start=call_start,
        call_end=call_end,
        duration_sec=duration_sec,
        queue=queue,
        channel=channel,
        language="en",
        region="NE",
        branch="boston_central",
        agent=AgentProfile(
            agent_id="A100",
            name="Agent 100",
            team="retail",
            tenure_months=12,
            location="Boston",
            quality_score=85,
        ),
        caller=CallerProfile(
            customer_id="U1000",
            segment=segment,
            tenure_years=2.5,
            age_band="35-44",
            products=[product],
            risk_score=0.2,
            churn_risk="low",
            nps_last=nps_last,
            delinquency_days=0,
            digital_usage_score=0.6,
            avg_monthly_balance_usd=2500,
            income_band="40k-70k",
        ),
        interaction=InteractionMeta(
            wait_time_sec=wait_time_sec,
            hold_time_sec=10,
            transfer_count=0,
            after_call_work_sec=60,
            first_call_resolution=fcr,
            escalated=escalated,
            escalation_reason="policy_dispute" if escalated else None,
            auth_method="otp_sms",
            auth_passed=True,
        ),
        case=CaseInfo(
            product=product,
            issue_category="fee_dispute",
            issue_subcategory="overdraft_fee",
            resolution="resolved",
            resolution_time_days=1,
        ),
        nlp=NlpSignals(
            sentiment_score=sentiment_score,
            sentiment_label="positive" if sentiment_score >= 0.15 else "negative" if sentiment_score <= -0.15 else "neutral",
            emotion_peaks=["concern"],
            topics=["fees"],
            tags=tags,
            summary="Caller asked about fees.",
            call_intent="fee_inquiry",
            resolution_summary="Resolved with explanation.",
            transcript="Agent: Hello. Customer: I have a fee question.",
        ),
        quality=QualitySignals(
            silence_ratio=0.08,
            overlap_ratio=0.03,
            speech_rate_wpm=160,
            script_adherence=0.8,
        ),
    )


def test_compute_kpis_basic():
    start = datetime(2024, 8, 1, tzinfo=timezone.utc)
    records = [
        _make_record("C1", start, 100, 10, 0.2, False, True, 8, ["complaint"]),
        _make_record("C2", start + timedelta(hours=2), 200, 30, -0.2, True, False, 4, ["complaint"]),
    ]
    kpis = compute_kpis(records)
    assert kpis.call_count == 2
    assert kpis.avg_duration_sec == pytest.approx(150.0)
    assert kpis.avg_wait_sec == pytest.approx(20.0)
    assert kpis.avg_sentiment_score == pytest.approx(0.0)
    assert kpis.escalation_rate == pytest.approx(0.5)
    assert kpis.fcr_rate == pytest.approx(0.5)
    assert kpis.avg_nps == pytest.approx(6.0)


def test_group_kpis_by_tag_min_calls():
    start = datetime(2024, 8, 1, tzinfo=timezone.utc)
    records = [
        _make_record("C1", start, 100, 10, 0.1, False, True, 7, ["complaint", "fee_waiver"]),
        _make_record("C2", start + timedelta(hours=1), 200, 30, -0.2, True, False, 3, ["complaint"]),
    ]
    groups = group_kpis(records, group_by="tag", min_calls=2)
    assert len(groups) == 1
    assert groups[0].group == "complaint"
    assert groups[0].call_count == 2


def test_trend_kpis_monthly_delta():
    july = datetime(2024, 7, 15, tzinfo=timezone.utc)
    august = datetime(2024, 8, 15, tzinfo=timezone.utc)
    records = [
        _make_record("C1", july, 100, 20, -0.1, False, True, 6, ["complaint"]),
        _make_record("C2", august, 200, 30, -0.3, True, False, 4, ["complaint"]),
    ]
    trend = trend_kpis(records, period="month")
    assert len(trend.points) == 2
    assert trend.points[0].period == "2024-07"
    assert trend.points[1].period == "2024-08"
    assert trend.points[1].deltas["call_count"].value == pytest.approx(0.0)
