from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION: str = "call_center.v2"

SentimentLabel = Literal["positive", "neutral", "negative"]
ChurnRisk = Literal["low", "medium", "high"]
ResolutionStatus = Literal["resolved", "pending", "failed"]
KycStatus = Literal["verified", "partial", "failed"]
CaseStatus = Literal["open", "pending", "closed", "reopened"]
CallPriority = Literal["low", "medium", "high", "vip"]
ShiftName = Literal["morning", "afternoon", "evening", "night"]
AbandonStage = Literal["ivr", "queue", "agent", "unknown"]


class CallCenterFilters(BaseModel):
    segments: list[str] | None = None
    tags: list[str] | None = None
    products: list[str] | None = None
    queues: list[str] | None = None
    channels: list[str] | None = None
    languages: list[str] | None = None
    regions: list[str] | None = None
    branches: list[str] | None = None
    issue_categories: list[str] | None = None
    issue_subcategories: list[str] | None = None
    churn_risks: list[str] | None = None
    customer_tiers: list[str] | None = None
    sentiment_labels: list[str] | None = None
    escalation_only: bool = False
    abandoned_only: bool = False
    min_sentiment: float | None = None
    max_sentiment: float | None = None
    min_wait_sec: int | None = None
    max_wait_sec: int | None = None
    min_duration_sec: int | None = None
    max_duration_sec: int | None = None
    min_quality_score: int | None = None
    min_script_adherence: float | None = None


class AgentProfile(BaseModel):
    agent_id: str
    name: str
    team: str
    tenure_months: int = Field(ge=0)
    location: str
    quality_score: int = Field(ge=0, le=100)
    tier: str = Field(default="L1")
    manager_id: str | None = None
    languages: list[str] = Field(default_factory=list)


class CallerProfile(BaseModel):
    customer_id: str
    segment: str
    tenure_years: float = Field(ge=0)
    age_band: str
    products: list[str] = Field(min_length=1)
    risk_score: float = Field(ge=0, le=1)
    churn_risk: ChurnRisk
    nps_last: int | None = Field(default=None, ge=0, le=10)
    delinquency_days: int = Field(ge=0)
    digital_usage_score: float = Field(ge=0, le=1)
    avg_monthly_balance_usd: float = Field(ge=0)
    income_band: str
    lifetime_value_usd: float = Field(ge=0)
    primary_language: str
    customer_tier: str
    complaints_90d: int = Field(ge=0)
    recent_calls_30d: int = Field(ge=0)
    kyc_status: KycStatus


class InteractionMeta(BaseModel):
    wait_time_sec: int | None = Field(default=None, ge=0)
    hold_time_sec: int | None = Field(default=None, ge=0)
    ivr_time_sec: int | None = Field(default=None, ge=0)
    transfer_count: int = Field(ge=0)
    after_call_work_sec: int | None = Field(default=None, ge=0)
    first_call_resolution: bool
    escalated: bool
    escalation_reason: str | None = None
    abandoned: bool = False
    abandon_stage: AbandonStage | None = None
    callback_requested: bool = False
    callback_completed: bool = False
    auth_method: str
    auth_passed: bool


class CaseInfo(BaseModel):
    case_id: str
    product: str
    issue_category: str
    issue_subcategory: str
    resolution: ResolutionStatus
    resolution_time_days: int | None = Field(default=None, ge=0)
    case_status: CaseStatus
    reopen_count: int = Field(ge=0)
    sla_breach: bool = False
    root_cause: str
    resolution_code: str
    follow_up_required: bool = False


class NlpSignals(BaseModel):
    sentiment_score: float = Field(ge=-1, le=1)
    sentiment_label: SentimentLabel
    emotion_peaks: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    summary: str
    call_intent: str
    resolution_summary: str
    transcript: str
    language_mix: list[str] = Field(default_factory=list)
    toxicity_score: float = Field(ge=0, le=1)
    compliance_flags: list[str] = Field(default_factory=list)
    key_phrases: list[str] = Field(default_factory=list)
    named_entities: list[str] = Field(default_factory=list)


class QualitySignals(BaseModel):
    silence_ratio: float = Field(ge=0, le=1)
    overlap_ratio: float = Field(ge=0, le=1)
    speech_rate_wpm: int = Field(ge=0)
    script_adherence: float = Field(ge=0, le=1)
    empathy_score: float = Field(ge=0, le=1)
    resolution_confidence: float = Field(ge=0, le=1)
    csat_pred: float = Field(ge=0, le=10)
    compliance_score: float = Field(ge=0, le=1)
    qa_flags: list[str] = Field(default_factory=list)


class FinancialImpact(BaseModel):
    fee_refund_usd: float = Field(ge=0)
    fraud_loss_usd: float = Field(ge=0)
    chargeback_amount_usd: float = Field(ge=0)
    retention_offer_usd: float = Field(ge=0)
    upsell_value_usd: float = Field(ge=0)
    revenue_impact_usd: float
    operational_cost_usd: float = Field(ge=0)


class CallCenterRecord(BaseModel):
    schema_version: ClassVar[str] = SCHEMA_VERSION

    call_id: str
    call_start: datetime
    call_end: datetime
    duration_sec: int = Field(ge=0)
    shift: ShiftName
    priority: CallPriority
    is_outbound: bool = False
    queue: str
    channel: str
    language: str
    region: str
    branch: str
    agent: AgentProfile
    caller: CallerProfile
    interaction: InteractionMeta
    case: CaseInfo
    nlp: NlpSignals
    quality: QualitySignals
    financials: FinancialImpact

    @field_validator("call_start", "call_end")
    @classmethod
    def ensure_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @model_validator(mode="after")
    def validate_timings(self):
        if self.call_end < self.call_start:
            raise ValueError("call_end must be after call_start")
        return self


class CallCenterDataset(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    source_system: str = Field(default="mock_call_center")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    records: list[CallCenterRecord]

    @field_validator("generated_at")
    @classmethod
    def ensure_dataset_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class DateRange(BaseModel):
    model_config = {"populate_by_name": True}

    start: str = Field(alias="from")
    end: str = Field(alias="to")


class GroupCount(BaseModel):
    group: str
    count: int


class SampleRecord(BaseModel):
    call_id: str
    queue: str
    segment: str
    issue: str
    sentiment: str
    language: str
    channel: str
    priority: str


class LoadSummary(BaseModel):
    total_records: int
    date_range: DateRange
    segment_breakdown: list[GroupCount]
    language_breakdown: list[GroupCount]
    channel_breakdown: list[GroupCount]
    queue_breakdown: list[GroupCount]
    top_tags: list[GroupCount]
    sentiment_distribution: list[GroupCount]
    avg_sentiment_score: float | None
    avg_wait_time_sec: float | None
    escalation_rate: float
    sample_records: list[SampleRecord]
    invalid_records: int = 0
    missing_transcripts: int = 0
    missing_wait_time: int = 0


class KpiSummary(BaseModel):
    call_count: int
    unique_customers: int
    repeat_caller_rate: float
    abandon_rate: float
    callback_rate: float
    avg_duration_sec: float | None
    avg_wait_sec: float | None
    avg_hold_sec: float | None
    avg_after_call_work_sec: float | None
    transfer_rate: float
    escalation_rate: float
    fcr_rate: float
    resolution_rate: float
    avg_resolution_days: float | None
    sla_20s_rate: float
    sla_60s_rate: float
    avg_sentiment_score: float | None
    negative_sentiment_rate: float
    complaint_rate: float
    fraud_risk_rate: float
    regulatory_risk_rate: float
    auth_fail_rate: float
    compliance_flag_rate: float
    avg_quality_score: float | None
    avg_script_adherence: float | None
    avg_empathy_score: float | None
    avg_speech_rate_wpm: float | None
    avg_nps: float | None
    total_refunds_usd: float
    total_fraud_loss_usd: float
    total_chargeback_usd: float
    total_revenue_impact_usd: float
    avg_revenue_impact_usd: float | None


class GroupKpi(BaseModel):
    group: str
    call_count: int
    unique_customers: int
    repeat_caller_rate: float
    abandon_rate: float
    callback_rate: float
    avg_duration_sec: float | None
    avg_wait_sec: float | None
    avg_hold_sec: float | None
    avg_after_call_work_sec: float | None
    transfer_rate: float
    escalation_rate: float
    fcr_rate: float
    resolution_rate: float
    avg_resolution_days: float | None
    sla_20s_rate: float
    sla_60s_rate: float
    avg_sentiment_score: float | None
    negative_sentiment_rate: float
    complaint_rate: float
    fraud_risk_rate: float
    regulatory_risk_rate: float
    auth_fail_rate: float
    compliance_flag_rate: float
    avg_quality_score: float | None
    avg_script_adherence: float | None
    avg_empathy_score: float | None
    avg_speech_rate_wpm: float | None
    avg_nps: float | None
    total_refunds_usd: float
    total_fraud_loss_usd: float
    total_chargeback_usd: float
    total_revenue_impact_usd: float
    avg_revenue_impact_usd: float | None


class AggregateSummary(BaseModel):
    overall_kpis: KpiSummary
    group_by: str
    focus_metric: str
    top_groups: list[GroupKpi]
    insights: list[str]


class TrendDelta(BaseModel):
    value: float | None
    pct: float | None


class TrendPoint(BaseModel):
    period: str
    kpis: KpiSummary
    deltas: dict[str, TrendDelta]


class TrendSummary(BaseModel):
    period: str
    points: list[TrendPoint]
