from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION: str = "call_center.v1"

SentimentLabel = Literal["positive", "neutral", "negative"]
ChurnRisk = Literal["low", "medium", "high"]
ResolutionStatus = Literal["resolved", "pending", "failed"]


class AgentProfile(BaseModel):
    agent_id: str
    name: str
    team: str
    tenure_months: int = Field(ge=0)
    location: str
    quality_score: int = Field(ge=0, le=100)


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


class InteractionMeta(BaseModel):
    wait_time_sec: int = Field(ge=0)
    hold_time_sec: int = Field(ge=0)
    transfer_count: int = Field(ge=0)
    after_call_work_sec: int = Field(ge=0)
    first_call_resolution: bool
    escalated: bool
    escalation_reason: str | None = None
    auth_method: str
    auth_passed: bool


class CaseInfo(BaseModel):
    product: str
    issue_category: str
    issue_subcategory: str
    resolution: ResolutionStatus
    resolution_time_days: int | None = Field(default=None, ge=0)


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


class QualitySignals(BaseModel):
    silence_ratio: float = Field(ge=0, le=1)
    overlap_ratio: float = Field(ge=0, le=1)
    speech_rate_wpm: int = Field(ge=0)
    script_adherence: float = Field(ge=0, le=1)


class CallCenterRecord(BaseModel):
    schema_version: ClassVar[str] = SCHEMA_VERSION

    call_id: str
    call_start: datetime
    call_end: datetime
    duration_sec: int = Field(ge=0)
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


class LoadSummary(BaseModel):
    total_records: int
    date_range: DateRange
    segment_breakdown: list[GroupCount]
    top_tags: list[GroupCount]
    sentiment_distribution: list[GroupCount]
    avg_sentiment_score: float | None
    avg_wait_time_sec: float | None
    escalation_rate: float
    sample_records: list[SampleRecord]
    invalid_records: int = 0


class KpiSummary(BaseModel):
    call_count: int
    avg_duration_sec: float | None
    avg_wait_sec: float | None
    avg_sentiment_score: float | None
    escalation_rate: float
    fcr_rate: float
    avg_nps: float | None


class GroupKpi(BaseModel):
    group: str
    call_count: int
    avg_duration_sec: float | None
    avg_wait_sec: float | None
    avg_sentiment_score: float | None
    escalation_rate: float
    fcr_rate: float
    avg_nps: float | None


class AggregateSummary(BaseModel):
    overall_kpis: KpiSummary
    group_by: str
    focus_metric: str
    top_groups: list[GroupKpi]
    insights: list[str]
