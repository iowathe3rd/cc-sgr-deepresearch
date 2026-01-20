import json
from typing import Literal

from pydantic import BaseModel, Field
from sgr_agent_core.agents.sgr_tool_calling_agent import SGRToolCallingAgent
from sgr_agent_core.base_tool import BaseTool
from sgr_agent_core.models import AgentStatesEnum


class CallCenterExpectedMetrics(BaseModel):
    model_config = {"extra": "forbid"}

    call_count: int
    unique_customers: int
    repeat_caller_rate: float
    abandon_rate: float
    callback_rate: float
    avg_duration_sec: float | None = None
    avg_wait_sec: float | None = None
    avg_hold_sec: float | None = None
    avg_after_call_work_sec: float | None = None
    transfer_rate: float
    avg_sentiment_score: float | None = None
    escalation_rate: float
    fcr_rate: float
    resolution_rate: float
    avg_resolution_days: float | None = None
    sla_20s_rate: float
    sla_60s_rate: float
    negative_sentiment_rate: float
    complaint_rate: float
    fraud_risk_rate: float
    regulatory_risk_rate: float
    auth_fail_rate: float
    compliance_flag_rate: float
    avg_quality_score: float | None = None
    avg_script_adherence: float | None = None
    avg_empathy_score: float | None = None
    avg_speech_rate_wpm: float | None = None
    avg_nps: float | None = None
    total_refunds_usd: float
    total_fraud_loss_usd: float
    total_chargeback_usd: float
    total_revenue_impact_usd: float
    avg_revenue_impact_usd: float | None = None

CALL_CENTER_JUDGE_SYSTEM_PROMPT = """You are a grading agent for call center analytics answers.
Grade only the metrics explicitly requested in the task.
Use the expected metrics JSON as the ground truth.

Grading rules:
- Counts must be exact.
- Rates within +/- 0.02 are correct.
- Averages within +/- 5.0 units are correct.
- Sentiment within +/- 0.05 is correct.
- Totals (financial impact/refunds/losses) within +/- 5% are correct.
- If any requested metric is missing or wildly incorrect, mark INCORRECT.
- If most requested metrics are correct but one is missing, mark PARTIAL.
- If all requested metrics are correct, mark CORRECT.

Return your decision by calling the tool once. Do not answer in free-form text.

Available tools:
{available_tools}
"""


class CallCenterGradeResult(BaseModel):
    reasoning: str = Field(..., description="Brief rationale for the grade")
    expected_metrics: CallCenterExpectedMetrics = Field(..., description="Expected metrics used for grading")
    predicted_excerpt: str = Field(..., description="Extracted numeric evidence from the answer")
    grade_answer: Literal["CORRECT", "INCORRECT", "PARTIAL"] = Field(
        ..., description="Grade of the answer"
    )


class CallCenterGradeTool(BaseTool):
    """Return the grade for a call center benchmark case."""

    tool_name = "call_center_grade"

    reasoning: str = Field(description="Brief rationale for the grade")
    expected_metrics: CallCenterExpectedMetrics = Field(description="Expected metrics used for grading")
    predicted_excerpt: str = Field(description="Extracted numeric evidence from the answer")
    grade_answer: Literal["CORRECT", "INCORRECT", "PARTIAL"] = Field(description="Grade of the answer")

    async def __call__(self, context, config, **_) -> str:
        payload = self.model_dump()
        payload_json = json.dumps(payload, ensure_ascii=False)
        context.state = AgentStatesEnum.COMPLETED
        context.execution_result = payload_json
        return payload_json


class CallCenterJudgeAgent(SGRToolCallingAgent):
    name = "call_center_judge_agent"
