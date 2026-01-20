import json
from typing import Literal

from pydantic import BaseModel, Field

from sgr_agent_core.agents.sgr_tool_calling_agent import SGRToolCallingAgent
from sgr_agent_core.base_tool import BaseTool
from sgr_agent_core.models import AgentStatesEnum

CALL_CENTER_JUDGE_SYSTEM_PROMPT = """You are a grading agent for call center analytics answers.
Grade only the metrics explicitly requested in the task.
Use the expected metrics JSON as the ground truth.

Grading rules:
- Rates within +/- 0.02 are correct.
- Averages within +/- 5.0 units are correct.
- Sentiment within +/- 0.05 is correct.
- If any requested metric is missing or wildly incorrect, mark INCORRECT.
- If most requested metrics are correct but one is missing, mark PARTIAL.
- If all requested metrics are correct, mark CORRECT.

Return your decision by calling the tool once. Do not answer in free-form text.

Available tools:
{available_tools}
"""


class CallCenterGradeResult(BaseModel):
    reasoning: str = Field(..., description="Brief rationale for the grade")
    expected_metrics: dict = Field(..., description="Expected metrics used for grading")
    predicted_excerpt: str = Field(..., description="Extracted numeric evidence from the answer")
    grade_answer: Literal["CORRECT", "INCORRECT", "PARTIAL"] = Field(
        ..., description="Grade of the answer"
    )


class CallCenterGradeTool(BaseTool):
    """Return the grade for a call center benchmark case."""

    tool_name = "call_center_grade"

    reasoning: str = Field(description="Brief rationale for the grade")
    expected_metrics: dict = Field(description="Expected metrics used for grading")
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
