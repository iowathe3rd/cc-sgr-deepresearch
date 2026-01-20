from typing import Literal

from openai import BadRequestError, OpenAI
from pydantic import BaseModel, Field

from benchmark.call_center_prompts import CALL_CENTER_GRADER_TEMPLATE


class CallCenterGradeModel(BaseModel):
    reasoning: str = Field(..., description="Brief rationale for the grade")
    expected_metrics: dict = Field(..., description="Expected metrics used for grading")
    predicted_excerpt: str = Field(..., description="Extracted numeric evidence from the answer")
    grade_answer: Literal["CORRECT", "INCORRECT", "PARTIAL"] = Field(..., description="Grade of the answer")


def grade_call_center_answer(
    predicted_answer: str,
    task: str,
    expected: dict,
    model_config: dict,
) -> CallCenterGradeModel:
    client = OpenAI(base_url=model_config["base_url"], api_key=model_config["api_key"])

    try:
        completion = client.beta.chat.completions.parse(
            model=model_config["model"],
            messages=[
                {
                    "role": "user",
                    "content": CALL_CENTER_GRADER_TEMPLATE(task, expected, predicted_answer),
                }
            ],
            response_format=CallCenterGradeModel,
        )
        return completion.choices[0].message.parsed
    except BadRequestError as exc:
        raise RuntimeError(
            "Judge model does not support structured outputs. "
            "Set JUDGE_MODEL_NAME to a model that supports response_format parsing."
        ) from exc
