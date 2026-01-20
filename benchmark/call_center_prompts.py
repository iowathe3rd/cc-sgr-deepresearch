import json


def CALL_CENTER_GRADER_TEMPLATE(task: str, expected: dict, predicted: str) -> str:
    expected_json = json.dumps(expected, indent=2)
    return f"""
You are grading a call center analytics answer.
Check whether the predicted answer reports the required metrics with reasonable numeric accuracy.

Rules:
- If a metric is missing or wildly incorrect, mark INCORRECT.
- Use tolerance: rates within ±0.02, averages within ±5.0 units, sentiment within ±0.05.
- If most metrics are correct but one is missing, mark PARTIAL.
- Only use the expected metrics below as ground truth.
- Return a JSON object with keys: reasoning, expected_metrics, predicted_excerpt, grade_answer.

Task:
{task}

Expected metrics (JSON):
{expected_json}

Predicted answer:
{predicted}
""".strip()
