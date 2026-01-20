import argparse
import asyncio
import json
import logging
import os
import random
import uuid
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from sgr_agent_core.agent_factory import Agent

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv

from benchmark.call_center_utils import CallCenterGradeModel, grade_call_center_answer
from benchmark.utils import save_result
from examples.call_center_mock.agents import CallCenterDeepResearchAgent
from examples.call_center_mock.contracts import CallCenterFilters, CallCenterRecord
from examples.call_center_mock.definitions import CALL_CENTER_SYSTEM_PROMPT
from examples.call_center_mock.generate_mock_data import generate_records
from examples.call_center_mock.metrics import compute_kpis, filter_records
from examples.call_center_mock.tools import (
    CallCenterAggregateTool,
    CallCenterLoadDatasetTool,
    CallCenterTrendTool,
)
from sgr_agent_core import AgentDefinition, AgentFactory, PromptsConfig
from sgr_agent_core.tools import AdaptPlanTool, FinalAnswerTool, GeneratePlanTool

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_DATASET_PATH = project_root / "examples" / "call_center_mock" / "mock_data" / "call_center_records.json"


def _load_records(path: Path) -> list[CallCenterRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_records = payload.get("records") if isinstance(payload, dict) else payload
    return [CallCenterRecord.model_validate(item) for item in raw_records]


def _generate_dataset(seed: int, count: int, start: str, end: str, output_path: Path) -> None:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    dataset = {
        "schema_version": "call_center.v1",
        "source_system": "mock_call_center",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "records": generate_records(count, start_dt, end_dt, seed),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")


def _expected_metrics(records: list[CallCenterRecord], case: dict) -> dict:
    filters = CallCenterFilters(**case.get("filters", {}))
    filtered = filter_records(records, case["date_from"], case["date_to"], filters)
    kpis = compute_kpis(filtered)
    return kpis.model_dump()


async def _run_case(case: dict, agent_def: AgentDefinition, judge_config: dict, records: list[CallCenterRecord]) -> Dict[str, Any]:
    task_messages = [{"role": "user", "content": case["task"]}]
    agent: Agent = await AgentFactory.create(agent_def=agent_def, task_messages=task_messages)

    expected = _expected_metrics(records, case)

    try:
        await agent.execute()
        predicted = agent._context.execution_result or ""
        grade_report: CallCenterGradeModel = grade_call_center_answer(predicted, case["task"], expected, judge_config)
        grade = grade_report.grade_answer
    except Exception as exc:
        return {
            "case_id": case["id"],
            "task": case["task"],
            "expected": expected,
            "predicted": "",
            "grade_str": "ERROR",
            "error": str(exc),
        }

    return {
        "case_id": case["id"],
        "task": case["task"],
        "expected": expected,
        "predicted": predicted,
        "grade_str": grade,
        "grade_report": grade_report.model_dump(),
        "error": None,
    }


async def main(args):
    load_dotenv()

    agent_config = {
        "base_url": os.getenv("AGENT_BASE_URL"),
        "api_key": os.getenv("AGENT_API_KEY"),
        "model": os.getenv("AGENT_MODEL_NAME"),
    }
    judge_config = {
        "base_url": os.getenv("JUDGE_BASE_URL"),
        "api_key": os.getenv("JUDGE_API_KEY"),
        "model": os.getenv("JUDGE_MODEL_NAME"),
    }

    if not all(agent_config.values()):
        raise RuntimeError("AGENT_BASE_URL, AGENT_API_KEY, AGENT_MODEL_NAME must be set.")
    if not all(judge_config.values()):
        raise RuntimeError("JUDGE_BASE_URL, JUDGE_API_KEY, JUDGE_MODEL_NAME must be set.")

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"call_center_bench_{uuid.uuid4().hex}.xlsx"
    logger.info("Benchmark report will be written to %s", output_path)
    os.environ["CALL_CENTER_DATASET_PATH"] = str(dataset_path)
    if args.generate_dataset:
        logger.info("Generating deterministic dataset for benchmark.")
        _generate_dataset(args.seed, args.count, args.start, args.end, dataset_path)

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    records = _load_records(dataset_path)

    case_ids = [item.strip() for item in args.case_ids.split(",")] if args.case_ids else []
    if case_ids:
        case_map = {case["id"]: case for case in cases}
        cases = [case_map[case_id] for case_id in case_ids if case_id in case_map]

    if args.case_count is not None:
        if args.case_count <= 0:
            raise ValueError("case-count must be positive.")
        if args.case_sample == "random":
            rng = random.Random(args.case_seed)
            cases = rng.sample(cases, min(args.case_count, len(cases)))
        else:
            cases = cases[: args.case_count]

    agent_def = AgentDefinition(
        name="call_center_deep_research",
        base_class=CallCenterDeepResearchAgent,
        tools=[
            CallCenterLoadDatasetTool,
            CallCenterAggregateTool,
            CallCenterTrendTool,
            GeneratePlanTool,
            AdaptPlanTool,
            FinalAnswerTool,
        ],
        prompts=PromptsConfig(system_prompt_str=CALL_CENTER_SYSTEM_PROMPT),
        llm=agent_config,
        execution={"max_clarifications": 0, "max_iterations": 6},
    )

    results = []
    for case in cases:
        result = await _run_case(case, agent_def, judge_config, records)
        results.append(result)
        save_result(results, output_path)
        logger.info("Completed case %s with grade %s", case["id"], result.get("grade_str"))

    logger.info("Benchmark complete. Results saved to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Call Center Agent Benchmark")
    parser.add_argument("--cases", type=str, default="benchmark/call_center_cases.json")
    parser.add_argument("--output-dir", type=str, default="evals", help="Directory for benchmark reports")
    parser.add_argument("--generate_dataset", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Dataset generation seed")
    parser.add_argument("--count", type=int, default=240, help="Dataset generation count")
    parser.add_argument("--start", type=str, default="2024-06-01", help="Dataset generation start date")
    parser.add_argument("--end", type=str, default="2024-09-30", help="Dataset generation end date")
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH), help="Dataset file path")
    parser.add_argument("--dataset-seed", type=int, dest="seed", help="Alias for --seed")
    parser.add_argument("--dataset-count", type=int, dest="count", help="Alias for --count")
    parser.add_argument("--dataset-start", type=str, dest="start", help="Alias for --start")
    parser.add_argument("--dataset-end", type=str, dest="end", help="Alias for --end")
    parser.add_argument("--case-count", type=int, default=None, help="Number of cases to run")
    parser.add_argument(
        "--case-sample",
        type=str,
        choices=["first", "random"],
        default="first",
        help="Case sampling strategy when case-count is set",
    )
    parser.add_argument("--case-seed", type=int, default=42, help="Seed for random case sampling")
    parser.add_argument("--case-ids", type=str, default="", help="Comma-separated case IDs to run")
    args = parser.parse_args()

    asyncio.run(main(args))
