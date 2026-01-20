# Call Center Deep Research (Mock Data)

This example adds a production-style agent that analyzes call center data for a bank.
It includes data contracts, deterministic mock data, exact KPI calculations, and a benchmark harness.

Key files:
- `examples/call_center_mock/contracts.py` - data contracts and output schemas
- `examples/call_center_mock/metrics.py` - KPI engine, grouping, trends
- `examples/call_center_mock/tools.py` - custom tools used by the agent
- `examples/call_center_mock/generate_mock_data.py` - synthetic data generator
- `benchmark/run_call_center_bench.py` - benchmark runner
- `benchmark/call_center_cases.json` - benchmark cases

## Requirements

- Python 3.11+
- A virtual environment with dependencies installed

If you need dependencies:
```bash
uv pip install -e ".[dev]" --python .venv/bin/python
```

## Data Contracts

The dataset follows `call_center.v1` schema (see `examples/call_center_mock/contracts.py`).
The generator writes an object with metadata and a `records` array:

```json
{
  "schema_version": "call_center.v1",
  "source_system": "mock_call_center",
  "generated_at": "2024-08-01T00:00:00Z",
  "records": [ ... ]
}
```

Records are validated by Pydantic before analysis; invalid records are counted in `invalid_records`.

## Tools

- `call_center_load_dataset`
  - Filters records by date range and optional filters (segment, tags, queue, channel, sentiment).
  - Outputs a structured summary with counts and KPIs.
- `call_center_aggregate`
  - Computes exact KPIs and group-level metrics for a chosen dimension.
- `call_center_trend`
  - Monthly KPI trend summary with deltas vs previous month.

## Generate Mock Data

```bash
PYTHONPATH=. .venv/bin/python examples/call_center_mock/generate_mock_data.py --count 240
```

You can override date range and seed:
```bash
PYTHONPATH=. .venv/bin/python examples/call_center_mock/generate_mock_data.py \
  --seed 42 --count 240 --start 2024-06-01 --end 2024-09-30
```

## Quick Tool Test (no LLM required)

```bash
PYTHONPATH=. .venv/bin/python examples/call_center_mock/demo_tools.py
```

You should see `LOAD RESULT`, `AGGREGATE RESULT`, and `TREND RESULT` JSON blocks.

## Full Agent Run (requires LLM API key)

### Option A: Direct agent demo
```bash
export OPENAI_API_KEY="your-key"
PYTHONPATH=. .venv/bin/python examples/call_center_mock/demo_agent.py
```

### Option B: OpenAI-compatible API server

1) Copy and edit the config file:
```bash
cp examples/call_center_mock/config.yaml examples/call_center_mock/my_config.yaml
```

2) Set `llm.api_key` in `examples/call_center_mock/my_config.yaml`, then run:
```bash
PYTHONPATH=. .venv/bin/python -m sgr_agent_core.server --config-file examples/call_center_mock/my_config.yaml
```

3) Send a request:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8010/v1", api_key="dummy")

response = client.chat.completions.create(
    model="call_center_deep_research",
    messages=[{"role": "user", "content": "Analyze August 2024 calls. Focus on sentiment drivers, escalations, and recommendations."}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Evaluation and Benchmark

The benchmark runs predefined cases against the agent and uses a judge agent to grade numeric accuracy.
The judge uses SGR tool-calling; `gpt-4o-mini` is the recommended judge model.

### What it measures
- Required KPIs vs expected values (tolerances are enforced by the judge prompt)
- Missing metrics or large deviations result in `INCORRECT`
- Minor omissions result in `PARTIAL`

### Run the benchmark
```bash
AGENT_BASE_URL="https://api.openai.com/v1" \
AGENT_API_KEY="your-agent-key" \
AGENT_MODEL_NAME="gpt-5-nano-2025-08-07" \
JUDGE_BASE_URL="https://api.openai.com/v1" \
JUDGE_API_KEY="your-judge-key" \
JUDGE_MODEL_NAME="gpt-4o-mini" \
PYTHONPATH=. .venv/bin/python benchmark/run_call_center_bench.py --generate_dataset
```

### Useful CLI options

Run a subset of cases:
```bash
PYTHONPATH=. .venv/bin/python benchmark/run_call_center_bench.py --case-count 2 --case-sample random --case-seed 7
```

Run specific case IDs:
```bash
PYTHONPATH=. .venv/bin/python benchmark/run_call_center_bench.py --case-ids aug_overall,jul_overall
```

Use a custom dataset file:
```bash
PYTHONPATH=. .venv/bin/python benchmark/run_call_center_bench.py --dataset-path /path/to/call_center_records.json
```

Generate a smaller dataset sample:
```bash
PYTHONPATH=. .venv/bin/python benchmark/run_call_center_bench.py --generate_dataset --dataset-count 120
```

### Benchmark inputs
- Cases: `benchmark/call_center_cases.json`
- Judge agent + prompt: `benchmark/call_center_judge.py`

### Output
- Reports are saved as `evals/call_center_bench_<uuid>.xlsx` (or `.csv` if `openpyxl` is missing).
- Columns: `case_id`, `task`, `expected`, `predicted`, `grade_str`, `grade_report`

## Metric Tests (deterministic)

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_call_center_metrics.py
```

## Logging (all agent runs)

Every agent run writes two log files (if `execution.logs_dir` is set):

- `logs/<timestamp>-<agent_id>-events.jsonl` - structured event stream (run_start, reasoning, tool_execution, error, run_end)
- `logs/<timestamp>-<agent_id>-log.json` - full summary with tools and reasoning snapshots

The standard rotating log is still written to `logs/sgr_agent_core.log` when the server uses
`logging_config.yaml`.
