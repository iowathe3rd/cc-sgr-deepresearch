# Call Center Deep Research (Mock Data)

This example adds a custom agent that performs deep analysis on a mock bank call center dataset.
It includes two custom tools:

- `call_center_load_dataset` - loads and filters the dataset
- `call_center_aggregate` - aggregates KPIs and trend signals

## Quick Tool Test (no LLM required)

```bash
PYTHONPATH=. python3 examples/call_center_mock/demo_tools.py
```

## Full Agent Run (requires LLM API key)

1) Copy and edit the config file:

```bash
cp examples/call_center_mock/config.yaml examples/call_center_mock/my_config.yaml
```

2) Set your API key and run:

```bash
PYTHONPATH=. python3 -m sgr_agent_core.server --config-file examples/call_center_mock/my_config.yaml
```

3) Send a request to the local OpenAI-compatible API:

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
