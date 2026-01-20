import asyncio

from sgr_agent_core.agent_definition import AgentConfig
from sgr_agent_core.models import AgentContext

from examples.call_center_mock.tools import (
    CallCenterAggregateTool,
    CallCenterFilters,
    CallCenterLoadDatasetTool,
)


async def main():
    context = AgentContext()
    config = AgentConfig()

    load_tool = CallCenterLoadDatasetTool(
        reasoning="Load August call data for baseline analysis.",
        date_from="2024-08-01",
        date_to="2024-08-31",
        filters=CallCenterFilters(tags=["complaint", "fraud_risk", "payment_delay"]),
        include_transcripts=False,
    )
    load_result = await load_tool(context, config)
    print("LOAD RESULT")
    print(load_result)

    aggregate_tool = CallCenterAggregateTool(
        reasoning="Identify the worst drivers of negative sentiment and escalations.",
        group_by="tag",
        focus_metric="sentiment",
        top_n=5,
        min_calls=1,
    )
    aggregate_result = await aggregate_tool(context, config)
    print("\nAGGREGATE RESULT")
    print(aggregate_result)


if __name__ == "__main__":
    asyncio.run(main())
