import asyncio
import os

from sgr_agent_core import AgentFactory, AgentDefinition, PromptsConfig
from sgr_agent_core.tools import AdaptPlanTool, ClarificationTool, FinalAnswerTool, GeneratePlanTool

from examples.call_center_mock.agents import CallCenterDeepResearchAgent
from examples.call_center_mock.tools import CallCenterAggregateTool, CallCenterLoadDatasetTool, CallCenterTrendTool

SYSTEM_PROMPT = """You are a bank call center analytics expert.
Use the available tools to analyze the provided call center dataset.
Focus on root causes, trends, segmentation, operational KPIs, and actionable recommendations.
Rely only on tool outputs and keep the analysis grounded in the data.

Available tools:
{available_tools}
"""


async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the agent demo.")

    agent_def = AgentDefinition(
        name="call_center_deep_research",
        base_class=CallCenterDeepResearchAgent,
        tools=[
            CallCenterLoadDatasetTool,
            CallCenterAggregateTool,
            CallCenterTrendTool,
            GeneratePlanTool,
            AdaptPlanTool,
            ClarificationTool,
            FinalAnswerTool,
        ],
        prompts=PromptsConfig(system_prompt_str=SYSTEM_PROMPT),
        llm={"api_key": api_key, "model": "gpt-5-nano-2025-08-07", "temperature": 0.3},
    )

    task_messages = [
        {
            "role": "user",
            "content": (
                "Analyze August 2024 call center data. Identify the top drivers of negative sentiment, "
                "escalations, and long wait times. Provide prioritized operational recommendations."
            ),
        }
    ]

    agent = await AgentFactory.create(agent_def=agent_def, task_messages=task_messages)
    result = await agent.execute()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
