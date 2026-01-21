import asyncio
import os

from sgr_agent_core import AgentFactory, AgentDefinition, PromptsConfig
from sgr_agent_core.tools import AdaptPlanTool, ClarificationTool, FinalAnswerTool, GeneratePlanTool

from examples.call_center_mock.agents import CallCenterDeepResearchAgent
from examples.call_center_mock.tools import CallCenterAggregateTool, CallCenterLoadDatasetTool, CallCenterTrendTool

SYSTEM_PROMPT = """You are a bank call center analytics expert.
Use the available tools to analyze the provided call center dataset.
Focus on root causes, multilingual transcripts, trends, segmentation, operational KPIs, SLA attainment,
quality/compliance signals, financial impact, and actionable recommendations.
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
        llm={"api_key": api_key, "model": "gpt-5-nano-2025-08-07", "max_completion_tokens": 8192, "temperature": 1},
        execution={"max_iterations": 15, "max_clarifications": 5, "logs_dir": "logs/call_center_mock", "reports_dir": "reports/call_center_mock"},
        search={
            "tavily_api_key": os.getenv("TAVILY_API_KEY"),
            "max_searches": 4,
            "max_results": 10,
            "content_limit": 3500,
        }
    )

    task_messages = [
        {
            "role": "user",
            "content": (
                "Perform a comprehensive multi-dimensional analysis of Q3 2024 call center performance. "
                "Analyze: (1) Segment performance by call type, agent tenure, and time-of-day patterns; "
                "(2) Identify correlation between SLA breaches and customer churn; "
                "(3) Decompose AHT variance across regions and languages; "
                "(4) Evaluate FCR impact on CSAT and repeat contact rates; "
                "(5) Assess compliance violations by severity and financial exposure; "
                "(6) Model escalation drivers and propose intervention strategies; "
                "(7) Quantify revenue leakage from abandoned calls and failed transfers; "
                "(8) Benchmark against industry standards; (9) Project 30/60/90-day forecasts with confidence intervals; "
                "(10) Deliver an executive summary with 10 prioritized initiatives ranked by ROI and implementation complexity."
            ),
        }
    ]
    agent = await AgentFactory.create(agent_def=agent_def, task_messages=task_messages)
    result = await agent.execute()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
