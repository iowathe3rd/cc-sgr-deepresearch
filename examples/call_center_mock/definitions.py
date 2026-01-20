from sgr_agent_core.agent_definition import AgentDefinition, PromptsConfig
from sgr_agent_core.tools import AdaptPlanTool, ClarificationTool, FinalAnswerTool, GeneratePlanTool

from examples.call_center_mock.agents import CallCenterDeepResearchAgent
from examples.call_center_mock.tools import CallCenterAggregateTool, CallCenterLoadDatasetTool, CallCenterTrendTool

CALL_CENTER_SYSTEM_PROMPT = """You are a bank call center analytics expert.
Use the available tools to analyze the provided call center dataset.
Focus on root causes, multilingual transcripts, trends, segmentation, operational KPIs, SLA attainment,
quality/compliance signals, financial impact, and actionable recommendations.
Rely only on tool outputs and keep the analysis grounded in the data.

Available tools:
{available_tools}
"""

DEFAULT_TOOLKIT = [
    CallCenterLoadDatasetTool,
    CallCenterAggregateTool,
    CallCenterTrendTool,
    GeneratePlanTool,
    AdaptPlanTool,
    ClarificationTool,
    FinalAnswerTool,
]


def get_call_center_agent_definition() -> AgentDefinition:
    return AgentDefinition(
        name="call_center_deep_research",
        base_class=CallCenterDeepResearchAgent,
        tools=DEFAULT_TOOLKIT,
        prompts=PromptsConfig(system_prompt_str=CALL_CENTER_SYSTEM_PROMPT),
    )
