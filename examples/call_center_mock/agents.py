from typing import Type

from openai import pydantic_function_tool
from openai.types.chat import ChatCompletionFunctionToolParam

from sgr_agent_core.agent_definition import AgentConfig
from sgr_agent_core.agents.sgr_tool_calling_agent import SGRToolCallingAgent
from sgr_agent_core.tools import (
    AdaptPlanTool,
    ClarificationTool,
    FinalAnswerTool,
    GeneratePlanTool,
)

try:
    from tools import CallCenterAggregateTool, CallCenterLoadDatasetTool, CallCenterTrendTool
except ImportError:  # pragma: no cover - fallback for package imports
    from examples.call_center_mock.tools import (
        CallCenterAggregateTool,
        CallCenterLoadDatasetTool,
        CallCenterTrendTool,
    )


class CallCenterDeepResearchAgent(SGRToolCallingAgent):
    """Deep research agent for bank call center analytics."""

    name: str = "call_center_deep_research"

    def __init__(
        self,
        task_messages: list,
        openai_client,
        agent_config: AgentConfig,
        toolkit: list[Type],
        def_name: str | None = None,
        **kwargs: dict,
    ):
        base_tools = [
            CallCenterLoadDatasetTool,
            CallCenterAggregateTool,
            CallCenterTrendTool,
            GeneratePlanTool,
            AdaptPlanTool,
            ClarificationTool,
            FinalAnswerTool,
        ]
        super().__init__(
            task_messages=task_messages,
            openai_client=openai_client,
            agent_config=agent_config,
            toolkit=base_tools + [tool for tool in toolkit if tool not in base_tools],
            def_name=def_name,
            **kwargs,
        )

    async def _prepare_tools(self) -> list[ChatCompletionFunctionToolParam]:
        tools = set(self.toolkit)
        if self._context.iteration >= self.config.execution.max_iterations:
            tools = {FinalAnswerTool}
        if self._context.clarifications_used >= self.config.execution.max_clarifications:
            tools -= {ClarificationTool}
        return [pydantic_function_tool(tool, name=tool.tool_name, description="") for tool in tools]
