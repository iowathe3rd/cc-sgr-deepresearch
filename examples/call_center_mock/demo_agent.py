import asyncio
import json
import os

from rich.console import Console
from rich.prompt import Prompt

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


console = Console()


def _parse_sse_chunk(raw_chunk: str) -> dict | None:
    data = raw_chunk.strip()
    if not data.startswith("data:"):
        return None
    payload = data.removeprefix("data:").strip()
    if payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


async def _stream_agent_output(agent) -> list[str] | None:
    clarification_questions: list[str] | None = None
    async for raw_chunk in agent.streaming_generator.stream():
        parsed = _parse_sse_chunk(raw_chunk)
        if not parsed:
            continue
        choice = parsed.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        content = delta.get("content")
        if content:
            console.print(content, end="", style="white")

        tool_calls = delta.get("tool_calls") or []
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            if function.get("name") != "clarificationtool":
                continue
            args = function.get("arguments") or "{}"
            try:
                payload = json.loads(args)
            except json.JSONDecodeError:
                continue
            clarification_questions = payload.get("questions") or []
    return clarification_questions


async def _run_agent_with_clarifications(agent) -> str | None:
    execution_task = asyncio.create_task(agent.execute())
    while True:
        clarification_questions = await _stream_agent_output(agent)
        if clarification_questions:
            console.print("\n[bold red]Clarification needed:[/bold red]")
            for index, question in enumerate(clarification_questions, 1):
                console.print(f"[bold]{index}.[/bold] {question}", style="yellow")
            clarification = Prompt.ask("[bold white]Enter your clarification[/bold white]")
            await agent.provide_clarification([{"role": "user", "content": clarification}])
            continue
        break
    return await execution_task


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
        execution={
            "max_iterations": 15,
            "max_clarifications": 5,
            "logs_dir": "logs/call_center_mock",
            "reports_dir": "reports/call_center_mock",
        },
        search={
            "tavily_api_key": os.getenv("TAVILY_API_KEY"),
            "max_searches": 4,
            "max_results": 10,
            "content_limit": 3500,
        },
    )

    console.print("\n[bold green]Call Center Research CLI[/bold green]", style="bold white")
    console.print("[white]Type 'exit' to quit.[/white]")

    while True:
        request = Prompt.ask("[bold white]Enter your request[/bold white]")
        if not request or request.strip().lower() == "exit":
            console.print("[bold white]Exiting.[/bold white]")
            break

        task_messages = [{"role": "user", "content": request}]
        agent = await AgentFactory.create(agent_def=agent_def, task_messages=task_messages)
        console.print(f"\n[bold green]Starting analysis:[/bold green] [italic]{request}[/italic]\n")
        result = await _run_agent_with_clarifications(agent)
        if result:
            console.print("\n[bold green]Analysis complete.[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
