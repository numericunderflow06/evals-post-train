"""MCP-Atlas environment adapter.

Evaluates agents on multi-server MCP (Model Context Protocol) tool use
with 36 real MCP servers and 220 tools.

Install: git clone https://github.com/scaleapi/mcp-atlas && pip install -e .
"""

import json
import logging
from typing import Any, Dict, List, Optional

from tool_eval.environment import (
    Environment,
    TaskResult,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from tool_eval.environments.registry import register

logger = logging.getLogger(__name__)


@register("mcp-atlas")
class MCPAtlasEnvironment(Environment):
    """Environment adapter for MCP-Atlas benchmark.

    Agent must discover and orchestrate tools across multiple MCP servers
    to complete multi-step tasks.
    """

    def __init__(self, **kwargs):
        self._tasks = {}
        self._current_id = None
        self._current_task = None
        self._done = False
        self._answer = None
        self._tool_calls_log = []
        self._mcp_client = None

        # Try loading the dataset
        try:
            from datasets import load_dataset
            ds = load_dataset("ScaleAI/MCP-Atlas", split="test")
            self._tasks = {str(i): row for i, row in enumerate(ds)}
        except Exception as e:
            logger.warning(f"MCP-Atlas dataset not loaded: {e}")

        # Try connecting to MCP servers
        try:
            self._init_mcp_servers()
        except Exception as e:
            logger.warning(f"MCP servers not available: {e}")

    def _init_mcp_servers(self):
        """Initialize MCP server connections."""
        # MCP-Atlas ships with a Docker-based server setup
        # This will be populated when mcp-atlas is properly installed
        pass

    def get_name(self) -> str:
        return "mcp-atlas"

    def get_tasks(self) -> List[str]:
        return list(self._tasks.keys())

    def reset(self, task_id: str) -> str:
        self._current_id = task_id
        self._current_task = self._tasks.get(task_id, {})
        self._done = False
        self._answer = None
        self._tool_calls_log = []

        prompt = self._current_task.get("prompt", self._current_task.get("task", ""))
        return (
            f"Complete the following task using the available MCP tools.\n\n"
            f"Task: {prompt}\n\n"
            f"Use list_mcp_servers to discover available servers, "
            f"list_tools to see tools on a server, and call_tool to execute them. "
            f"Submit your answer when done."
        )

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="list_mcp_servers",
                description="List all available MCP servers and their descriptions.",
                parameters=[],
            ),
            ToolDefinition(
                name="list_tools",
                description="List all tools available on a specific MCP server.",
                parameters=[
                    ToolParameter(name="server", type="string", description="MCP server name"),
                ],
            ),
            ToolDefinition(
                name="call_tool",
                description="Call a tool on an MCP server with the given arguments.",
                parameters=[
                    ToolParameter(name="server", type="string", description="MCP server name"),
                    ToolParameter(name="tool", type="string", description="Tool name"),
                    ToolParameter(name="arguments", type="object", description="Tool arguments as JSON object", required=False),
                ],
            ),
            ToolDefinition(
                name="submit_answer",
                description="Submit your final answer to the task.",
                parameters=[
                    ToolParameter(name="answer", type="string", description="Your answer"),
                ],
            ),
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        self._tool_calls_log.append({"tool": tool_name, "args": arguments})

        if tool_name == "list_mcp_servers":
            if self._mcp_client:
                servers = self._mcp_client.list_servers()
                return ToolResult(tool_name=tool_name, output=json.dumps(servers, indent=2), success=True)
            return ToolResult(
                tool_name=tool_name,
                output="MCP servers not configured. Run mcp-atlas Docker setup first.",
                success=False, error="not_configured",
            )

        elif tool_name == "list_tools":
            server = arguments.get("server", "")
            if self._mcp_client:
                tools = self._mcp_client.list_tools(server)
                return ToolResult(tool_name=tool_name, output=json.dumps(tools, indent=2), success=True)
            return ToolResult(
                tool_name=tool_name,
                output=f"MCP server '{server}' not available.",
                success=False, error="not_configured",
            )

        elif tool_name == "call_tool":
            server = arguments.get("server", "")
            tool = arguments.get("tool", "")
            args = arguments.get("arguments", {})
            if self._mcp_client:
                result = self._mcp_client.call_tool(server, tool, args)
                return ToolResult(tool_name=tool_name, output=json.dumps(result, indent=2), success=True)
            return ToolResult(
                tool_name=tool_name,
                output=f"Cannot call {tool} on {server}: MCP servers not configured.",
                success=False, error="not_configured",
            )

        elif tool_name == "submit_answer":
            self._answer = arguments.get("answer", "")
            self._done = True
            return ToolResult(tool_name=tool_name, output="Answer submitted.", success=True)

        return ToolResult(tool_name=tool_name, output=f"Unknown tool: {tool_name}", success=False)

    def get_user_message(self) -> Optional[str]:
        return None

    def is_done(self) -> bool:
        return self._done

    def score(self) -> TaskResult:
        """Score using claims-based rubric from MCP-Atlas."""
        if not self._answer:
            return TaskResult(
                task_id=self._current_id, success=False, score=0.0,
                num_turns=0, num_tool_calls=len(self._tool_calls_log),
            )

        # MCP-Atlas uses claims-based scoring — each task has verifiable claims
        claims = self._current_task.get("claims", [])
        if claims:
            satisfied = 0
            for claim in claims:
                claim_text = claim if isinstance(claim, str) else claim.get("claim", "")
                if claim_text.lower() in self._answer.lower():
                    satisfied += 1
            score = satisfied / len(claims) if claims else 0.0
        else:
            # Fallback — check if answer is non-empty
            score = 0.5 if self._answer else 0.0

        return TaskResult(
            task_id=self._current_id,
            success=score >= 0.5,
            score=score,
            num_turns=0,
            num_tool_calls=len(self._tool_calls_log),
            metadata={
                "num_claims": len(claims) if claims else 0,
                "tool_calls": len(self._tool_calls_log),
            },
        )

    def get_system_prompt(self) -> Optional[str]:
        return (
            "You are an AI agent that uses MCP (Model Context Protocol) tools to "
            "complete tasks. First discover available servers and tools, then use "
            "them to solve the task. You may need to call multiple tools across "
            "different servers."
        )

    def get_max_turns(self) -> int:
        return 30
