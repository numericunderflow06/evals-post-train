"""TerminalBench 2 environment adapter.

Evaluates agents on terminal-based tasks across software engineering,
biology, security, and gaming domains.

Install: pip install harbor
"""

import logging
import subprocess
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


@register("terminalbench")
class TerminalBenchEnvironment(Environment):
    """Environment adapter for TerminalBench 2.

    The agent gets a bash tool and must solve terminal-based tasks.
    Full evaluation uses the Harbor framework with containers.
    """

    def __init__(
        self,
        timeout: int = 120,
        **kwargs,
    ):
        self.timeout = timeout
        self._tasks = {}
        self._current_id = None
        self._current_task = None
        self._done = False
        self._bash_history = []
        self._last_output = ""

        # Try to load tasks from harbor
        try:
            import harbor
            self._harbor_available = True
        except ImportError:
            self._harbor_available = False
            logger.warning(
                "harbor not installed. Install: pip install harbor. "
                "Using stub mode — tasks must be provided manually."
            )

    def get_name(self) -> str:
        return "terminalbench-2"

    def get_tasks(self) -> List[str]:
        if self._harbor_available:
            try:
                import harbor
                return [t.id for t in harbor.get_tasks()]
            except Exception:
                pass
        return list(self._tasks.keys())

    def reset(self, task_id: str) -> str:
        self._current_id = task_id
        self._done = False
        self._bash_history = []
        self._last_output = ""

        if self._harbor_available:
            try:
                import harbor
                task = harbor.get_task(task_id)
                self._current_task = task
                return task.description
            except Exception as e:
                logger.warning(f"Failed to load task from harbor: {e}")

        return f"Complete task: {task_id}"

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="bash",
                description="Execute a bash command in the terminal environment.",
                parameters=[
                    ToolParameter(
                        name="command",
                        type="string",
                        description="The bash command to execute",
                    ),
                ],
            ),
            ToolDefinition(
                name="done",
                description="Signal that you have completed the task.",
                parameters=[
                    ToolParameter(
                        name="answer",
                        type="string",
                        description="Your final answer or completion message",
                        required=False,
                    ),
                ],
            ),
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        if tool_name == "bash":
            command = arguments.get("command", "")
            self._bash_history.append(command)
            try:
                result = subprocess.run(
                    ["bash", "-c", command],
                    capture_output=True, text=True,
                    timeout=self.timeout,
                )
                output = result.stdout
                if result.stderr:
                    output += f"\nSTDERR:\n{result.stderr}"
                self._last_output = output
                if len(output) > 10000:
                    output = output[:5000] + "\n...[truncated]...\n" + output[-2000:]
                return ToolResult(tool_name="bash", output=output, success=True)
            except subprocess.TimeoutExpired:
                return ToolResult(
                    tool_name="bash",
                    output=f"Timed out after {self.timeout}s",
                    success=False, error="timeout",
                )

        elif tool_name == "done":
            self._done = True
            return ToolResult(tool_name="done", output="Task marked as complete.", success=True)

        return ToolResult(tool_name=tool_name, output=f"Unknown tool: {tool_name}", success=False)

    def get_user_message(self) -> Optional[str]:
        return None

    def is_done(self) -> bool:
        return self._done

    def score(self) -> TaskResult:
        return TaskResult(
            task_id=self._current_id,
            success=self._done,
            score=1.0 if self._done else 0.0,
            num_turns=0,
            num_tool_calls=len(self._bash_history),
            metadata={
                "bash_commands": len(self._bash_history),
                "note": "Full scoring requires harbor framework with containers",
            },
        )

    def get_system_prompt(self) -> Optional[str]:
        return (
            "You are an expert at solving terminal-based tasks. Use bash commands "
            "to complete the given task. When finished, call the done tool."
        )

    def get_max_turns(self) -> int:
        return 50
