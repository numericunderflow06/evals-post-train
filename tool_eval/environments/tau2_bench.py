"""tau2-bench environment adapter.

Wraps the tau2-bench benchmark (sierra-research/tau2-bench) which evaluates
conversational agents in customer service scenarios with tool APIs.

Install: pip install tau2-bench
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


@register("tau2-bench")
class Tau2BenchEnvironment(Environment):
    """Environment adapter for tau2-bench.

    Supports "airline", "retail", and "telecom" domains.
    """

    def __init__(self, domain: str = "retail", **kwargs):
        try:
            from tau_bench.envs import get_env
        except ImportError:
            raise ImportError(
                "tau2-bench not installed. Run: pip install tau2-bench"
            )

        self.domain = domain
        self._env = None
        self._task_id = None
        self._done = False
        self._user_messages: List[str] = []
        self._agent_responses: List[str] = []
        self._tool_call_count = 0

        # Load the environment
        self._tau_env = get_env(domain)

    def get_name(self) -> str:
        return f"tau2-bench-{self.domain}"

    def get_tasks(self) -> List[str]:
        """Return available task IDs."""
        return [str(i) for i in range(len(self._tau_env.tasks))]

    def reset(self, task_id: str) -> str:
        """Reset for a new task and return initial user message."""
        idx = int(task_id)
        self._task_id = task_id
        self._done = False
        self._tool_call_count = 0
        self._user_messages = []
        self._agent_responses = []

        task = self._tau_env.tasks[idx]
        self._current_task = task

        # Reset the environment state
        self._tau_env.reset(task)

        # The initial user message comes from the task
        initial_msg = task.user_message
        self._user_messages.append(initial_msg)

        return initial_msg

    def get_tools(self) -> List[ToolDefinition]:
        """Convert tau-bench tools to our ToolDefinition format."""
        tools = []
        for tool_spec in self._tau_env.tools:
            params = []
            if hasattr(tool_spec, "parameters"):
                schema = tool_spec.parameters
                if isinstance(schema, dict):
                    properties = schema.get("properties", {})
                    required = schema.get("required", [])
                    for name, prop in properties.items():
                        params.append(ToolParameter(
                            name=name,
                            type=prop.get("type", "string"),
                            description=prop.get("description", ""),
                            required=name in required,
                            enum=prop.get("enum"),
                        ))

            tools.append(ToolDefinition(
                name=tool_spec.name if hasattr(tool_spec, "name") else str(tool_spec),
                description=getattr(tool_spec, "description", ""),
                parameters=params,
            ))

        return tools

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool call via the tau-bench environment."""
        self._tool_call_count += 1
        try:
            result = self._tau_env.step(tool_name, arguments)
            output = str(result)
            return ToolResult(
                tool_name=tool_name,
                output=output,
                success=True,
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                output=f"Error: {e}",
                success=False,
                error=str(e),
            )

    def get_user_message(self) -> Optional[str]:
        """Get next user message from the simulated user."""
        try:
            msg = self._tau_env.get_user_message()
            if msg:
                self._user_messages.append(msg)
                return msg
        except Exception:
            pass
        return None

    def is_done(self) -> bool:
        """Check if the conversation is complete."""
        try:
            return self._tau_env.is_done()
        except Exception:
            return self._done

    def on_agent_response(self, response: str) -> None:
        """Record agent text response."""
        self._agent_responses.append(response)
        try:
            self._tau_env.on_agent_response(response)
        except Exception:
            pass

    def score(self) -> TaskResult:
        """Score the agent's performance."""
        try:
            reward = self._tau_env.score()
            score = float(reward) if isinstance(reward, (int, float)) else 0.0
        except Exception as e:
            logger.warning(f"Scoring failed for task {self._task_id}: {e}")
            score = 0.0

        return TaskResult(
            task_id=self._task_id,
            success=score >= 1.0,
            score=score,
            num_turns=len(self._agent_responses),
            num_tool_calls=self._tool_call_count,
            metadata={
                "domain": self.domain,
                "num_user_messages": len(self._user_messages),
            },
        )

    def get_system_prompt(self) -> Optional[str]:
        """Get the system prompt for the agent."""
        try:
            return self._tau_env.system_prompt
        except Exception:
            return (
                f"You are a helpful customer service agent for the {self.domain} domain. "
                f"Use the available tools to help the user with their request. "
                f"Be polite, efficient, and follow company policies."
            )

    def get_max_turns(self) -> int:
        return 30
