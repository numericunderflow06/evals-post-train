"""Abstract base class for benchmark environments.

Each benchmark defines tools, handles tool execution, and scores results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolParameter:
    name: str
    type: str  # "string", "integer", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None


@dataclass
class ToolDefinition:
    """OpenAI-compatible function/tool definition."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        for p in self.parameters:
            prop = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_schema(self) -> dict:
        """Convert to Anthropic tool use format."""
        properties = {}
        required = []
        for p in self.parameters:
            prop = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass
class ToolResult:
    """Result of executing a tool call."""
    tool_name: str
    output: str
    success: bool = True
    error: Optional[str] = None


@dataclass
class TaskResult:
    """Final result for a single evaluation task."""
    task_id: str
    success: bool
    score: float  # 0.0 to 1.0
    num_turns: int
    num_tool_calls: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    trajectory: List[dict] = field(default_factory=list)


class Environment(ABC):
    """Abstract base class for benchmark environments."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the benchmark name."""

    @abstractmethod
    def get_tasks(self) -> List[str]:
        """Return list of task IDs in this benchmark."""

    @abstractmethod
    def reset(self, task_id: str) -> str:
        """Reset environment for a new task.

        Returns the initial system prompt / task description.
        """

    @abstractmethod
    def get_tools(self) -> List[ToolDefinition]:
        """Return tool definitions available for the current task."""

    @abstractmethod
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool call and return the result."""

    @abstractmethod
    def get_user_message(self) -> Optional[str]:
        """Get the next user message (for conversational benchmarks).

        Returns None if there is no user message (agent-only turn).
        """

    @abstractmethod
    def is_done(self) -> bool:
        """Check if the current task episode is complete."""

    @abstractmethod
    def score(self) -> TaskResult:
        """Score the agent's performance on the current task."""

    def get_system_prompt(self) -> Optional[str]:
        """Optional system prompt for the agent."""
        return None

    def get_max_turns(self) -> int:
        """Maximum number of agent turns before force-stopping."""
        return 50

    def on_agent_response(self, response: str) -> None:
        """Hook called when the agent produces a text response (no tool call)."""
        pass
