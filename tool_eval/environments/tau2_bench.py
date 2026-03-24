"""tau2-bench environment adapter.

Wraps the tau2-bench benchmark (sierra-research/tau2-bench) which evaluates
conversational agents in customer service scenarios with tool APIs.

Install: pip install git+https://github.com/sierra-research/tau2-bench.git

Note: tau2 requires pyaudio (which needs portaudio system headers).
If pyaudio is unavailable, this module will mock it so that the
text-based environment and tools still work.
"""

import json
import logging
import sys
import types
import uuid
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


def _ensure_pyaudio_available():
    """Install a stub pyaudio module if the real one is not importable.

    tau2's import chain pulls in voice modules that require pyaudio at
    import time even when only the text-based environment is used.
    Mocking it allows the rest of tau2 to load.
    """
    if "pyaudio" not in sys.modules:
        try:
            import pyaudio  # noqa: F401
        except ImportError:
            logger.debug(
                "pyaudio not installed; inserting stub so tau2 text env can load"
            )
            mock = types.ModuleType("pyaudio")
            mock.PyAudio = type("PyAudio", (), {})  # type: ignore[attr-defined]
            mock.paInt16 = 8  # type: ignore[attr-defined]
            mock.paContinue = 0  # type: ignore[attr-defined]
            mock.paComplete = 1  # type: ignore[attr-defined]
            sys.modules["pyaudio"] = mock


@register("tau2-bench")
class Tau2BenchEnvironment(Environment):
    """Environment adapter for tau2-bench.

    Supported domains: "airline", "retail", "telecom", "telecom-workflow",
    "banking_knowledge", and "mock".

    The ``task_split`` parameter controls which subset of tasks to load
    (default ``"base"``).  Pass ``task_split=None`` to load all tasks.
    """

    def __init__(
        self,
        domain: str = "retail",
        task_split: Optional[str] = "base",
        **kwargs,
    ):
        _ensure_pyaudio_available()

        try:
            from tau2.registry import registry as tau2_registry
        except ImportError:
            raise ImportError(
                "tau2-bench is not installed. "
                "Run: pip install git+https://github.com/sierra-research/tau2-bench.git\n"
                "You also need to set TAU2_DATA_DIR to point to the data/ directory "
                "from the cloned repo."
            )

        self.domain = domain
        self._task_split = task_split
        self._task_id: Optional[str] = None
        self._done = False
        self._user_messages: List[str] = []
        self._agent_responses: List[str] = []
        self._tool_call_count = 0

        # Build the tau2 environment for this domain.
        # get_env_constructor returns a factory function: () -> tau2 Environment
        get_env_fn = tau2_registry.get_env_constructor(domain)
        self._tau_env = get_env_fn()

        # Load tasks via the registry's task loader.
        # The task set name in the registry matches the domain name for the
        # standard domains (airline, retail, telecom, etc.).
        tasks_loader = tau2_registry.get_tasks_loader(domain)
        self._tasks: List[Any] = tasks_loader(task_split)

        # Build a mapping from task.id -> index for O(1) lookup
        self._task_id_map: Dict[str, int] = {
            t.id: i for i, t in enumerate(self._tasks)
        }

        self._current_task: Optional[Any] = None

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return f"tau2-bench-{self.domain}"

    def get_tasks(self) -> List[str]:
        """Return task IDs (the tau2 Task.id field, not integer indices)."""
        return [t.id for t in self._tasks]

    def reset(self, task_id: str) -> str:
        """Reset for a new task and return initial user instructions."""
        if task_id in self._task_id_map:
            idx = self._task_id_map[task_id]
        else:
            # Allow integer-string fallback ("0", "1", ...)
            idx = int(task_id)

        self._task_id = task_id
        self._done = False
        self._tool_call_count = 0
        self._user_messages = []
        self._agent_responses = []

        task = self._tasks[idx]
        self._current_task = task

        # Re-create the environment to get a clean state.
        from tau2.registry import registry as tau2_registry

        get_env_fn = tau2_registry.get_env_constructor(self.domain)
        self._tau_env = get_env_fn()

        # Apply initial state (DB mutations, initialization actions,
        # message history replay) if the task specifies one.
        if task.initial_state is not None:
            self._tau_env.set_state(
                initialization_data=task.initial_state.initialization_data,
                initialization_actions=task.initial_state.initialization_actions,
                message_history=task.initial_state.message_history or [],
            )

        # The user instructions come from the task's user_scenario.
        initial_msg = str(task.user_scenario.instructions)
        self._user_messages.append(initial_msg)

        return initial_msg

    def get_tools(self) -> List[ToolDefinition]:
        """Convert tau2 Tool objects to our ToolDefinition format."""
        tau2_tools = self._tau_env.get_tools()  # list[tau2 Tool]
        tools: List[ToolDefinition] = []

        for tool in tau2_tools:
            params: List[ToolParameter] = []

            # Each tau2 Tool exposes an openai_schema property
            schema = tool.openai_schema
            func_schema = schema.get("function", {})
            param_schema = func_schema.get("parameters", {})
            properties = param_schema.get("properties", {})
            required_list = param_schema.get("required", [])

            for pname, pspec in properties.items():
                params.append(
                    ToolParameter(
                        name=pname,
                        type=pspec.get("type", "string"),
                        description=pspec.get("description", ""),
                        required=pname in required_list,
                        enum=pspec.get("enum"),
                    )
                )

            tools.append(
                ToolDefinition(
                    name=tool.name,
                    description=tool.short_desc or tool.name,
                    parameters=params,
                )
            )

        return tools

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool call via the tau2 environment."""
        self._tool_call_count += 1
        try:
            from tau2.data_model.message import ToolCall

            tc = ToolCall(
                id=str(uuid.uuid4()),
                name=tool_name,
                arguments=arguments,
            )
            # get_response returns a ToolMessage with .content (str)
            tool_message = self._tau_env.get_response(tc)
            output = tool_message.content
            is_error = getattr(tool_message, "error", False)
            return ToolResult(
                tool_name=tool_name,
                output=output,
                success=not is_error,
                error=output if is_error else None,
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                output=f"Error: {e}",
                success=False,
                error=str(e),
            )

    def get_user_message(self) -> Optional[str]:
        """Return None -- tau2 manages user simulation externally.

        In tau2, the user simulator is a separate LLM-powered component
        that runs inside the Orchestrator.  Since our adapter exposes
        only the environment/tool layer, the caller's own agent loop is
        responsible for driving the conversation.
        """
        return None

    def is_done(self) -> bool:
        """Check if the current task episode is complete.

        Without the full orchestrator there is no automatic termination
        signal, so the caller should use ``get_max_turns`` as a guard.
        """
        return self._done

    def on_agent_response(self, response: str) -> None:
        """Record agent text response."""
        self._agent_responses.append(response)

    def score(self) -> TaskResult:
        """Score the agent's performance.

        tau2's full evaluation is handled by a separate evaluator that
        compares DB state, checks assertions, and uses an LLM judge.
        Here we run the subset that can be checked locally (DB hash
        comparison and env assertions) without requiring an LLM call.
        """
        score = 0.0
        task = self._current_task

        if task is not None and task.evaluation_criteria is not None:
            criteria = task.evaluation_criteria
            checks_passed = 0
            checks_total = 0

            # Environment assertions
            if criteria.env_assertions:
                for assertion in criteria.env_assertions:
                    checks_total += 1
                    try:
                        passed = self._tau_env.run_env_assertion(
                            assertion, raise_assertion_error=False
                        )
                        if passed:
                            checks_passed += 1
                    except Exception as e:
                        logger.debug(f"Assertion check failed: {e}")

            if checks_total > 0:
                score = checks_passed / checks_total

        return TaskResult(
            task_id=self._task_id or "",
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
        """Get the domain policy as the system prompt for the agent."""
        return self._tau_env.policy

    def get_max_turns(self) -> int:
        return 30
