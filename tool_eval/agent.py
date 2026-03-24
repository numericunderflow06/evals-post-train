"""Core agent loop for tool-use evaluation.

Drives multi-turn interaction: model generates tool calls, environment
executes them, results are fed back until the task is done.
"""

import logging
import time
import uuid
from typing import Dict, List, Optional

from tool_eval.environment import Environment, TaskResult, ToolResult
from tool_eval.models import Message, ModelBackend, ModelResponse, ToolCall

logger = logging.getLogger(__name__)


class AgentLoop:
    """Orchestrates the model-environment interaction loop."""

    def __init__(
        self,
        model: ModelBackend,
        environment: Environment,
        max_turns: Optional[int] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        verbose: bool = False,
    ):
        self.model = model
        self.env = environment
        self.max_turns = max_turns or environment.get_max_turns()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

    def run_task(self, task_id: str) -> TaskResult:
        """Run a single evaluation task through the agent loop.

        Returns a TaskResult with score, trajectory, and metadata.
        """
        # Reset environment
        initial_prompt = self.env.reset(task_id)
        tools = self.env.get_tools()
        system_prompt = self.env.get_system_prompt()

        # Build initial messages
        messages: List[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        # Add initial task description
        messages.append(Message(role="user", content=initial_prompt))

        trajectory = []
        num_turns = 0
        num_tool_calls = 0
        total_tokens = 0
        start_time = time.time()

        while not self.env.is_done() and num_turns < self.max_turns:
            num_turns += 1

            # Get model response
            response = self.model.generate(
                messages=messages,
                tools=tools if tools else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.usage:
                total_tokens += response.usage.get("total_tokens", 0)

            if self.verbose:
                logger.info(f"Turn {num_turns}: content={response.content[:100] if response.content else None}, "
                            f"tool_calls={len(response.tool_calls)}")

            # Record trajectory
            turn_record = {
                "turn": num_turns,
                "content": response.content,
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ],
            }

            if response.tool_calls:
                # Execute tool calls
                assistant_msg = Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
                messages.append(assistant_msg)

                tool_results = []
                for tc in response.tool_calls:
                    num_tool_calls += 1
                    try:
                        result = self.env.execute_tool(tc.name, tc.arguments)
                    except Exception as e:
                        result = ToolResult(
                            tool_name=tc.name,
                            output=f"Error: {e}",
                            success=False,
                            error=str(e),
                        )

                    tool_results.append({
                        "name": tc.name,
                        "output": result.output[:500],
                        "success": result.success,
                    })

                    # Add tool result message
                    messages.append(Message(
                        role="tool",
                        content=result.output,
                        tool_call_id=tc.id,
                        name=tc.name,
                    ))

                    if self.verbose:
                        logger.info(f"  Tool {tc.name}: success={result.success}, "
                                    f"output={result.output[:100]}")

                turn_record["tool_results"] = tool_results

            else:
                # No tool calls — agent is responding with text
                self.env.on_agent_response(response.content or "")
                messages.append(Message(
                    role="assistant",
                    content=response.content,
                ))

                # Check for next user message (conversational benchmarks)
                user_msg = self.env.get_user_message()
                if user_msg:
                    messages.append(Message(role="user", content=user_msg))
                    turn_record["user_followup"] = user_msg

            trajectory.append(turn_record)

        # Score
        elapsed = time.time() - start_time
        result = self.env.score()
        result.num_turns = num_turns
        result.num_tool_calls = num_tool_calls
        result.trajectory = trajectory
        result.metadata["elapsed_seconds"] = elapsed
        result.metadata["total_tokens"] = total_tokens
        result.metadata["model"] = self.model.get_name()

        return result

    def run_all(
        self,
        task_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, TaskResult]:
        """Run the agent on multiple tasks.

        Args:
            task_ids: Specific tasks to run. If None, runs all.
            limit: Maximum number of tasks to run.

        Returns:
            Dict mapping task_id to TaskResult.
        """
        if task_ids is None:
            task_ids = self.env.get_tasks()
        if limit:
            task_ids = task_ids[:limit]

        results = {}
        for i, task_id in enumerate(task_ids):
            logger.info(f"[{i+1}/{len(task_ids)}] Running task: {task_id}")
            try:
                result = self.run_task(task_id)
                results[task_id] = result
                logger.info(
                    f"  score={result.score:.2f}, turns={result.num_turns}, "
                    f"tool_calls={result.num_tool_calls}"
                )
            except Exception as e:
                logger.error(f"  FAILED: {e}")
                results[task_id] = TaskResult(
                    task_id=task_id,
                    success=False,
                    score=0.0,
                    num_turns=0,
                    num_tool_calls=0,
                    metadata={"error": str(e)},
                )

        return results
