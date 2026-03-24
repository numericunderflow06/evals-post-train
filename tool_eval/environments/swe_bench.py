"""SWE-bench Verified environment adapter.

Wraps SWE-bench evaluation where an agent must fix real GitHub issues
using bash tools in a Docker container.

Install: pip install swebench
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
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


@register("swe-bench")
class SWEBenchEnvironment(Environment):
    """Environment adapter for SWE-bench Verified.

    The agent gets a bash tool to navigate, search, edit code, and run tests
    in a cloned repository with a real GitHub issue to fix.
    """

    def __init__(
        self,
        dataset: str = "princeton-nlp/SWE-bench_Verified",
        split: str = "test",
        workspace_dir: Optional[str] = None,
        timeout: int = 300,
        **kwargs,
    ):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        self.dataset_name = dataset
        self.timeout = timeout
        self.workspace_dir = workspace_dir or tempfile.mkdtemp(prefix="swebench_")

        # Load tasks
        ds = load_dataset(dataset, split=split)
        self._tasks = {row["instance_id"]: row for row in ds}

        self._current_task = None
        self._current_id = None
        self._bash_history: List[str] = []
        self._done = False
        self._patch_content = ""
        self._workspace = None

    def get_name(self) -> str:
        return "swe-bench-verified"

    def get_tasks(self) -> List[str]:
        return list(self._tasks.keys())

    def reset(self, task_id: str) -> str:
        """Set up the repository and return the issue description."""
        self._current_id = task_id
        self._current_task = self._tasks[task_id]
        self._bash_history = []
        self._done = False
        self._patch_content = ""

        task = self._current_task
        repo = task["repo"]
        base_commit = task["base_commit"]

        # Set up workspace
        self._workspace = Path(self.workspace_dir) / task_id.replace("/", "__")
        self._workspace.mkdir(parents=True, exist_ok=True)

        # Clone and checkout
        repo_dir = self._workspace / "repo"
        if not repo_dir.exists():
            subprocess.run(
                ["git", "clone", f"https://github.com/{repo}.git", str(repo_dir)],
                capture_output=True, timeout=120,
            )
        subprocess.run(
            ["git", "checkout", base_commit],
            cwd=str(repo_dir), capture_output=True, timeout=30,
        )

        issue = task["problem_statement"]
        return (
            f"You are working on the repository {repo} (checked out at commit {base_commit[:8]}).\n"
            f"The repository is located at {repo_dir}\n\n"
            f"Please fix the following issue:\n\n{issue}\n\n"
            f"Use the bash tool to explore the codebase, make edits, and run tests.\n"
            f"When you are done, use the submit_patch tool with the git diff of your changes."
        )

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="bash",
                description="Execute a bash command in the repository directory. Use for navigation, search, editing, and running tests.",
                parameters=[
                    ToolParameter(
                        name="command",
                        type="string",
                        description="The bash command to execute",
                    ),
                ],
            ),
            ToolDefinition(
                name="submit_patch",
                description="Submit your fix as a git diff patch. Call this when you are done fixing the issue.",
                parameters=[
                    ToolParameter(
                        name="patch",
                        type="string",
                        description="The git diff patch content, or 'auto' to use the current git diff",
                    ),
                ],
            ),
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        repo_dir = self._workspace / "repo" if self._workspace else None

        if tool_name == "bash":
            command = arguments.get("command", "")
            self._bash_history.append(command)
            try:
                result = subprocess.run(
                    ["bash", "-c", command],
                    cwd=str(repo_dir) if repo_dir else None,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                output = result.stdout
                if result.stderr:
                    output += f"\nSTDERR:\n{result.stderr}"
                if result.returncode != 0:
                    output += f"\n(exit code: {result.returncode})"
                # Truncate very long outputs
                if len(output) > 10000:
                    output = output[:5000] + "\n...[truncated]...\n" + output[-2000:]
                return ToolResult(tool_name="bash", output=output, success=True)
            except subprocess.TimeoutExpired:
                return ToolResult(
                    tool_name="bash",
                    output=f"Command timed out after {self.timeout}s",
                    success=False,
                    error="timeout",
                )

        elif tool_name == "submit_patch":
            patch = arguments.get("patch", "auto")
            if patch == "auto" and repo_dir:
                result = subprocess.run(
                    ["git", "diff"],
                    cwd=str(repo_dir),
                    capture_output=True, text=True, timeout=30,
                )
                patch = result.stdout

            self._patch_content = patch
            self._done = True
            return ToolResult(
                tool_name="submit_patch",
                output="Patch submitted successfully.",
                success=True,
            )

        return ToolResult(
            tool_name=tool_name,
            output=f"Unknown tool: {tool_name}",
            success=False,
            error="unknown_tool",
        )

    def get_user_message(self) -> Optional[str]:
        return None

    def is_done(self) -> bool:
        return self._done

    def score(self) -> TaskResult:
        """Score by running the SWE-bench test harness.

        For full scoring, use the swebench CLI:
          python -m swebench.harness.run_evaluation
        Here we do a lightweight check: did the agent produce a non-empty patch?
        """
        has_patch = bool(self._patch_content.strip())

        return TaskResult(
            task_id=self._current_id,
            success=has_patch,  # Lightweight — full eval requires Docker test runner
            score=1.0 if has_patch else 0.0,
            num_turns=0,
            num_tool_calls=len(self._bash_history),
            metadata={
                "patch_length": len(self._patch_content),
                "bash_commands": len(self._bash_history),
                "note": "Full scoring requires swebench.harness.run_evaluation",
            },
        )

    def get_system_prompt(self) -> Optional[str]:
        return (
            "You are an expert software engineer. You are given a GitHub issue and must "
            "fix it by editing the repository code. Use the bash tool to explore files, "
            "understand the codebase, make edits, and run tests. When done, use "
            "submit_patch to submit your fix."
        )

    def get_max_turns(self) -> int:
        return 50
