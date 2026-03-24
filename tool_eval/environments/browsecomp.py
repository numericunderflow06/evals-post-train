"""BrowseComp environment adapter.

Evaluates agents on hard web browsing tasks that require multi-step
information retrieval across multiple web pages.

Dataset: openai/browsecomp (on HuggingFace)
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


@register("browsecomp")
class BrowseCompEnvironment(Environment):
    """Environment for BrowseComp web browsing benchmark.

    Agent gets web_search and browse_page tools to find answers
    to hard factual questions requiring multi-step web navigation.
    """

    def __init__(self, search_backend: str = "serper", **kwargs):
        """
        Args:
            search_backend: "serper" (default) or "google" — which search API to use.
        """
        self.search_backend = search_backend
        self._tasks = {}
        self._current_id = None
        self._current_task = None
        self._done = False
        self._answer = None
        self._search_count = 0
        self._browse_count = 0

        try:
            from datasets import load_dataset
            ds = load_dataset("openai/browsecomp", split="test")
            self._tasks = {str(i): row for i, row in enumerate(ds)}
        except Exception as e:
            logger.warning(f"BrowseComp dataset not loaded: {e}")

    def get_name(self) -> str:
        return "browsecomp"

    def get_tasks(self) -> List[str]:
        return list(self._tasks.keys())

    def reset(self, task_id: str) -> str:
        self._current_id = task_id
        self._current_task = self._tasks[task_id]
        self._done = False
        self._answer = None
        self._search_count = 0
        self._browse_count = 0

        question = self._current_task.get("question", self._current_task.get("input", ""))
        return (
            f"Answer the following question. You will need to search the web and "
            f"browse multiple pages to find the answer.\n\n"
            f"Question: {question}\n\n"
            f"Use web_search to find relevant pages, browse_page to read them, "
            f"and submit_answer when you have the answer."
        )

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="web_search",
                description="Search the web for information. Returns a list of search results with titles, URLs, and snippets.",
                parameters=[
                    ToolParameter(name="query", type="string", description="Search query"),
                    ToolParameter(name="num_results", type="integer", description="Number of results (default 10)", required=False),
                ],
            ),
            ToolDefinition(
                name="browse_page",
                description="Fetch and read the text content of a web page.",
                parameters=[
                    ToolParameter(name="url", type="string", description="URL to fetch"),
                ],
            ),
            ToolDefinition(
                name="submit_answer",
                description="Submit your final answer to the question.",
                parameters=[
                    ToolParameter(name="answer", type="string", description="Your answer"),
                ],
            ),
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        if tool_name == "web_search":
            self._search_count += 1
            query = arguments.get("query", "")
            return self._do_web_search(query, arguments.get("num_results", 10))

        elif tool_name == "browse_page":
            self._browse_count += 1
            url = arguments.get("url", "")
            return self._do_browse(url)

        elif tool_name == "submit_answer":
            self._answer = arguments.get("answer", "")
            self._done = True
            return ToolResult(tool_name="submit_answer", output="Answer submitted.", success=True)

        return ToolResult(tool_name=tool_name, output=f"Unknown tool: {tool_name}", success=False)

    def _do_web_search(self, query: str, num_results: int) -> ToolResult:
        """Perform web search via configured backend."""
        import os

        if self.search_backend == "serper":
            api_key = os.environ.get("SERPER_API_KEY")
            if not api_key:
                return ToolResult(
                    tool_name="web_search",
                    output="SERPER_API_KEY not set. Set it to enable web search.",
                    success=False, error="no_api_key",
                )
            try:
                import requests
                resp = requests.post(
                    "https://google.serper.dev/search",
                    json={"q": query, "num": num_results},
                    headers={"X-API-KEY": api_key},
                    timeout=15,
                )
                data = resp.json()
                results = []
                for item in data.get("organic", [])[:num_results]:
                    results.append(
                        f"Title: {item.get('title', '')}\n"
                        f"URL: {item.get('link', '')}\n"
                        f"Snippet: {item.get('snippet', '')}\n"
                    )
                return ToolResult(
                    tool_name="web_search",
                    output="\n---\n".join(results) if results else "No results found.",
                    success=True,
                )
            except Exception as e:
                return ToolResult(tool_name="web_search", output=f"Search error: {e}", success=False)

        return ToolResult(
            tool_name="web_search",
            output=f"Unsupported search backend: {self.search_backend}",
            success=False,
        )

    def _do_browse(self, url: str) -> ToolResult:
        """Fetch and extract text from a web page."""
        try:
            import requests
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()

            # Try to extract text with BeautifulSoup
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
            except ImportError:
                # Fallback to raw text
                text = resp.text

            # Truncate
            if len(text) > 15000:
                text = text[:7500] + "\n...[truncated]...\n" + text[-3000:]

            return ToolResult(tool_name="browse_page", output=text, success=True)
        except Exception as e:
            return ToolResult(tool_name="browse_page", output=f"Browse error: {e}", success=False)

    def get_user_message(self) -> Optional[str]:
        return None

    def is_done(self) -> bool:
        return self._done

    def score(self) -> TaskResult:
        if not self._answer:
            return TaskResult(
                task_id=self._current_id, success=False, score=0.0,
                num_turns=0, num_tool_calls=self._search_count + self._browse_count,
            )

        target = self._current_task.get("answer", self._current_task.get("target", ""))
        # Case-insensitive containment check
        match = target.lower().strip() in self._answer.lower().strip() if target else False

        return TaskResult(
            task_id=self._current_id,
            success=match,
            score=1.0 if match else 0.0,
            num_turns=0,
            num_tool_calls=self._search_count + self._browse_count,
            metadata={
                "searches": self._search_count,
                "pages_browsed": self._browse_count,
            },
        )

    def get_system_prompt(self) -> Optional[str]:
        return (
            "You are a research assistant skilled at finding hard-to-find information "
            "on the web. Use web_search and browse_page to find the answer. Follow "
            "links, cross-reference sources, and dig deep. Submit your answer when confident."
        )

    def get_max_turns(self) -> int:
        return 30
