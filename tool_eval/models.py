"""Model backends for tool-use evaluation.

Supports OpenAI-compatible APIs (covers vLLM, Together, etc.),
Anthropic API, and local HuggingFace models with tool calling.
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tool_eval.environment import ToolDefinition


@dataclass
class ToolCall:
    """A tool/function call from the model."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ModelResponse:
    """Structured response from the model."""
    content: Optional[str] = None  # Text content (may be None if tool calls)
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Any = None  # Raw API response for debugging
    usage: Optional[Dict[str, int]] = None  # Token usage


@dataclass
class Message:
    """Chat message in the conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # tool name for tool messages

    def to_openai(self) -> dict:
        msg = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg

    def to_anthropic(self) -> dict:
        if self.role == "system":
            return None  # Handled separately in Anthropic API
        msg = {"role": self.role}
        if self.role == "assistant" and self.tool_calls:
            content = []
            if self.content:
                content.append({"type": "text", "text": self.content})
            for tc in self.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            msg["content"] = content
        elif self.role == "tool":
            msg["role"] = "user"
            msg["content"] = [{
                "type": "tool_result",
                "tool_use_id": self.tool_call_id,
                "content": self.content or "",
            }]
        else:
            msg["content"] = self.content or ""
        return msg


class ModelBackend(ABC):
    """Abstract model backend for generating responses with tool calling."""

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Generate a response, optionally with tool calls."""

    @abstractmethod
    def get_name(self) -> str:
        """Return model identifier."""


class OpenAIBackend(ModelBackend):
    """OpenAI-compatible API backend (also works with vLLM, Together, etc.)."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        kwargs = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [t.to_openai_schema() for t in tools]

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ModelResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            raw=response,
            usage=usage,
        )

    def get_name(self) -> str:
        return self.model


class AnthropicBackend(ModelBackend):
    """Anthropic API backend."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        self.model = model
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        # Extract system message
        system = None
        api_messages = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                anthropic_msg = m.to_anthropic()
                if anthropic_msg:
                    api_messages.append(anthropic_msg)

        # Merge consecutive user messages (Anthropic requires alternating roles)
        merged = []
        for msg in api_messages:
            if merged and merged[-1]["role"] == msg["role"] == "user":
                # Merge content
                prev = merged[-1]["content"]
                curr = msg["content"]
                if isinstance(prev, str):
                    prev = [{"type": "text", "text": prev}]
                if isinstance(curr, str):
                    curr = [{"type": "text", "text": curr}]
                merged[-1]["content"] = prev + curr
            else:
                merged.append(msg)

        kwargs = {
            "model": self.model,
            "messages": merged,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [t.to_anthropic_schema() for t in tools]

        response = self.client.messages.create(**kwargs)

        content_text = None
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return ModelResponse(
            content=content_text,
            tool_calls=tool_calls,
            raw=response,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        )

    def get_name(self) -> str:
        return self.model


class TextToolCallingBackend(ModelBackend):
    """Text-based tool calling for models without native function calling.

    Works with any model served via an OpenAI-compatible chat/completions
    endpoint (vLLM, TGI, etc.) that doesn't support the `tools` parameter.
    Tool definitions are injected into the system prompt and tool calls are
    parsed from the model's text output.

    This is the recommended backend for open models like OLMo 3.

    The model is instructed to output tool calls as:
        <tool_call>
        {"name": "tool_name", "arguments": {"arg1": "value1"}}
        </tool_call>

    And text responses normally. Multiple tool calls per turn are supported.
    """

    TOOL_CALL_START = "<tool_call>"
    TOOL_CALL_END = "</tool_call>"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
            base_url=base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        )
        self._call_counter = 0

    def _format_tools_prompt(self, tools: List[ToolDefinition]) -> str:
        """Format tool definitions as text for the system prompt."""
        if not tools:
            return ""

        lines = [
            "# Available Tools",
            "",
            "You have access to the following tools. To call a tool, output:",
            "",
            "<tool_call>",
            '{"name": "tool_name", "arguments": {"arg1": "value1"}}',
            "</tool_call>",
            "",
            "You may call multiple tools in one response. After each tool call, "
            "you will receive the result and can continue reasoning.",
            "",
            "If you want to respond to the user without calling a tool, "
            "just write your response normally without any <tool_call> tags.",
            "",
            "## Tool Definitions",
            "",
        ]

        for tool in tools:
            lines.append(f"### {tool.name}")
            lines.append(f"{tool.description}")
            if tool.parameters:
                lines.append("Parameters:")
                for p in tool.parameters:
                    req = " (required)" if p.required else " (optional)"
                    lines.append(f"  - {p.name} ({p.type}){req}: {p.description}")
            lines.append("")

        return "\n".join(lines)

    def _build_messages(
        self, messages: List[Message], tools: Optional[List[ToolDefinition]]
    ) -> List[dict]:
        """Build plain chat messages with tool info in system prompt."""
        tool_prompt = self._format_tools_prompt(tools) if tools else ""
        result = []

        for m in messages:
            if m.role == "system":
                # Append tool definitions to system prompt
                content = m.content or ""
                if tool_prompt:
                    content = content + "\n\n" + tool_prompt if content else tool_prompt
                result.append({"role": "system", "content": content})
            elif m.role == "assistant":
                # Reconstruct text including tool call markers
                content = m.content or ""
                if m.tool_calls:
                    for tc in m.tool_calls:
                        call_json = json.dumps(
                            {"name": tc.name, "arguments": tc.arguments},
                            ensure_ascii=False,
                        )
                        content += f"\n{self.TOOL_CALL_START}\n{call_json}\n{self.TOOL_CALL_END}"
                result.append({"role": "assistant", "content": content})
            elif m.role == "tool":
                # Format tool results as user messages
                tool_name = m.name or "tool"
                result.append({
                    "role": "user",
                    "content": f"[Tool result from {tool_name}]:\n{m.content}",
                })
            else:
                result.append({"role": m.role, "content": m.content or ""})

        # Merge consecutive same-role messages (some models require alternation)
        merged = []
        for msg in result:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append(msg)

        return merged

    def _parse_tool_calls(self, text: str) -> tuple:
        """Parse tool calls from model text output.

        Returns (remaining_text, list_of_ToolCall).
        """
        import re

        tool_calls = []
        remaining = text

        # Find all <tool_call>...</tool_call> blocks
        pattern = re.compile(
            rf"{re.escape(self.TOOL_CALL_START)}\s*(.*?)\s*{re.escape(self.TOOL_CALL_END)}",
            re.DOTALL,
        )

        for match in pattern.finditer(text):
            raw = match.group(1).strip()
            try:
                parsed = json.loads(raw)
                name = parsed.get("name", "")
                arguments = parsed.get("arguments", parsed.get("params", {}))
                self._call_counter += 1
                tool_calls.append(ToolCall(
                    id=f"text_call_{self._call_counter}",
                    name=name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                ))
            except json.JSONDecodeError:
                # Try to salvage — look for JSON-like content
                try:
                    # Sometimes models wrap in markdown code blocks
                    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        name = parsed.get("name", "")
                        arguments = parsed.get("arguments", parsed.get("params", {}))
                        self._call_counter += 1
                        tool_calls.append(ToolCall(
                            id=f"text_call_{self._call_counter}",
                            name=name,
                            arguments=arguments if isinstance(arguments, dict) else {},
                        ))
                except (json.JSONDecodeError, AttributeError):
                    pass

        # Remove tool call blocks from remaining text
        remaining = pattern.sub("", text).strip()

        # Also try to parse bare JSON tool calls without tags (common model behavior)
        if not tool_calls:
            # Look for {"name": "...", "arguments": {...}} patterns
            bare_pattern = re.compile(
                r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}',
            )
            for match in bare_pattern.finditer(text):
                name = match.group(1)
                try:
                    arguments = json.loads(match.group(2))
                    self._call_counter += 1
                    tool_calls.append(ToolCall(
                        id=f"text_call_{self._call_counter}",
                        name=name,
                        arguments=arguments,
                    ))
                    remaining = text[:match.start()] + text[match.end():]
                except json.JSONDecodeError:
                    pass

        return remaining, tool_calls

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        api_messages = self._build_messages(messages, tools)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        raw_text = choice.message.content or ""

        # Parse tool calls from text
        remaining_text, tool_calls = self._parse_tool_calls(raw_text)

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ModelResponse(
            content=remaining_text if remaining_text else None,
            tool_calls=tool_calls,
            raw=response,
            usage=usage,
        )

    def get_name(self) -> str:
        return self.model


def create_backend(
    backend_type: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ModelBackend:
    """Factory function to create a model backend.

    Args:
        backend_type: "openai", "anthropic", "vllm", or "text"
        model: Model name/path
        api_key: API key (uses env var if not provided)
        base_url: API base URL (for vLLM or custom endpoints)

    Backend types:
        openai     - OpenAI API with native function calling
        anthropic  - Anthropic API with native tool use
        vllm       - vLLM with native function calling (requires model support)
        text       - Text-based tool calling via prompt engineering. Works with
                     ANY model (OLMo 3, Llama, Qwen, etc.) served via an
                     OpenAI-compatible endpoint. Recommended for open models
                     without native function calling support.
    """
    if backend_type in ("openai", "vllm"):
        if backend_type == "vllm" and base_url is None:
            base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        return OpenAIBackend(model=model, api_key=api_key, base_url=base_url)
    elif backend_type == "anthropic":
        return AnthropicBackend(model=model, api_key=api_key)
    elif backend_type == "text":
        if base_url is None:
            base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        return TextToolCallingBackend(model=model, api_key=api_key, base_url=base_url)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
