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


def create_backend(
    backend_type: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ModelBackend:
    """Factory function to create a model backend.

    Args:
        backend_type: "openai", "anthropic", or "vllm"
        model: Model name/path
        api_key: API key (uses env var if not provided)
        base_url: API base URL (for vLLM or custom endpoints)
    """
    if backend_type in ("openai", "vllm"):
        if backend_type == "vllm" and base_url is None:
            base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        return OpenAIBackend(model=model, api_key=api_key, base_url=base_url)
    elif backend_type == "anthropic":
        return AnthropicBackend(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
