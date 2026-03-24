"""Benchmark environment adapters.

Each module wraps an existing benchmark's evaluation harness behind
the common Environment interface.
"""

from tool_eval.environments.registry import ENVIRONMENTS, get_environment

__all__ = ["ENVIRONMENTS", "get_environment"]
