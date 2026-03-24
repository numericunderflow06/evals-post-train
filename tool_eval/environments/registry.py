"""Benchmark environment registry."""

from typing import Dict, Type

from tool_eval.environment import Environment


ENVIRONMENTS: Dict[str, Type[Environment]] = {}


def register(name: str):
    """Decorator to register an environment class."""
    def decorator(cls):
        ENVIRONMENTS[name] = cls
        return cls
    return decorator


def get_environment(name: str, **kwargs) -> Environment:
    """Create an environment instance by name."""
    if name not in ENVIRONMENTS:
        available = ", ".join(sorted(ENVIRONMENTS.keys()))
        raise ValueError(
            f"Unknown environment: {name}. Available: {available}"
        )
    return ENVIRONMENTS[name](**kwargs)
