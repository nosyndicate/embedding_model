"""Registry for storing and retrieving classes by name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    """A registry for storing and retrieving classes by name."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register a class."""

        def decorator(cls: type[T]) -> type[T]:
            if name in self._registry:
                raise ValueError(
                    f"Name '{name}' already registered in {self.name} registry"
                )
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        """Get a class by name."""
        if name not in self._registry:
            raise KeyError(
                f"Name '{name}' not found in {self.name} registry. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def build(self, name: str, **kwargs: Any) -> Any:
        """Build an instance by name with kwargs."""
        cls = self.get(name)
        return cls(**kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, entries={list(self._registry.keys())})"
