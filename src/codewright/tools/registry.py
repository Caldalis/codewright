from __future__ import annotations

from codewright.tools.errors import FatalToolError
from codewright.tools.handler import ToolHandler
from codewright.tools.spec import ToolSpec


class ToolRegistry:


    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}
        self._frozen: bool = False

    def register(self, handler: ToolHandler) -> ToolRegistry:
        if self._frozen:
            raise RuntimeError(
                f"ToolRegistry is frozen; cannot register {handler.tool_name!r} "
                "after the first turn has started. Register all handlers before "
                "submitting OpUserTurn (or before calling run_turn directly)."
            )
        name = handler.tool_name
        if name in self._handlers:
            raise ValueError(f"duplicate tool: {name}")
        self._handlers[name] = handler
        return self

    def freeze(self) -> None:
        self._frozen = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def get(self, tool_name: str) -> ToolHandler:
        try:
            return self._handlers[tool_name]
        except KeyError as exc:
            raise FatalToolError(f"unknown tool: {tool_name}") from exc

    def has(self, tool_name: str) -> bool:
        return tool_name in self._handlers

    def names(self) -> list[str]:
        return list(self._handlers.keys())

    def specs(self) -> list[ToolSpec]:
        return [h.spec() for h in self._handlers.values()]
