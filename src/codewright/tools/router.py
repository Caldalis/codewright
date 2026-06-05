from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from codewright.agent.turn_context import TurnContext
from codewright.llm.base import ToolCallBlock
from codewright.tools.errors import RespondToModelError
from codewright.tools.invocation import ToolInvocation
from codewright.tools.registry import ToolRegistry

if TYPE_CHECKING:  # pragma: no cover
    from codewright.agent.session import Session


class ToolRouter:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def build_invocation(
        self,
        call: ToolCallBlock,
        session: Session,
        turn_context: TurnContext,
    ) -> ToolInvocation:
        args = _parse_arguments(call.arguments_json)

        if not self._registry.has(call.tool_name):
            raise RespondToModelError(f"unknown tool: {call.tool_name}")
        return ToolInvocation(
            session=session,
            turn_context=turn_context,
            call_id=call.call_id,
            tool_name=call.tool_name,
            arguments=args,
            cancellation_token=turn_context.cancellation_token.child(),
        )


def _parse_arguments(arguments_json: str) -> dict[str, Any]:
    if not arguments_json:
        return {}
    try:
        parsed = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise RespondToModelError(f"invalid tool arguments JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RespondToModelError(
            f"tool arguments must decode to an object, got {type(parsed).__name__}"
        )
    return parsed
