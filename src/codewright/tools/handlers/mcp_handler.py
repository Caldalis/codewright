from __future__ import annotations

from codewright.mcp import McpConnectionManager
from codewright.mcp.client import ToolInfo
from codewright.tools.errors import RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ToolSpec


class McpToolHandler(ToolHandler):
    """One handler per MCP tool. Created by ``McpConnectionManager.start()``."""

    def __init__(
        self, tool_info: ToolInfo, manager: McpConnectionManager
    ) -> None:
        from codewright.mcp.handler import qualified_name

        self._info = tool_info
        self._manager = manager
        self._qualified = qualified_name(tool_info.server_name, tool_info.name)

    @property
    def tool_name(self) -> str:
        return self._qualified

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._qualified,
            description=self._info.description
            or f"MCP tool from {self._info.server_name}",
            parameters=self._info.input_schema,

            supports_parallel=self._info.supports_parallel,

            requires_approval=False,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        try:
            result = await self._manager.call(self._qualified, invocation.arguments)
        except Exception as exc:  # transport-layer error from the manager
            raise RespondToModelError(
                f"{self._qualified}: MCP call failed: {exc}"
            ) from exc

        body = result.to_text()
        if result.is_error:

            raise RespondToModelError(body or "MCP tool returned isError=true")
        structured = (
            {"structured": result.structured_content}
            if result.structured_content is not None
            else {}
        )
        return ToolResult(success=True, body=body, structured_data=structured)


__all__ = ["McpToolHandler"]
