from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from mcp import types

CLIENT_NAME = "codewright"
CLIENT_VERSION = "0.1.0"


@dataclass(frozen=True)
class ToolInfo:

    name: str
    description: str | None
    input_schema: dict[str, Any]
    server_name: str

    supports_parallel: bool = False


@dataclass(frozen=True)
class CallToolResult:

    is_error: bool
    content: list[dict[str, Any]] = field(default_factory=list)
    structured_content: Any = None

    def to_text(self) -> str:
        out: list[str] = []
        for block in self.content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text")
                if isinstance(text, str):
                    out.append(text)
            elif btype == "resource":
                res = block.get("resource") or {}
                uri = res.get("uri") or ""
                out.append(f"<resource uri={uri!r}/>")
            else:
                out.append(f"<content type={btype!r}/>")
        return "\n".join(out) if out else ""


class McpClient(abc.ABC):
    """One client per MCP server. Owns its transport (subprocess or HTTP)."""

    @abc.abstractmethod
    async def initialize(self) -> dict[str, Any]:
        """Run the protocol handshake. Returns the server's ``serverInfo``."""

    @abc.abstractmethod
    async def list_tools(self) -> list[ToolInfo]:
        """Return the tools advertised by this server."""

    @abc.abstractmethod
    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> CallToolResult:
        """Invoke one tool. Caller passes the *bare* tool name (no server prefix)."""

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Terminate the transport. Idempotent."""


def sdk_client_info() -> types.Implementation:
    return types.Implementation(name=CLIENT_NAME, version=CLIENT_VERSION)


def server_info_from_initialize(result: types.InitializeResult) -> dict[str, Any]:
    return _model_to_dict(result.serverInfo)


def tool_list_from_sdk(
    server_name: str, result: types.ListToolsResult
) -> list[ToolInfo]:
    tools_raw = result.tools or []
    out: list[ToolInfo] = []
    for entry in tools_raw:
        name = entry.name
        if not isinstance(name, str) or not name:
            continue
        annotations = entry.annotations
        supports_parallel = bool(
            annotations.readOnlyHint if annotations is not None else False
        )
        out.append(
            ToolInfo(
                name=name,
                description=entry.description,
                input_schema=entry.inputSchema or {"type": "object"},
                server_name=server_name,
                supports_parallel=supports_parallel,
            )
        )
    return out


def call_result_from_sdk(result: types.CallToolResult) -> CallToolResult:
    return CallToolResult(
        is_error=bool(result.isError),
        content=[_model_to_dict(block) for block in result.content],
        structured_content=result.structuredContent,
    )


def _model_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, mode="json")
    if isinstance(value, dict):
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


__all__ = [
    "CLIENT_NAME",
    "CLIENT_VERSION",
    "CallToolResult",
    "McpClient",
    "ToolInfo",
    "call_result_from_sdk",
    "sdk_client_info",
    "server_info_from_initialize",
    "tool_list_from_sdk",
]
