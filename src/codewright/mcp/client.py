from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


MCP_PROTOCOL_VERSION = "2024-11-05"
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


def _initialize_params() -> dict[str, Any]:
    return {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {
            "roots": {"listChanged": False},
        },
        "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
    }


def _parse_tool_list(
    server_name: str, result: dict[str, Any]
) -> list[ToolInfo]:
    tools_raw = result.get("tools") or []
    out: list[ToolInfo] = []
    for entry in tools_raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        schema = entry.get("inputSchema") or {"type": "object"}
        annotations = entry.get("annotations") or {}
        supports_parallel = bool(annotations.get("readOnlyHint", False))
        out.append(
            ToolInfo(
                name=name,
                description=entry.get("description"),
                input_schema=schema,
                server_name=server_name,
                supports_parallel=supports_parallel,
            )
        )
    return out


def _parse_call_result(envelope: dict[str, Any]) -> CallToolResult:
    content_raw = envelope.get("content") or []
    if not isinstance(content_raw, list):
        content_raw = []
    return CallToolResult(
        is_error=bool(envelope.get("isError", False)),
        content=list(content_raw),
        structured_content=envelope.get("structuredContent"),
    )


__all__ = [
    "CLIENT_NAME",
    "CLIENT_VERSION",
    "MCP_PROTOCOL_VERSION",
    "CallToolResult",
    "McpClient",
    "ToolInfo",
    "_initialize_params",
    "_parse_call_result",
    "_parse_tool_list",
]
