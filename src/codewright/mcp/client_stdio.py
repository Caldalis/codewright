from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Any

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

from codewright.mcp._session import SdkMcpClient
from codewright.mcp.config import McpServerConfig

_REQUEST_TIMEOUT_SEC = 60.0


class StdioMcpClient(SdkMcpClient):
    """SDK-backed stdio MCP client."""

    def __init__(self, config: McpServerConfig) -> None:
        if config.transport != "stdio" or not config.command:
            raise ValueError(
                f"StdioMcpClient requires stdio transport with command; got {config!r}"
            )
        self._config = config
        super().__init__(
            server_name=config.name,
            open_transport=self._open_transport,
            read_timeout_sec=_REQUEST_TIMEOUT_SEC,
        )

    def _open_transport(self) -> AbstractAsyncContextManager[Any]:
        return stdio_client(
            StdioServerParameters(
                command=self._config.command or "",
                args=list(self._config.args),
                env=self._config.env or None,
            )
        )


__all__ = ["StdioMcpClient"]
