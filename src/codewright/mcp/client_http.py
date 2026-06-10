from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from mcp.client.streamable_http import streamable_http_client

from codewright.mcp._session import SdkMcpClient
from codewright.mcp.config import McpServerConfig


_REQUEST_TIMEOUT_SEC = 60.0


class HttpMcpClient(SdkMcpClient):
    """SDK-backed streamable HTTP MCP client."""

    def __init__(self, config: McpServerConfig) -> None:
        if config.transport != "streamable_http" or not config.url:
            raise ValueError(
                f"HttpMcpClient requires streamable_http transport with url; got {config!r}"
            )
        self._config = config
        super().__init__(
            server_name=config.name,
            open_transport=self._open_transport,
            read_timeout_sec=_REQUEST_TIMEOUT_SEC,
        )

    @asynccontextmanager
    async def _open_transport(self) -> AsyncIterator[tuple[Any, Any]]:

        timeout = httpx.Timeout(_REQUEST_TIMEOUT_SEC, read=_REQUEST_TIMEOUT_SEC)
        async with httpx.AsyncClient(
            headers=self._headers() or None, timeout=timeout
        ) as http_client:
            async with streamable_http_client(
                self._config.url or "", http_client=http_client
            ) as (read_stream, write_stream, _get_session_id):
                yield read_stream, write_stream

    def _headers(self) -> dict[str, str]:
        token = self._config.bearer_token()
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}


__all__ = ["HttpMcpClient"]
