from __future__ import annotations

from typing import Any

import httpx

from codewright.mcp._jsonrpc import (
    IdAllocator,
    encode_notification,
    encode_request,
    extract_result,
    parse_response,
)
from codewright.mcp.client import (
    CallToolResult,
    McpClient,
    ToolInfo,
    _initialize_params,
    _parse_call_result,
    _parse_tool_list,
)
from codewright.mcp.config import McpServerConfig


class HttpMcpClient(McpClient):

    _REQUEST_TIMEOUT = 60.0

    def __init__(
        self,
        config: McpServerConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if config.transport != "streamable_http" or not config.url:
            raise ValueError(
                f"HttpMcpClient requires streamable_http transport with url; got {config!r}"
            )
        self._config = config
        self._http = http_client or httpx.AsyncClient(timeout=self._REQUEST_TIMEOUT)
        self._owns_http = http_client is None
        self._ids = IdAllocator()
        self._session_id: str | None = None

    @property
    def server_name(self) -> str:
        return self._config.name

    async def initialize(self) -> dict[str, Any]:
        result = await self._request("initialize", _initialize_params())
        await self._notify("notifications/initialized", None)
        if isinstance(result, dict):
            info = result.get("serverInfo")
            if isinstance(info, dict):
                return info
        return {}

    async def list_tools(self) -> list[ToolInfo]:
        result = await self._request("tools/list", {})
        if not isinstance(result, dict):
            return []
        return _parse_tool_list(self._config.name, result)

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> CallToolResult:
        result = await self._request(
            "tools/call", {"name": name, "arguments": arguments}
        )
        if not isinstance(result, dict):
            return CallToolResult(is_error=True, content=[])
        return _parse_call_result(result)

    async def shutdown(self) -> None:
        if self._owns_http:
            try:
                await self._http.aclose()
            except Exception:
                pass



    async def _request(self, method: str, params: dict[str, Any] | None) -> Any:
        rpc_id = self._ids.next()
        body = encode_request(rpc_id, method, params)
        envelope = await self._post(body)
        return extract_result(envelope)

    async def _notify(self, method: str, params: dict[str, Any] | None) -> None:
        body = encode_notification(method, params)
        try:
            await self._http.post(
                self._config.url or "", headers=self._headers(), content=body
            )
        except httpx.HTTPError:
            pass

    async def _post(self, body: str) -> dict[str, Any]:
        url = self._config.url or ""
        response = await self._http.post(
            url, headers=self._headers(), content=body
        )
        response.raise_for_status()

        if (sid := response.headers.get("Mcp-Session-Id")):
            self._session_id = sid
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return _read_first_data_event(response.text)
        return parse_response(response.text.strip())

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        token = self._config.bearer_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers


def _read_first_data_event(sse_text: str) -> dict[str, Any]:
    for raw in sse_text.splitlines():
        raw = raw.strip()
        if not raw or not raw.startswith("data:"):
            continue
        payload = raw[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        return parse_response(payload)
    raise ValueError("SSE response had no data: events")


__all__ = ["HttpMcpClient"]
