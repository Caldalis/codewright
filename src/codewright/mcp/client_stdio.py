from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from typing import Any

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


class StdioMcpClient(McpClient):

    _REQUEST_TIMEOUT = 60.0

    def __init__(self, config: McpServerConfig) -> None:
        if config.transport != "stdio" or not config.command:
            raise ValueError(
                f"StdioMcpClient requires stdio transport with command; got {config!r}"
            )
        self._config = config
        self._proc: asyncio.subprocess.Process | None = None
        self._stdin: asyncio.StreamWriter | None = None
        self._stdout: asyncio.StreamReader | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._ids = IdAllocator()
        self._write_lock = asyncio.Lock()
        self._closed = False

    @property
    def server_name(self) -> str:
        return self._config.name

    async def initialize(self) -> dict[str, Any]:
        await self._spawn()
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
        if self._closed:
            return
        self._closed = True

        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("MCP client shutting down"))
        self._pending.clear()
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._stdin is not None:
            try:
                self._stdin.close()
            except Exception:
                pass
        if self._proc is not None:
            try:
                self._proc.terminate()
            except (ProcessLookupError, OSError):
                pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=3.0)
            except (TimeoutError, asyncio.CancelledError):
                try:
                    self._proc.kill()
                except (ProcessLookupError, OSError):
                    pass

    async def _spawn(self) -> None:
        if self._proc is not None:
            return
        env = dict(os.environ)
        env.update(self._config.env)

        self._proc = await asyncio.create_subprocess_exec(
            self._config.command,
            *self._config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr,
            env=env,
        )
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError(
                f"MCP server {self._config.name!r} did not expose stdio pipes"
            )
        self._stdin = self._proc.stdin
        self._stdout = self._proc.stdout
        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"mcp_reader[{self._config.name}]"
        )

    async def _read_loop(self) -> None:
        assert self._stdout is not None
        while not self._closed:
            try:
                line = await self._stdout.readline()
            except (asyncio.IncompleteReadError, ConnectionResetError):
                break
            if not line:
                # EOF
                break
            try:
                envelope = parse_response(line.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            self._dispatch(envelope)

    def _dispatch(self, envelope: Mapping[str, Any]) -> None:
        rpc_id = envelope.get("id")
        if rpc_id is None:

            return
        fut = self._pending.pop(int(rpc_id), None)
        if fut is None or fut.done():
            return
        fut.set_result(dict(envelope))

    async def _request(self, method: str, params: dict[str, Any] | None) -> Any:
        if self._stdin is None:
            raise RuntimeError(f"MCP client {self._config.name!r} not initialized")
        loop = asyncio.get_running_loop()
        rpc_id = self._ids.next()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[rpc_id] = fut
        async with self._write_lock:
            self._stdin.write(encode_request(rpc_id, method, params).encode("utf-8"))
            await self._stdin.drain()
        try:
            envelope = await asyncio.wait_for(fut, timeout=self._REQUEST_TIMEOUT)
        finally:
            self._pending.pop(rpc_id, None)
        return extract_result(envelope)

    async def _notify(self, method: str, params: dict[str, Any] | None) -> None:
        if self._stdin is None:
            return
        async with self._write_lock:
            self._stdin.write(encode_notification(method, params).encode("utf-8"))
            await self._stdin.drain()


__all__ = ["StdioMcpClient"]
