from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from codewright.mcp.client import CallToolResult, McpClient, ToolInfo
from codewright.mcp.client_http import HttpMcpClient
from codewright.mcp.client_stdio import StdioMcpClient
from codewright.mcp.config import McpServerConfig

QUALIFIED_NAME_SEPARATOR = "__"

WarningSink = Callable[[str], Awaitable[None] | None]


def qualified_name(server_name: str, tool_name: str) -> str:
    return f"{server_name}{QUALIFIED_NAME_SEPARATOR}{tool_name}"


def split_qualified_name(qualified: str) -> tuple[str, str]:
    server, sep, tool = qualified.partition(QUALIFIED_NAME_SEPARATOR)
    if not sep or not tool:
        raise ValueError(
            f"invalid MCP tool name {qualified!r}; expected <server>__<tool>"
        )
    return server, tool


class McpConnectionManager:

    def __init__(
        self,
        configs: list[McpServerConfig],
        *,
        client_factory: Callable[[McpServerConfig], McpClient] | None = None,
    ) -> None:
        self._configs = list(configs)
        self._client_factory = client_factory or _default_factory
        self._clients: dict[str, McpClient] = {}
        self._tools: list[ToolInfo] = []
        self._started = False
        self._on_warning: WarningSink | None = None

    @property
    def server_names(self) -> list[str]:
        return [c.name for c in self._configs]

    async def start(self, on_warning: WarningSink | None = None) -> None:

        if self._started:
            return
        self._started = True
        self._on_warning = on_warning
        if not self._configs:
            return

        async def _bring_up(
            cfg: McpServerConfig,
        ) -> tuple[str, McpClient | None, list[ToolInfo]]:
            client = self._client_factory(cfg)
            try:

                async with asyncio.timeout(cfg.startup_timeout_sec):
                    await client.initialize()
                    tools = await client.list_tools()
            except (Exception, BaseExceptionGroup) as exc:
                # Isolate this server's startup failure
                await _emit_warning(
                    on_warning,
                    f"[mcp] server {cfg.name!r} failed to start: {exc}",
                )
                try:
                    await client.shutdown()
                except Exception:
                    pass
                return cfg.name, None, []
            return cfg.name, client, tools

        tasks: list[asyncio.Task[tuple[str, McpClient | None, list[ToolInfo]]]] = []
        async with asyncio.TaskGroup() as tg:
            for cfg in self._configs:
                tasks.append(tg.create_task(_bring_up(cfg)))

        for task in tasks:
            name, client, tools = task.result()
            if client is None:
                continue
            self._clients[name] = client
            self._tools.extend(tools)

    def all_tools(self) -> list[ToolInfo]:
        return list(self._tools)

    def get_client(self, server_name: str) -> McpClient:
        if server_name not in self._clients:
            raise KeyError(f"unknown MCP server: {server_name!r}")
        return self._clients[server_name]

    async def call(
        self, qualified: str, arguments: dict[str, Any]
    ) -> CallToolResult:
        server, tool = split_qualified_name(qualified)
        client = self.get_client(server)
        return await client.call_tool(tool, arguments)

    async def shutdown(self) -> None:
        if not self._clients:
            return
        await asyncio.gather(
            *(self._safe_shutdown(c) for c in self._clients.values()),
            return_exceptions=False,
        )
        self._clients.clear()
        self._tools.clear()

    async def _safe_shutdown(self, client: McpClient) -> None:
        try:
            await client.shutdown()
        except Exception as exc:
            await _emit_warning(
                self._on_warning,
                f"[mcp] server {client.server_name!r} shutdown error: {exc}",
            )


    async def list_mcp_resources(self, server_name: str) -> list[dict[str, Any]]:
        raise NotImplementedError("mcp resources not implemented")

    async def read_mcp_resource(
        self, server_name: str, uri: str
    ) -> dict[str, Any]:
        raise NotImplementedError("mcp resources not implemented")


def _default_factory(cfg: McpServerConfig) -> McpClient:
    if cfg.transport == "stdio":
        return StdioMcpClient(cfg)
    if cfg.transport == "streamable_http":
        return HttpMcpClient(cfg)
    raise ValueError(f"unknown MCP transport: {cfg.transport!r}")


async def _emit_warning(sink: WarningSink | None, message: str) -> None:
    if sink is None:
        return
    out = sink(message)
    if asyncio.iscoroutine(out):
        await out


__all__ = [
    "QUALIFIED_NAME_SEPARATOR",
    "McpConnectionManager",
    "WarningSink",
    "qualified_name",
    "split_qualified_name",
]
