from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from datetime import timedelta
from typing import Any

from mcp import ClientSession

from codewright.mcp.client import (
    CallToolResult,
    McpClient,
    ToolInfo,
    call_result_from_sdk,
    sdk_client_info,
    server_info_from_initialize,
    tool_list_from_sdk,
)

TransportFactory = Callable[[], AbstractAsyncContextManager[Any]]

_SHUTDOWN_GRACE_SEC = 5.0

class SdkMcpSession:

    def __init__(
        self,
        *,
        server_name: str,
        open_transport: TransportFactory,
        read_timeout_sec: float | None,
    ) -> None:
        self._server_name = server_name
        self._open_transport = open_transport
        self._read_timeout_sec = read_timeout_sec
        self._runner: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._close = asyncio.Event()
        self._session: ClientSession | None = None
        self._server_info: dict[str, Any] = {}
        self._error: BaseException | None = None
        self._close_error: BaseException | None = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError(
                f"MCP client {self._server_name!r} not initialized"
            )
        return self._session

    async def initialize(self) -> dict[str, Any]:
        if self._runner is not None:
            return self._server_info
        self._runner = asyncio.create_task(
            self._run(), name=f"mcp_session[{self._server_name}]"
        )
        await self._ready.wait()
        if self._error is not None:
            raise _surface(self._error)
        return self._server_info

    async def _run(self) -> None:
        try:
            async with AsyncExitStack() as stack:
                streams = await stack.enter_async_context(self._open_transport())
                session = await stack.enter_async_context(
                    ClientSession(
                        streams[0],
                        streams[1],
                        read_timeout_seconds=(
                            timedelta(seconds=self._read_timeout_sec)
                            if self._read_timeout_sec is not None
                            else None
                        ),
                        client_info=sdk_client_info(),
                    )
                )
                init_result = await session.initialize()
                self._server_info = server_info_from_initialize(init_result)
                self._session = session
                self._ready.set()

                await self._close.wait()
        except BaseException as exc:
            if not self._ready.is_set():
                self._error = exc
                self._ready.set()
            elif not isinstance(exc, asyncio.CancelledError):
                self._close_error = exc
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            self._session = None

    async def shutdown(self) -> None:
        runner = self._runner
        if runner is None:
            return
        self._runner = None
        self._close.set()
        timed_out = False
        if self._ready.is_set() and self._error is None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(runner), timeout=_SHUTDOWN_GRACE_SEC
                )
            except (TimeoutError, asyncio.CancelledError):
                timed_out = True
                runner.cancel()
            except Exception:
                pass
        else:
            runner.cancel()
        if not runner.done():
            try:
                await runner
            except (asyncio.CancelledError, Exception):
                pass

        surfaced = _meaningful_close_error(self._close_error)
        self._close_error = None
        if surfaced is not None:
            raise surfaced
        if timed_out:
            raise TimeoutError(
                f"MCP server {self._server_name!r} did not shut down within "
                f"{_SHUTDOWN_GRACE_SEC:.0f}s; forced cancel (subprocess may linger)"
            )

class SdkMcpClient(McpClient):

    def __init__(
        self,
        *,
        server_name: str,
        open_transport: TransportFactory,
        read_timeout_sec: float | None,
    ) -> None:
        self._server_name = server_name
        self._sdk = SdkMcpSession(
            server_name=server_name,
            open_transport=open_transport,
            read_timeout_sec=read_timeout_sec,
        )

    @property
    def server_name(self) -> str:
        return self._server_name

    async def initialize(self) -> dict[str, Any]:
        return await self._sdk.initialize()

    async def list_tools(self) -> list[ToolInfo]:
        result = await self._sdk.session.list_tools()
        return tool_list_from_sdk(self._server_name, result)

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> CallToolResult:
        result = await self._sdk.session.call_tool(name, arguments=arguments)
        return call_result_from_sdk(result)

    async def shutdown(self) -> None:
        await self._sdk.shutdown()


def _surface(exc: BaseException) -> BaseException:

    if isinstance(exc, Exception):
        return exc
    return RuntimeError(f"MCP session failed to initialize: {exc!r}")


def _meaningful_close_error(exc: BaseException | None) -> Exception | None:

    if exc is None or isinstance(exc, asyncio.CancelledError):
        return None
    if isinstance(exc, BaseExceptionGroup):
        _cancels, rest = exc.split(asyncio.CancelledError)
        if rest is None:
            return None
        return rest if isinstance(rest, Exception) else RuntimeError(repr(rest))
    if isinstance(exc, Exception):
        return exc
    return RuntimeError(repr(exc))


__all__ = ["SdkMcpClient", "SdkMcpSession", "TransportFactory"]
