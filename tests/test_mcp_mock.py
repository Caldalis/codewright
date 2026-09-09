"""MCP: discovery + tools/call + error path + hanging-server isolation."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from codewright.mcp.client import CallToolResult
from codewright.mcp.client_http import HttpMcpClient
from codewright.mcp.client_stdio import StdioMcpClient
from codewright.mcp.config import McpServerConfig
from codewright.mcp.handler import (
    McpConnectionManager,
    qualified_name,
    split_qualified_name,
)
from codewright.tools.handlers.mcp_handler import McpToolHandler
from codewright.tools.invocation import ToolInvocation
from codewright.tools.registry import ToolRegistry

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FAKE_SERVER = FIXTURES / "fake_mcp_server.py"
HANG_SERVER = FIXTURES / "hanging_mcp_server.py"


def _fake_cfg(
    name: str = "fake",
    timeout: float = 10.0,
    *,
    pidfile: Path | None = None,
) -> McpServerConfig:
    return McpServerConfig(
        name=name,
        transport="stdio",
        command=sys.executable,
        args=(str(FAKE_SERVER),),
        env={"CW_MCP_PIDFILE": str(pidfile)} if pidfile else {},
        startup_timeout_sec=timeout,
    )


def _hang_cfg(
    name: str = "hang",
    timeout: float = 1.0,
    *,
    pidfile: Path | None = None,
) -> McpServerConfig:
    return McpServerConfig(
        name=name,
        transport="stdio",
        command=sys.executable,
        args=(str(HANG_SERVER),),
        env={"CW_MCP_PIDFILE": str(pidfile)} if pidfile else {},
        startup_timeout_sec=timeout,
    )


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True,
            text=True,
        ).stdout
        return f'"{pid}"' in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid(pidfile: Path, tries: int = 100) -> int:
    for _ in range(tries):
        try:
            text = pidfile.read_text(encoding="utf-8").strip()
            if text:
                return int(text)
        except (FileNotFoundError, ValueError):
            pass
        time.sleep(0.05)
    raise AssertionError(f"server never wrote its PID to {pidfile}")


_REAP_BUDGET_SEC = 8.0


async def _wait_until_dead(pid: int) -> bool:
    deadline = time.monotonic() + _REAP_BUDGET_SEC
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        await asyncio.sleep(0.1)
    return not _pid_alive(pid)


def _closed_local_url() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}/mcp"


class TestQualifiedName:
    def test_round_trip(self):
        assert qualified_name("github", "create_issue") == "github__create_issue"
        s, t = split_qualified_name("github__create_issue")
        assert (s, t) == ("github", "create_issue")

    def test_invalid_qualified_raises(self):
        with pytest.raises(ValueError):
            split_qualified_name("notqualified")


class TestStdioClientDirect:
    @pytest.mark.asyncio
    async def test_initialize_and_list_tools(self):
        client = StdioMcpClient(_fake_cfg())
        try:
            info = await client.initialize()
            assert info.get("name") == "fake"
            tools = await client.list_tools()
            assert {t.name for t in tools} == {"echo", "fail"}
            assert all(t.server_name == "fake" for t in tools)
        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_call_echo(self):
        client = StdioMcpClient(_fake_cfg())
        try:
            await client.initialize()
            result = await client.call_tool("echo", {"text": "hi there"})
            assert result.is_error is False
            assert result.to_text() == "hi there"
        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_call_fail_returns_error_flag(self):
        client = StdioMcpClient(_fake_cfg())
        try:
            await client.initialize()
            result = await client.call_tool("fail", {"reason": "boom"})
            assert result.is_error is True
            assert "boom" in result.to_text()
        finally:
            await client.shutdown()


class TestConnectionManager:
    @pytest.mark.asyncio
    async def test_start_then_call_dispatches_by_server(self):
        mgr = McpConnectionManager([_fake_cfg("alpha")])
        try:
            await mgr.start()
            tools = mgr.all_tools()
            assert {t.name for t in tools} == {"echo", "fail"}
            result = await mgr.call("alpha__echo", {"text": "yo"})
            assert isinstance(result, CallToolResult)
            assert result.to_text() == "yo"
        finally:
            await mgr.shutdown()

    @pytest.mark.asyncio
    async def test_hanging_server_does_not_block_others(self):
        warnings: list[str] = []

        async def warn(msg: str) -> None:
            warnings.append(msg)

        mgr = McpConnectionManager([_fake_cfg("good"), _hang_cfg("bad", 1.0)])
        try:
            await mgr.start(on_warning=warn)
            names = {t.server_name for t in mgr.all_tools()}
            assert "good" in names
            assert "bad" not in names
            assert any("bad" in w for w in warnings)
        finally:
            await mgr.shutdown()

    @pytest.mark.asyncio
    async def test_resources_stub_raises_not_implemented(self):
        mgr = McpConnectionManager([])
        try:
            await mgr.start()
            with pytest.raises(NotImplementedError):
                await mgr.list_mcp_resources("anything")
            with pytest.raises(NotImplementedError):
                await mgr.read_mcp_resource("anything", "uri")
        finally:
            await mgr.shutdown()


class TestSessionStartMcp:
    """Production wiring: Session.start_mcp discovers + registers in one call."""

    @pytest.mark.asyncio
    async def test_start_mcp_registers_handlers_and_shutdown_cleans_up(self):
        from codewright.agent.session import Session
        from codewright.protocol import PermissionProfile

        session = Session(
            session_id="s",
            cwd=Path("."),
            permission_profile=PermissionProfile.READ_ONLY,
        )
        try:
            await session.start_mcp([_fake_cfg("svc")])
            # Tools auto-registered under <server>__<tool>.
            assert session.tool_registry.has("svc__echo")
            assert session.tool_registry.has("svc__fail")
            # Manager exposed for tests / TUI introspection.
            assert session.mcp is not None
        finally:
            await session.shutdown()
        # After shutdown, manager is released and subprocess is gone.
        assert session.mcp is None

    @pytest.mark.asyncio
    async def test_start_mcp_empty_configs_is_noop(self):
        from codewright.agent.session import Session
        from codewright.protocol import PermissionProfile

        session = Session(
            session_id="s",
            cwd=Path("."),
            permission_profile=PermissionProfile.READ_ONLY,
        )
        try:
            await session.start_mcp([])
            assert session.mcp is None
            assert session.tool_registry.names() == []
        finally:
            await session.shutdown()

    @pytest.mark.asyncio
    async def test_start_mcp_failing_server_emits_warning_event(self):
        from codewright.agent.session import Session
        from codewright.protocol import EvWarning, PermissionProfile

        session = Session(
            session_id="s",
            cwd=Path("."),
            permission_profile=PermissionProfile.READ_ONLY,
        )
        try:
            await session.start_mcp(
                [_fake_cfg("good"), _hang_cfg("bad", timeout=1.0)]
            )
            # 'good' is registered; 'bad' is skipped.
            assert session.tool_registry.has("good__echo")
            assert not any(
                n.startswith("bad__") for n in session.tool_registry.names()
            )
            # The warning fires as a real EvWarning on the EQ. Drain a few
            # events to look for it (session_configured is the first event).
            saw_warning = False
            for _ in range(10):
                ev = await asyncio.wait_for(session.next_event(), timeout=1.0)
                if isinstance(ev.msg, EvWarning) and "bad" in ev.msg.message:
                    saw_warning = True
                    break
            assert saw_warning, "expected EvWarning naming 'bad' server"
        finally:
            await session.shutdown()


class TestMcpToolHandler:
    @pytest.mark.asyncio
    async def test_handler_registers_and_dispatches(self):
        mgr = McpConnectionManager([_fake_cfg("svc")])
        registry = ToolRegistry()
        try:
            await mgr.start()
            for info in mgr.all_tools():
                registry.register(McpToolHandler(info, mgr))

            assert registry.has("svc__echo")
            spec = registry.get("svc__echo").spec()
            assert spec.name == "svc__echo"
            assert spec.parameters["type"] == "object"

            handler = registry.get("svc__echo")
            from codewright.agent.cancellation import CancellationToken
            from codewright.agent.turn_context import TurnContext
            from codewright.protocol import AskForApproval, PermissionProfile

            inv = ToolInvocation(
                session=None,
                turn_context=TurnContext(
                    turn_id="t",
                    cwd=Path("."),
                    model="m",
                    permission_profile=PermissionProfile.READ_ONLY,
                    approval_policy=AskForApproval.NEVER,
                    cancellation_token=CancellationToken(),
                ),
                call_id="c",
                tool_name="svc__echo",
                arguments={"text": "world"},
                cancellation_token=CancellationToken(),
            )
            result = await handler.handle(inv)
            assert result.success is True
            assert result.body == "world"
        finally:
            await mgr.shutdown()

    @pytest.mark.asyncio
    async def test_handler_isError_raises_respond_to_model(self):
        from codewright.tools.errors import RespondToModelError

        mgr = McpConnectionManager([_fake_cfg("svc")])
        try:
            await mgr.start()
            info = next(t for t in mgr.all_tools() if t.name == "fail")
            handler = McpToolHandler(info, mgr)
            from codewright.agent.cancellation import CancellationToken
            from codewright.agent.turn_context import TurnContext
            from codewright.protocol import AskForApproval, PermissionProfile

            inv = ToolInvocation(
                session=None,
                turn_context=TurnContext(
                    turn_id="t",
                    cwd=Path("."),
                    model="m",
                    permission_profile=PermissionProfile.READ_ONLY,
                    approval_policy=AskForApproval.NEVER,
                    cancellation_token=CancellationToken(),
                ),
                call_id="c",
                tool_name="svc__fail",
                arguments={"reason": "kaboom"},
                cancellation_token=CancellationToken(),
            )
            with pytest.raises(RespondToModelError) as exc:
                await handler.handle(inv)
            assert "kaboom" in str(exc.value)
        finally:
            await mgr.shutdown()


class TestShutdownReapsSubprocess:
    """Lifecycle: the SDK session is opened on a transient task but closed from
    another task; closing must not raise and must actually reap the server."""

    @pytest.mark.asyncio
    async def test_cross_task_open_then_close_reaps_subprocess(self, tmp_path):
        pidfile = tmp_path / "pid.txt"
        holder: dict[str, StdioMcpClient] = {}

        async def bring_up() -> None:
            client = StdioMcpClient(_fake_cfg(pidfile=pidfile))
            await client.initialize()
            await client.list_tools()
            holder["client"] = client

        # Open inside a TaskGroup child task, mirroring McpConnectionManager.
        async with asyncio.TaskGroup() as tg:
            tg.create_task(bring_up())

        client = holder["client"]
        pid = _read_pid(pidfile)
        assert _pid_alive(pid)

        # Close from a *different* task than the one that opened it.
        await client.shutdown()
        assert await _wait_until_dead(pid), "stdio subprocess leaked after shutdown"

    @pytest.mark.asyncio
    async def test_manager_shutdown_reaps_subprocess(self, tmp_path):
        pidfile = tmp_path / "pid.txt"
        mgr = McpConnectionManager([_fake_cfg("svc", pidfile=pidfile)])
        await mgr.start()
        pid = _read_pid(pidfile)
        assert _pid_alive(pid)
        await mgr.shutdown()
        assert await _wait_until_dead(pid), "subprocess leaked after manager shutdown"

    @pytest.mark.asyncio
    async def test_hung_server_subprocess_is_reaped_on_startup_failure(
        self, tmp_path
    ):
        # A server that hangs in initialize() must still be terminated when the
        # startup timeout trips (it ignores stdin EOF, so close alone won't do).
        pidfile = tmp_path / "pid.txt"
        mgr = McpConnectionManager([_hang_cfg("bad", 1.0, pidfile=pidfile)])
        try:
            await mgr.start()
            assert "bad" not in {t.server_name for t in mgr.all_tools()}
            pid = _read_pid(pidfile)
            assert await _wait_until_dead(pid), "hung subprocess leaked"
        finally:
            await mgr.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_error_is_surfaced_as_warning(self):
        class _BoomClient(StdioMcpClient):
            async def shutdown(self) -> None:
                raise RuntimeError("boom on close")

        warnings: list[str] = []
        mgr = McpConnectionManager(
            [_fake_cfg("svc")],
            client_factory=lambda cfg: _BoomClient(cfg),
        )
        await mgr.start(on_warning=lambda m: warnings.append(m))
        await mgr.shutdown()
        assert any("svc" in w and "shutdown error" in w for w in warnings)


class TestHttpClient:
    """The streamable-http client shares the SDK lifecycle with stdio; here we
    cover its transport-specific bits (auth header) and the error/close path."""

    def test_headers_include_bearer_token(self, monkeypatch):
        monkeypatch.setenv("CW_TEST_TOKEN", "s3cret")
        client = HttpMcpClient(
            McpServerConfig(
                name="h",
                transport="streamable_http",
                url="http://example/mcp",
                bearer_token_env_var="CW_TEST_TOKEN",
            )
        )
        assert client._headers() == {"Authorization": "Bearer s3cret"}

    def test_headers_empty_without_token(self):
        client = HttpMcpClient(
            McpServerConfig(
                name="h", transport="streamable_http", url="http://example/mcp"
            )
        )
        assert client._headers() == {}

    @pytest.mark.asyncio
    async def test_unreachable_initialize_fails_and_shutdown_is_clean(self):
        client = HttpMcpClient(
            McpServerConfig(
                name="h",
                transport="streamable_http",
                url=_closed_local_url(),
                startup_timeout_sec=5.0,
            )
        )
        # The exact failure type (connection refused, possibly wrapped in an
        # ExceptionGroup by the transport's task group) is not a stable
        # contract, so assert only that initialize surfaces a failure rather
        # than hanging or returning.
        failure: Exception | None = None
        try:
            await asyncio.wait_for(client.initialize(), timeout=10.0)
        except Exception as exc:
            failure = exc
        assert failure is not None, "initialize against a closed port must fail"
        # Cross-task / post-failure shutdown must be clean and idempotent.
        await client.shutdown()
        await client.shutdown()
