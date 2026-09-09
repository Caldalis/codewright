"""run_shell: success, exit code, spawn failure, timeout, cancellation, cap."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import pytest

from codewright.agent.cancellation import CancellationToken
from codewright.agent.session import Session
from codewright.agent.turn_context import TurnContext
from codewright.protocol import AskForApproval, PermissionProfile
from codewright.tools.handlers.run_shell import RunShellHandler
from codewright.tools.invocation import ToolInvocation
from codewright.workspace import WorkspaceManager

_IS_WINDOWS = sys.platform == "win32"


async def _session(tmp_path: Path) -> Session:
    wm = WorkspaceManager(tmp_path, PermissionProfile.WORKSPACE_WRITE)
    sess = Session(
        session_id="rs",
        cwd=tmp_path,
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        workspace=wm,
    )
    await sess.next_event()
    return sess


def _inv(
    sess, command, cwd, cancellation=None, timeout_ms=30_000, shell=False
) -> ToolInvocation:
    return ToolInvocation(
        session=sess,
        turn_context=TurnContext(
            turn_id="t",
            cwd=cwd,
            model="m",
            permission_profile=PermissionProfile.WORKSPACE_WRITE,
            approval_policy=AskForApproval.NEVER,
            cancellation_token=CancellationToken(),
        ),
        call_id=uuid.uuid4().hex,
        tool_name="run_shell",
        arguments={"command": command, "timeout_ms": timeout_ms, "shell": shell},
        cancellation_token=cancellation or CancellationToken(),
    )


def _python_exe() -> str:
    return sys.executable or shutil.which("python") or "python"


@pytest.mark.asyncio
async def test_simple_success(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    try:
        h = RunShellHandler()
        result = await h.handle(
            _inv(sess, [_python_exe(), "-c", "print('hello')"], tmp_path)
        )
        assert result.success is True
        assert "hello" in result.body
        assert result.structured_data["exit_code"] == 0
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_nonzero_exit_reports_code(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    try:
        h = RunShellHandler()
        result = await h.handle(
            _inv(sess, [_python_exe(), "-c", "import sys; sys.exit(7)"], tmp_path)
        )
        # Tool ran cleanly; exit code is surfaced in the body.
        assert result.success is True
        assert result.structured_data["exit_code"] == 7
        assert "Exit code: 7" in result.body
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_missing_executable_returns_tool_failure(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    try:
        h = RunShellHandler()
        command = "codewright_command_that_should_not_exist_12345"
        result = await h.handle(_inv(sess, [command], tmp_path))
        assert result.success is False
        assert result.structured_data["spawn_error"] is True
        assert result.structured_data["exit_code"] is None
        assert result.structured_data["error_type"] == "FileNotFoundError"
        assert "Command failed to start" in result.body
        assert command in result.body
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_timeout_kills_process(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    try:
        h = RunShellHandler()
        start = time.monotonic()
        result = await h.handle(
            _inv(
                sess,
                [_python_exe(), "-c", "import time; time.sleep(10)"],
                tmp_path,
                timeout_ms=300,
            )
        )
        elapsed = time.monotonic() - start
        assert result.success is False
        assert result.structured_data["timed_out"] is True
        assert elapsed < 5.0
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_cancellation_quickly_returns(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    try:
        h = RunShellHandler()
        token = CancellationToken()
        task = asyncio.create_task(
            h.handle(
                _inv(
                    sess,
                    [_python_exe(), "-c", "import time; time.sleep(30)"],
                    tmp_path,
                    cancellation=token,
                    timeout_ms=30_000,
                )
            )
        )
        await asyncio.sleep(0.3)
        token.cancel()
        result = await asyncio.wait_for(task, timeout=10.0)
        assert result.structured_data["cancelled"] is True
        assert result.success is False
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
@pytest.mark.skipif(_IS_WINDOWS, reason="POSIX-only progress-group kill check")
async def test_unix_kills_grandchild(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    try:
        h = RunShellHandler()
        # Use python to spawn a grandchild and print its pid; the grandchild
        # blocks until killed. We cancel and then confirm the pid is gone.
        script = (
            "import os, sys, time;"
            "pid = os.fork();"
            "f=sys.stdout;"
            "(f.write(str(os.getpid())+'\\n') or f.flush()) if pid==0 else None;"
            "time.sleep(60) if pid==0 else (print(pid), sys.stdout.flush(), time.sleep(60))"
        )
        token = CancellationToken()
        task = asyncio.create_task(
            h.handle(
                _inv(
                    sess,
                    [_python_exe(), "-c", script],
                    tmp_path,
                    cancellation=token,
                    timeout_ms=10_000,
                )
            )
        )
        # Give the script time to print the grandchild pid.
        await asyncio.sleep(0.7)
        token.cancel()
        result = await asyncio.wait_for(task, timeout=10.0)
        body = result.body
        # The first line of stdout is the printed grandchild PID.
        grandchild = int(body.split("Output:")[1].strip().split()[0])
        # Allow a moment for the kernel to reap.
        await asyncio.sleep(0.3)
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild, 0)
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_output_cap(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    try:
        h = RunShellHandler()
        script = "import sys; sys.stdout.write('x'*200000)"
        result = await h.handle(
            _inv(sess, [_python_exe(), "-c", script], tmp_path, timeout_ms=15_000)
        )
        # The combined body must include the cap notice and stay under ~110 KB.
        assert "[output capped at 100 KB]" in result.body
        # truncate_middle keeps the cap message; total size capped.
        assert len(result.body) <= 110_000
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_stdin_devnull_does_not_block(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    try:
        h = RunShellHandler()
        # 'cat' on POSIX or 'find' on Windows that reads stdin would block
        # without DEVNULL. Use python to keep this cross-platform.
        result = await h.handle(
            _inv(
                sess,
                [_python_exe(), "-c", "import sys; print(repr(sys.stdin.read()))"],
                tmp_path,
                timeout_ms=5_000,
            )
        )
        assert result.success is True
        # stdin was closed -> read returns ''.
        assert "''" in result.body
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_shell_mode_interprets_operators(tmp_path: Path) -> None:
    # shell=True must route through the platform shell so `&&` chains both
    # commands instead of being passed as literal argv to the first program.
    sess = await _session(tmp_path)
    try:
        h = RunShellHandler()
        result = await h.handle(
            _inv(sess, ["echo alpha && echo beta"], tmp_path, shell=True)
        )
        assert result.success is True
        assert "alpha" in result.body
        assert "beta" in result.body
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
@pytest.mark.skipif(not _IS_WINDOWS, reason="PATHEXT (.cmd) resolution is Windows-only")
async def test_windows_resolves_cmd_shim_on_path(tmp_path: Path, monkeypatch) -> None:
    # A bare command name that only exists as a `.cmd` (like npm/npx/tsc) must
    # be resolved via PATHEXT; CreateProcess does not do this on its own, so
    # without resolution the spawn fails with FileNotFoundError.
    shim = tmp_path / "greet.cmd"
    shim.write_text("@echo off\r\necho resolved-via-pathext\r\n", encoding="ascii")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    sess = await _session(tmp_path)
    try:
        h = RunShellHandler()
        result = await h.handle(_inv(sess, ["greet"], tmp_path, timeout_ms=10_000))
        assert result.success is True, result.body
        assert result.structured_data["exit_code"] == 0
        assert "resolved-via-pathext" in result.body
    finally:
        await sess.shutdown()


@pytest.mark.skipif(not _IS_WINDOWS, reason="non-UTF-8 fallback decode is Windows-only")
def test_decode_output_falls_back_to_locale_encoding() -> None:
    # Output produced in the console's preferred (non-UTF-8) code page must not
    # be mangled into U+FFFD replacement characters.
    import locale

    from codewright.tools.handlers.run_shell import _decode_output

    enc = locale.getpreferredencoding(False)
    if enc.lower().replace("-", "") in ("utf8", "cp65001"):
        # UTF-8 console: valid UTF-8 must pass through untouched.
        data = "café".encode()
        expected = "café"
    else:
        # Legacy code page (e.g. cp936/cp1252): bytes that are invalid UTF-8 but
        # valid in the active code page must decode via the fallback, not U+FFFD.
        data = b"\xb0\xa1"
        expected = data.decode(enc)
    assert _decode_output(data) == expected
    assert "�" not in _decode_output(data)


@pytest.mark.skipif(not _IS_WINDOWS, reason="code-page fallback is Windows-only")
def test_decode_output_keeps_utf8_when_only_one_byte_is_bad() -> None:
    # A stray non-UTF-8 byte (or a 100 KB cap that splits a multibyte char) must
    # NOT flip the whole stream to the code page. The UTF-8 content survives and
    # only the bad spot degrades to a single U+FFFD.
    from codewright.tools.handlers.run_shell import _decode_output

    good = "日志中文输出" * 50
    data = good.encode() + b"\xff"
    out = _decode_output(data)
    assert good in out
    assert out.count("�") <= 2
