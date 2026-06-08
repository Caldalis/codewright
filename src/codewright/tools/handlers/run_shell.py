from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from pydantic import Field

from codewright.protocol import PendingAction, ReviewDecision
from codewright.tools.errors import FatalToolError, RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec
from codewright.tools.truncate import truncate_middle

_DEFAULT_TIMEOUT_MS = 30_000
_OUTPUT_CAP_BYTES = 100 * 1024  # 100 KB total
_DRAIN_GRACE_SECONDS = 2.0
_IS_WINDOWS = sys.platform == "win32"


class RunShellParams(ParameterModel):
    command: list[str] = Field(..., description="argv list to spawn (not parsed by a shell)")
    cwd: str | None = Field(
        None, description="working directory (relative paths resolve against workspace root)"
    )
    timeout_ms: int = Field(
        _DEFAULT_TIMEOUT_MS, ge=1, description="hard timeout before the child tree is killed"
    )


class RunShellHandler(ToolHandler):
    @property
    def tool_name(self) -> str:
        return "run_shell"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description=(
                "Spawn an external command as a child process. argv is passed "
                "directly to the OS — no shell expansion. Use this tool for "
                "build / test / file inspection. To edit files, use apply_patch."
            ),
            parameters=RunShellParams.to_json_schema(),
            supports_parallel=False,
            requires_approval=True,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        params = _coerce_params(invocation.arguments)
        if not params.command:
            raise RespondToModelError("run_shell: command must not be empty")

        workspace = invocation.session.workspace
        if params.cwd:
            cwd = workspace.canonicalize(params.cwd)
        else:
            cwd = workspace.canonicalize(invocation.turn_context.cwd)

        action = PendingAction(
            action_id=uuid.uuid4().hex,
            kind="exec",
            summary=" ".join(shlex.quote(c) for c in params.command),
            details={
                "command": list(params.command),
                "cwd": str(cwd),
                "timeout_ms": params.timeout_ms,
            },
        )
        decision = await workspace.check_action(action, invocation.session)
        if decision == ReviewDecision.DENIED:
            raise RespondToModelError(
                f"run_shell denied by user/policy: {action.summary}"
            )
        if decision == ReviewDecision.ABORT:
            raise FatalToolError("user aborted run_shell")

        result = await self._exec(
            params, cwd, invocation.cancellation_token
        )
        workspace.audit(
            {
                "tool": "run_shell",
                "call_id": invocation.call_id,
                "command": list(params.command),
                "cwd": str(cwd),
                "exit_code": result.structured_data.get("exit_code"),
                "wall_time_s": result.structured_data.get("wall_time_s"),
                "cancelled": result.structured_data.get("cancelled", False),
            }
        )
        return result

    async def _exec(
        self,
        params: RunShellParams,
        cwd: Path,
        cancellation_token,
    ) -> ToolResult:
        start = time.monotonic()
        try:
            proc, pgid = await _spawn(params.command, cwd)
        except OSError as exc:
            wall_time = time.monotonic() - start
            command_text = " ".join(shlex.quote(c) for c in params.command)
            body = (
                "Command failed to start\n"
                f"Command: {command_text}\n"
                f"Error: {type(exc).__name__}: {exc}"
            )
            structured = {
                "exit_code": None,
                "wall_time_s": round(wall_time, 3),
                "cancelled": False,
                "timed_out": False,
                "spawn_error": True,
                "error_type": type(exc).__name__,
            }
            return ToolResult(success=False, body=body, structured_data=structured)

        stdout_buf: list[bytes] = []
        stderr_buf: list[bytes] = []
        cap_state = {"bytes": 0, "capped": False}

        async def drain(stream: asyncio.StreamReader | None, buf: list[bytes]) -> None:
            if stream is None:
                return
            try:
                while True:
                    chunk = await stream.read(4096)
                    if not chunk:
                        return
                    if cap_state["capped"]:
                        continue
                    remaining = _OUTPUT_CAP_BYTES - cap_state["bytes"]
                    if remaining <= 0:
                        cap_state["capped"] = True
                        continue
                    if len(chunk) > remaining:
                        buf.append(chunk[:remaining])
                        cap_state["bytes"] = _OUTPUT_CAP_BYTES
                        cap_state["capped"] = True
                    else:
                        buf.append(chunk)
                        cap_state["bytes"] += len(chunk)
            except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
                return

        stdout_task = asyncio.create_task(drain(proc.stdout, stdout_buf))
        stderr_task = asyncio.create_task(drain(proc.stderr, stderr_buf))

        wait_proc = asyncio.create_task(proc.wait())
        wait_cancel = asyncio.create_task(cancellation_token.wait())

        wait_timeout = asyncio.create_task(asyncio.sleep(params.timeout_ms / 1000.0))

        try:
            done, _pending = await asyncio.wait(
                [wait_proc, wait_cancel, wait_timeout],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for t in (wait_cancel, wait_timeout):
                if not t.done():
                    t.cancel()

        cancelled = wait_cancel in done and proc.returncode is None
        timed_out = wait_timeout in done and proc.returncode is None
        if proc.returncode is None:
            _terminate_tree(proc, pgid)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except TimeoutError:
                pass


        for task in (stdout_task, stderr_task):
            try:
                await asyncio.wait_for(task, timeout=_DRAIN_GRACE_SECONDS)
            except TimeoutError:
                task.cancel()

        wall_time = time.monotonic() - start
        combined = b"".join(stdout_buf) + b"".join(stderr_buf)
        text = combined.decode("utf-8", errors="replace")
        if cap_state["capped"]:
            text += "\n[output capped at 100 KB]"
        text = truncate_middle(text)

        if cancelled:
            body = (
                f"Cancelled by user after {wall_time:.1f}s\n"
                f"Command: {' '.join(shlex.quote(c) for c in params.command)}\n"
                f"Output:\n{text}"
            )
            structured = {
                "exit_code": -1,
                "wall_time_s": round(wall_time, 3),
                "cancelled": True,
                "timed_out": False,
            }
            return ToolResult(success=False, body=body, structured_data=structured)
        if timed_out:
            body = (
                f"Timed out after {params.timeout_ms} ms\n"
                f"Command: {' '.join(shlex.quote(c) for c in params.command)}\n"
                f"Output:\n{text}"
            )
            structured = {
                "exit_code": -1,
                "wall_time_s": round(wall_time, 3),
                "cancelled": False,
                "timed_out": True,
            }
            return ToolResult(success=False, body=body, structured_data=structured)

        exit_code = proc.returncode if proc.returncode is not None else -1
        body = (
            f"Exit code: {exit_code}\n"
            f"Wall time: {wall_time:.3f}s\n"
            f"Output:\n{text}"
        )
        structured = {
            "exit_code": exit_code,
            "wall_time_s": round(wall_time, 3),
            "cancelled": False,
            "timed_out": False,
        }

        return ToolResult(success=True, body=body, structured_data=structured)




def _coerce_params(arguments: dict) -> RunShellParams:
    try:
        return RunShellParams(**arguments)
    except Exception as exc:
        raise RespondToModelError(f"run_shell: invalid arguments: {exc}") from exc


async def _spawn(command: list[str], cwd: Path) -> tuple[asyncio.subprocess.Process, int | None]:
    """Spawn ``command`` with stdin DEVNULL and an isolated process group."""
    kwargs: dict = {
        "cwd": str(cwd),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if _IS_WINDOWS:
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        )
        proc = await asyncio.create_subprocess_exec(*command, **kwargs)
        return proc, None

    kwargs["preexec_fn"] = os.setsid
    proc = await asyncio.create_subprocess_exec(*command, **kwargs)
    return proc, proc.pid


def _terminate_tree(proc: asyncio.subprocess.Process, pgid: int | None) -> None:
    if proc.returncode is not None:
        return
    if _IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except ProcessLookupError:
                return
        return
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
    try:
        proc.kill()
    except ProcessLookupError:
        return
