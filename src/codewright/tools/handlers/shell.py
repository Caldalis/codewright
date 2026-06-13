from __future__ import annotations

import uuid
from pathlib import Path

from pydantic import Field

from codewright.protocol import PendingAction, ReviewDecision
from codewright.tools.errors import FatalToolError, RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.handlers._file_tools import coerce_params
from codewright.tools.handlers._shell import (
    BODY_BUDGET_CHARS,
    DEFAULT_TIMEOUT_MS,
    ForegroundResult,
    ShellJob,
    ShellManager,
    decode_output,
    render_output,
)
from codewright.tools.handlers._shell_safety import analyze_command
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec


class ShellParams(ParameterModel):
    command: str = Field(
        ...,
        min_length=1,
        description=(
            "Full command line (may be multiline). Pipes, redirects, globs and "
            "&& work. cd / export / activated venvs persist across calls within "
            "the same session."
        ),
    )
    description: str | None = Field(
        None,
        description=(
            "One-line summary of what the command does; shown in approval "
            "prompts and audit logs, not echoed back to you."
        ),
    )
    session: str = Field(
        "main",
        pattern=r"^[A-Za-z0-9._-]{1,64}$",
        description=(
            "Named session. Each session keeps its own cwd/env state. Use "
            "separate sessions for independent workstreams."
        ),
    )
    timeout_ms: int = Field(
        DEFAULT_TIMEOUT_MS,
        ge=1,
        description=(
            "Hard timeout (ms) before the process tree is killed. Default 2 min; "
            "raise it (e.g. 600000) for long builds / installs / full test "
            "suites. Ignored when background=true."
        ),
    )
    background: bool = Field(
        False,
        description=(
            "Run without waiting: returns a job_id immediately. Use for dev "
            "servers / watchers. Poll with shell_output; stop with shell_kill. "
            "Background commands do not update session state."
        ),
    )
    restart: bool = Field(
        False,
        description="Discard this session's saved cwd/env state before running.",
    )


class ShellHandler(ToolHandler):
    def __init__(self, manager: ShellManager) -> None:
        self._manager = manager

    @property
    def tool_name(self) -> str:
        return "shell"

    async def aclose(self) -> None:
        """Kill background jobs at root-session shutdown (see Session.shutdown)."""
        await self._manager.aclose()

    def spec(self) -> ToolSpec:
        dialect = self._manager.dialect
        return ToolSpec(
            name=self.tool_name,
            description=(
                f"Run a command line with {dialect.label}. "
                "State persists across calls within a named session (each call is "
                "a fresh process; cwd, exported vars and functions are captured "
                "and replayed), and the resulting cwd is echoed in every result. "
                "stdin is closed and interactive prompts are suppressed "
                "(CI=true, GIT_TERMINAL_PROMPT=0, PAGER=cat, …) — to undo one for "
                "a session, export an overriding value. Output merges stdout+stderr "
                "in order, strips ANSI/progress noise, and is truncated head+tail; "
                "the full output of every job can be paged with shell_output. "
                "Use this for builds, tests, git, and package managers. Prefer "
                "read_file / search_text / list_dir / find_files for file "
                "inspection, and apply_patch for edits (never via this tool)."
            ),
            parameters=ShellParams.to_json_schema(),
            supports_parallel=False,
            requires_approval=True,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        params = coerce_params(ShellParams, invocation.arguments, self.tool_name)
        command = params.command.strip()
        if not command:
            raise RespondToModelError("shell: command must not be empty or blank")

        workspace = invocation.session.workspace
        key = self._manager.session_key(invocation.session.session_id, params.session)
        if params.restart:
            self._manager.reset_session(workspace.root, key)

        turn_cwd = workspace.canonicalize(invocation.turn_context.cwd)
        session_cwd = self._manager.session_cwd(workspace.root, key)
        effective_cwd = Path(session_cwd) if session_cwd else turn_cwd

        analysis = analyze_command(command)
        if analysis.backgrounded:
            raise RespondToModelError(
                "shell: '&' backgrounding is not supported — it detaches a "
                "process the tool cannot track or kill. To run a long-lived "
                "command (server / watcher), drop the '&' and pass "
                "background=true; then poll with shell_output and stop with "
                "shell_kill."
            )
        summary = command if not params.description else f"{command}  — {params.description}"
        action = PendingAction(
            action_id=uuid.uuid4().hex,
            kind="exec",
            summary=summary,
            details={
                "command": command,
                "segments": list(analysis.heads),
                "flagged": list(analysis.flagged),
                "readonly": analysis.readonly,
                "cwd": str(effective_cwd),
                "session": params.session,
                "background": params.background,
                "timeout_ms": params.timeout_ms,
            },
        )
        decision = await workspace.check_action(action, invocation.session)
        if decision == ReviewDecision.DENIED:
            raise RespondToModelError(f"shell denied by user/policy: {summary}")
        if decision == ReviewDecision.ABORT:
            raise FatalToolError("user aborted shell command")

        if params.background:
            job = await self._manager.spawn_background(
                root=workspace.root,
                session_key=key,
                session_name=params.session,
                command=command,
                start_cwd=effective_cwd,
            )
            result = self._background_result(job)
        else:
            fres = await self._manager.run_foreground(
                root=workspace.root,
                session_key=key,
                session_name=params.session,
                command=command,
                start_cwd=effective_cwd,
                timeout_ms=params.timeout_ms,
                cancellation_token=invocation.cancellation_token,
            )
            result = self._foreground_result(fres, params)

        workspace.audit(
            {
                "tool": self.tool_name,
                "call_id": invocation.call_id,
                "command": command,
                "description": params.description,
                "session": params.session,
                "segments": list(analysis.heads),
                "flagged": list(analysis.flagged),
                "cwd": str(effective_cwd),
                "background": params.background,
                "job_id": result.structured_data.get("job_id"),
                "exit_code": result.structured_data.get("exit_code"),
                "status": result.structured_data.get("status"),
            }
        )
        return result

    def _degraded_banner(self) -> str:
        if self._manager.dialect.persistent:
            return ""
        return (
            "[shell degraded: cmd.exe fallback, bash not found — cd/export do "
            "NOT persist and safety checks are limited; set CODEWRIGHT_SHELL_PATH "
            "or install Git Bash]\n"
        )

    def _background_result(self, job: ShellJob) -> ToolResult:
        if job.status == "spawn_error":
            return self._spawn_error_result(job)
        body = (
            f"Started background job {job.job_id} (session {job.session_name})\n"
            f"Command: {job.command}\n"
            f'Poll output with shell_output(job_id="{job.job_id}"); '
            f'stop with shell_kill(target="{job.job_id}").'
        )
        return ToolResult(
            success=True,
            body=self._degraded_banner() + body,
            structured_data={
                "job_id": job.job_id,
                "status": "running",
                "session": job.session_name,
                "background": True,
                "exit_code": None,
            },
        )

    def _foreground_result(
        self, fres: ForegroundResult, params: ShellParams
    ) -> ToolResult:
        job = fres.job
        if job.status == "spawn_error":
            return self._spawn_error_result(job)

        text = render_output(job.head_buf, job.tail_buf, job.dropped_middle)
        truncated = job.dropped_middle or len(text) >= BODY_BUDGET_CHARS - 100
        header = [
            f"Exit code: {job.exit_code}",
            f"Wall time: {job.wall_time_s:.3f}s",
            f"Cwd: {fres.cwd}",
        ]
        if params.session != "main":
            header.append(f"Session: {params.session}")

        if job.status == "timeout":
            header.insert(
                0,
                f"Timed out after {params.timeout_ms} ms; process tree killed.\n"
                "Hint: for a legitimately long command (build / install / test "
                "suite), retry with a larger timeout_ms; for a server / watcher, "
                "use background=true.",
            )
        elif job.status == "killed":
            header.insert(0, "Cancelled by user.")

        if text.strip():
            body = "\n".join(header) + f"\nOutput:\n{text}"
        elif job.status == "done" and job.exit_code == 0:
            body = "\n".join(header) + "\n(command succeeded, no output)"
        else:
            body = "\n".join(header) + "\n(no output)"
        if truncated:
            body += (
                f"\n[output: {job.spill_bytes} bytes total, head+tail shown; "
                f'page the full output with shell_output(job_id="{job.job_id}")]'
            )

        return ToolResult(
            success=job.status == "done",
            body=self._degraded_banner() + body,
            structured_data={
                "job_id": job.job_id,
                "status": job.status,
                "exit_code": job.exit_code,
                "wall_time_s": job.wall_time_s,
                "cwd": fres.cwd,
                "session": params.session,
                "background": False,
                "timed_out": job.status == "timeout",
                "cancelled": job.status == "killed",
                "output_bytes": job.spill_bytes,
                "truncated": truncated,
            },
        )

    def _spawn_error_result(self, job: ShellJob) -> ToolResult:
        error = decode_output(job.head_buf)
        body = (
            "Command failed to start\n"
            f"Command: {job.command}\n"
            f"Error: {error}"
        )
        return ToolResult(
            success=False,
            body=self._degraded_banner() + body,
            structured_data={
                "job_id": job.job_id,
                "status": "spawn_error",
                "exit_code": None,
                "spawn_error": True,
            },
        )
