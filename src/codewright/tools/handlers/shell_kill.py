from __future__ import annotations

from pydantic import Field

from codewright.tools.errors import RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.handlers._file_tools import coerce_params
from codewright.tools.handlers._shell import ShellManager
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec


class ShellKillParams(ParameterModel):
    target: str = Field(
        ...,
        min_length=1,
        description=(
            "A job_id returned by `shell`, or a session name — the latter kills "
            "every running job started from that session."
        ),
    )

class ShellKillHandler(ToolHandler):
    def __init__(self, manager: ShellManager) -> None:
        self._manager = manager

    @property
    def tool_name(self) -> str:
        return "shell_kill"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description=(
                "Kill a running `shell` job (by job_id) or all running jobs of a "
                "session (by session name). The whole process tree is terminated."
            ),
            parameters=ShellKillParams.to_json_schema(),
            supports_parallel=True,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        params = coerce_params(ShellKillParams, invocation.arguments, self.tool_name)

        job = self._manager.get_job(params.target)
        if job is not None:
            killed = self._manager.kill_job(job)
            body = (
                f"Killed job {job.job_id} ({job.command})"
                if killed
                else f"Job {job.job_id} was already finished ({job.status})"
            )
            return ToolResult(
                success=True,
                body=body,
                structured_data={
                    "job_id": job.job_id,
                    "killed": killed,
                    "status": job.status,
                },
            )

        key = self._manager.session_key(
            invocation.session.session_id, params.target
        )
        jobs = self._manager.jobs_for_session(key)
        if not jobs:
            raise RespondToModelError(
                f"shell_kill: '{params.target}' matches no job_id and no session "
                "with jobs"
            )
        killed_ids = [j.job_id for j in jobs if self._manager.kill_job(j)]
        body = (
            f"Killed {len(killed_ids)} running job(s) in session "
            f"'{params.target}': {', '.join(killed_ids) or 'none'}"
        )
        return ToolResult(
            success=True,
            body=body,
            structured_data={
                "session": params.target,
                "killed_job_ids": killed_ids,
            },
        )
