from __future__ import annotations

from pydantic import Field

from codewright.tools.errors import RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.handlers._file_tools import coerce_params
from codewright.tools.handlers._shell import (
    BODY_BUDGET_CHARS,
    PAGE_BYTES,
    ShellManager,
    clean_output,
    decode_output,
)
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec
from codewright.tools.truncate import truncate_middle


class ShellOutputParams(ParameterModel):
    job_id: str = Field(..., min_length=1, description="Job id returned by `shell`.")
    cursor: int = Field(
        0,
        ge=0,
        description=(
            "Byte offset to read from. Pass the next_cursor of the previous "
            "call to read incrementally; 0 re-reads from the start."
        ),
    )


class ShellOutputHandler(ToolHandler):
    def __init__(self, manager: ShellManager) -> None:
        self._manager = manager

    @property
    def tool_name(self) -> str:
        return "shell_output"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description=(
                "Read output of a `shell` job by cursor. Works for two cases: "
                "polling a background job incrementally (pass the previous "
                "next_cursor), and paging through the full output of a completed "
                "command whose inline result was truncated."
            ),
            parameters=ShellOutputParams.to_json_schema(),
            supports_parallel=True,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        params = coerce_params(ShellOutputParams, invocation.arguments, self.tool_name)
        job = self._manager.get_job(params.job_id)
        if job is None:
            raise RespondToModelError(
                f"shell_output: unknown job_id '{params.job_id}'"
            )

        try:
            with job.spill_path.open("rb") as fh:
                fh.seek(params.cursor)
                raw = fh.read(PAGE_BYTES)
        except OSError:
            raw = b""

        next_cursor = params.cursor + len(raw)
        stored_bytes = _spill_stored(job)
        eof = not job.running and next_cursor >= stored_bytes

        status = job.status if not job.running else "running"
        header = [f"Job {job.job_id} (session {job.session_name}): {status}"]
        if job.exit_code is not None:
            header.append(f"Exit code: {job.exit_code}")
        header.append(
            f"Bytes [{params.cursor}, {next_cursor}) of {stored_bytes} stored"
            + (f" ({job.spill_bytes} produced)" if job.spill_truncated else "")
        )
        if job.spill_truncated:
            header.append(
                "[output beyond the storage cap was discarded as it streamed]"
            )

        if raw:
            text = clean_output(decode_output(raw))
            if len(text) > BODY_BUDGET_CHARS:
                text = truncate_middle(text, BODY_BUDGET_CHARS)
            body = "\n".join(header) + f"\nOutput:\n{text}"
        elif job.running:
            body = "\n".join(header) + "\n(no new output yet)"
        else:
            body = "\n".join(header) + "\n(no further output)"

        return ToolResult(
            success=True,
            body=body,
            structured_data={
                "job_id": job.job_id,
                "status": status,
                "exit_code": job.exit_code,
                "cursor": params.cursor,
                "next_cursor": next_cursor,
                "stored_bytes": stored_bytes,
                "total_bytes": job.spill_bytes,
                "spill_truncated": job.spill_truncated,
                "eof": eof,
            },
        )


def _spill_stored(job) -> int:
    try:
        return job.spill_path.stat().st_size
    except OSError:
        return 0
