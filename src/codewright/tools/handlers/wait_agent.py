from __future__ import annotations

from pydantic import Field

from codewright.protocol.agent_messages import AgentPath, AgentPathError
from codewright.tools.errors import RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec


_DEFAULT_WAIT_TIMEOUT_MS = 300_000

class WaitAgentParams(ParameterModel):
    targets: list[str] = Field(
        ...,
        description=(
            "AgentPath strings to wait on (e.g. `[\"/root/explorer_db\", "
            "\"/root/worker_api\"]`). Returns when at least one reaches a "
            "terminal status (completed / failed / closed)."
        ),
    )
    timeout_ms: int | None = Field(
        None,
        description=(
            "Optional timeout in milliseconds. If omitted, a default cap is "
            "applied so the wait can never block forever. When the timeout "
            "elapses the tool returns whatever agents are terminal at that "
            "moment, which may be empty."
        ),
    )

class WaitAgentHandler(ToolHandler):
    @property
    def tool_name(self) -> str:
        return "wait_agent"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description=(
                "Block the current turn until at least one of the listed "
                "subagents reaches a terminal status (completed / failed / "
                "closed). Returns each terminal agent's path + status + last "
                "assistant message. Safe to call in parallel with other "
                "read-only tools."
            ),
            parameters=WaitAgentParams.to_json_schema(),
            supports_parallel=True,
            requires_approval=False,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        try:
            params = WaitAgentParams(**invocation.arguments)
        except Exception as exc:
            raise RespondToModelError(
                f"wait_agent: invalid arguments: {exc}"
            ) from exc
        paths: list[AgentPath] = []
        for s in params.targets:
            try:
                paths.append(AgentPath.parse(s))
            except AgentPathError as exc:
                raise RespondToModelError(
                    f"wait_agent: invalid target {s!r}: {exc}"
                ) from exc
        control = invocation.session.agent_control
        self_path = str(invocation.session.agent_path)
        for p in paths:
            if str(p) == self_path:
                raise RespondToModelError(
                    f"wait_agent: cannot wait on yourself ({p})"
                )
            if not control.has(p):
                raise RespondToModelError(f"wait_agent: no live agent at {p}")
        timeout_ms = (
            params.timeout_ms
            if params.timeout_ms is not None
            else _DEFAULT_WAIT_TIMEOUT_MS
        )
        terminal = await control.wait_agent(paths, timeout_ms)
        if not terminal:
            return ToolResult(
                success=True,
                body="wait_agent: timeout elapsed; no agents terminal yet.",
            )
        lines = [
            f"- {info.path} [{info.status}]: "
            f"{(info.last_message or '').strip() or '<no message>'}"
            for info in terminal
        ]
        return ToolResult(
            success=True,
            body="\n".join(lines),
            structured_data={
                "terminal": [
                    {
                        "path": str(info.path),
                        "status": info.status,
                        "last_message": info.last_message,
                        "role": info.role,
                    }
                    for info in terminal
                ]
            },
        )
