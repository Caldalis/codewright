from __future__ import annotations

from codewright.tools.handler import ToolHandler
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec


class ListAgentsParams(ParameterModel):
    """No parameters. Defined explicitly so the JSON schema is well-formed."""


class ListAgentsHandler(ToolHandler):
    @property
    def tool_name(self) -> str:
        return "list_agents"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description=(
                "Return the current subagent tree as a list of "
                "{path, role, status, last_message} entries. Safe to call "
                "in parallel."
            ),
            parameters=ListAgentsParams.to_json_schema(),
            supports_parallel=True,
            requires_approval=False,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:

        agents = invocation.session.agent_control.list_agents()
        if not agents:
            return ToolResult(success=True, body="no subagents")
        lines = [
            f"- {info.path} [{info.status}] role={info.role}: "
            f"{(info.last_message or '').strip() or '<no message yet>'}"
            for info in agents
        ]
        return ToolResult(
            success=True,
            body="\n".join(lines),
            structured_data={
                "agents": [
                    {
                        "path": str(info.path),
                        "role": info.role,
                        "task_name": info.task_name,
                        "status": info.status,
                        "last_message": info.last_message,
                    }
                    for info in agents
                ]
            },
        )
