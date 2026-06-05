from __future__ import annotations

from pydantic import Field

from codewright.protocol.agent_messages import AgentPath, AgentPathError
from codewright.tools.errors import RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec


class CloseAgentParams(ParameterModel):
    target: str = Field(
        ...,
        description=(
            "AgentPath of the subagent to shut down (e.g. `/root/worker_api`)."
        ),
    )


class CloseAgentHandler(ToolHandler):
    @property
    def tool_name(self) -> str:
        return "close_agent"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description=(
                "Shut down a subagent. The agent's submission loop stops "
                "after its current turn; pending mailbox messages are "
                "discarded. Safe to call on an already-closed agent."
            ),
            parameters=CloseAgentParams.to_json_schema(),
            supports_parallel=False,
            requires_approval=False,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        try:
            params = CloseAgentParams(**invocation.arguments)
        except Exception as exc:
            raise RespondToModelError(
                f"close_agent: invalid arguments: {exc}"
            ) from exc
        try:
            target_path = AgentPath.parse(params.target)
        except AgentPathError as exc:
            raise RespondToModelError(
                f"close_agent: invalid target {params.target!r}: {exc}"
            ) from exc
        control = invocation.session.agent_control
        if not control.has(target_path):
            raise RespondToModelError(f"close_agent: no live agent at {target_path}")
        await control.close_agent(target_path)
        return ToolResult(success=True, body=f"closed {target_path}")
