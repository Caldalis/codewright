from __future__ import annotations

from pydantic import Field

from codewright.protocol.agent_messages import AgentPath, AgentPathError
from codewright.tools.errors import RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec


class FollowupTaskParams(ParameterModel):
    target: str = Field(
        ...,
        description="AgentPath of the recipient, e.g. `/root/worker_api`.",
    )
    content: str = Field(
        ...,
        description=(
            "The new task / instruction. The recipient will start a fresh "
            "turn to execute it."
        ),
    )


class FollowupTaskHandler(ToolHandler):
    @property
    def tool_name(self) -> str:
        return "followup_task"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description=(
                "Send a new task to another agent and force it to start a "
                "turn to execute it. Use this for the next step of an "
                "ongoing investigation, or to push a refined instruction "
                "into a worker that just completed a previous task."
            ),
            parameters=FollowupTaskParams.to_json_schema(),
            supports_parallel=False,
            requires_approval=False,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        try:
            params = FollowupTaskParams(**invocation.arguments)
        except Exception as exc:
            raise RespondToModelError(
                f"followup_task: invalid arguments: {exc}"
            ) from exc
        try:
            target_path = AgentPath.parse(params.target)
        except AgentPathError as exc:
            raise RespondToModelError(
                f"followup_task: invalid target path {params.target!r}: {exc}"
            ) from exc
        control = invocation.session.agent_control
        if not control.has(target_path):
            raise RespondToModelError(
                f"followup_task: no live agent at {target_path}"
            )
        if control.get_info(target_path).status == "closed":
            # A closed agent's submission loop has stopped; a new turn would
            # never run (and would silently flip its status back to running).
            raise RespondToModelError(
                f"followup_task: agent {target_path} is closed; spawn a new "
                "agent instead"
            )
        await control.followup_task(
            target_path, params.content, author=invocation.session.agent_path
        )
        return ToolResult(
            success=True,
            body=f"queued followup task for {target_path}; new turn started.",
        )
