from __future__ import annotations

from pydantic import Field

from codewright.protocol.agent_messages import AgentPath, AgentPathError
from codewright.tools.errors import RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec


class SendMessageParams(ParameterModel):
    target: str = Field(
        ...,
        description=(
            "AgentPath of the recipient, e.g. `/root/explorer_db`. The "
            "message is queued and shown to the recipient at the start of "
            "its next turn; it does NOT cause the recipient to start a new "
            "turn on its own — use `followup_task` for that."
        ),
    )
    content: str = Field(..., description="The message body.")


class SendMessageHandler(ToolHandler):
    @property
    def tool_name(self) -> str:
        return "send_message"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description=(
                "Append a message to another agent's inbox without waking it. "
                "The recipient sees the message on its next turn (e.g., the "
                "one started by `followup_task`). Use this when you want to "
                "share context without forcing a turn."
            ),
            parameters=SendMessageParams.to_json_schema(),
            supports_parallel=False,
            requires_approval=False,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        try:
            params = SendMessageParams(**invocation.arguments)
        except Exception as exc:
            raise RespondToModelError(
                f"send_message: invalid arguments: {exc}"
            ) from exc
        try:
            target_path = AgentPath.parse(params.target)
        except AgentPathError as exc:
            raise RespondToModelError(
                f"send_message: invalid target path {params.target!r}: {exc}"
            ) from exc
        control = invocation.session.agent_control
        if not control.has(target_path):
            raise RespondToModelError(
                f"send_message: no live agent at {target_path}"
            )
        await control.send_message(
            target_path, params.content, author=invocation.session.agent_path
        )
        return ToolResult(success=True, body=f"queued message for {target_path}")
