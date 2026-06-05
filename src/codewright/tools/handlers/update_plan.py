from __future__ import annotations

from pydantic import Field

from codewright.protocol import EvPlanUpdate, PlanItem
from codewright.tools.errors import RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec


class UpdatePlanParams(ParameterModel):
    plan: list[PlanItem] = Field(
        ...,
        description="Ordered list of {step, status} dicts; status is pending|in_progress|completed",
    )
    explanation: str | None = Field(
        None, description="Short note shown alongside the plan in the UI"
    )


class UpdatePlanHandler(ToolHandler):
    @property
    def tool_name(self) -> str:
        return "update_plan"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description=(
                "Publish the agent's current plan. Each step is "
                "{step: <short imperative>, status: pending|in_progress|completed}. "
                "Call again whenever a step transitions or the plan is revised."
            ),
            parameters=UpdatePlanParams.to_json_schema(),
            supports_parallel=False,
            requires_approval=False,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        try:
            params = UpdatePlanParams(**invocation.arguments)
        except Exception as exc:
            raise RespondToModelError(
                f"update_plan: invalid arguments: {exc}"
            ) from exc

        session = invocation.session
        session.plan = list(params.plan)
        await session.emit_event(
            EvPlanUpdate(plan=list(params.plan), explanation=params.explanation)
        )
        body = "plan updated ({} item{})".format(
            len(params.plan), "" if len(params.plan) == 1 else "s"
        )
        return ToolResult(
            success=True,
            body=body,
            structured_data={
                "items": [item.model_dump() for item in params.plan],
                "explanation": params.explanation,
            },
        )
