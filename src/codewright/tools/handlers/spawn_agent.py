from __future__ import annotations

from pydantic import Field

from codewright.agent.roles import RoleRegistry, load_builtin_roles
from codewright.tools.errors import RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec


class SpawnAgentParams(ParameterModel):
    role: str = Field(
        ...,
        description="Role name. See the tool description for the full list.",
    )
    task_name: str = Field(
        ...,
        description=(
            "Short kebab/snake-case identifier for this task; becomes part of "
            "the AgentPath (e.g., role=explorer, task_name=db -> "
            "/root/explorer_db). Lowercase letters, digits, underscores."
        ),
    )
    message: str = Field(
        ..., description="Initial instruction / goal for the spawned agent."
    )


class SpawnAgentHandler(ToolHandler):

    def __init__(self, role_registry: RoleRegistry | None = None) -> None:
        self._roles = role_registry or load_builtin_roles()

    @property
    def tool_name(self) -> str:
        return "spawn_agent"

    def spec(self) -> ToolSpec:
        role_lines = "\n".join(
            f"- `{role.name}`: {role.description}" for role in self._roles.all_roles()
        )
        description = (
            "Spawn a child agent to work on a sub-task in parallel. Use this "
            "when the work splits naturally (an investigation, a scoped "
            "execution) or when the same kind of work can run on several "
            "inputs at once. The child runs in its own isolated session — it "
            "does not see your history. Communicate by `followup_task` / "
            "`send_message`, and observe completion via `wait_agent` / "
            "`list_agents`.\n\nAvailable roles:\n" + role_lines + "\n\n"
            "Pick the role that best fits the sub-task."
        )
        return ToolSpec(
            name=self.tool_name,
            description=description,
            parameters=SpawnAgentParams.to_json_schema(),
            supports_parallel=False,
            requires_approval=False,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        try:
            params = SpawnAgentParams(**invocation.arguments)
        except Exception as exc:
            raise RespondToModelError(
                f"spawn_agent: invalid arguments: {exc}"
            ) from exc
        if not self._roles.has(params.role):
            available = ", ".join(r.name for r in self._roles.all_roles())
            raise RespondToModelError(
                f"spawn_agent: unknown role {params.role!r}; available: {available}"
            )
        try:
            path = await invocation.session.agent_control.spawn_agent(
                role=params.role,
                task_name=params.task_name,
                initial_message=params.message,
                parent_path=invocation.session.agent_path,
            )
        except ValueError as exc:
            raise RespondToModelError(f"spawn_agent: {exc}") from exc
        return ToolResult(
            success=True,
            body=(
                f"Spawned agent at {path}. Use `wait_agent` to await its "
                f"completion or `list_agents` to track progress."
            ),
            structured_data={"path": str(path), "role": params.role},
        )
