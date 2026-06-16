from __future__ import annotations

from pydantic import Field

from codewright.agent.skills import SkillRegistry
from codewright.tools.errors import RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec

_PROVISIONAL_MARK = "[未验证]"

class SkillParams(ParameterModel):
    name: str = Field(
        ...,
        description="Exact name of the skill to load (see the menu in this tool's description).",
    )

class SkillHandler(ToolHandler):

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    @property
    def tool_name(self) -> str:
        return "skill"

    def _menu_items(self):
        return [
            s
            for s in self._registry.all_skills()
            if not (s.type == "fact" and s.status == "trusted")
        ]

    def spec(self) -> ToolSpec:
        items = self._menu_items()
        if items:
            lines = []
            for s in items:
                mark = f" {_PROVISIONAL_MARK}" if s.status == "provisional" else ""
                lines.append(f"- `{s.name}`{mark}: {s.description}")
            menu = "\n".join(lines)
        else:
            menu = "(no skills available yet)"
        description = (
            "Load a skill: project-specific instructions and workflows for THIS "
            "workspace (learned automatically, or authored by hand under "
            "./skills). Skills use progressive disclosure — only the menu below "
            "is always visible; call this tool with a skill `name` to read its "
            "full instructions into context, then follow them. Load only what is "
            "relevant to the current task.\n\n"
            f"Skills marked {_PROVISIONAL_MARK} (provisional) were auto-learned "
            "from a previous task and have NOT been confirmed by successful "
            "re-use; treat their advice as a hint that may be wrong and verify it "
            "(e.g. by running tests) before relying on it.\n\n"
            "Available skills:\n" + menu
        )
        return ToolSpec(
            name=self.tool_name,
            description=description,
            parameters=SkillParams.to_json_schema(),
            supports_parallel=True,
            requires_approval=False,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        try:
            params = SkillParams(**invocation.arguments)
        except Exception as exc:
            raise RespondToModelError(f"skill: invalid arguments: {exc}") from exc

        skill = self._registry.get(params.name)
        if skill is None or skill.status == "quarantined":
            available = ", ".join(s.name for s in self._menu_items()) or "<none>"
            raise RespondToModelError(
                f"skill: unknown skill {params.name!r}; available: {available}"
            )

        body = self._registry.load_body(skill)
        self._registry.record_retrieval(skill.name)
        distillation = getattr(invocation.session, "distillation", None)
        if distillation is not None:
            distillation.note_retrieval(skill.name)
        if skill.status == "provisional":
            body += (
                f"\n\n[note] This skill is {_PROVISIONAL_MARK} (provisional): "
                "auto-learned and not yet confirmed. Verify before relying on it."
            )
        return ToolResult(
            success=True,
            body=body,
            structured_data={
                "name": skill.name,
                "status": skill.status,
                "type": skill.type,
                "base_dir": str(skill.base_dir),
            },
        )
