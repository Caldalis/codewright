from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import TYPE_CHECKING

from codewright.llm.base import CanonicalMessage

if TYPE_CHECKING:
    from codewright.agent.turn_context import TurnContext


_PROMPTS_DIR = Path(__file__).resolve().parent
_DEFAULT_SYSTEM_PROMPT_PATH = _PROMPTS_DIR / "system.md"
_APPLY_PATCH_INSTRUCTIONS_PATH = _PROMPTS_DIR / "apply_patch_tool_instructions.md"
_ROLES_DIR = _PROMPTS_DIR / "roles"


def load_default_system_prompt() -> str:
    return _DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def load_apply_patch_instructions() -> str:
    if not _APPLY_PATCH_INSTRUCTIONS_PATH.exists():
        return ""
    return _APPLY_PATCH_INSTRUCTIONS_PATH.read_text(encoding="utf-8")


def _read_role_snippet(role: str) -> str:
    path = _ROLES_DIR / f"{role}.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_role_text(role: str) -> str:
    """Public reader for a role's markdown snippet (e.g. the distiller prompt)."""
    return _read_role_snippet(role)


def _render_developer_layer(
    turn_context: TurnContext,
    agents_md: str | None,
    learned_facts: str | None = None,
) -> str:
    parts: list[str] = []
    parts.append(
        "<turn_settings>\n"
        f"permission_profile: {turn_context.permission_profile.value}\n"
        f"approval_policy: {turn_context.approval_policy.value}\n"
        f"cwd: {turn_context.cwd}\n"
        f"model: {turn_context.model}\n"
        f"role: {turn_context.role}\n"
        "</turn_settings>"
    )

    apply_patch_help = load_apply_patch_instructions()
    if apply_patch_help.strip():
        parts.append(
            "<apply_patch_instructions>\n"
            f"{apply_patch_help.strip()}\n"
            "</apply_patch_instructions>"
        )

    role_snippet = _read_role_snippet(turn_context.role)
    if role_snippet.strip():
        parts.append(f"<role_instructions>\n{role_snippet.strip()}\n</role_instructions>")

    if agents_md and agents_md.strip():
        parts.append(f"<user_instructions>\n{agents_md.strip()}\n</user_instructions>")

    if learned_facts and learned_facts.strip():
        parts.append(f"<learned_facts>\n{learned_facts.strip()}\n</learned_facts>")

    return "\n\n".join(parts)


def _render_environment_block(turn_context: TurnContext) -> str:
    today = _dt.date.today().isoformat()
    shell = os.environ.get("SHELL") or os.environ.get("COMSPEC") or "unknown"
    return (
        "<environment_context>\n"
        f"cwd: {turn_context.cwd}\n"
        f"date: {today}\n"
        f"shell: {shell}\n"
        "</environment_context>"
    )


class PromptBuilder:


    def __init__(self, system_template: str) -> None:
        self._system_template = system_template

    def build(
        self,
        turn_context: TurnContext,
        history: tuple[CanonicalMessage, ...] | list[CanonicalMessage],
        user_input: str,
        *,
        agents_md: str | None = None,
        learned_facts: str | None = None,
    ) -> list[CanonicalMessage]:
        messages: list[CanonicalMessage] = [
            CanonicalMessage(role="system", content=self._system_template),
            CanonicalMessage(
                role="developer",
                content=_render_developer_layer(turn_context, agents_md, learned_facts),
            ),
        ]
        messages.extend(history)
        if user_input:
            wrapped = f"{user_input}\n\n{_render_environment_block(turn_context)}"
            messages.append(CanonicalMessage(role="user", content=wrapped))
        return messages
