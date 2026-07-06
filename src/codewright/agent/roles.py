from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ReasoningEffort = Literal["low", "medium", "high"]

_ROLES_DIR = Path(__file__).resolve().parent.parent / "prompts" / "roles"


@dataclass(frozen=True)
class RoleConfig:


    name: str
    description: str
    system_prompt_path: Path
    reasoning_effort: ReasoningEffort | None = None


class RoleRegistry:

    def __init__(self) -> None:
        self._roles: dict[str, RoleConfig] = {}

    def register(self, role: RoleConfig) -> RoleRegistry:

        self._roles[role.name] = role
        return self

    def get(self, name: str) -> RoleConfig:
        try:
            return self._roles[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._roles)) or "<empty>"
            raise KeyError(
                f"unknown role: {name!r}; available: {available}"
            ) from exc

    def has(self, name: str) -> bool:
        return name in self._roles

    def all_roles(self) -> list[RoleConfig]:
        return list(self._roles.values())

    def __len__(self) -> int:
        return len(self._roles)


_BUILTIN_ROLES: tuple[RoleConfig, ...] = (
    RoleConfig(
        name="default",
        description=(
            "General-purpose coding agent. Use when the task does not clearly "
            "fit the explorer (read-only investigation) or worker "
            "(scoped execution) roles."
        ),
        system_prompt_path=_ROLES_DIR / "default.md",
    ),
    RoleConfig(
        name="explorer",
        description=(
            "Read-only investigation agent. Spawn one (or several in parallel) "
            "to understand parts of the codebase without modifying anything; "
            "the agent reports findings with explicit file path citations."
        ),
        system_prompt_path=_ROLES_DIR / "explorer.md",
    ),
    RoleConfig(
        name="worker",
        description=(
            "Scoped execution agent. Use when you have a concrete sub-task "
            "with a clear definition of done; the agent executes it and "
            "returns a one-line summary of the outcome."
        ),
        system_prompt_path=_ROLES_DIR / "worker.md",
    ),
)


def load_builtin_roles() -> RoleRegistry:
    reg = RoleRegistry()
    for role in _BUILTIN_ROLES:
        reg.register(role)
    return reg


def load_user_roles(codewright_home: Path) -> RoleRegistry:

    reg = RoleRegistry()
    roles_dir = codewright_home / "agents"
    if not roles_dir.exists():
        return reg
    for path in sorted(roles_dir.glob("*.toml")):
        try:
            with open(path, "rb") as fh:
                data = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        name = data.get("name") or path.stem
        description = data.get("description")
        prompt_text = data.get("system_prompt")
        prompt_path = data.get("system_prompt_path")
        if not isinstance(name, str) or not isinstance(description, str):
            continue

        if prompt_path is not None and isinstance(prompt_path, str):
            sp_path = Path(prompt_path)
            if not sp_path.is_absolute():
                sp_path = roles_dir / sp_path
        elif prompt_text is not None and isinstance(prompt_text, str):
            sp_path = roles_dir / f"_{name}_inline.md"
            try:
                sp_path.write_text(prompt_text, encoding="utf-8")
            except OSError:
                continue
        else:
            continue
        effort_raw = data.get("reasoning_effort")
        effort: ReasoningEffort | None = (
            effort_raw if effort_raw in ("low", "medium", "high") else None
        )
        reg.register(
            RoleConfig(
                name=name,
                description=description,
                system_prompt_path=sp_path,
                reasoning_effort=effort,
            )
        )
    return reg


__all__ = [
    "ReasoningEffort",
    "RoleConfig",
    "RoleRegistry",
    "load_builtin_roles",
    "load_user_roles",
]
