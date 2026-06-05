from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codewright.protocol import PendingAction, PermissionProfile, ReviewDecision
from codewright.workspace.permissions import action_signature, assess_action

if TYPE_CHECKING:  # pragma: no cover
    from codewright.agent.session import Session

_AGENTS_MD_CANDIDATES: tuple[str, ...] = ("AGENTS.md", ".agents.md")
_AUDIT_FILE_REL = Path(".codewright") / "audit.jsonl"


from codewright.tools.errors import RespondToModelError  # noqa: E402


class WorkspaceManager:

    def __init__(self, root: Path, permission_profile: PermissionProfile) -> None:
        self._root = Path(root).resolve(strict=False)
        self._profile = permission_profile
        self._session_approvals: set[str] = set()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def permission_profile(self) -> PermissionProfile:
        return self._profile



    def canonicalize(self, path: str | Path) -> Path:

        raw = Path(path)
        if not raw.is_absolute():
            raw = self._root / raw

        resolved = raw.resolve(strict=False)
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise RespondToModelError(
                f"path escapes workspace root: {path!s} -> {resolved}"
            ) from exc
        return resolved


    def resolve_agents_md(self, cwd: Path | None = None) -> str | None:

        try:
            start = (Path(cwd).resolve(strict=False) if cwd else self._root)
        except OSError:
            return None
        try:
            start.relative_to(self._root)
        except ValueError:
            return None

        current = start
        while True:
            for name in _AGENTS_MD_CANDIDATES:
                candidate = current / name
                if candidate.is_file():
                    try:
                        return candidate.read_text(encoding="utf-8")
                    except OSError:
                        return None
            if current == self._root:
                return None
            parent = current.parent
            if parent == current:
                return None
            current = parent


    async def check_action(
        self,
        action: PendingAction,
        session: Session,
    ) -> ReviewDecision:
        cwd = Path(action.details.get("cwd") or self._root)
        verdict = assess_action(
            self._profile,
            action,
            self._session_approvals,
            cwd,
            self._root,
        )
        if verdict == "auto_allow":
            return ReviewDecision.APPROVED
        if verdict == "auto_deny":
            return ReviewDecision.DENIED
        decision = await session.request_approval(action)
        if decision == ReviewDecision.APPROVED_FOR_SESSION:
            self._session_approvals.add(action_signature(action))
        return decision


    def audit(self, event: dict) -> None:

        record = {
            "ts": datetime.now(UTC).isoformat(),
            **event,
        }
        path = self._root / _AUDIT_FILE_REL
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            return
