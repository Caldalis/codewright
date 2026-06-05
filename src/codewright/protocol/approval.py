from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class ReviewDecision(StrEnum):

    APPROVED = "approved"
    APPROVED_FOR_SESSION = "approved_for_session"
    DENIED = "denied"
    ABORT = "abort"

class PermissionProfile(StrEnum):

    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    DANGEROUS = "dangerous"


PendingActionKind = Literal["exec", "patch", "network"]

@dataclass(frozen=True, slots=True)
class PendingAction:

    action_id: str
    kind: PendingActionKind
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
