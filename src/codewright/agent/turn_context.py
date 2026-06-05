from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codewright.agent.cancellation import CancellationToken
from codewright.protocol import AskForApproval, PermissionProfile


@dataclass(frozen=True)
class TurnContext:
    turn_id: str
    cwd: Path
    model: str
    permission_profile: PermissionProfile
    approval_policy: AskForApproval
    cancellation_token: CancellationToken
    max_context_tokens: int = 128_000
    compact_threshold: float = 0.9
    role: str = "default"
