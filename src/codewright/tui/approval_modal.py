from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML

from codewright.protocol import (
    OpExecApprovalResponse,
    OpPatchApprovalResponse,
    PendingAction,
    ReviewDecision,
)

if TYPE_CHECKING:  # pragma: no cover
    from rich.live import Live

    from codewright.agent.session import Session


_CHAR_TO_DECISION: dict[str, ReviewDecision] = {
    "y": ReviewDecision.APPROVED,
    "s": ReviewDecision.APPROVED_FOR_SESSION,
    "n": ReviewDecision.DENIED,
    "a": ReviewDecision.ABORT,
}


def _render_action(action: PendingAction) -> str:
    head = f"[{action.kind}] {action.summary}"
    if not action.details:
        return head
    detail_lines = "\n".join(f"  {k}: {v}" for k, v in sorted(action.details.items()))
    return f"{head}\n{detail_lines}"


async def request_decision(
    session: Session,
    live: Live | None,
    request_id: str,
    action: PendingAction,
    *,
    is_exec: bool,
) -> None:

    if live is not None:
        live.stop()
    try:
        prompt = PromptSession[str]()
        print(_render_action(action))
        decision: ReviewDecision | None = None
        while decision is None:
            answer = await prompt.prompt_async(
                HTML(
                    "<b>Approve?</b> [<ansigreen>y</ansigreen>=yes, "
                    "<ansigreen>s</ansigreen>=session, "
                    "<ansired>n</ansired>=no, "
                    "<ansired>a</ansired>=abort] "
                )
            )
            decision = _CHAR_TO_DECISION.get((answer or "").strip().lower()[:1])
    finally:
        if live is not None:
            live.start()
    if is_exec:
        await session.submit(
            OpExecApprovalResponse(request_id=request_id, decision=decision)
        )
    else:
        await session.submit(
            OpPatchApprovalResponse(request_id=request_id, decision=decision)
        )


__all__ = ["request_decision"]
