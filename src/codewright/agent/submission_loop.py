from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from codewright.protocol import (
    AskForApproval,
    EvError,
    EvShutdownComplete,
    EvTurnCompleted,
    EvTurnStarted,
    EvWarning,
    OpCompact,
    OpExecApprovalResponse,
    OpInterAgentCommunication,
    OpInterrupt,
    OpOverrideTurnContext,
    OpPatchApprovalResponse,
    OpShutdown,
    OpUserTurn,
    Submission,
)

if TYPE_CHECKING:
    from codewright.agent.session import Session


async def submission_loop(session: Session) -> None:
    while True:
        sub: Submission = await session._tx_sub.get()
        try:
            should_exit = await _dispatch(session, sub)
        except Exception as exc:
            await session.emit_event(
                EvError(message=f"submission_loop error on {type(sub.op).__name__}: {exc}"),
                sub.id,
            )
            should_exit = False
        if should_exit:
            break


async def _dispatch(session: Session, sub: Submission) -> bool:
    op = sub.op

    if isinstance(op, OpInterrupt):
        session.cancellation_token.cancel()
        return False

    if isinstance(op, OpShutdown):
        await session.emit_event(EvShutdownComplete(), sub.id)
        return True

    if isinstance(op, OpUserTurn):
        if session.llm_enabled:
            await _drive_user_turn(session, sub, op)
        else:

            await session.emit_event(EvTurnStarted(turn_id=sub.id), sub.id)
            await session.emit_event(
                EvTurnCompleted(
                    turn_id=sub.id,
                    last_agent_message="[mock] no real LLM wired yet",
                ),
                sub.id,
            )
        return False

    if isinstance(op, OpCompact):
        await session.emit_event(
            EvWarning(message="compact not yet implemented "), sub.id
        )
        return False

    if isinstance(op, OpExecApprovalResponse):
        resolved = session._resolve_pending_approval(op.request_id, op.decision)
        if not resolved:
            await session.emit_event(
                EvWarning(
                    message=f"exec approval response for unknown request_id={op.request_id!r}"
                ),
                sub.id,
            )
        return False

    if isinstance(op, OpPatchApprovalResponse):
        resolved = session._resolve_pending_approval(op.request_id, op.decision)
        if not resolved:
            await session.emit_event(
                EvWarning(
                    message=f"patch approval response for unknown request_id={op.request_id!r}"
                ),
                sub.id,
            )
        return False

    if isinstance(op, OpInterAgentCommunication):

        from codewright.protocol import InterAgentMessage

        msg = InterAgentMessage(
            author=op.author,
            recipient=op.recipient,
            content=op.content,
            trigger_turn=op.trigger_turn,
        )
        target_session = session
        if session.has_agent_control and session.agent_control.has(op.recipient):
            target_session = session.agent_control.get_session(op.recipient)
        if op.trigger_turn:
            target_session.mailbox.push_with_wake(msg)
        else:
            target_session.mailbox.push(msg)
        return False

    if isinstance(op, OpOverrideTurnContext):
        raise NotImplementedError("override_turn_context not in MVP")

    await session.emit_event(
        EvError(message=f"unhandled Op variant: {type(op).__name__}"), sub.id
    )
    return False


async def _drive_user_turn(session: Session, sub: Submission, op: OpUserTurn) -> None:

    from codewright.agent.turn import run_turn
    from codewright.agent.turn_context import TurnContext

    user_input = "\n".join(item.text for item in op.items)
    cancellation = session.cancellation_token.child()

    context_manager = session.context
    turn_context = TurnContext(
        turn_id=uuid.uuid4().hex,
        cwd=op.cwd or session.cwd,
        model=op.model or session.model,
        permission_profile=op.permission_profile or session.permission_profile,
        approval_policy=op.approval_policy or session.approval_policy or AskForApproval.ON_REQUEST,
        cancellation_token=cancellation,
        max_context_tokens=context_manager.max_context_tokens,
        compact_threshold=context_manager.compact_threshold,
        role=session.role,
    )

    await run_turn(session, turn_context, user_input, sub_id=sub.id)
