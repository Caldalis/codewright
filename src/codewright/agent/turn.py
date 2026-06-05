from __future__ import annotations

from typing import TYPE_CHECKING

from codewright.agent.turn_context import TurnContext
from codewright.context.compact import compact_history
from codewright.llm.base import CanonicalMessage, ContentBlock, ToolCallBlock
from codewright.persistence.rollout import RolloutLine
from codewright.protocol import (
    EvAgentMessage,
    EvAgentMessageDelta,
    EvError,
    EvTokenCount,
    EvToolCallCompleted,
    EvToolCallStarted,
    EvTurnAborted,
    EvTurnCompleted,
    EvTurnStarted,
    EvWarning,
)
from codewright.tools.errors import FatalToolError, RespondToModelError

if TYPE_CHECKING:
    from codewright.agent.session import Session


async def run_turn(
    session: Session,
    turn_context: TurnContext,
    user_input: str,
    sub_id: str = "",
) -> str | None:

    session.tool_registry.freeze()
    await session.emit_event(EvTurnStarted(turn_id=turn_context.turn_id), sub_id)

    mail_lines: list[str] = []
    for m in session.mailbox.drain_pending():
        mail_lines.append(f"[message from {m.author}]: {m.content}")
    mail_block = "\n".join(mail_lines)
    if mail_block and user_input:
        pending_user_input = f"{mail_block}\n\n{user_input}"
    elif mail_block:
        pending_user_input = mail_block
    else:
        pending_user_input = user_input
    last_message_text: str | None = None
    agents_md = _maybe_agents_md(session, turn_context)
    compacted_this_turn = False

    while True:
        if turn_context.cancellation_token.is_cancelled():
            await session.emit_event(
                EvTurnAborted(turn_id=turn_context.turn_id, reason="interrupted"),
                sub_id,
            )
            return None

        if not compacted_this_turn and session.context.should_compact():
            await session.emit_event(
                EvWarning(message="[compact] context limit reached; compacting"),
                sub_id,
            )
            await compact_history(
                session.context, session.summarizer, turn_context
            )
            await session.record_rollout(
                RolloutLine(
                    type="compaction",
                    payload={
                        "replacement": [
                            {"role": m.role, "content": _flatten_content(m)}
                            for m in session.context.snapshot()
                        ]
                    },
                )
            )
            compacted_this_turn = True

        messages = session.prompt_builder.build(
            turn_context,
            session.context.snapshot(),
            pending_user_input,
            agents_md=agents_md,
        )

        if pending_user_input:
            session.context.append(
                CanonicalMessage(role="user", content=pending_user_input)
            )
            await session.record_rollout(
                RolloutLine(
                    type="user_msg", payload={"content": pending_user_input}
                )
            )
        pending_user_input = ""

        stream = session.llm.stream(
            messages,
            tools=list(session.tool_specs()),
            turn_context=turn_context,
        )

        message_text = ""
        tool_calls: list[ToolCallBlock] = []
        stream_error: str | None = None
        stream_iter = await _maybe_await(stream)

        async for ev in stream_iter:
            if ev.kind == "text_delta" and ev.text:
                message_text += ev.text
                await session.emit_event(EvAgentMessageDelta(delta=ev.text), sub_id)
            elif ev.kind == "tool_call_completed" and ev.tool_call is not None:
                tool_calls.append(ev.tool_call)
            elif ev.kind == "usage" and ev.usage is not None:
                await session.emit_event(
                    EvTokenCount(
                        input=ev.usage.input,
                        output=ev.usage.output,
                        total=ev.usage.total,
                    ),
                    sub_id,
                )
            elif ev.kind == "error":
                stream_error = ev.error or "unknown provider error"
                break

        if stream_error is not None:
            await session.emit_event(EvError(message=stream_error), sub_id)
            await session.emit_event(
                EvTurnAborted(turn_id=turn_context.turn_id, reason="error"),
                sub_id,
            )
            return None

        if message_text:
            await session.emit_event(EvAgentMessage(content=message_text), sub_id)
            last_message_text = message_text

        if not tool_calls:
            if message_text:
                session.context.append(
                    CanonicalMessage(role="assistant", content=message_text)
                )
                await session.record_rollout(
                    RolloutLine(
                        type="assistant_msg", payload={"content": message_text}
                    )
                )
            await session.emit_event(
                EvTurnCompleted(
                    turn_id=turn_context.turn_id,
                    last_agent_message=last_message_text,
                ),
                sub_id,
            )
            return last_message_text

        assistant_with_calls = CanonicalMessage(
            role="assistant",
            content=(ContentBlock(text=message_text or None),),
            tool_calls=tuple(tool_calls),
        )
        session.context.append(assistant_with_calls)
        await session.record_rollout(
            RolloutLine(
                type="assistant_msg",
                payload={
                    "content": message_text,
                    "tool_calls": [
                        {
                            "call_id": tc.call_id,
                            "tool_name": tc.tool_name,
                            "arguments_json": tc.arguments_json,
                        }
                        for tc in tool_calls
                    ],
                },
            )
        )

        invocations = []
        for call in tool_calls:
            await session.emit_event(
                EvToolCallStarted(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    arguments={},
                ),
                sub_id,
            )
            try:
                invocations.append(
                    session.tool_router.build_invocation(call, session, turn_context)
                )
            except RespondToModelError as exc:

                _append_tool_failure(session, call, str(exc))
                await session.emit_event(
                    EvToolCallCompleted(
                        call_id=call.call_id, success=False, body=str(exc)
                    ),
                    sub_id,
                )

        if invocations:
            try:
                results = await session.tool_executor.dispatch_batch(invocations)
            except FatalToolError as exc:
                await session.emit_event(EvError(message=str(exc)), sub_id)
                await session.emit_event(
                    EvTurnAborted(turn_id=turn_context.turn_id, reason="error"),
                    sub_id,
                )
                return None

            for inv, result in zip(invocations, results, strict=True):
                session.context.append(
                    CanonicalMessage(
                        role="tool",
                        content=result.body,
                        name=inv.tool_name,
                        tool_call_id=inv.call_id,
                    )
                )
                await session.record_rollout(
                    RolloutLine(
                        type="tool_result",
                        payload={
                            "call_id": inv.call_id,
                            "tool_name": inv.tool_name,
                            "body": result.body,
                            "success": result.success,
                        },
                    )
                )
                await session.emit_event(
                    EvToolCallCompleted(
                        call_id=inv.call_id,
                        success=result.success,
                        body=result.body,
                    ),
                    sub_id,
                )



def _flatten_content(msg: CanonicalMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.text for b in msg.content if b.text)


def _maybe_agents_md(session: Session, turn_context: TurnContext) -> str | None:
    try:
        return session.workspace.resolve_agents_md(turn_context.cwd)
    except RuntimeError:
        return None


def _append_tool_failure(session: Session, call: ToolCallBlock, body: str) -> None:
    session.context.append(
        CanonicalMessage(
            role="tool",
            content=body,
            name=call.tool_name,
            tool_call_id=call.call_id,
        )
    )


async def _maybe_await(value):
    if hasattr(value, "__aiter__"):
        return value
    return await value
