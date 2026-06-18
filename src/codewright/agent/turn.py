from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING

from codewright.agent.turn_context import TurnContext
from codewright.context.compact import compact_history
from codewright.llm.base import CanonicalMessage, ContentBlock, ToolCallBlock
from codewright.persistence.rollout import RolloutLine
from codewright.protocol import (
    EvAgentMessage,
    EvAgentMessageDelta,
    EvCompactionCompleted,
    EvCompactionStarted,
    EvError,
    EvTokenCount,
    EvToolCallCompleted,
    EvToolCallStarted,
    EvTurnAborted,
    EvTurnCompleted,
    EvTurnStarted,
)
from codewright.tools.errors import FatalToolError, RespondToModelError

if TYPE_CHECKING:
    from codewright.agent.session import Session


class _TurnInterrupted(Exception):
    pass


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
    learned_facts = _maybe_learned_facts(session)
    distill = session.distillation
    if distill is not None and pending_user_input:
        distill.note_user(pending_user_input)
    compacted_this_turn = False

    while True:
        if turn_context.cancellation_token.is_cancelled():
            await _emit_turn_interrupted(session, turn_context, sub_id)
            return None

        if not compacted_this_turn and session.context.should_compact():
            tokens_before = session.context.total_tokens()
            await session.emit_event(
                EvCompactionStarted(reason="auto", tokens_before=tokens_before),
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
            await session.emit_event(
                EvCompactionCompleted(
                    tokens_before=tokens_before,
                    tokens_after=session.context.total_tokens(),
                ),
                sub_id,
            )
            compacted_this_turn = True

        messages = session.prompt_builder.build(
            turn_context,
            session.context.snapshot(),
            pending_user_input,
            agents_md=agents_md,
            learned_facts=learned_facts,
            plan=session.plan,
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
        try:
            stream_iter = await _maybe_await(
                stream, turn_context.cancellation_token
            )
        except _TurnInterrupted:
            await _emit_turn_interrupted(session, turn_context, sub_id)
            return None

        try:
            async for ev in _iter_stream_with_cancellation(
                stream_iter, turn_context.cancellation_token
            ):
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
        except _TurnInterrupted:
            await _emit_turn_interrupted(session, turn_context, sub_id)
            return None

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
            try:
                inv = session.tool_router.build_invocation(
                    call, session, turn_context
                )
            except RespondToModelError as exc:
                await session.emit_event(
                    EvToolCallStarted(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        arguments={},
                    ),
                    sub_id,
                )
                _append_tool_failure(session, call, str(exc))
                await session.emit_event(
                    EvToolCallCompleted(
                        call_id=call.call_id, success=False, body=str(exc)
                    ),
                    sub_id,
                )
                continue
            await session.emit_event(
                EvToolCallStarted(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    arguments=inv.arguments,
                ),
                sub_id,
            )
            invocations.append(inv)

        if invocations:
            try:
                results = await _dispatch_tools_with_cancellation(
                    session, invocations, turn_context.cancellation_token
                )
            except _TurnInterrupted:
                await _emit_turn_interrupted(session, turn_context, sub_id)
                return None
            except asyncio.CancelledError:
                if turn_context.cancellation_token.is_cancelled():
                    await _emit_turn_interrupted(session, turn_context, sub_id)
                    return None
                raise
            except FatalToolError as exc:
                await session.emit_event(EvError(message=str(exc)), sub_id)
                await session.emit_event(
                    EvTurnAborted(turn_id=turn_context.turn_id, reason="error"),
                    sub_id,
                )
                return None

            for inv, result in zip(invocations, results, strict=True):
                if distill is not None:
                    distill.note_tool(inv.tool_name, inv.arguments, result)
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

            if distill is not None:
                await distill.on_batch_end(session, turn_context)



def _flatten_content(msg: CanonicalMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.text for b in msg.content if b.text)


def _maybe_agents_md(session: Session, turn_context: TurnContext) -> str | None:
    try:
        return session.workspace.resolve_agents_md(turn_context.cwd)
    except RuntimeError:
        return None


def _maybe_learned_facts(session: Session) -> str | None:
    try:
        return session.skill_registry.always_on_text()
    except Exception:
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


async def _maybe_await(value, cancellation_token):
    if hasattr(value, "__aiter__"):
        return value

    pending_stream = asyncio.ensure_future(value)
    wait_cancel = asyncio.create_task(cancellation_token.wait())
    try:
        done, _pending = await asyncio.wait(
            [pending_stream, wait_cancel],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if wait_cancel in done:
            pending_stream.cancel()
            with suppress(asyncio.CancelledError):
                await pending_stream
            raise _TurnInterrupted()
        return pending_stream.result()
    except asyncio.CancelledError:
        pending_stream.cancel()
        if cancellation_token.is_cancelled():
            raise _TurnInterrupted() from None
        raise
    finally:
        if not wait_cancel.done():
            wait_cancel.cancel()


async def _emit_turn_interrupted(
    session: Session, turn_context: TurnContext, sub_id: str
) -> None:
    await session.emit_event(
        EvTurnAborted(turn_id=turn_context.turn_id, reason="interrupted"),
        sub_id,
    )


async def _iter_stream_with_cancellation(stream_iter, cancellation_token):
    iterator = stream_iter.__aiter__()
    while True:
        if cancellation_token.is_cancelled():
            raise _TurnInterrupted()

        next_event = asyncio.create_task(iterator.__anext__())
        wait_cancel = asyncio.create_task(cancellation_token.wait())
        try:
            done, _pending = await asyncio.wait(
                [next_event, wait_cancel],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if wait_cancel in done:
                next_event.cancel()
                with suppress(asyncio.CancelledError):
                    await next_event
                await _close_stream_iterator(iterator)
                raise _TurnInterrupted()
            try:
                yield next_event.result()
            except StopAsyncIteration:
                return
        except asyncio.CancelledError:
            next_event.cancel()
            if cancellation_token.is_cancelled():
                raise _TurnInterrupted() from None
            raise
        finally:
            if not wait_cancel.done():
                wait_cancel.cancel()


async def _close_stream_iterator(iterator) -> None:
    aclose = getattr(iterator, "aclose", None)
    if aclose is None:
        return
    with suppress(Exception, asyncio.CancelledError):
        await aclose()


async def _dispatch_tools_with_cancellation(
    session: Session, invocations, cancellation_token
):
    dispatch = asyncio.create_task(session.tool_executor.dispatch_batch(invocations))
    wait_cancel = asyncio.create_task(cancellation_token.wait())
    try:
        done, _pending = await asyncio.wait(
            [dispatch, wait_cancel],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if wait_cancel in done:
            dispatch.cancel()
            with suppress(asyncio.CancelledError):
                await dispatch
            raise _TurnInterrupted()
        return dispatch.result()
    except asyncio.CancelledError:
        dispatch.cancel()
        if cancellation_token.is_cancelled():
            raise _TurnInterrupted() from None
        raise
    finally:
        if not wait_cancel.done():
            wait_cancel.cancel()
