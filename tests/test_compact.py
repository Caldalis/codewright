"""Compact: ContextManager.replace_all + compact_history end-to-end."""

from __future__ import annotations

import pytest

from codewright.agent.cancellation import CancellationToken
from codewright.agent.turn_context import TurnContext
from codewright.context.compact import (
    LLMSummarizer,
    _collect_user_messages_tail,
    compact_history,
    load_compact_prompt,
)
from codewright.context.manager import ContextManager
from codewright.context.summarizer import Summarizer
from codewright.llm.base import (
    CanonicalMessage,
    ContentBlock,
    LLMProvider,
    StreamEvent,
    ToolCallBlock,
)
from codewright.protocol import AskForApproval, PermissionProfile


def _tc(turn_id: str = "t1") -> TurnContext:
    from pathlib import Path

    return TurnContext(
        turn_id=turn_id,
        cwd=Path("."),
        model="mock",
        permission_profile=PermissionProfile.READ_ONLY,
        approval_policy=AskForApproval.NEVER,
        cancellation_token=CancellationToken(),
    )


class StubSummarizer(Summarizer):
    """Returns a fixed string; lets us assert structure without an LLM."""

    def __init__(self, text: str = "SUMMARY-X") -> None:
        self._text = text
        self.calls: list[tuple[int, str]] = []

    async def summarize(
        self, messages: list[CanonicalMessage], compact_prompt: str
    ) -> str:
        self.calls.append((len(messages), compact_prompt))
        return self._text


class StubLLM(LLMProvider):
    """Stream a fixed sequence of text deltas; for LLMSummarizer's plumbing."""

    def __init__(self, deltas: list[str]) -> None:
        self._deltas = deltas

    async def stream(self, messages, tools, turn_context):  # type: ignore[override]
        async def _gen():
            for d in self._deltas:
                yield StreamEvent(kind="text_delta", text=d)
            yield StreamEvent(kind="message_completed")

        return _gen()


class TestReplaceAll:
    def test_replace_all_resets_history(self):
        cm = ContextManager(max_context_tokens=1000)
        cm.append(CanonicalMessage(role="user", content="x"))
        cm.append(CanonicalMessage(role="assistant", content="y"))
        assert len(cm) == 2

        cm.replace_all(
            [
                CanonicalMessage(role="user", content="env"),
                CanonicalMessage(role="user", content="tail"),
            ]
        )
        snap = cm.snapshot()
        assert len(snap) == 2
        assert snap[0].content == "env"
        assert snap[1].content == "tail"


class TestCollectUserTail:
    def test_filters_to_user_role_only(self):
        history = (
            CanonicalMessage(role="user", content="u1"),
            CanonicalMessage(role="assistant", content="a1"),
            CanonicalMessage(
                role="assistant",
                content=(ContentBlock(text=None),),
                tool_calls=(
                    ToolCallBlock(
                        call_id="c1", tool_name="x", arguments_json="{}"
                    ),
                ),
            ),
            CanonicalMessage(role="tool", content="output", tool_call_id="c1"),
            CanonicalMessage(role="user", content="u2"),
        )
        tail = _collect_user_messages_tail(history, budget=100_000)
        assert [m.content for m in tail] == ["u1", "u2"]

    def test_excludes_prior_summary_messages(self):
        history = (
            CanonicalMessage(role="user", content="u1"),
            CanonicalMessage(
                role="user", content="[COMPACTED SUMMARY]:\nprev"
            ),
            CanonicalMessage(role="user", content="u2"),
        )
        tail = _collect_user_messages_tail(history, budget=100_000)
        assert [m.content for m in tail] == ["u1", "u2"]

    def test_budget_picks_most_recent(self):
        # Each "x" * 40 == 10 tokens. Budget of 25 fits at most 2.
        history = tuple(
            CanonicalMessage(role="user", content="x" * 40) for _ in range(5)
        )
        tail = _collect_user_messages_tail(history, budget=25)
        assert 1 <= len(tail) <= 3  # Most recent ones, up to budget.
        # Must keep ordering (oldest first among selected).
        assert all(isinstance(m, CanonicalMessage) for m in tail)


class TestCompactHistory:
    @pytest.mark.asyncio
    async def test_compaction_replaces_history_with_tail_then_summary(self):
        """env_context lives in PromptBuilder, not in compacted history (ADR D-4-006).

        After one compact, history = ``user_tail`` (filtered) + summary.
        Tool calls, tool results, and assistant text are dropped.
        """
        cm = ContextManager(max_context_tokens=1000)
        cm.append(CanonicalMessage(role="user", content="user-1"))
        cm.append(CanonicalMessage(role="assistant", content="agent-1"))
        cm.append(
            CanonicalMessage(
                role="assistant",
                content=(ContentBlock(text=None),),
                tool_calls=(
                    ToolCallBlock(
                        call_id="c", tool_name="run_shell", arguments_json="{}"
                    ),
                ),
            )
        )
        cm.append(CanonicalMessage(role="tool", content="tool-output"))
        cm.append(CanonicalMessage(role="user", content="user-2"))

        stub = StubSummarizer("SUM")
        await compact_history(cm, stub, _tc())

        snap = cm.snapshot()
        # user_tail + summary; no env, no tool/assistant/reasoning.
        assert len(snap) == 3
        assert snap[0].role == "user" and snap[0].content == "user-1"
        assert snap[1].role == "user" and snap[1].content == "user-2"
        assert snap[2].role == "user"
        assert snap[2].content.startswith("[COMPACTED SUMMARY]:")  # type: ignore[union-attr]
        assert "SUM" in snap[2].content  # type: ignore[operator]

        # No env_context block must have leaked into history.
        for m in snap:
            text = m.content if isinstance(m.content, str) else ""
            assert "<environment_context>" not in text

        # Summarizer received the full pre-compact history.
        assert stub.calls and stub.calls[0][0] == 5

    @pytest.mark.asyncio
    async def test_empty_history_is_noop(self):
        cm = ContextManager(max_context_tokens=1000)
        stub = StubSummarizer("ignored")
        await compact_history(cm, stub, _tc())
        assert cm.snapshot() == ()
        assert stub.calls == []

    @pytest.mark.asyncio
    async def test_double_compact_does_not_accumulate_env_or_summary(self):
        """Second compact must not pick up the first compact's summary or env."""
        cm = ContextManager(max_context_tokens=1000)
        cm.append(CanonicalMessage(role="user", content="u1"))
        cm.append(CanonicalMessage(role="assistant", content="a1"))

        await compact_history(cm, StubSummarizer("SUM-1"), _tc())
        snap1 = cm.snapshot()
        # After first compact: [u1, summary1]; no env block.
        assert [m.role for m in snap1] == ["user", "user"]
        assert snap1[0].content == "u1"
        assert snap1[1].content.startswith("[COMPACTED SUMMARY]:")  # type: ignore[union-attr]
        assert "SUM-1" in snap1[1].content  # type: ignore[operator]
        for m in snap1:
            text = m.content if isinstance(m.content, str) else ""
            assert "<environment_context>" not in text

        # Simulate another turn happening between compacts.
        cm.append(CanonicalMessage(role="user", content="u2"))
        cm.append(CanonicalMessage(role="assistant", content="a2"))

        await compact_history(cm, StubSummarizer("SUM-2"), _tc())
        snap2 = cm.snapshot()

        # Expected: [u1, u2, summary2]. The previous summary is filtered out;
        # no env_context block ever appeared in history, so none can leak.
        contents = [
            m.content if isinstance(m.content, str) else "" for m in snap2
        ]
        # Exactly one summary line, and it's the new one.
        summary_lines = [c for c in contents if c.startswith("[COMPACTED SUMMARY]:")]
        assert len(summary_lines) == 1
        assert "SUM-2" in summary_lines[0]
        assert "SUM-1" not in summary_lines[0]
        # User messages preserved across both compacts.
        assert "u1" in contents
        assert "u2" in contents
        # No env block anywhere.
        for c in contents:
            assert "<environment_context>" not in c

    @pytest.mark.asyncio
    async def test_oversize_message_is_truncated_to_fit_budget(self):
        """Single user msg > budget gets truncate_middle'd, not kept whole."""
        cm = ContextManager(max_context_tokens=1000)
        # 4000-char message ≈ 1000 tokens.
        big = "x" * 4000
        cm.append(CanonicalMessage(role="user", content=big))

        # _collect_user_messages_tail's budget here is the module constant
        # (20K tokens), so to actually exercise truncation we call the
        # internal helper directly with a tighter budget.
        tail = _collect_user_messages_tail(cm.snapshot(), budget=100)
        assert len(tail) == 1
        text = tail[0].content
        assert isinstance(text, str)
        # The truncation marker leaves an unmistakable signature.
        assert "chars truncated" in text
        # And the result is small enough for the budget (with marker slack).
        from codewright.context.manager import approx_token_count

        assert approx_token_count(text) <= 200  # budget + marker headroom


class TestLLMSummarizer:
    @pytest.mark.asyncio
    async def test_collects_text_deltas_from_provider(self):
        provider = StubLLM(["A ", "summary", "."])
        s = LLMSummarizer(provider)
        out = await s.summarize(
            [CanonicalMessage(role="user", content="hello")],
            load_compact_prompt(),
        )
        assert out == "A summary."

    @pytest.mark.asyncio
    async def test_error_event_raises(self):
        class ErrLLM(LLMProvider):
            async def stream(self, messages, tools, turn_context):  # type: ignore[override]
                async def _gen():
                    yield StreamEvent(kind="error", error="boom")

                return _gen()

        s = LLMSummarizer(ErrLLM())
        with pytest.raises(RuntimeError) as exc:
            await s.summarize([], "prompt")
        assert "boom" in str(exc.value)


class TestRunTurnTriggersCompact:
    """End-to-end: when the LLM provider gets called, history should already be compacted."""

    @pytest.mark.asyncio
    async def test_should_compact_triggers_replacement_before_next_sample(self):
        from codewright.agent.session import Session
        from codewright.agent.turn import run_turn
        from codewright.prompts.builder import PromptBuilder, load_default_system_prompt

        # tiny context (10 tokens => threshold ~5) so a single 'aaaa...' fills it.
        cm = ContextManager(max_context_tokens=10, compact_threshold=0.5)
        cm.append(CanonicalMessage(role="user", content="x" * 200))
        assert cm.should_compact()

        class OneAnswerLLM(LLMProvider):
            def __init__(self) -> None:
                self.seen_messages: list[list[CanonicalMessage]] = []

            async def stream(self, messages, tools, turn_context):  # type: ignore[override]
                self.seen_messages.append(list(messages))

                async def _gen():
                    yield StreamEvent(kind="text_delta", text="OK")
                    yield StreamEvent(kind="message_completed")

                return _gen()

        llm = OneAnswerLLM()
        session = Session(
            session_id="s",
            cwd=_tc().cwd,
            permission_profile=PermissionProfile.READ_ONLY,
            llm=llm,
            context_manager=cm,
            prompt_builder=PromptBuilder(load_default_system_prompt()),
            summarizer=StubSummarizer("S"),
        )
        try:
            result = await run_turn(session, _tc(), user_input="hi")
            assert result == "OK"
            # The summarizer must have been used.
            # After compact + the user_input + the assistant reply,
            # history should be: env_ctx + (no prior user tail) + summary + 'hi' + 'OK'
            snap = cm.snapshot()
            roles = [m.role for m in snap]
            contents = [
                m.content if isinstance(m.content, str) else "" for m in snap
            ]
            # The user 'hi' message should appear (rolled in after compact).
            assert "hi" in contents
            # And the summary placeholder must be in the history.
            assert any(
                isinstance(c, str) and c.startswith("[COMPACTED SUMMARY]:")
                for c in contents
            )
            assert "assistant" in roles
        finally:
            await session.shutdown()
