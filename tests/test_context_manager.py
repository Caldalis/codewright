"""ContextManager + History invariants."""

from __future__ import annotations

import pytest

from codewright.context import ContextManager, History, approx_token_count
from codewright.llm.base import CanonicalMessage


class TestApproxTokenCount:
    def test_simple_division(self):
        assert approx_token_count("a" * 100) == 25

    def test_empty(self):
        assert approx_token_count("") == 0


class TestContextManager:
    def test_append_then_snapshot(self):
        cm = ContextManager(max_context_tokens=1000)
        cm.append(CanonicalMessage(role="user", content="hello"))
        snap = cm.snapshot()
        assert len(snap) == 1
        assert snap[0].content == "hello"

    def test_snapshot_is_immutable_tuple(self):
        cm = ContextManager(max_context_tokens=1000)
        cm.append(CanonicalMessage(role="user", content="hello"))
        snap = cm.snapshot()
        assert isinstance(snap, tuple)

    def test_should_compact_threshold(self):
        cm = ContextManager(max_context_tokens=100, compact_threshold=0.5)
        # 100 chars => 25 tokens => below 50
        cm.append(CanonicalMessage(role="user", content="a" * 100))
        assert not cm.should_compact()
        # bring total to 200 chars => 50 tokens => at threshold
        cm.append(CanonicalMessage(role="user", content="a" * 100))
        assert cm.should_compact()

    def test_history_total_tokens_counts_content_blocks(self):
        from codewright.llm.base import ContentBlock

        h = History()
        h.append(
            CanonicalMessage(
                role="assistant", content=(ContentBlock(text="a" * 40),)
            )
        )
        assert h.total_tokens() == 10

    def test_history_total_tokens_counts_tool_call_arguments(self):
        from codewright.llm.base import ToolCallBlock

        h = History()
        h.append(
            CanonicalMessage(
                role="assistant",
                content="",
                tool_calls=(
                    ToolCallBlock(
                        call_id="c1",
                        tool_name="apply_patch",
                        arguments_json="a" * 400,
                    ),
                ),
            )
        )

        assert h.total_tokens() >= approx_token_count("a" * 400)

    def test_should_compact_counts_tool_call_arguments(self):
        from codewright.llm.base import ToolCallBlock

        cm = ContextManager(max_context_tokens=100, compact_threshold=0.5)
        cm.append(
            CanonicalMessage(
                role="assistant",
                content="",
                tool_calls=(
                    ToolCallBlock(
                        call_id="c1",
                        tool_name="apply_patch",
                        arguments_json="a" * 240,
                    ),
                ),
            )
        )

        assert cm.should_compact()

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            ContextManager(max_context_tokens=100, compact_threshold=1.5)
        with pytest.raises(ValueError):
            ContextManager(max_context_tokens=0)
