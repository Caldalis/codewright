"""Smoke tests for the provider-neutral message types."""

from __future__ import annotations

import dataclasses

import pytest

from codewright.llm.base import (
    CanonicalMessage,
    ContentBlock,
    StreamEvent,
    TokenUsage,
    ToolCallBlock,
)


class TestCanonicalMessage:
    def test_is_frozen_dataclass(self):
        assert dataclasses.is_dataclass(CanonicalMessage)
        assert CanonicalMessage.__dataclass_params__.frozen

    def test_equality_by_value(self):
        a = CanonicalMessage(role="user", content="hi")
        b = CanonicalMessage(role="user", content="hi")
        assert a == b

    def test_assignment_raises(self):
        msg = CanonicalMessage(role="user", content="hi")
        with pytest.raises(dataclasses.FrozenInstanceError):
            msg.content = "no"

    def test_tool_call_block_immutable(self):
        tc = ToolCallBlock(call_id="c1", tool_name="run_shell", arguments_json='{"cmd": "ls"}')
        with pytest.raises(dataclasses.FrozenInstanceError):
            tc.call_id = "x"

    def test_content_block_default_none(self):
        block = ContentBlock(text="hello")
        assert block.text == "hello"
        assert block.tool_calls is None

    def test_tool_calls_tuple(self):
        tc = ToolCallBlock(call_id="c1", tool_name="x", arguments_json="{}")
        msg = CanonicalMessage(role="assistant", content="", tool_calls=(tc,))
        assert msg.tool_calls == (tc,)


class TestStreamEvent:
    def test_text_delta(self):
        ev = StreamEvent(kind="text_delta", text="hi")
        assert ev.kind == "text_delta"
        assert ev.text == "hi"

    def test_usage(self):
        u = TokenUsage(input=10, output=5, total=15)
        ev = StreamEvent(kind="usage", usage=u)
        assert ev.usage == u
