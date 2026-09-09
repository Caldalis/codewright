"""ChatCompletionsAdapter SSE translation, message rendering."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from codewright.llm.base import CanonicalMessage, ContentBlock, ToolCallBlock
from codewright.llm.chat_completions import (
    ChatCompletionsAdapter,
    _to_provider_messages,
    _to_provider_tools,
)
from codewright.tools.handlers import ApplyPatchHandler


def _sse(payload: dict[str, Any]) -> bytes:
    return ("data: " + json.dumps(payload) + "\n\n").encode("utf-8")


def _make_transport(chunks: list[bytes], status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = b"".join(chunks) + b"data: [DONE]\n\n"
        return httpx.Response(
            status,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    return httpx.MockTransport(handler)


class TestProviderMessageRendering:
    def test_developer_renders_as_second_system(self):
        msgs = [
            CanonicalMessage(role="system", content="SYS"),
            CanonicalMessage(role="developer", content="DEV"),
            CanonicalMessage(role="user", content="hi"),
        ]
        out = _to_provider_messages(msgs)
        assert out[0]["role"] == "system"
        assert "SYS" in out[0]["content"]
        assert "<developer_instructions>" in out[0]["content"]
        assert "DEV" in out[0]["content"]
        assert out[1] == {"role": "user", "content": "hi"}

    def test_assistant_with_tool_calls(self):
        tc = ToolCallBlock(call_id="c1", tool_name="run_shell", arguments_json='{"cmd":"ls"}')
        msgs = [CanonicalMessage(role="assistant", content="", tool_calls=(tc,))]
        out = _to_provider_messages(msgs)
        assert out[0]["role"] == "assistant"
        assert out[0]["tool_calls"][0]["id"] == "c1"
        assert out[0]["tool_calls"][0]["function"]["name"] == "run_shell"
        assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"cmd":"ls"}'

    def test_tool_role_keeps_call_id(self):
        msgs = [
            CanonicalMessage(
                role="tool",
                content="stdout",
                name="run_shell",
                tool_call_id="c1",
            )
        ]
        out = _to_provider_messages(msgs)
        assert out[0] == {"role": "tool", "tool_call_id": "c1", "content": "stdout"}

    def test_content_blocks_flatten_to_string(self):
        msg = CanonicalMessage(
            role="user",
            content=(ContentBlock(text="hello "), ContentBlock(text="world")),
        )
        out = _to_provider_messages([msg])
        assert out[0]["content"] == "hello world"

    def test_tool_spec_parameters_are_sent_to_provider(self):
        out = _to_provider_tools([ApplyPatchHandler().spec()])
        parameters = out[0]["function"]["parameters"]
        properties = parameters["properties"]
        assert "patch" in properties
        assert parameters["required"] == ["patch"]
        assert "name" not in properties
        assert "description" not in properties
        assert "parameters" not in properties


@pytest.mark.asyncio
async def test_text_stream_translation():
    chunks = [
        _sse({"choices": [{"delta": {"content": "hel"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]}),
        _sse(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        ),
    ]
    transport = _make_transport(chunks)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = ChatCompletionsAdapter(
            api_key="test", model="m", http_client=client
        )
        stream = await adapter.stream([CanonicalMessage(role="user", content="hi")], tools=[], turn_context=None)
        events = []
        async for ev in stream:
            events.append(ev)

    kinds = [ev.kind for ev in events]
    # Two text deltas, one usage, one message_completed
    assert kinds.count("text_delta") == 2
    assert "usage" in kinds
    assert kinds[-1] == "message_completed"

    text = "".join(ev.text for ev in events if ev.kind == "text_delta" and ev.text)
    assert text == "hello"


@pytest.mark.asyncio
async def test_empty_choices_usage_chunk_is_skipped_without_crashing():
    chunks = [
        _sse({"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]}),
        _sse(
            {
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ),
    ]
    transport = _make_transport(chunks)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = ChatCompletionsAdapter(api_key="test", model="m", http_client=client)
        stream = await adapter.stream(
            [CanonicalMessage(role="user", content="hi")], tools=[], turn_context=None
        )
        events = []
        async for ev in stream:
            events.append(ev)

    kinds = [ev.kind for ev in events]
    assert kinds == ["text_delta", "usage", "message_completed"]
    assert events[0].text == "hi"
    assert events[1].usage is not None
    assert events[1].usage.total == 2


@pytest.mark.asyncio
async def test_tool_call_stream_translation():
    chunks = [
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "function": {"name": "run_shell", "arguments": '{"cmd":'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"ls"}'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
    ]
    transport = _make_transport(chunks)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = ChatCompletionsAdapter(api_key="t", model="m", http_client=client)
        stream = await adapter.stream(
            [CanonicalMessage(role="user", content="ls")], tools=[], turn_context=None
        )
        events = []
        async for ev in stream:
            events.append(ev)

    kinds = [ev.kind for ev in events]
    assert "tool_call_started" in kinds
    assert kinds.count("tool_call_arguments_delta") == 2
    assert kinds.count("tool_call_completed") == 1
    completed = next(ev for ev in events if ev.kind == "tool_call_completed")
    assert completed.tool_call is not None
    assert completed.tool_call.call_id == "c1"
    assert completed.tool_call.tool_name == "run_shell"
    assert completed.tool_call.arguments_json == '{"cmd":"ls"}'


@pytest.mark.asyncio
async def test_http_error_yields_error_event():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = ChatCompletionsAdapter(api_key="t", model="m", http_client=client)
        stream = await adapter.stream(
            [CanonicalMessage(role="user", content="hi")], tools=[], turn_context=None
        )
        events = []
        async for ev in stream:
            events.append(ev)

    assert any(ev.kind == "error" for ev in events)


@pytest.mark.asyncio
async def test_sse_error_frame_yields_error_event():
    chunks = [
        b'data: {"error":{"code":10012,"message":"EngineInternalError:Bad Request"}}\n\n',
    ]
    transport = _make_transport(chunks)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = ChatCompletionsAdapter(api_key="t", model="m", http_client=client)
        stream = await adapter.stream(
            [CanonicalMessage(role="user", content="hi")], tools=[], turn_context=None
        )
        events = []
        async for ev in stream:
            events.append(ev)

    assert [ev.kind for ev in events] == ["error"]
    assert events[0].error is not None
    assert "10012" in events[0].error
