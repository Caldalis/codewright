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


# ------------------------------------------------------- provider passthrough
#
# Thinking models return their trace on the assistant message under a key the
# OpenAI schema does not define, and some refuse the next request if an
# assistant message carrying tool_calls comes back without it. The spelling
# varies by provider and only the issuing provider's spelling is accepted, so
# the adapter carries the key verbatim rather than normalising it.


@pytest.mark.asyncio
async def test_reasoning_deltas_are_joined_and_reported_once():
    chunks = [
        _sse({"choices": [{"delta": {"reasoning_content": "I should "}}]}),
        _sse({"choices": [{"delta": {"reasoning_content": "check the file."}}]}),
        _sse({"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]}),
    ]
    transport = _make_transport(chunks)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = ChatCompletionsAdapter(api_key="t", model="m", http_client=client)
        stream = await adapter.stream(
            [CanonicalMessage(role="user", content="hi")], tools=[], turn_context=None
        )
        events = [ev async for ev in stream]

    done = events[-1]
    assert done.kind == "message_completed"
    # Joined like content is, not last-write-wins.
    assert done.provider_extras == {"reasoning_content": "I should check the file."}


@pytest.mark.asyncio
async def test_each_provider_keeps_its_own_spelling():
    for key in ("reasoning", "thinking", "encrypted_content"):
        chunks = [
            _sse({"choices": [{"delta": {key: "trace"}}]}),
            _sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}),
        ]
        transport = _make_transport(chunks)
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = ChatCompletionsAdapter(api_key="t", model="m", http_client=client)
            stream = await adapter.stream(
                [CanonicalMessage(role="user", content="hi")],
                tools=[],
                turn_context=None,
            )
            events = [ev async for ev in stream]
        assert events[-1].provider_extras == {key: "trace"}, key


@pytest.mark.asyncio
async def test_non_streaming_shape_is_read_too():
    """A provider may put the whole message in the final frame rather than
    streaming it in deltas."""
    chunks = [
        _sse({"choices": [{"message": {"role": "assistant",
                                       "reasoning_content": "thought",
                                       "content": "ok"},
                           "finish_reason": "stop"}]}),
    ]
    transport = _make_transport(chunks)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = ChatCompletionsAdapter(api_key="t", model="m", http_client=client)
        stream = await adapter.stream(
            [CanonicalMessage(role="user", content="hi")], tools=[], turn_context=None
        )
        events = [ev async for ev in stream]

    assert events[-1].provider_extras == {"reasoning_content": "thought"}


@pytest.mark.asyncio
async def test_nothing_is_invented_when_the_provider_sends_nothing():
    """The field must stay absent for providers that never use it -- OpenAI
    rejects unknown keys on a message outright."""
    chunks = [
        _sse({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}),
    ]
    transport = _make_transport(chunks)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = ChatCompletionsAdapter(api_key="t", model="m", http_client=client)
        stream = await adapter.stream(
            [CanonicalMessage(role="user", content="hi")], tools=[], turn_context=None
        )
        events = [ev async for ev in stream]

    assert events[-1].provider_extras is None


def test_extras_are_echoed_on_a_tool_call_message():
    """The shape that actually gets rejected: assistant + tool_calls."""
    msgs = [
        CanonicalMessage(
            role="assistant",
            content=(ContentBlock(text=None),),
            tool_calls=(ToolCallBlock(call_id="c1", tool_name="shell",
                                      arguments_json="{}"),),
            provider_extras={"reasoning_content": "trace"},
        ),
    ]
    out = _to_provider_messages(msgs)
    assert out[0]["reasoning_content"] == "trace"
    assert out[0]["tool_calls"][0]["id"] == "c1"


def test_extras_are_echoed_on_a_plain_assistant_message():
    out = _to_provider_messages([
        CanonicalMessage(role="assistant", content="hi",
                         provider_extras={"reasoning": "trace"}),
    ])
    assert out[0]["reasoning"] == "trace"


def test_no_extras_means_no_extra_keys():
    out = _to_provider_messages([
        CanonicalMessage(role="assistant", content="hi"),
        CanonicalMessage(role="user", content="hi"),
    ])
    for entry in out:
        assert set(entry) <= {"role", "content"}, entry


def test_extras_never_leak_onto_a_user_or_tool_message():
    """Only the assistant turn carries them. A user message with the key set --
    which should not happen -- must not put it on the wire."""
    out = _to_provider_messages([
        CanonicalMessage(role="user", content="hi",
                         provider_extras={"reasoning_content": "x"}),
        CanonicalMessage(role="tool", content="r", tool_call_id="c1",
                         provider_extras={"reasoning_content": "x"}),
    ])
    assert "reasoning_content" not in out[0]
    assert "reasoning_content" not in out[1]


@pytest.mark.asyncio
async def test_a_trailing_message_frame_replaces_the_deltas():
    """A provider may stream the trace AND repeat the whole message at the end.

    `delta` is an increment and `message` is the finished value, so reading both
    as increments reports the trace twice.
    """
    chunks = [
        _sse({"choices": [{"delta": {"reasoning_content": "I should "}}]}),
        _sse({"choices": [{"delta": {"reasoning_content": "check it."}}]}),
        _sse({"choices": [{"message": {"role": "assistant",
                                       "reasoning_content": "I should check it.",
                                       "content": "ok"},
                           "finish_reason": "stop"}]}),
    ]
    transport = _make_transport(chunks)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = ChatCompletionsAdapter(api_key="t", model="m", http_client=client)
        stream = await adapter.stream(
            [CanonicalMessage(role="user", content="hi")], tools=[], turn_context=None
        )
        events = [ev async for ev in stream]

    assert events[-1].provider_extras == {
        "reasoning_content": "I should check it."
    }
