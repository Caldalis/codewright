"""ResponsesApiAdapter: payload shape + SSE → StreamEvent translation."""

from __future__ import annotations

import json

import httpx
import pytest

from codewright.llm import create_llm_provider
from codewright.llm.base import CanonicalMessage, ContentBlock, ToolCallBlock
from codewright.llm.responses_api import (
    ResponsesApiAdapter,
    _to_responses_input,
    _to_responses_tools,
)
from codewright.tools.handlers import ApplyPatchHandler


class _ToolStub:
    def __init__(self, name: str, params: dict) -> None:
        self.name = name
        self.description = "stub"
        self._params = params

    def model_json_schema(self) -> dict:
        return self._params


class TestPayloadShape:
    def test_first_system_message_becomes_instructions(self):
        messages = [
            CanonicalMessage(role="system", content="long-lived"),
            CanonicalMessage(role="developer", content="dev block"),
            CanonicalMessage(role="user", content="hello"),
        ]
        instructions, items = _to_responses_input(messages)
        assert instructions == "long-lived"
        assert items[0]["role"] == "developer"
        assert items[0]["content"][0]["text"] == "dev block"
        assert items[1]["role"] == "user"
        assert items[1]["content"][0]["type"] == "input_text"

    def test_assistant_with_tool_calls_emits_function_call(self):
        msg = CanonicalMessage(
            role="assistant",
            content=(ContentBlock(text="picking a tool"),),
            tool_calls=(
                ToolCallBlock(
                    call_id="c1", tool_name="run_shell", arguments_json='{"x":1}'
                ),
            ),
        )
        _, items = _to_responses_input([msg])
        # Expect: message + function_call.
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "assistant"
        assert items[0]["content"][0]["type"] == "output_text"
        assert items[1]["type"] == "function_call"
        assert items[1]["call_id"] == "c1"
        assert items[1]["name"] == "run_shell"
        assert items[1]["arguments"] == '{"x":1}'

    def test_tool_role_emits_function_call_output(self):
        msg = CanonicalMessage(
            role="tool",
            content="stdout text",
            name="run_shell",
            tool_call_id="c1",
        )
        _, items = _to_responses_input([msg])
        assert items == [
            {"type": "function_call_output", "call_id": "c1", "output": "stdout text"}
        ]

    def test_tools_flatten_to_responses_shape(self):
        tools = [_ToolStub("run_shell", {"type": "object", "properties": {}})]
        out = _to_responses_tools(tools)
        assert out == [
            {
                "type": "function",
                "name": "run_shell",
                "description": "stub",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            }
        ]

    def test_tool_spec_parameters_are_sent_to_provider(self):
        out = _to_responses_tools([ApplyPatchHandler().spec()])
        parameters = out[0]["parameters"]
        properties = parameters["properties"]
        assert "patch" in properties
        assert parameters["required"] == ["patch"]
        assert "name" not in properties
        assert "description" not in properties
        assert "parameters" not in properties


SSE_BODY = (
    "event: response.output_text.delta\n"
    "data: {\"delta\": \"Hi\"}\n"
    "\n"
    "event: response.output_text.delta\n"
    "data: {\"delta\": \" there\"}\n"
    "\n"
    "event: response.output_item.done\n"
    "data: {\"item\": {\"type\": \"function_call\", \"call_id\": \"call_a\", "
    "\"name\": \"run_shell\", \"arguments\": \"{}\"}}\n"
    "\n"
    "event: response.completed\n"
    "data: {\"response\": {\"usage\": {\"input_tokens\": 10, "
    "\"output_tokens\": 5, \"total_tokens\": 15}}}\n"
    "\n"
)


class TestStreamingTranslation:
    @pytest.mark.asyncio
    async def test_full_round_trip_with_mock_transport(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(
                200,
                content=SSE_BODY,
                headers={"content-type": "text/event-stream"},
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            adapter = ResponsesApiAdapter(
                api_key="sk-test",
                model="gpt-5",
                base_url="https://example.invalid/v1",
                http_client=http,
            )
            events = []
            stream = await adapter.stream(
                [
                    CanonicalMessage(role="system", content="sys"),
                    CanonicalMessage(role="user", content="hi"),
                ],
                tools=[_ToolStub("run_shell", {"type": "object"})],
                turn_context=None,
            )
            async for ev in stream:
                events.append(ev)

        # Captured outbound request.
        assert captured["url"].endswith("/responses")
        assert captured["body"]["model"] == "gpt-5"
        assert captured["body"]["instructions"] == "sys"
        assert captured["body"]["input"][0]["role"] == "user"
        assert captured["body"]["tools"][0]["name"] == "run_shell"
        assert captured["headers"]["authorization"] == "Bearer sk-test"

        # Stream events.
        kinds = [ev.kind for ev in events]
        assert kinds[0] == "text_delta"
        assert kinds.count("text_delta") == 2
        assert "tool_call_completed" in kinds
        assert "usage" in kinds
        assert kinds[-1] == "message_completed"

        tool_done = next(ev for ev in events if ev.kind == "tool_call_completed")
        assert tool_done.tool_call.tool_name == "run_shell"
        assert tool_done.tool_call.call_id == "call_a"

        usage = next(ev for ev in events if ev.kind == "usage")
        assert usage.usage.input == 10
        assert usage.usage.output == 5
        assert usage.usage.total == 15

    @pytest.mark.asyncio
    async def test_http_error_yields_error_event(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            adapter = ResponsesApiAdapter(
                api_key="x", model="m", http_client=http
            )
            stream = await adapter.stream(
                [CanonicalMessage(role="user", content="hi")],
                tools=[],
                turn_context=None,
            )
            events = [ev async for ev in stream]
        assert any(ev.kind == "error" for ev in events)


class TestFactory:
    def test_create_llm_provider_dispatches(self):
        responses = create_llm_provider(
            "responses", api_key="x", model="gpt-5"
        )
        assert isinstance(responses, ResponsesApiAdapter)
        chat = create_llm_provider(
            "chat_completions", api_key="x", model="gpt-4"
        )
        from codewright.llm import ChatCompletionsAdapter

        assert isinstance(chat, ChatCompletionsAdapter)

    def test_unknown_api_style_raises(self):
        with pytest.raises(ValueError):
            create_llm_provider("websocket", api_key="x", model="m")  # type: ignore[arg-type]
