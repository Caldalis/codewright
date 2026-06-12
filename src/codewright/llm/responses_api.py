from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from codewright.llm.base import (
    CanonicalMessage,
    ContentBlock,
    LLMProvider,
    StreamEvent,
    TokenUsage,
    ToolCallBlock,
)


def _content_to_str(content: str | tuple[ContentBlock, ...]) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.text for b in content if b.text is not None)


def _to_responses_input(
    messages: list[CanonicalMessage],
) -> tuple[str, list[dict[str, Any]]]:

    instructions = ""
    items: list[dict[str, Any]] = []
    instructions_set = False
    for m in messages:
        if m.role == "system" and not instructions_set:
            instructions = _content_to_str(m.content)
            instructions_set = True
            continue
        if m.role == "system":

            items.append(
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": _content_to_str(m.content)}
                    ],
                }
            )
            continue
        if m.role == "developer":
            items.append(
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": _content_to_str(m.content)}
                    ],
                }
            )
            continue
        if m.role == "user":
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": _content_to_str(m.content)}
                    ],
                }
            )
            continue
        if m.role == "assistant":
            text = _content_to_str(m.content)
            if text:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            for tc in m.tool_calls or ():
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.call_id,
                        "name": tc.tool_name,
                        "arguments": tc.arguments_json,
                    }
                )
            continue
        if m.role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": m.tool_call_id or "",
                    "output": _content_to_str(m.content),
                }
            )
            continue

    return instructions, items


def _to_responses_tools(tools: list[Any]) -> list[dict[str, Any]]:

    out: list[dict[str, Any]] = []
    for t in tools:
        name = getattr(t, "name", None)
        if not name:
            continue
        description = getattr(t, "description", "") or ""
        parameters = _tool_parameters(t)
        out.append(
            {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": parameters,
                "strict": False,
            }
        )
    return out


def _tool_parameters(tool: Any) -> dict[str, Any]:
    parameters = getattr(tool, "parameters", None)
    if isinstance(parameters, dict):
        return parameters or {"type": "object", "properties": {}}
    if hasattr(parameters, "to_json_schema"):
        schema = parameters.to_json_schema()
        if isinstance(schema, dict):
            return schema

    if hasattr(parameters, "model_json_schema"):
        schema = parameters.model_json_schema()
        if isinstance(schema, dict):
            return schema
    return {"type": "object", "properties": {}}


class ResponsesApiAdapter(LLMProvider):

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        request_timeout: float = 120.0,
        store: bool = False,
        prompt_cache_key: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._http = http_client or httpx.AsyncClient(timeout=request_timeout)
        self._owns_http = http_client is None
        self._store = store
        self._prompt_cache_key = prompt_cache_key

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def stream(
        self,
        messages: list[CanonicalMessage],
        tools: list[Any],
        turn_context: Any,
    ) -> AsyncIterator[StreamEvent]:
        del turn_context
        instructions, input_items = _to_responses_input(messages)
        provider_tools = _to_responses_tools(tools)

        payload: dict[str, Any] = {
            "model": self._model,
            "instructions": instructions,
            "input": input_items,
            "tools": provider_tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "store": self._store,
            "stream": True,
            "include": [],
        }
        if self._prompt_cache_key:
            payload["prompt_cache_key"] = self._prompt_cache_key

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        url = f"{self._base_url}/responses"
        return self._iter_stream(url, headers, payload)

    async def _iter_stream(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        try:
            async with self._http.stream(
                "POST", url, headers=headers, json=payload
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    yield StreamEvent(
                        kind="error",
                        error=f"HTTP {response.status_code}: {body.decode('utf-8', errors='replace')}",
                    )
                    return

                event_name: str | None = None
                tool_call_index = 0
                async for line in response.aiter_lines():
                    if not line:
                        # blank line ends one SSE event in the OpenAI stream.
                        event_name = None
                        continue
                    if line.startswith(":"):
                        # SSE comment; keep-alive.
                        continue
                    if line.startswith("event:"):
                        event_name = line[len("event:") :].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    async for ev in _translate_event(
                        event_name, chunk, tool_call_index
                    ):
                        if ev.kind == "tool_call_completed":
                            tool_call_index += 1
                        yield ev
        except httpx.HTTPError as exc:
            yield StreamEvent(kind="error", error=f"HTTP error: {exc}")
            return

        yield StreamEvent(kind="message_completed")


async def _translate_event(
    event_name: str | None, chunk: dict[str, Any], next_tool_index: int
) -> AsyncIterator[StreamEvent]:

    kind = event_name or chunk.get("type")

    if kind == "response.output_text.delta":
        delta = chunk.get("delta")
        if isinstance(delta, str) and delta:
            yield StreamEvent(kind="text_delta", text=delta)
        return

    if kind == "response.output_item.done":
        item = chunk.get("item") or {}
        if item.get("type") == "function_call":
            yield StreamEvent(
                kind="tool_call_completed",
                tool_call=ToolCallBlock(
                    call_id=str(item.get("call_id") or ""),
                    tool_name=str(item.get("name") or ""),
                    arguments_json=str(item.get("arguments") or ""),
                ),
                tool_call_index=next_tool_index,
            )
        return

    if kind == "response.completed":
        response = chunk.get("response") or {}
        usage = response.get("usage") or chunk.get("usage")
        if isinstance(usage, dict):
            yield StreamEvent(
                kind="usage",
                usage=TokenUsage(
                    input=int(usage.get("input_tokens", 0)),
                    output=int(usage.get("output_tokens", 0)),
                    total=int(
                        usage.get(
                            "total_tokens",
                            int(usage.get("input_tokens", 0))
                            + int(usage.get("output_tokens", 0)),
                        )
                    ),
                ),
            )
        return

    if kind in ("response.error", "error"):
        message = (
            chunk.get("error", {}).get("message")
            if isinstance(chunk.get("error"), dict)
            else None
        ) or chunk.get("message")
        yield StreamEvent(
            kind="error", error=str(message or "responses api error")
        )
        return


__all__ = ["ResponsesApiAdapter"]
