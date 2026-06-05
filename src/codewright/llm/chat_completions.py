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
    parts: list[str] = []
    for block in content:
        if block.text is not None:
            parts.append(block.text)
    return "".join(parts)


def _to_provider_messages(messages: list[CanonicalMessage]) -> list[dict[str, Any]]:

    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "developer":
          
            out.append({"role": "system", "content": _content_to_str(m.content)})
            continue
        if m.role == "assistant" and m.tool_calls:
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": _content_to_str(m.content) or None,
                "tool_calls": [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": tc.arguments_json,
                        },
                    }
                    for tc in m.tool_calls
                ],
            }
            out.append(entry)
            continue
        if m.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id or "",
                    "content": _content_to_str(m.content),
                }
            )
            continue
        out.append({"role": m.role, "content": _content_to_str(m.content)})
    return out


def _to_provider_tools(tools: list[Any]) -> list[dict[str, Any]]:

    out: list[dict[str, Any]] = []
    for t in tools:
        name = getattr(t, "name", None)
        description = getattr(t, "description", "")
        if hasattr(t, "model_json_schema"):
            parameters = t.model_json_schema()
        else:
            parameters = getattr(t, "parameters", {"type": "object"})
        if not name:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return out


class ChatCompletionsAdapter(LLMProvider):

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        request_timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._http = http_client or httpx.AsyncClient(timeout=request_timeout)
        self._owns_http = http_client is None

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
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": _to_provider_messages(messages),
            "stream": True,
        }
        provider_tools = _to_provider_tools(tools)
        if provider_tools:
            payload["tools"] = provider_tools

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        url = f"{self._base_url}/chat/completions"

        return self._iter_stream(url, headers, payload)

    async def _iter_stream(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:

        tool_names: dict[int, str] = {}
        tool_ids: dict[int, str] = {}
        tool_args: dict[int, list[str]] = {}
        finish_reason: str | None = None

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

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        # SSE comment line / keep-alive — provider-dependent.
                        continue

                    async for event in _translate_chunk(
                        chunk, tool_names, tool_ids, tool_args
                    ):
                        if event.kind == "text_delta":
                            yield event
                        elif event.kind == "tool_call_started":
                            yield event
                        elif event.kind == "tool_call_arguments_delta":
                            yield event

                    choices = chunk.get("choices") or []
                    if choices:
                        finish_reason = choices[0].get("finish_reason") or finish_reason

                    usage = chunk.get("usage")
                    if usage:
                        yield StreamEvent(
                            kind="usage",
                            usage=TokenUsage(
                                input=int(usage.get("prompt_tokens", 0)),
                                output=int(usage.get("completion_tokens", 0)),
                                total=int(usage.get("total_tokens", 0)),
                            ),
                        )

        except httpx.HTTPError as exc:
            yield StreamEvent(kind="error", error=f"HTTP error: {exc}")
            return

        if finish_reason == "tool_calls":
            for idx in sorted(tool_names):
                yield StreamEvent(
                    kind="tool_call_completed",
                    tool_call=ToolCallBlock(
                        call_id=tool_ids.get(idx, ""),
                        tool_name=tool_names[idx],
                        arguments_json="".join(tool_args.get(idx, [])),
                    ),
                    tool_call_index=idx,
                )

        yield StreamEvent(kind="message_completed")


async def _translate_chunk(
    chunk: dict[str, Any],
    tool_names: dict[int, str],
    tool_ids: dict[int, str],
    tool_args: dict[int, list[str]],
) -> AsyncIterator[StreamEvent]:
    choices = chunk.get("choices") or []
    if not choices:
        return
    delta = choices[0].get("delta") or {}

    text = delta.get("content")
    if isinstance(text, str) and text:
        yield StreamEvent(kind="text_delta", text=text)

    for tc in delta.get("tool_calls") or []:
        idx = int(tc.get("index", 0))
        func = tc.get("function") or {}
        tc_id = tc.get("id")
        if tc_id:
            tool_ids.setdefault(idx, tc_id)
        func_name = func.get("name")
        if func_name and idx not in tool_names:
            tool_names[idx] = func_name
            yield StreamEvent(
                kind="tool_call_started",
                tool_call=ToolCallBlock(
                    call_id=tool_ids.get(idx, ""),
                    tool_name=func_name,
                    arguments_json="",
                ),
                tool_call_index=idx,
            )
        func_args = func.get("arguments")
        if func_args:
            tool_args.setdefault(idx, []).append(func_args)
            yield StreamEvent(
                kind="tool_call_arguments_delta",
                arguments_delta=func_args,
                tool_call_index=idx,
            )
