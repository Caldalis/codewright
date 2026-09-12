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


# Keys a provider may add to its own assistant message, which it then expects
# back. An allow-list rather than "everything we do not recognise": a delta also
# carries role, refusal and provider bookkeeping, and echoing those back is a
# good way to get a request rejected for a new reason.
#
# The spellings differ by provider and a provider only accepts the one it
# issued, so each is carried under its own name. Extend this when a provider
# turns up using another.
_PASSTHROUGH_KEYS = (
    "reasoning_content",   # DeepSeek, GLM, Kimi, Grok, Doubao
    "reasoning",           # some OpenAI-compatible gateways
    "thinking",            # seen in streaming deltas
    "encrypted_content",   # Doubao, alongside reasoning_content
)


def _to_provider_messages(messages: list[CanonicalMessage]) -> list[dict[str, Any]]:

    system_parts: list[str] = []
    body_messages: list[CanonicalMessage] = []
    for m in messages:
        if m.role == "system":
            text = _content_to_str(m.content)
            if text:
                system_parts.append(text)
            continue
        if m.role == "developer":
            text = _content_to_str(m.content)
            if text:
                system_parts.append(
                    "<developer_instructions>\n"
                    f"{text}\n"
                    "</developer_instructions>"
                )
            continue
        body_messages.append(m)

    out: list[dict[str, Any]] = []
    if system_parts:
        out.append({"role": "system", "content": "\n\n".join(system_parts)})

    for m in body_messages:
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
            # Hand back whatever this provider attached. Only ever present
            # because the same provider produced it, so a provider that does
            # not use these keys never sees them.
            if m.provider_extras:
                entry.update(m.provider_extras)
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
        entry = {"role": m.role, "content": _content_to_str(m.content)}
        if m.role == "assistant" and m.provider_extras:
            entry.update(m.provider_extras)
        out.append(entry)
    return out


def _to_provider_tools(tools: list[Any]) -> list[dict[str, Any]]:

    out: list[dict[str, Any]] = []
    for t in tools:
        name = getattr(t, "name", None)
        description = getattr(t, "description", "")
        parameters = _tool_parameters(t)
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
            "stream_options": {"include_usage": True},
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
        # Arrives in fragments like content does, so join rather than overwrite.
        extras: dict[str, list[str]] = {}

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
                    if line.startswith("data:"):
                        data = line[len("data:") :].strip()
                    elif line.lstrip().startswith("{"):
                        data = line.strip()
                    else:
                        continue
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        # SSE comment line / keep-alive — provider-dependent.
                        continue

                    provider_error = _provider_error_message(chunk)
                    if provider_error is not None:
                        yield StreamEvent(kind="error", error=provider_error)
                        return

                    _collect_passthrough(chunk, extras)

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

        joined = {k: "".join(v) for k, v in extras.items() if "".join(v)}
        yield StreamEvent(
            kind="message_completed",
            provider_extras=joined or None,
        )


def _collect_passthrough(chunk: dict[str, Any], into: dict[str, list[str]]) -> None:
    """Accumulate the provider's own fields off one chunk.

    The two frame shapes do not mean the same thing, and treating them alike
    double-counts. A `delta` is an increment, so it appends. A `message` is the
    finished value, so it replaces -- a provider that streams deltas and then
    repeats the whole message in its final frame would otherwise report the
    trace twice. A provider that does not stream sends only `message`, and
    replacing an empty accumulator is the same as appending to it.
    """
    choices = chunk.get("choices") or []
    if not choices:
        return
    choice = choices[0]

    delta = choice.get("delta")
    if isinstance(delta, dict):
        for key in _PASSTHROUGH_KEYS:
            value = delta.get(key)
            if isinstance(value, str) and value:
                into.setdefault(key, []).append(value)

    message = choice.get("message")
    if isinstance(message, dict):
        for key in _PASSTHROUGH_KEYS:
            value = message.get(key)
            if isinstance(value, str) and value:
                into[key] = [value]


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


def _provider_error_message(chunk: dict[str, Any]) -> str | None:
    err = chunk.get("error")
    if err is None and chunk.get("type") == "error":
        err = chunk
    if err is None:
        return None
    if isinstance(err, dict):
        message = err.get("message") or json.dumps(err, ensure_ascii=False)
        code = err.get("code")
        if code is not None:
            return f"provider error {code}: {message}"
        return f"provider error: {message}"
    return f"provider error: {err}"
