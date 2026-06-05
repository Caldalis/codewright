from typing import Any, Literal

import httpx

from codewright.llm.base import (
    CanonicalMessage,
    ContentBlock,
    LLMProvider,
    ProviderRequest,
    Role,
    StreamEvent,
    StreamEventKind,
    TokenUsage,
    ToolCallBlock,
)
from codewright.llm.chat_completions import ChatCompletionsAdapter
from codewright.llm.responses_api import ResponsesApiAdapter

ApiStyle = Literal["chat_completions", "responses"]


def create_llm_provider(
    api_style: ApiStyle,
    *,
    api_key: str,
    model: str,
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    extra: dict[str, Any] | None = None,
) -> LLMProvider:

    extras = extra or {}
    if api_style == "chat_completions":
        return ChatCompletionsAdapter(
            api_key=api_key,
            model=model,
            base_url=base_url,
            http_client=http_client,
        )
    if api_style == "responses":
        return ResponsesApiAdapter(
            api_key=api_key,
            model=model,
            base_url=base_url,
            http_client=http_client,
            store=bool(extras.get("store", False)),
            prompt_cache_key=extras.get("prompt_cache_key"),
        )
    raise ValueError(f"unknown api_style: {api_style!r}")


__all__ = [
    "ApiStyle",
    "CanonicalMessage",
    "ChatCompletionsAdapter",
    "ContentBlock",
    "LLMProvider",
    "ProviderRequest",
    "ResponsesApiAdapter",
    "Role",
    "StreamEvent",
    "StreamEventKind",
    "TokenUsage",
    "ToolCallBlock",
    "create_llm_provider",
]
