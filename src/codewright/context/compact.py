from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codewright.context.manager import ContextManager, approx_token_count
from codewright.context.summarizer import Summarizer
from codewright.llm.base import CanonicalMessage, LLMProvider
from codewright.tools.truncate import truncate_middle

if TYPE_CHECKING:  # pragma: no cover
    from codewright.agent.turn_context import TurnContext


_USER_TAIL_TOKEN_BUDGET = 20_000
_SUMMARY_PREFIX = "[COMPACTED SUMMARY]:"
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "compact.md"


def load_compact_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


class LLMSummarizer(Summarizer):

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def summarize(
        self, messages: list[CanonicalMessage], compact_prompt: str
    ) -> str:

        prompt_messages = [
            *messages,
            CanonicalMessage(role="user", content=compact_prompt),
        ]
        stream = self._llm.stream(prompt_messages, tools=[], turn_context=None)
        if hasattr(stream, "__aiter__"):
            iterator = stream
        else:
            iterator = await stream

        parts: list[str] = []
        async for ev in iterator:
            if ev.kind == "text_delta" and ev.text:
                parts.append(ev.text)
            elif ev.kind == "error":
                raise RuntimeError(
                    f"summarizer LLM error: {ev.error or 'unknown'}"
                )
        return "".join(parts).strip()


def _message_text(msg: CanonicalMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.text for b in msg.content if b.text)


def _is_summary_message(msg: CanonicalMessage) -> bool:
    return msg.role == "user" and _message_text(msg).startswith(_SUMMARY_PREFIX)


def _collect_user_messages_tail(
    snapshot: tuple[CanonicalMessage, ...] | list[CanonicalMessage],
    budget: int,
) -> list[CanonicalMessage]:

    selected: list[CanonicalMessage] = []
    remaining = budget
    for msg in reversed(list(snapshot)):
        if msg.role != "user":
            continue
        if _is_summary_message(msg):
            continue
        text = _message_text(msg)
        cost = approx_token_count(text)
        if cost <= remaining:
            selected.append(msg)
            remaining -= cost
            continue
        if not selected:

            max_chars = max(1, remaining * 4)
            truncated = truncate_middle(text, max_chars=max_chars)
            selected.append(
                CanonicalMessage(role="user", content=truncated)
            )
        break
    selected.reverse()
    return selected


async def compact_history(
    context: ContextManager,
    summarizer: Summarizer,
    turn_context: TurnContext | None = None,
) -> None:

    del turn_context
    snapshot = context.snapshot()
    if not snapshot:
        return
    user_tail = _collect_user_messages_tail(snapshot, _USER_TAIL_TOKEN_BUDGET)
    summary_text = await summarizer.summarize(list(snapshot), load_compact_prompt())
    if not summary_text:
        summary_text = "(no summary available)"

    new_history: list[CanonicalMessage] = [
        *user_tail,
        CanonicalMessage(
            role="user", content=f"{_SUMMARY_PREFIX}\n{summary_text}"
        ),
    ]
    context.replace_all(new_history)


run_compact = compact_history


__all__ = [
    "LLMSummarizer",
    "compact_history",
    "load_compact_prompt",
    "run_compact",
]
