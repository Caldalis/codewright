from __future__ import annotations

import abc

from codewright.llm.base import CanonicalMessage


class Summarizer(abc.ABC):

    @abc.abstractmethod
    async def summarize(
        self, messages: list[CanonicalMessage], compact_prompt: str
    ) -> str: ...
