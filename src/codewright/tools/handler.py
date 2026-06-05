from __future__ import annotations

import abc

from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ToolSpec


class ToolHandler(abc.ABC):

    @property
    @abc.abstractmethod
    def tool_name(self) -> str: ...

    @abc.abstractmethod
    def spec(self) -> ToolSpec: ...

    @abc.abstractmethod
    async def handle(self, invocation: ToolInvocation) -> ToolResult: ...
