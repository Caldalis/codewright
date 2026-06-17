from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from codewright.protocol.approval import PendingAction


class PlanItemStatus(StrEnum):

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

class PlanItem(BaseModel):

    model_config = ConfigDict(frozen=True)
    step: str
    status: PlanItemStatus



class _EventBase(BaseModel):
    model_config = ConfigDict(frozen=True)


class EvTurnStarted(_EventBase):
    type: Literal["turn_started"] = "turn_started"
    turn_id: str


class EvTurnCompleted(_EventBase):
    type: Literal["turn_completed"] = "turn_completed"
    turn_id: str
    last_agent_message: str | None = None


TurnAbortReason = Literal["interrupted", "error"]


class EvTurnAborted(_EventBase):
    type: Literal["turn_aborted"] = "turn_aborted"
    turn_id: str
    reason: TurnAbortReason


class EvAgentMessage(_EventBase):
    type: Literal["agent_message"] = "agent_message"
    content: str


class EvAgentMessageDelta(_EventBase):
    type: Literal["agent_message_delta"] = "agent_message_delta"
    delta: str


class EvAgentReasoning(_EventBase):

    type: Literal["agent_reasoning"] = "agent_reasoning"
    content: str


class EvToolCallStarted(_EventBase):
    type: Literal["tool_call_started"] = "tool_call_started"
    call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class EvToolCallCompleted(_EventBase):
    type: Literal["tool_call_completed"] = "tool_call_completed"
    call_id: str
    success: bool
    body: str


class EvExecApprovalRequest(_EventBase):
    type: Literal["exec_approval_request"] = "exec_approval_request"
    request_id: str
    action: PendingAction


class EvPatchApprovalRequest(_EventBase):
    type: Literal["patch_approval_request"] = "patch_approval_request"
    request_id: str
    action: PendingAction


class EvPlanUpdate(_EventBase):
    type: Literal["plan_update"] = "plan_update"
    plan: list[PlanItem]
    explanation: str | None = None


class EvTokenCount(_EventBase):
    type: Literal["token_count"] = "token_count"
    input: int
    output: int
    total: int


class EvCompactionStarted(_EventBase):
    type: Literal["compaction_started"] = "compaction_started"
    reason: Literal["auto", "manual"] = "auto"
    tokens_before: int = 0


class EvCompactionCompleted(_EventBase):
    type: Literal["compaction_completed"] = "compaction_completed"
    tokens_before: int = 0
    tokens_after: int = 0


class EvError(_EventBase):
    type: Literal["error"] = "error"
    message: str


class EvWarning(_EventBase):
    type: Literal["warning"] = "warning"
    message: str


class EvSessionConfigured(_EventBase):

    type: Literal["session_configured"] = "session_configured"
    session_id: str
    model: str
    cwd: str
    permission_profile: str


class EvShutdownComplete(_EventBase):
    type: Literal["shutdown_complete"] = "shutdown_complete"


EventMsg = Annotated[
    EvTurnStarted | EvTurnCompleted | EvTurnAborted | EvAgentMessage | EvAgentMessageDelta | EvAgentReasoning | EvToolCallStarted | EvToolCallCompleted | EvExecApprovalRequest | EvPatchApprovalRequest | EvPlanUpdate | EvTokenCount | EvCompactionStarted | EvCompactionCompleted | EvError | EvWarning | EvSessionConfigured | EvShutdownComplete,
    Field(discriminator="type"),
]

class Event(BaseModel):

    model_config = ConfigDict(frozen=True)
    id: str
    msg: EventMsg
