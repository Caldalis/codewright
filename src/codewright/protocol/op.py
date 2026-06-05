from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from codewright.protocol.agent_messages import AgentPath
from codewright.protocol.approval import PermissionProfile, ReviewDecision


class AskForApproval(StrEnum):


    UNTRUSTED = "untrusted"
    ON_REQUEST = "on_request"
    NEVER = "never"



class UserInputText(BaseModel):

    model_config = ConfigDict(frozen=True)
    type: Literal["text"] = "text"
    text: str


UserInput = Annotated[UserInputText, Field(discriminator="type")]


class _OpBase(BaseModel):

    model_config = ConfigDict(frozen=True)


class OpUserTurn(_OpBase):


    type: Literal["user_turn"] = "user_turn"
    items: list[UserInputText]
    cwd: Path | None = None
    approval_policy: AskForApproval | None = None
    permission_profile: PermissionProfile | None = None
    model: str | None = None
    final_output_json_schema: dict[str, Any] | None = None


class OpInterrupt(_OpBase):

    type: Literal["interrupt"] = "interrupt"


class OpCompact(_OpBase):

    type: Literal["compact"] = "compact"


class OpShutdown(_OpBase):

    type: Literal["shutdown"] = "shutdown"


class OpExecApprovalResponse(_OpBase):
    type: Literal["exec_approval_response"] = "exec_approval_response"
    request_id: str
    decision: ReviewDecision


class OpPatchApprovalResponse(_OpBase):

    type: Literal["patch_approval_response"] = "patch_approval_response"
    request_id: str
    decision: ReviewDecision


class OpInterAgentCommunication(_OpBase):

    type: Literal["inter_agent_communication"] = "inter_agent_communication"
    author: AgentPath
    recipient: AgentPath
    content: str
    trigger_turn: bool


class OpOverrideTurnContext(_OpBase):

    type: Literal["override_turn_context"] = "override_turn_context"
    cwd: Path | None = None
    approval_policy: AskForApproval | None = None
    permission_profile: PermissionProfile | None = None
    model: str | None = None


Op = Annotated[
    OpUserTurn | OpInterrupt | OpCompact | OpShutdown | OpExecApprovalResponse | OpPatchApprovalResponse | OpInterAgentCommunication | OpOverrideTurnContext,
    Field(discriminator="type"),
]


class Submission(BaseModel):

    model_config = ConfigDict(frozen=True)
    id: str
    op: Op
