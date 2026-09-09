"""Protocol layer round-trip + invariants."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from codewright.protocol import (
    AgentPath,
    AgentPathError,
    AskForApproval,
    EvAgentMessage,
    EvAgentMessageDelta,
    Event,
    EvExecApprovalRequest,
    EvSessionConfigured,
    EvShutdownComplete,
    EvTurnAborted,
    EvTurnCompleted,
    EvTurnStarted,
    InterAgentMessage,
    OpCompact,
    OpExecApprovalResponse,
    OpInterAgentCommunication,
    OpInterrupt,
    OpOverrideTurnContext,
    OpPatchApprovalResponse,
    OpShutdown,
    OpUserTurn,
    PendingAction,
    PermissionProfile,
    PlanItem,
    PlanItemStatus,
    ReviewDecision,
    Submission,
    ToolCall,
    UserInputText,
)

# -- Op round-trip ----------------------------------------------------------


def test_op_user_turn_roundtrip():
    sub = Submission(
        id="s1",
        op=OpUserTurn(
            items=[UserInputText(text="hello")],
            cwd=Path("/tmp/x"),
            approval_policy=AskForApproval.ON_REQUEST,
            permission_profile=PermissionProfile.WORKSPACE_WRITE,
            model="gpt-4o",
            final_output_json_schema={"type": "object", "properties": {}},
        ),
    )
    raw = sub.model_dump_json()
    back = Submission.model_validate_json(raw)
    assert back == sub
    assert isinstance(back.op, OpUserTurn)
    assert back.op.items[0].text == "hello"


def test_op_simple_variants_roundtrip():
    for op in (OpInterrupt(), OpCompact(), OpShutdown()):
        sub = Submission(id="x", op=op)
        back = Submission.model_validate_json(sub.model_dump_json())
        assert type(back.op) is type(op)


def test_op_approval_response_carries_decision():
    sub = Submission(
        id="x",
        op=OpExecApprovalResponse(request_id="req-1", decision=ReviewDecision.APPROVED),
    )
    back = Submission.model_validate_json(sub.model_dump_json())
    assert isinstance(back.op, OpExecApprovalResponse)
    assert back.op.decision is ReviewDecision.APPROVED


def test_op_patch_approval_response_roundtrip():
    sub = Submission(
        id="x",
        op=OpPatchApprovalResponse(request_id="req-2", decision=ReviewDecision.DENIED),
    )
    back = Submission.model_validate_json(sub.model_dump_json())
    assert isinstance(back.op, OpPatchApprovalResponse)
    assert back.op.decision is ReviewDecision.DENIED


def test_op_inter_agent_communication_roundtrip():
    op = OpInterAgentCommunication(
        author=AgentPath.root(),
        recipient=AgentPath.root().child("worker_api"),
        content="please continue",
        trigger_turn=True,
    )
    sub = Submission(id="x", op=op)
    back = Submission.model_validate_json(sub.model_dump_json())
    assert isinstance(back.op, OpInterAgentCommunication)
    assert back.op.recipient.segments == ("root", "worker_api")
    assert back.op.trigger_turn is True


def test_op_override_turn_context_roundtrip():
    sub = Submission(
        id="x",
        op=OpOverrideTurnContext(
            cwd=Path("/tmp/y"),
            approval_policy=AskForApproval.NEVER,
            permission_profile=PermissionProfile.READ_ONLY,
            model="gpt-4o",
        ),
    )
    back = Submission.model_validate_json(sub.model_dump_json())
    assert isinstance(back.op, OpOverrideTurnContext)
    assert back.op.approval_policy is AskForApproval.NEVER


def test_op_is_frozen():
    op = OpInterrupt()
    with pytest.raises(ValidationError):
        op.type = "shutdown"  # type: ignore[misc]


# -- Event round-trip -------------------------------------------------------


def test_event_lifecycle_variants_roundtrip():
    events = [
        EvTurnStarted(turn_id="t"),
        EvTurnCompleted(turn_id="t", last_agent_message="hi"),
        EvTurnAborted(turn_id="t", reason="interrupted"),
        EvAgentMessage(content="ok"),
        EvAgentMessageDelta(delta="ok"),
        EvShutdownComplete(),
    ]
    for ev in events:
        wrapped = Event(id="s", msg=ev)
        back = Event.model_validate_json(wrapped.model_dump_json())
        assert type(back.msg) is type(ev)


def test_event_exec_approval_request_carries_action():
    action = PendingAction(
        action_id="a1",
        kind="exec",
        summary="rm -rf /tmp/x",
        details={"argv": ["rm", "-rf", "/tmp/x"]},
    )
    ev = EvExecApprovalRequest(request_id="r1", action=action)
    wrapped = Event(id="s", msg=ev)
    back = Event.model_validate_json(wrapped.model_dump_json())
    assert isinstance(back.msg, EvExecApprovalRequest)
    assert back.msg.action.details["argv"] == ["rm", "-rf", "/tmp/x"]


def test_event_session_configured_serializes_permission_profile_as_string():
    ev = EvSessionConfigured(
        session_id="s",
        model="gpt-4o",
        cwd="/tmp",
        permission_profile=PermissionProfile.WORKSPACE_WRITE.value,
    )
    wrapped = Event(id="", msg=ev)
    raw = wrapped.model_dump_json()
    assert '"permission_profile":"workspace_write"' in raw


# -- AgentPath --------------------------------------------------------------


def test_agent_path_root_and_child():
    root = AgentPath.root()
    assert str(root) == "/root"
    assert root.is_root()
    child = root.child("explorer_db")
    assert str(child) == "/root/explorer_db"
    assert not child.is_root()
    assert child.parent() == root
    assert root.parent() is None


def test_agent_path_parse_roundtrip():
    p = AgentPath.parse("/root/worker_api/explorer_search")
    assert p.segments == ("root", "worker_api", "explorer_search")
    assert str(p) == "/root/worker_api/explorer_search"


def test_agent_path_rejects_bad_segments():
    with pytest.raises(AgentPathError):
        AgentPath(segments=("notroot",))
    with pytest.raises(AgentPathError):
        AgentPath(segments=("root", "BAD-NAME"))  # uppercase + dash not allowed
    with pytest.raises(AgentPathError):
        AgentPath(segments=("root", "role"))  # missing _task
    with pytest.raises(AgentPathError):
        AgentPath.parse("missing-leading-slash")


def test_inter_agent_message_holds_paths():
    msg = InterAgentMessage(
        author=AgentPath.root(),
        recipient=AgentPath.root().child("worker_one"),
        content="ping",
        trigger_turn=False,
    )
    assert msg.content == "ping"
    assert msg.trigger_turn is False


# -- Supporting types -------------------------------------------------------


def test_tool_call_dataclass_immutable():
    import dataclasses

    tc = ToolCall(call_id="c1", tool_name="run_shell", arguments={"command": ["ls"]})
    with pytest.raises(dataclasses.FrozenInstanceError):
        tc.call_id = "c2"  # type: ignore[misc]


def test_plan_item_status_enum():
    item = PlanItem(step="design DB schema", status=PlanItemStatus.IN_PROGRESS)
    assert item.status is PlanItemStatus.IN_PROGRESS
    raw = item.model_dump_json()
    assert '"status":"in_progress"' in raw
