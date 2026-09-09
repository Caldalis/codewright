"""update_plan: updates session.plan + emits EvPlanUpdate."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from codewright.agent.cancellation import CancellationToken
from codewright.agent.session import Session
from codewright.agent.turn_context import TurnContext
from codewright.protocol import AskForApproval, EvPlanUpdate, PermissionProfile, PlanItemStatus
from codewright.tools.handlers.update_plan import UpdatePlanHandler
from codewright.tools.invocation import ToolInvocation


def _inv(sess, args, cwd):
    return ToolInvocation(
        session=sess,
        turn_context=TurnContext(
            turn_id="t",
            cwd=cwd,
            model="m",
            permission_profile=PermissionProfile.WORKSPACE_WRITE,
            approval_policy=AskForApproval.NEVER,
            cancellation_token=CancellationToken(),
        ),
        call_id=uuid.uuid4().hex,
        tool_name="update_plan",
        arguments=args,
        cancellation_token=CancellationToken(),
    )


@pytest.mark.asyncio
async def test_update_plan_persists_and_emits(tmp_path: Path) -> None:
    sess = Session(
        session_id="s",
        cwd=tmp_path,
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
    )
    try:
        await sess.next_event()  # drain SessionConfigured
        h = UpdatePlanHandler()
        result = await h.handle(
            _inv(
                sess,
                {
                    "plan": [
                        {"step": "scope", "status": "completed"},
                        {"step": "build", "status": "in_progress"},
                    ],
                    "explanation": "starting work",
                },
                tmp_path,
            )
        )
        assert result.success is True
        assert len(sess.plan) == 2
        assert sess.plan[1].status == PlanItemStatus.IN_PROGRESS

        ev = await sess.next_event()
        assert isinstance(ev.msg, EvPlanUpdate)
        assert ev.msg.plan[0].step == "scope"
        assert ev.msg.explanation == "starting work"
    finally:
        await sess.shutdown()
