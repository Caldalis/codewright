"""PromptBuilder three-layer assembly."""

from __future__ import annotations

from pathlib import Path

from codewright.agent.cancellation import CancellationToken
from codewright.agent.turn_context import TurnContext
from codewright.llm.base import CanonicalMessage
from codewright.prompts.builder import PromptBuilder, load_default_system_prompt
from codewright.protocol import AskForApproval, PermissionProfile


def _ctx() -> TurnContext:
    return TurnContext(
        turn_id="t1",
        cwd=Path("/tmp/work"),
        model="gpt-x",
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        approval_policy=AskForApproval.ON_REQUEST,
        cancellation_token=CancellationToken(),
    )


def test_default_system_prompt_nonempty():
    text = load_default_system_prompt()
    assert text.strip()
    assert "codewright" in text.lower()


def test_three_layer_order_without_history():
    pb = PromptBuilder("STATIC SYSTEM")
    msgs = pb.build(_ctx(), history=[], user_input="hello")
    assert len(msgs) == 3
    assert msgs[0].role == "system"
    assert msgs[0].content == "STATIC SYSTEM"
    assert msgs[1].role == "developer"
    assert msgs[2].role == "user"
    assert "hello" in msgs[2].content
    assert "<environment_context>" in msgs[2].content


def test_history_inserted_between_developer_and_user():
    pb = PromptBuilder("SYS")
    history = [
        CanonicalMessage(role="user", content="prior"),
        CanonicalMessage(role="assistant", content="response"),
    ]
    msgs = pb.build(_ctx(), history=history, user_input="next")
    assert [m.role for m in msgs] == ["system", "developer", "user", "assistant", "user"]
    assert msgs[2].content == "prior"
    assert msgs[3].content == "response"


def test_developer_layer_carries_turn_settings():
    pb = PromptBuilder("SYS")
    msgs = pb.build(_ctx(), history=[], user_input="hi")
    dev_content = msgs[1].content
    assert "permission_profile: workspace_write" in dev_content
    assert "approval_policy: on_request" in dev_content
    assert "model: gpt-x" in dev_content


def test_agents_md_injection_when_provided():
    pb = PromptBuilder("SYS")
    msgs = pb.build(
        _ctx(),
        history=[],
        user_input="hi",
        agents_md="Use tabs not spaces.",
    )
    assert "<user_instructions>" in msgs[1].content
    assert "Use tabs not spaces." in msgs[1].content


def test_no_user_message_when_input_empty():
    pb = PromptBuilder("SYS")
    msgs = pb.build(_ctx(), history=[], user_input="")
    # Just system + developer, no user.
    assert [m.role for m in msgs] == ["system", "developer"]


def test_plan_injected_as_tail_user_message():
    from codewright.protocol import PlanItem, PlanItemStatus

    pb = PromptBuilder("SYS")
    plan = [
        PlanItem(step="scope it", status=PlanItemStatus.COMPLETED),
        PlanItem(step="write code", status=PlanItemStatus.IN_PROGRESS),
        PlanItem(step="add tests", status=PlanItemStatus.PENDING),
    ]
    msgs = pb.build(_ctx(), history=[], user_input="go", plan=plan)
    # The plan rides at the very tail (after the user input) for max recency.
    assert [m.role for m in msgs] == ["system", "developer", "user", "user"]
    tail = msgs[-1].content
    assert "<current_plan>" in tail
    assert "[completed] scope it" in tail
    assert "[in_progress] write code" in tail
    assert "[pending] add tests" in tail
    # Reminds the model it can revise the plan via the tool.
    assert "update_plan" in tail


def test_plan_not_injected_when_empty():
    pb = PromptBuilder("SYS")
    msgs = pb.build(_ctx(), history=[], user_input="go", plan=[])
    assert [m.role for m in msgs] == ["system", "developer", "user"]
