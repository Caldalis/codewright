"""TurnContext frozen-ness + defaults."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from codewright.agent.cancellation import CancellationToken
from codewright.agent.turn_context import TurnContext
from codewright.protocol import AskForApproval, PermissionProfile


def _ctx(**overrides) -> TurnContext:
    defaults = {
        "turn_id": "t1",
        "cwd": Path("/tmp"),
        "model": "gpt-x",
        "permission_profile": PermissionProfile.WORKSPACE_WRITE,
        "approval_policy": AskForApproval.ON_REQUEST,
        "cancellation_token": CancellationToken(),
    }
    defaults.update(overrides)
    return TurnContext(**defaults)


def test_turn_context_is_frozen_dataclass():
    assert dataclasses.is_dataclass(TurnContext)
    assert TurnContext.__dataclass_params__.frozen


def test_cannot_mutate():
    ctx = _ctx()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.model = "other"


def test_defaults():
    ctx = _ctx()
    assert ctx.max_context_tokens == 128_000
    assert ctx.compact_threshold == 0.9
    assert ctx.role == "default"


def test_value_equality():
    tok = CancellationToken()
    a = _ctx(cancellation_token=tok)
    b = _ctx(cancellation_token=tok)
    assert a == b
