"""Rollout JSONL: append-only writer + round-trip replay."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from codewright.agent.cancellation import CancellationToken
from codewright.agent.session import Session
from codewright.agent.turn import run_turn
from codewright.agent.turn_context import TurnContext
from codewright.context.manager import ContextManager
from codewright.llm.base import LLMProvider, StreamEvent
from codewright.persistence.rollout import (
    RolloutLine,
    RolloutRecorder,
    RolloutWriteError,
    SessionMeta,
)
from codewright.persistence.session_store import SessionStore
from codewright.prompts.builder import PromptBuilder, load_default_system_prompt
from codewright.protocol import AskForApproval, PermissionProfile


def _meta(session_id: str = "sess-1", cwd: Path | None = None) -> SessionMeta:
    return SessionMeta(
        session_id=session_id,
        cwd=str(cwd or Path(".").resolve()),
        model="mock",
        permission_profile=PermissionProfile.READ_ONLY.value,
        start_time=time.time(),
    )


def _tc() -> TurnContext:
    return TurnContext(
        turn_id="t1",
        cwd=Path("."),
        model="mock",
        permission_profile=PermissionProfile.READ_ONLY,
        approval_policy=AskForApproval.NEVER,
        cancellation_token=CancellationToken(),
    )


@pytest.mark.asyncio
async def test_plan_update_persists_and_replays(tmp_path: Path):
    """update_plan state must survive a resume (write -> replay round-trip)."""
    from codewright.protocol import PlanItem, PlanItemStatus

    path = tmp_path / "rollout.jsonl"
    rec = await RolloutRecorder.create(path, _meta())
    sess = Session(
        session_id="sess-1",
        cwd=tmp_path,
        permission_profile=PermissionProfile.READ_ONLY,
        rollout=rec,
    )
    try:
        plan = [
            PlanItem(step="scope", status=PlanItemStatus.COMPLETED),
            PlanItem(step="build", status=PlanItemStatus.IN_PROGRESS),
        ]
        await sess.record_plan_update(plan, "midway")
        await rec.flush()
    finally:
        await sess.shutdown()

    lines = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
    plan_lines = [ln for ln in lines if ln["type"] == "plan_update"]
    assert len(plan_lines) == 1
    assert plan_lines[0]["payload"]["plan"][1]["status"] == "in_progress"
    assert plan_lines[0]["payload"]["explanation"] == "midway"

    fresh = Session(
        session_id="sess-1",
        cwd=tmp_path,
        permission_profile=PermissionProfile.READ_ONLY,
    )
    try:
        fresh.replay(
            [RolloutLine(**ln) for ln in lines if ln["type"] != "session_meta"]
        )
        assert [p.step for p in fresh.plan] == ["scope", "build"]
        assert fresh.plan[1].status == PlanItemStatus.IN_PROGRESS
    finally:
        await fresh.shutdown()


class TestRolloutRecorderBasics:
    @pytest.mark.asyncio
    async def test_create_writes_session_meta_first_line(self, tmp_path: Path):
        path = tmp_path / "rollout.jsonl"
        rec = await RolloutRecorder.create(path, _meta())
        try:
            await rec.append(
                RolloutLine(type="user_msg", payload={"content": "hi"})
            )
            await rec.flush()
        finally:
            await rec.shutdown()

        contents = path.read_text(encoding="utf-8").splitlines()
        assert len(contents) >= 2
        first = json.loads(contents[0])
        assert first["type"] == "session_meta"
        assert first["payload"]["model"] == "mock"

    @pytest.mark.asyncio
    async def test_create_refuses_existing_file(self, tmp_path: Path):
        path = tmp_path / "rollout.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        with pytest.raises(FileExistsError):
            await RolloutRecorder.create(path, _meta())

    @pytest.mark.asyncio
    async def test_resume_appends_without_meta(self, tmp_path: Path):
        path = tmp_path / "rollout.jsonl"
        rec = await RolloutRecorder.create(path, _meta())
        await rec.flush()
        await rec.shutdown()

        rec2 = await RolloutRecorder.resume(path)
        try:
            await rec2.append(
                RolloutLine(type="user_msg", payload={"content": "second"})
            )
            await rec2.flush()
        finally:
            await rec2.shutdown()

        lines = path.read_text(encoding="utf-8").splitlines()
        types = [json.loads(ln)["type"] for ln in lines]
        # session_meta only appears once: written by create, not by resume.
        assert types.count("session_meta") == 1
        assert types[-1] == "user_msg"

    @pytest.mark.asyncio
    async def test_append_after_writer_failure_raises(
        self, tmp_path: Path, monkeypatch
    ):
        path = tmp_path / "rollout.jsonl"
        rec = await RolloutRecorder.create(path, _meta())
        # Force a terminal failure by setting the internal mailbox directly.
        # (Simulates disk-full / permission-revoked outcomes.)
        rec._terminal_failure[0] = OSError("disk full")
        with pytest.raises(RolloutWriteError):
            await rec.append(
                RolloutLine(type="user_msg", payload={"content": "x"})
            )
        await rec.shutdown()


class TestSessionStore:
    @pytest.mark.asyncio
    async def test_list_then_load_round_trip(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        meta = _meta(session_id="abc")
        rec = await store.create_recorder(meta)
        try:
            await rec.append(
                RolloutLine(type="user_msg", payload={"content": "msg-1"})
            )
            await rec.append(
                RolloutLine(
                    type="assistant_msg", payload={"content": "answer"}
                )
            )
            await rec.flush()
        finally:
            await rec.shutdown()

        listed = await store.list_sessions()
        assert [m.session_id for m in listed] == ["abc"]
        loaded_meta, lines = await store.load_session("abc")
        assert loaded_meta.session_id == "abc"
        types = [ln.type for ln in lines]
        assert types == ["session_meta", "user_msg", "assistant_msg"]

    @pytest.mark.asyncio
    async def test_load_missing_session_raises(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        with pytest.raises(FileNotFoundError):
            await store.load_session("missing")


class TwoTurnLLM(LLMProvider):
    """First call replies "first", second call replies "second"."""

    def __init__(self) -> None:
        self._n = 0

    async def stream(self, messages, tools, turn_context):  # type: ignore[override]
        self._n += 1
        text = "first" if self._n == 1 else "second"

        async def _gen():
            yield StreamEvent(kind="text_delta", text=text)
            yield StreamEvent(kind="message_completed")

        return _gen()


class TestEndToEndResume:
    @pytest.mark.asyncio
    async def test_run_turn_records_rollout_and_replay_rebuilds_history(
        self, tmp_path: Path
    ):
        store = SessionStore(tmp_path)
        meta = _meta(session_id="e2e")
        rec = await store.create_recorder(meta)

        cm = ContextManager(max_context_tokens=10_000)
        session = Session(
            session_id="e2e",
            cwd=Path("."),
            permission_profile=PermissionProfile.READ_ONLY,
            llm=TwoTurnLLM(),
            context_manager=cm,
            prompt_builder=PromptBuilder(load_default_system_prompt()),
            rollout=rec,
        )
        try:
            await run_turn(session, _tc(), user_input="hello")
            await run_turn(session, _tc(), user_input="again")
            await rec.flush()
        finally:
            await session.shutdown()
            await rec.shutdown()

        # File contains expected types.
        path = store.session_path("e2e")
        types = [json.loads(ln)["type"] for ln in path.read_text(encoding="utf-8").splitlines()]
        assert types[0] == "session_meta"
        # Two user msgs and two assistant msgs.
        assert types.count("user_msg") == 2
        assert types.count("assistant_msg") == 2

        # Resume into a fresh session: replay rebuilds history.
        _meta_loaded, lines = await store.load_session("e2e")
        cm2 = ContextManager(max_context_tokens=10_000)
        resumed = Session(
            session_id="e2e",
            cwd=Path("."),
            permission_profile=PermissionProfile.READ_ONLY,
            context_manager=cm2,
        )
        try:
            resumed.replay(lines)
            snap = cm2.snapshot()
            roles = [m.role for m in snap]
            assert roles == ["user", "assistant", "user", "assistant"]
            contents = [
                m.content if isinstance(m.content, str) else "" for m in snap
            ]
            assert contents == ["hello", "first", "again", "second"]
        finally:
            await resumed.shutdown()


class TestAppendOnlyMode:
    def test_rollout_module_uses_append_mode(self):
        """G_arch §M2: file is opened with mode 'a' (append-only)."""
        src = Path(__file__).resolve().parent.parent / "src" / "codewright" / "persistence" / "rollout.py"
        text = src.read_text(encoding="utf-8")
        assert "open(path, \"a\"" in text or "open(path, 'a'" in text


class TestFirstLineEnforcement:
    """_SHARED.md §M2: first line MUST be session_meta. Fail-fast on violation."""

    @pytest.mark.asyncio
    async def test_first_line_not_session_meta_raises(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        path = store.session_path("bad")
        path.parent.mkdir(parents=True, exist_ok=True)
        # File starts with a user_msg line — invalid, even though a meta line
        # appears later we must not silently accept this.
        line1 = RolloutLine(type="user_msg", payload={"content": "no meta yet"})
        line2 = RolloutLine(type="session_meta", payload=_meta("bad").model_dump())
        path.write_text(
            line1.model_dump_json() + "\n" + line2.model_dump_json() + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError) as exc:
            await store.load_session("bad")
        assert "expected 'session_meta'" in str(exc.value)

    @pytest.mark.asyncio
    async def test_empty_file_raises(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        path = store.session_path("empty")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError) as exc:
            await store.load_session("empty")
        assert "empty rollout" in str(exc.value)

    @pytest.mark.asyncio
    async def test_malformed_first_line_raises(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        path = store.session_path("garbage")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json at all\n", encoding="utf-8")
        with pytest.raises(ValueError) as exc:
            await store.load_session("garbage")
        assert "not a valid rollout line" in str(exc.value)


# ---------------------------------------------------------------------------
# End-to-end: 3 turns (incl. tool call) → close → resume → 4th turn
# ---------------------------------------------------------------------------


class ScriptedLLM(LLMProvider):
    """Hard-coded transcript: turn 1 calls a tool then replies; turns 2/3 text-only."""

    def __init__(self, transcript: list[str | tuple[str, str, str]]):
        """``transcript`` is a flat sequence of LLM samples.

        Each entry is either a plain string (assistant text reply) or a
        3-tuple ``(call_id, tool_name, arguments_json)`` (one tool call).
        """
        self._transcript = list(transcript)
        self._n = 0

    async def stream(self, messages, tools, turn_context):  # type: ignore[override]
        from codewright.llm.base import StreamEvent, ToolCallBlock

        entry = self._transcript[self._n]
        self._n += 1

        async def _gen():
            if isinstance(entry, tuple):
                yield StreamEvent(
                    kind="tool_call_completed",
                    tool_call=ToolCallBlock(
                        call_id=entry[0],
                        tool_name=entry[1],
                        arguments_json=entry[2],
                    ),
                )
                yield StreamEvent(kind="message_completed")
            else:
                yield StreamEvent(kind="text_delta", text=entry)
                yield StreamEvent(kind="message_completed")

        return _gen()


class EchoHandler:
    """Local lookalike of a ToolHandler. Built inline to avoid imports at module level."""

    def __init__(self) -> None:
        from codewright.tools.spec import ToolSpec

        self._spec = ToolSpec(
            name="echo_tool",
            description="echo back",
            parameters={"type": "object"},
            supports_parallel=False,
            requires_approval=False,
        )

    @property
    def tool_name(self) -> str:
        return "echo_tool"

    def spec(self):
        return self._spec

    async def handle(self, invocation):
        from codewright.tools.result import ToolResult

        text = invocation.arguments.get("text", "") if isinstance(invocation.arguments, dict) else ""
        return ToolResult(success=True, body=f"echoed: {text}")


class TestThreePlusOneTurnResume:
    """Full lifecycle: 3 turns (incl. tool call) → shutdown → resume_session → 4th turn."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, tmp_path: Path):
        import json as _json

        from codewright.agent.resume import resume_session
        from codewright.llm.base import CanonicalMessage

        store = SessionStore(tmp_path)
        meta = _meta(session_id="3plus1")
        recorder = await store.create_recorder(meta)

        # Turn 1: tool call → tool result → final text "first"
        # Turn 2: text "second"
        # Turn 3: text "third"
        llm_a = ScriptedLLM(
            transcript=[
                ("echo-1", "echo_tool", '{"text": "alpha"}'),
                "first",
                "second",
                "third",
            ]
        )

        cm = ContextManager(max_context_tokens=10_000)
        session = Session(
            session_id="3plus1",
            cwd=Path("."),
            permission_profile=PermissionProfile.READ_ONLY,
            llm=llm_a,
            context_manager=cm,
            prompt_builder=PromptBuilder(load_default_system_prompt()),
            rollout=recorder,
        )
        session.tool_registry.register(EchoHandler())  # type: ignore[arg-type]

        try:
            r1 = await run_turn(session, _tc(), user_input="turn 1")
            assert r1 == "first"
            r2 = await run_turn(session, _tc(), user_input="turn 2")
            assert r2 == "second"
            r3 = await run_turn(session, _tc(), user_input="turn 3")
            assert r3 == "third"
            await recorder.flush()
        finally:
            await session.shutdown()  # also flushes + closes recorder

        # File assertion: 1 meta + (4 turn1 + 2 turn2 + 2 turn3) = 9 lines.
        # Turn 1 sequence: user_msg, assistant_msg(with tool_calls),
        # tool_result, assistant_msg.
        path = store.session_path("3plus1")
        types_before = [
            _json.loads(ln)["type"]
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert types_before == [
            "session_meta",
            "user_msg",
            "assistant_msg",
            "tool_result",
            "assistant_msg",
            "user_msg",
            "assistant_msg",
            "user_msg",
            "assistant_msg",
        ]

        # Resume into a fresh Session (different LLM instance + different
        # ContextManager; only the on-disk rollout is shared).
        llm_b = ScriptedLLM(transcript=["fourth"])
        resumed = await resume_session(
            workspace_root=tmp_path,
            session_id="3plus1",
            llm=llm_b,
            prompt_builder=PromptBuilder(load_default_system_prompt()),
        )

        # After replay, in-memory history must match the 8 content events
        # from the persisted run (session_meta is skipped).
        history_after_replay = resumed.context.snapshot()
        roles_after_replay = [m.role for m in history_after_replay]
        assert roles_after_replay == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        # Tool message preserves the tool_call_id so future model samples
        # see a coherent transcript.
        tool_msg = history_after_replay[2]
        assert tool_msg.tool_call_id == "echo-1"
        assert "echoed: alpha" in (
            tool_msg.content if isinstance(tool_msg.content, str) else ""
        )

        # Turn 4: text-only. The resumed session has no echo_tool registered
        # (we re-registered nothing); turn 4 doesn't need it because the LLM
        # transcript is text-only.
        try:
            r4 = await run_turn(resumed, _tc(), user_input="turn 4")
            assert r4 == "fourth"
        finally:
            await resumed.shutdown()

        # File now has the 4th turn's two lines appended to the same file.
        types_after = [
            _json.loads(ln)["type"]
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert types_after == [*types_before, "user_msg", "assistant_msg"]

        # End-state history: 8 replayed + 2 new = 10 messages.
        final_snap = resumed.context.snapshot()
        assert len(final_snap) == 10
        assert final_snap[-2] == CanonicalMessage(role="user", content="turn 4")
        assert final_snap[-1].role == "assistant"
        assert final_snap[-1].content == "fourth"
