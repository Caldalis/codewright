"""Phase 2 (LEARN half) of self-improving skills: red->green detection, the
deterministic write gate (evidence/anti-cheat/dedup/validity), and the
DistillationCoordinator end-to-end with a fake LLM."""

from __future__ import annotations

import json
from pathlib import Path

from codewright.agent.cancellation import CancellationToken
from codewright.agent.distill import (
    _DISTILL_MAX_ATTEMPTS,
    _EDIT_STORE_CAP,
    _EVENT_BYTE_BUDGET,
    _TRACKER_MAX_CMDS,
    _TRACKER_OUTPUT_CAP,
    STALE_AGE_SECONDS,
    DistillationCoordinator,
    VerificationTracker,
    _distill_candidates,
    _Transition,
    anti_cheat_ok,
    decide_status,
    extract_test_counts,
    gate_candidate,
    is_test_command,
    parse_candidates,
    render_skill_md,
    set_skill_status,
    write_provisional_skill,
)
from codewright.agent.skills import load_skills
from codewright.agent.turn_context import TurnContext
from codewright.llm.base import StreamEvent
from codewright.protocol import AskForApproval, PermissionProfile
from codewright.tools.result import ToolResult


def _ctx() -> TurnContext:
    return TurnContext(
        turn_id="t",
        cwd=Path("."),
        model="m",
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        approval_policy=AskForApproval.NEVER,
        cancellation_token=CancellationToken(),
    )


def _shell(exit_code: int, body: str) -> ToolResult:
    return ToolResult(
        success=(exit_code == 0), body=body, structured_data={"exit_code": exit_code}
    )


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self._text = text

    def stream(self, messages, tools, turn_context):
        text = self._text

        async def _gen():
            yield StreamEvent(kind="text_delta", text=text)

        return _gen()


class _FakeSession:
    def __init__(self, cwd: Path, llm: _FakeLLM) -> None:
        self.cwd = cwd
        self.llm = llm
        self.skill_registry = load_skills(cwd)
        self.events: list = []

    async def emit_event(self, msg) -> None:
        self.events.append(msg)


def _candidates_json(*cands: dict) -> str:
    return json.dumps({"candidates": list(cands)})


# --------------------------------------------------------------------------- #
# test-runner recognition + count parsing
# --------------------------------------------------------------------------- #


def test_is_test_command_recognises_runners():
    assert is_test_command("uv run pytest tests/")
    assert is_test_command("npm test")
    assert is_test_command("go test ./...")
    assert is_test_command("cargo nextest run")
    assert is_test_command("python -m pytest -q")
    assert not is_test_command("ls -la")
    assert not is_test_command("python build.py")
    assert is_test_command("mytest --all", extra=("mytest",))


def test_extract_test_counts():
    assert extract_test_counts("=== 3 failed, 5 passed in 1.2s ===") == _counts(5, 3, 0)
    assert extract_test_counts("=== 8 passed in 0.5s ===") == _counts(8, 0, 0)
    assert extract_test_counts("1 error, 2 passed") == _counts(2, 0, 1)
    assert extract_test_counts("no summary here") is None


def _counts(p, f, e):
    from codewright.agent.distill import TestCounts

    return TestCounts(p, f, e)


def test_anti_cheat_count_decrease_blocks():
    assert anti_cheat_ok("3 failed, 5 passed", "8 passed", edited_test_files=False) is True
    assert anti_cheat_ok("3 failed, 5 passed", "5 passed", edited_test_files=False) is False


def test_anti_cheat_unparseable_blocks_only_when_tests_edited():
    assert anti_cheat_ok("FAILED", "OK", edited_test_files=False) is True
    assert anti_cheat_ok("FAILED", "OK", edited_test_files=True) is False


# --------------------------------------------------------------------------- #
# candidate parsing + gate + write
# --------------------------------------------------------------------------- #


def test_parse_candidates_plain_fenced_and_garbage():
    a = parse_candidates(_candidates_json({"name": "x", "description": "d", "body": "b", "type": "fact"}))
    assert len(a) == 1 and a[0]["name"] == "x" and a[0]["type"] == "fact"
    b = parse_candidates('```json\n{"candidates":[{"name":"y","body":"b"}]}\n```')
    assert len(b) == 1 and b[0]["name"] == "y" and b[0]["type"] == "skill"  # default
    assert parse_candidates("total garbage, no json") == []
    assert parse_candidates('preamble {"candidates": []} trailer') == []


def test_gate_candidate_rules():
    base = {"type": "skill", "name": "good-name", "description": "d", "body": "b"}
    assert gate_candidate(base, set()) is True
    assert gate_candidate({**base, "name": "Bad_Name"}, set()) is False  # invalid chars
    assert gate_candidate({**base, "name": "-bad"}, set()) is False  # leading hyphen
    assert gate_candidate(base, {"good-name"}) is False  # duplicate / would overwrite
    assert gate_candidate({**base, "body": ""}, set()) is False
    assert gate_candidate({**base, "description": ""}, set()) is False
    assert gate_candidate({**base, "type": "weird"}, set()) is False


def test_write_provisional_skill_roundtrips(tmp_path: Path):
    cand = {
        "type": "lesson",
        "name": "my-skill",
        "description": "What it does: use when X.",
        "body": "Do Y.",
    }
    path = write_provisional_skill(tmp_path, cand, "pytest: red -> green @ now")
    assert path is not None and path.exists()
    skill = load_skills(tmp_path).get("my-skill")
    assert skill is not None
    assert skill.source == "learned" and skill.status == "provisional" and skill.type == "lesson"
    assert skill.description == "What it does: use when X."  # colon survived quoting
    assert "Do Y." in load_skills(tmp_path).load_body(skill)


def test_render_skill_md_is_parseable_and_marked_provisional(tmp_path: Path):
    md = render_skill_md({"name": "n", "description": "d", "body": "b", "type": "fact"}, "ev")
    assert "cw-status: provisional" in md
    assert "cw-source: learned" in md


# --------------------------------------------------------------------------- #
# VerificationTracker
# --------------------------------------------------------------------------- #


def test_verification_tracker_flags_red_then_green():
    t = VerificationTracker()
    assert t.observe("pytest", 1, "1 failed") is None  # first run: red, no transition
    transition = t.observe("pytest", 0, "1 passed")  # red -> green
    assert transition is not None and transition.kind == "red_green"
    assert transition.prev_output == "1 failed" and transition.cur_output == "1 passed"
    assert t.observe("pytest", 0, "1 passed") is None  # already green: no new transition
    regression = t.observe("pytest", 1, "1 failed")  # green -> red
    assert regression is not None and regression.kind == "green_red"
    assert t.observe("ls -la", 1, "x") is None  # non-test command ignored
    assert t.observe("pytest", None, "x") is None  # missing exit code ignored


# --------------------------------------------------------------------------- #
# DistillationCoordinator end-to-end (fake LLM + fake session)
# --------------------------------------------------------------------------- #


async def test_coordinator_writes_provisional_skill_on_red_green(tmp_path: Path):
    llm = _FakeLLM(
        _candidates_json(
            {
                "type": "lesson",
                "name": "auth-test-fixtures",
                "description": "How to run auth tests; they need the db_session fixture.",
                "body": "Auth tests require the db_session fixture from tests/conftest.py.",
            }
        )
    )
    sess = _FakeSession(tmp_path, llm)
    coord = DistillationCoordinator(tmp_path)
    coord.note_user("fix the failing auth test")
    coord.note_tool("apply_patch", {"patch": "*** Update File: src/auth.py\n+fix"}, _shell(0, "applied"))
    coord.note_tool("shell", {"command": "uv run pytest tests/test_auth.py"}, _shell(1, "3 failed, 5 passed in 1s"))
    coord.note_tool("shell", {"command": "uv run pytest tests/test_auth.py"}, _shell(0, "8 passed in 1s"))

    await coord.on_batch_end(sess, _ctx())
    await coord.drain()

    skill = load_skills(tmp_path).get("auth-test-fixtures")
    assert skill is not None
    assert skill.source == "learned" and skill.status == "provisional"
    assert any("auth-test-fixtures" in getattr(e, "message", "") for e in sess.events)


async def test_coordinator_no_distill_without_green(tmp_path: Path):
    sess = _FakeSession(tmp_path, _FakeLLM("{}"))
    coord = DistillationCoordinator(tmp_path)
    coord.note_tool("shell", {"command": "pytest"}, _shell(1, "3 failed"))  # stays red
    await coord.on_batch_end(sess, _ctx())
    await coord.drain()
    assert load_skills(tmp_path).all_skills() == []


async def test_coordinator_anti_cheat_blocks_test_count_drop(tmp_path: Path):
    llm = _FakeLLM(_candidates_json({"type": "fact", "name": "x", "description": "d", "body": "b"}))
    sess = _FakeSession(tmp_path, llm)
    coord = DistillationCoordinator(tmp_path)
    coord.note_tool("shell", {"command": "pytest"}, _shell(1, "3 failed, 5 passed"))  # total 8
    coord.note_tool("shell", {"command": "pytest"}, _shell(0, "5 passed"))  # total 5 < 8
    await coord.on_batch_end(sess, _ctx())
    await coord.drain()
    assert load_skills(tmp_path).all_skills() == []


async def test_coordinator_gate_does_not_overwrite_existing(tmp_path: Path):
    existing = tmp_path / "skills" / "auth-test-fixtures"
    existing.mkdir(parents=True)
    (existing / "SKILL.md").write_text(
        "---\nname: auth-test-fixtures\ndescription: existing\n---\n\nOLD BODY\n",
        encoding="utf-8",
    )
    llm = _FakeLLM(
        _candidates_json(
            {"type": "lesson", "name": "auth-test-fixtures", "description": "d", "body": "NEW BODY"}
        )
    )
    sess = _FakeSession(tmp_path, llm)
    coord = DistillationCoordinator(tmp_path)
    coord.note_tool("shell", {"command": "pytest"}, _shell(1, "1 failed, 1 passed"))
    coord.note_tool("shell", {"command": "pytest"}, _shell(0, "2 passed"))
    await coord.on_batch_end(sess, _ctx())
    await coord.drain()
    body = (existing / "SKILL.md").read_text(encoding="utf-8")
    assert "OLD BODY" in body and "NEW BODY" not in body


# --------------------------------------------------------------------------- #
# Phase 3: maturity (promote / quarantine / stale) + status rewriting
# --------------------------------------------------------------------------- #


def test_decide_status_rules():
    assert decide_status("provisional", "learned", 2, 0, None) == "trusted"  # promote
    assert decide_status("provisional", "learned", 1, 0, None) is None  # not enough yet
    assert decide_status("provisional", "learned", 0, 2, None) == "quarantined"  # failures
    assert decide_status("trusted", "learned", 1, 3, None) == "quarantined"  # trusted can fall
    assert decide_status("provisional", "authored", 5, 0, None) is None  # authored unmanaged
    assert decide_status("quarantined", "learned", 9, 0, None) is None  # terminal
    assert decide_status("provisional", "learned", 0, 0, STALE_AGE_SECONDS + 1) == "quarantined"
    assert decide_status("provisional", "learned", 0, 0, 10.0) is None  # fresh enough


def test_set_skill_status_rewrites_in_place(tmp_path: Path):
    path = write_provisional_skill(
        tmp_path, {"type": "fact", "name": "foo", "description": "d", "body": "b"}, "ev"
    )
    assert load_skills(tmp_path).get("foo").status == "provisional"
    assert set_skill_status(path, "trusted") is True
    assert load_skills(tmp_path).get("foo").status == "trusted"


async def test_promotion_after_repeated_success_makes_fact_always_on(tmp_path: Path):
    write_provisional_skill(
        tmp_path,
        {"type": "fact", "name": "use-uv", "description": "how to test", "body": "uv run pytest"},
        "ev",
    )
    sess = _FakeSession(tmp_path, _FakeLLM('{"candidates": []}'))
    coord = DistillationCoordinator(tmp_path)
    # two clean red->green episodes (distinct commands) that both used the skill
    for cmd in ("pytest tests/a.py", "pytest tests/b.py"):
        coord.note_retrieval("use-uv")
        coord.note_tool("shell", {"command": cmd}, _shell(1, "1 failed"))
        coord.note_tool("shell", {"command": cmd}, _shell(0, "1 passed"))
        await coord.on_batch_end(sess, _ctx())
        await coord.drain()
    reg = load_skills(tmp_path)
    assert reg.get("use-uv").status == "trusted"
    assert "uv run pytest" in (reg.always_on_text() or "")  # trusted fact now always-on


async def test_quarantine_after_failure_correlation(tmp_path: Path):
    write_provisional_skill(
        tmp_path, {"type": "skill", "name": "bad-skill", "description": "d", "body": "b"}, "ev"
    )
    sess = _FakeSession(tmp_path, _FakeLLM('{"candidates": []}'))
    coord = DistillationCoordinator(tmp_path)
    # two green->red regressions (distinct commands) right after using the skill
    for cmd in ("pytest tests/a.py", "pytest tests/b.py"):
        coord.note_tool("shell", {"command": cmd}, _shell(0, "1 passed"))  # warm: now green
        coord.note_retrieval("bad-skill")
        coord.note_tool("shell", {"command": cmd}, _shell(1, "1 failed"))  # green -> red
        await coord.on_batch_end(sess, _ctx())
        await coord.drain()
    reg = load_skills(tmp_path)
    assert reg.get("bad-skill").status == "quarantined"
    assert reg.all_skills() == []  # quarantined skills are hidden from the default listing


# --------------------------------------------------------------------------- #
# episode-buffer bounds (memory) + slice retention (recent edits / commands / why)
# --------------------------------------------------------------------------- #


def _ok():
    return _shell(0, "applied")


def test_edit_stored_is_truncated(tmp_path: Path):
    coord = DistillationCoordinator(tmp_path)
    coord.note_tool("apply_patch", {"patch": "X" * 50000}, _ok())
    edits = [e for e in coord._events if e.kind == "edit"]
    assert len(edits) == 1
    assert len(edits[0].text) <= _EDIT_STORE_CAP + 64  # truncated on store, not stored whole


def test_events_bounded_when_no_green(tmp_path: Path):
    coord = DistillationCoordinator(tmp_path)
    for i in range(200):
        coord.note_tool("apply_patch", {"patch": f"*** Update File: src/f{i}.py\n" + "Y" * 3000}, _ok())
    coord.note_tool("apply_patch", {"patch": "*** Update File: src/final.py\nLAST_FIX_MARKER"}, _ok())
    nontask_bytes = sum(len(e.text) for e in coord._events if e.kind != "task")
    assert nontask_bytes <= _EVENT_BYTE_BUDGET + _EDIT_STORE_CAP  # memory is bounded
    assert any("LAST_FIX_MARKER" in e.text for e in coord._events if e.kind == "edit")  # newest kept


def test_trim_keeps_task_events(tmp_path: Path):
    coord = DistillationCoordinator(tmp_path)
    coord.note_user("THE ORIGINAL GOAL")
    for _ in range(100):
        coord.note_tool("apply_patch", {"patch": "*** Update File: src/f.py\n" + "Z" * 3000}, _ok())
    assert any(e.kind == "task" and "THE ORIGINAL GOAL" in e.text for e in coord._events)


def test_slice_prefers_recent_edits(tmp_path: Path):
    coord = DistillationCoordinator(tmp_path)
    for i in range(20):
        coord.note_tool("apply_patch", {"patch": f"*** Update File: src/f{i}.py\n" + "A" * 3000}, _ok())
    coord.note_tool("apply_patch", {"patch": "*** Update File: src/fix.py\nUNIQUE_FIX_MARKER"}, _ok())
    slice_text = coord._render_slice(_Transition("pytest", "red_green", "1 failed", "1 passed"))
    assert "UNIQUE_FIX_MARKER" in slice_text  # the recent fix is kept, not truncated away


def test_slice_includes_recent_commands(tmp_path: Path):
    coord = DistillationCoordinator(tmp_path)
    coord.note_tool("shell", {"command": "npm run build"}, _shell(0, "ok"))
    slice_text = coord._render_slice(_Transition("pytest", "red_green", "1 failed", "1 passed"))
    assert "npm run build" in slice_text  # command-type lessons have a trail in the slice


def test_slice_failing_output_keeps_the_reason(tmp_path: Path):
    coord = DistillationCoordinator(tmp_path)
    long_fail = "HEADER\n" + ("noise\n" * 2000) + "AssertionError: THE REAL REASON"
    slice_text = coord._render_slice(_Transition("pytest", "red_green", long_fail, "1 passed"))
    assert "THE REAL REASON" in slice_text  # truncate_middle keeps the tail where the error is


# --------------------------------------------------------------------------- #
# distiller retry: correct format on retry, but respect a valid "nothing to learn"
# --------------------------------------------------------------------------- #


class _SeqLLM:
    """Returns each text in sequence across successive stream() calls."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.calls = 0

    def stream(self, messages, tools, turn_context):
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1

        async def _gen():
            yield StreamEvent(kind="text_delta", text=text)

        return _gen()


async def test_distill_retries_on_bad_format_then_succeeds():
    good = _candidates_json({"type": "fact", "name": "x", "description": "d", "body": "b"})
    llm = _SeqLLM(["not json at all", good])
    result = await _distill_candidates(llm, _ctx(), "slice", "(none)")
    assert llm.calls == 2  # retried once after the unparseable reply
    assert len(result) == 1 and result[0]["name"] == "x"


async def test_distill_accepts_empty_result_without_retry():
    # A valid "nothing to learn" answer must be respected, NOT retried into a write.
    llm = _SeqLLM(['{"candidates": []}', _candidates_json({"name": "nope", "description": "d", "body": "b"})])
    result = await _distill_candidates(llm, _ctx(), "slice", "(none)")
    assert result == []
    assert llm.calls == 1  # accepted the empty envelope; the second reply is never used


async def test_distill_gives_up_after_max_attempts():
    llm = _SeqLLM(["garbage, never valid"])
    result = await _distill_candidates(llm, _ctx(), "slice", "(none)")
    assert result == []
    assert llm.calls == _DISTILL_MAX_ATTEMPTS


# --------------------------------------------------------------------------- #
# bug fixes: VerificationTracker memory bound + distillation surviving corrupt metrics
# --------------------------------------------------------------------------- #


def test_tracker_stores_truncated_output():
    t = VerificationTracker()
    t.observe("pytest", 0, "x" * 50000 + "\n8 passed")
    stored = t._last["pytest"][1]
    assert len(stored) <= _TRACKER_OUTPUT_CAP + 100  # truncated on store, not kept whole
    assert "8 passed" in stored  # the tail (count anti_cheat needs) is preserved


def test_tracker_caps_distinct_commands():
    t = VerificationTracker()
    for i in range(_TRACKER_MAX_CMDS + 20):
        t.observe(f"pytest tests/test_{i}.py", 0, "1 passed")
    assert len(t._last) <= _TRACKER_MAX_CMDS


def test_tracker_lru_keeps_recently_used_command():
    t = VerificationTracker()
    t.observe("pytest tests/test_hot.py", 1, "1 failed")
    # overflow the cap, but keep re-touching the hot command so LRU never evicts it
    for i in range(_TRACKER_MAX_CMDS):
        t.observe(f"pytest tests/test_{i}.py", 0, "1 passed")
        t.observe("pytest tests/test_hot.py", 1, "1 failed")
    assert len(t._last) <= _TRACKER_MAX_CMDS
    assert "pytest tests/test_hot.py" in t._last  # hot command kept; a cold one was evicted


async def test_on_batch_end_survives_corrupt_metrics(tmp_path: Path):
    # A hand-corrupted .cw-metrics.json must never break the user's turn (bug 1 / fix c).
    write_provisional_skill(
        tmp_path, {"type": "skill", "name": "x", "description": "d", "body": "b"}, "ev"
    )
    (tmp_path / "skills" / ".cw-metrics.json").write_text('{"x": "not-a-dict"}', encoding="utf-8")
    sess = _FakeSession(tmp_path, _FakeLLM('{"candidates": []}'))
    coord = DistillationCoordinator(tmp_path)
    coord.note_retrieval("x")
    coord.note_tool("shell", {"command": "pytest"}, _shell(1, "1 failed"))
    coord.note_tool("shell", {"command": "pytest"}, _shell(0, "1 passed"))
    await coord.on_batch_end(sess, _ctx())  # must NOT raise despite corrupt metrics
    await coord.drain()
    assert any("distillation step failed" in getattr(e, "message", "") for e in sess.events)
