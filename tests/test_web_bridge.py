"""Tests for the WebSocket bridge (codewright.web).

Sessions are constructed without an LLM provider, so OpUserTurn takes the
mock path in submission_loop ("[mock] no real LLM wired yet") — the bridge's
transport behaviour is exercised end to end without any network calls.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from starlette.testclient import TestClient

from codewright.agent.session import Session
from codewright.protocol import PermissionProfile
from codewright.web import create_app


def _make_client(tmp_path: Path) -> tuple[TestClient, list[str | None]]:
    """Returns the test client plus a log of session_ids the factory saw."""
    factory_calls: list[str | None] = []

    async def session_factory(session_id: str | None) -> Session:
        factory_calls.append(session_id)
        if session_id == "missing":
            raise FileNotFoundError(f"session not found: {session_id}")
        return Session(
            session_id=session_id or uuid.uuid4().hex,
            cwd=tmp_path,
            permission_profile=PermissionProfile.WORKSPACE_WRITE,
        )

    return TestClient(create_app(session_factory, workspace_root=tmp_path)), factory_calls


def _write_rollout(tmp_path: Path, session_id: str, start_time: float, lines: list[dict]) -> None:
    sessions_dir = tmp_path / ".codewright" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "ts": start_time,
        "type": "session_meta",
        "payload": {
            "session_id": session_id,
            "cwd": str(tmp_path),
            "model": "mock",
            "permission_profile": "workspace_write",
            "start_time": start_time,
        },
    }
    with open(sessions_dir / f"{session_id}.jsonl", "w", encoding="utf-8") as fh:
        for line in [meta, *lines]:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def _recv_until(ws, wanted_type: str, limit: int = 20) -> dict:
    for _ in range(limit):
        frame = ws.receive_json()
        if frame["msg"]["type"] == wanted_type:
            return frame
    raise AssertionError(f"never received {wanted_type!r} within {limit} frames")


class TestWebBridge:
    def test_index_serves_test_page(self, tmp_path):
        client, _ = _make_client(tmp_path)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "WebSocket" in resp.text

    def test_session_configured_is_first_frame(self, tmp_path):
        client, _ = _make_client(tmp_path)
        with client.websocket_connect("/ws") as ws:
            frame = ws.receive_json()
            assert frame["msg"]["type"] == "session_configured"
            assert frame["msg"]["cwd"] == str(tmp_path)

    def test_user_turn_round_trip(self, tmp_path):
        client, _ = _make_client(tmp_path)
        with client.websocket_connect("/ws") as ws:
            _recv_until(ws, "session_configured")
            ws.send_json(
                {"type": "user_turn", "items": [{"type": "text", "text": "hello"}]}
            )
            _recv_until(ws, "turn_started")
            frame = _recv_until(ws, "turn_completed")
            assert "[mock]" in frame["msg"]["last_agent_message"]

    def test_invalid_json_answers_bridge_error(self, tmp_path):
        client, _ = _make_client(tmp_path)
        with client.websocket_connect("/ws") as ws:
            _recv_until(ws, "session_configured")
            ws.send_text("{not json")
            frame = ws.receive_json()
            assert frame["msg"]["type"] == "bridge_error"
            assert "invalid JSON" in frame["msg"]["message"]

    def test_disallowed_op_is_rejected(self, tmp_path):
        client, _ = _make_client(tmp_path)
        with client.websocket_connect("/ws") as ws:
            _recv_until(ws, "session_configured")
            ws.send_json({"type": "shutdown"})
            frame = ws.receive_json()
            assert frame["msg"]["type"] == "bridge_error"
            assert "not allowed" in frame["msg"]["message"]

    def test_malformed_op_keeps_connection_alive(self, tmp_path):
        client, _ = _make_client(tmp_path)
        with client.websocket_connect("/ws") as ws:
            _recv_until(ws, "session_configured")
            ws.send_json({"type": "user_turn"})  # missing required "items"
            frame = ws.receive_json()
            assert frame["msg"]["type"] == "bridge_error"
            assert "invalid op" in frame["msg"]["message"]
            ws.send_json(
                {"type": "user_turn", "items": [{"type": "text", "text": "still alive"}]}
            )
            _recv_until(ws, "turn_completed")


class TestSessionRest:
    def test_list_sessions_empty(self, tmp_path):
        client, _ = _make_client(tmp_path)
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.json() == {"sessions": []}

    def test_list_sessions_titles_and_order(self, tmp_path):
        _write_rollout(
            tmp_path, "older", start_time=1000.0,
            lines=[{"ts": 1001.0, "type": "user_msg",
                    "payload": {"content": "修复 parse_date 的 bug\n顺便跑测试"}}],
        )
        _write_rollout(tmp_path, "newer", start_time=2000.0, lines=[])
        client, _ = _make_client(tmp_path)
        data = client.get("/api/sessions").json()["sessions"]
        assert [s["session_id"] for s in data] == ["newer", "older"]
        assert data[1]["title"] == "修复 parse_date 的 bug"
        assert data[0]["title"] == ""  # no user_msg yet
        assert data[0]["active"] is False

    def test_history_blocks(self, tmp_path):
        _write_rollout(
            tmp_path, "s1", start_time=1000.0,
            lines=[
                {"ts": 1, "type": "user_msg", "payload": {"content": "do it"}},
                {"ts": 2, "type": "assistant_msg",
                 "payload": {"content": "", "tool_calls": [{"call_id": "c1",
                  "tool_name": "shell", "arguments_json": "{}"}]}},
                {"ts": 3, "type": "tool_result",
                 "payload": {"call_id": "c1", "tool_name": "shell",
                             "body": "Exit code: 0", "success": True}},
                {"ts": 4, "type": "assistant_msg", "payload": {"content": "done"}},
                {"ts": 5, "type": "compaction", "payload": {"replacement": []}},
            ],
        )
        client, _ = _make_client(tmp_path)
        blocks = client.get("/api/sessions/s1/history").json()["blocks"]
        assert [b["kind"] for b in blocks] == ["user", "tool", "agent"]
        assert blocks[1]["name"] == "shell"
        assert blocks[2]["text"] == "done"

    def test_history_unknown_session_404(self, tmp_path):
        client, _ = _make_client(tmp_path)
        assert client.get("/api/sessions/nope/history").status_code == 404


class TestResume:
    def test_factory_receives_session_id(self, tmp_path):
        client, calls = _make_client(tmp_path)
        with client.websocket_connect("/ws?session_id=abc123") as ws:
            frame = ws.receive_json()
            assert frame["msg"]["type"] == "session_configured"
            assert frame["msg"]["session_id"] == "abc123"
        assert calls == ["abc123"]

    def test_resume_failure_sends_error_and_closes(self, tmp_path):
        client, _ = _make_client(tmp_path)
        with client.websocket_connect("/ws?session_id=missing") as ws:
            frame = ws.receive_json()
            assert frame["msg"]["type"] == "bridge_error"
            assert "cannot resume" in frame["msg"]["message"]

    def test_double_open_same_session_rejected(self, tmp_path):
        client, _ = _make_client(tmp_path)
        with client.websocket_connect("/ws?session_id=dup1") as ws1:
            _recv_until(ws1, "session_configured")
            with client.websocket_connect("/ws?session_id=dup1") as ws2:
                frame = ws2.receive_json()
                assert frame["msg"]["type"] == "bridge_error"
                assert "already open" in frame["msg"]["message"]

    def test_session_reopenable_after_disconnect(self, tmp_path):
        client, _ = _make_client(tmp_path)
        with client.websocket_connect("/ws?session_id=re1") as ws:
            _recv_until(ws, "session_configured")
        with client.websocket_connect("/ws?session_id=re1") as ws:
            frame = ws.receive_json()
            assert frame["msg"]["type"] == "session_configured"
