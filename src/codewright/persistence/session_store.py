from __future__ import annotations

import json
from pathlib import Path

from codewright.persistence.rollout import RolloutLine, RolloutRecorder, SessionMeta


class SessionStore:

    SESSIONS_SUBDIR = (".codewright", "sessions")

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = Path(workspace_root)

    @property
    def workspace_root(self) -> Path:
        return self._workspace_root

    def session_dir(self) -> Path:
        return self._workspace_root.joinpath(*self.SESSIONS_SUBDIR)

    def session_path(self, session_id: str) -> Path:
        return self.session_dir() / f"{session_id}.jsonl"

    async def list_sessions(self) -> list[SessionMeta]:
        directory = self.session_dir()
        if not directory.exists():
            return []
        metas: list[SessionMeta] = []
        for path in sorted(directory.glob("*.jsonl")):
            meta = _read_first_meta(path)
            if meta is not None:
                metas.append(meta)
        return metas

    async def load_session(
        self, session_id: str
    ) -> tuple[SessionMeta, list[RolloutLine]]:

        path = self.session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"session not found: {session_id} ({path})")
        lines: list[RolloutLine] = []
        meta: SessionMeta | None = None
        with open(path, encoding="utf-8") as fh:  # noqa: ASYNC230
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                if meta is None:
                    try:
                        line = RolloutLine.model_validate_json(raw)
                    except Exception as exc:
                        raise ValueError(
                            f"first line of {path} is not a valid rollout line: {exc}"
                        ) from exc
                    if line.type != "session_meta":
                        raise ValueError(
                            f"first line of {path} has type={line.type!r}; "
                            "expected 'session_meta'"
                        )
                    try:
                        meta = SessionMeta.model_validate(line.payload)
                    except Exception as exc:
                        raise ValueError(
                            f"invalid session_meta payload in {path}: {exc}"
                        ) from exc
                    lines.append(line)
                    continue
                try:
                    line = RolloutLine.model_validate_json(raw)
                except Exception:
  
                    continue
                lines.append(line)
        if meta is None:
            raise ValueError(f"empty rollout file: {path}")
        return meta, lines

    async def create_recorder(self, meta: SessionMeta) -> RolloutRecorder:
        path = self.session_path(meta.session_id)
        return await RolloutRecorder.create(path, meta)

    async def resume_recorder(self, session_id: str) -> RolloutRecorder:
        return await RolloutRecorder.resume(self.session_path(session_id))


def _read_first_meta(path: Path) -> SessionMeta | None:
    try:
        with open(path, encoding="utf-8") as fh:
            for _ in range(8):  # search a small head window
                raw = fh.readline()
                if not raw:
                    return None
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "session_meta":
                    body = payload.get("payload") or {}
                    try:
                        return SessionMeta.model_validate(body)
                    except Exception:
                        return None
    except OSError:
        return None
    return None


__all__ = ["SessionStore"]
