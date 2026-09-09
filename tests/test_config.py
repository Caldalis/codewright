from __future__ import annotations

import sys
from pathlib import Path

import pytest

import codewright.config as config_mod
from codewright.cli import _bootstrap_session, _resume_session_from_config
from codewright.config import CliOverrides, CodewrightConfig, load_config
from codewright.mcp.config import parse_mcp_servers
from codewright.persistence.rollout import SessionMeta
from codewright.persistence.session_store import SessionStore
from codewright.protocol import PermissionProfile

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FAKE_SERVER = FIXTURES / "fake_mcp_server.py"


def _isolate_config(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(config_mod, "DEFAULT_CONFIG_PATH", path)
    for name in (
        "CODEWRIGHT_PROVIDER",
        "CODEWRIGHT_API_STYLE",
        "CODEWRIGHT_MODEL",
        "CODEWRIGHT_API_KEY",
        "CODEWRIGHT_API_KEY_ENV",
        "CODEWRIGHT_PROVIDER_BASE_URL",
        "CODEWRIGHT_BASE_URL",
        "CODEWRIGHT_MAX_CONTEXT_TOKENS",
        "CODEWRIGHT_COMPACT_THRESHOLD",
        "CODEWRIGHT_DEFAULT_ROLE",
        "CODEWRIGHT_PERMISSION_PROFILE",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_load_config_from_user_toml(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _isolate_config(monkeypatch, config_path)
    monkeypatch.setenv("CUSTOM_API_KEY", "file-key")
    config_path.write_text(
        """
[llm]
provider = "dashscope"
api_style = "chat_completions"
model = "qwen3.7-plus"
api_key_env = "CUSTOM_API_KEY"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

[agent]
max_context_tokens = 64000
compact_threshold = 0.75
default_role = "explorer"

[workspace]
permission_profile = "read_only"

[mcp_servers.fake]
transport = { type = "stdio", command = "python", args = ["server.py"] }
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.provider == "dashscope"
    assert cfg.model == "qwen3.7-plus"
    assert cfg.api_key == "file-key"
    assert cfg.provider_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert cfg.max_context_tokens == 64_000
    assert cfg.compact_threshold == 0.75
    assert cfg.default_role == "explorer"
    assert cfg.permission_profile == PermissionProfile.READ_ONLY
    assert len(cfg.mcp_servers) == 1
    assert cfg.mcp_servers[0].command == "python"


def test_config_precedence_cli_then_env_then_file(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _isolate_config(monkeypatch, config_path)
    config_path.write_text(
        """
[llm]
model = "file-model"
base_url = "https://file.example/v1"

[agent]
max_context_tokens = 1000

[workspace]
permission_profile = "read_only"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEWRIGHT_MODEL", "env-model")
    monkeypatch.setenv("CODEWRIGHT_PROVIDER_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("CODEWRIGHT_MAX_CONTEXT_TOKENS", "2000")
    monkeypatch.setenv("CODEWRIGHT_PERMISSION_PROFILE", "dangerous")

    cfg = load_config(
        CliOverrides(
            model="cli-model",
            provider_base_url="https://cli.example/v1",
            permission_profile="workspace_write",
        )
    )

    assert cfg.model == "cli-model"
    assert cfg.provider_base_url == "https://cli.example/v1"
    assert cfg.max_context_tokens == 2_000
    assert cfg.permission_profile == PermissionProfile.WORKSPACE_WRITE


def test_empty_environment_value_does_not_mask_file(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _isolate_config(monkeypatch, config_path)
    config_path.write_text('[llm]\nmodel = "file-model"\n', encoding="utf-8")
    monkeypatch.setenv("CODEWRIGHT_MODEL", "")

    cfg = load_config()

    assert cfg.model == "file-model"


def test_mcp_inline_transport_table_matches_example() -> None:
    configs = parse_mcp_servers(
        {
            "mcp_servers": {
                "linear": {
                    "transport": {
                        "type": "streamable_http",
                        "url": "https://mcp.linear.app/sse",
                        "bearer_token_env_var": "LINEAR_TOKEN",
                    }
                }
            }
        }
    )

    assert configs[0].name == "linear"
    assert configs[0].transport == "streamable_http"
    assert configs[0].url == "https://mcp.linear.app/sse"
    assert configs[0].bearer_token_env_var == "LINEAR_TOKEN"


@pytest.mark.asyncio
async def test_bootstrap_session_starts_configured_mcp(tmp_path: Path) -> None:
    cfg = CodewrightConfig(
        model="mock",
        api_key="",
        mcp_servers=parse_mcp_servers(
            {
                "mcp_servers": {
                    "svc": {
                        "transport": {
                            "type": "stdio",
                            "command": sys.executable,
                            "args": [str(FAKE_SERVER)],
                        }
                    }
                }
            }
        ),
    )

    session = await _bootstrap_session(
        workspace=tmp_path,
        config=cfg,
        persist=False,
    )
    try:
        assert session.tool_registry.has("svc__echo")
        assert session.mcp is not None
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_resume_session_from_config_applies_runtime_config(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    recorder = await store.create_recorder(
        SessionMeta(
            session_id="resume-config",
            cwd=str(tmp_path),
            model="old-model",
            permission_profile=PermissionProfile.READ_ONLY.value,
            start_time=0.0,
        )
    )
    await recorder.shutdown()

    cfg = CodewrightConfig(
        model="new-model",
        api_key="",
        max_context_tokens=32_000,
        compact_threshold=0.5,
        default_role="explorer",
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
    )

    session = await _resume_session_from_config(
        workspace=tmp_path,
        session_id="resume-config",
        config=cfg,
    )
    try:
        assert session.model == "new-model"
        assert session.permission_profile == PermissionProfile.WORKSPACE_WRITE
        assert session.workspace.permission_profile == PermissionProfile.WORKSPACE_WRITE
        assert session.context.max_context_tokens == 32_000
        assert session.context.compact_threshold == 0.5
        assert session.role == "explorer"
    finally:
        await session.shutdown()
