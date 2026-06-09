from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from codewright.mcp.config import McpServerConfig, parse_mcp_servers
from codewright.protocol import PermissionProfile

ApiStyle = Literal["chat_completions", "responses"]

DEFAULT_CONFIG_PATH = Path.home() / ".codewright" / "config.toml"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_API_STYLE: ApiStyle = "chat_completions"
DEFAULT_MAX_CONTEXT_TOKENS = 128_000
DEFAULT_COMPACT_THRESHOLD = 0.9
DEFAULT_ROLE = "default"
DEFAULT_PERMISSION_PROFILE = PermissionProfile.WORKSPACE_WRITE


@dataclass(frozen=True)
class CodewrightConfig:
    provider: str = "openai"
    api_style: ApiStyle = DEFAULT_API_STYLE
    model: str = DEFAULT_MODEL
    api_key: str = ""
    api_key_env: str | None = None
    provider_base_url: str | None = None
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    compact_threshold: float = DEFAULT_COMPACT_THRESHOLD
    default_role: str = DEFAULT_ROLE
    permission_profile: PermissionProfile = DEFAULT_PERMISSION_PROFILE
    mcp_servers: list[McpServerConfig] = field(default_factory=list)


@dataclass(frozen=True)
class CliOverrides:
    api_style: str | None = None
    model: str | None = None
    provider_base_url: str | None = None
    max_context_tokens: int | None = None
    permission_profile: str | None = None


def load_config(overrides: CliOverrides | None = None) -> CodewrightConfig:
    """Load effective config with precedence: CLI > environment > config file."""
    file_data = _read_config_file(DEFAULT_CONFIG_PATH)
    file_cfg = _config_from_toml(file_data)
    env_cfg = _env_overrides(file_cfg)
    cli_cfg = _cli_overrides(overrides or CliOverrides())
    return CodewrightConfig(
        provider=_pick(cli_cfg.get("provider"), env_cfg.get("provider"), file_cfg.provider),
        api_style=_coerce_api_style(
            _pick(cli_cfg.get("api_style"), env_cfg.get("api_style"), file_cfg.api_style)
        ),
        model=str(_pick(cli_cfg.get("model"), env_cfg.get("model"), file_cfg.model)),
        api_key=str(_pick(cli_cfg.get("api_key"), env_cfg.get("api_key"), file_cfg.api_key)),
        api_key_env=_pick(
            cli_cfg.get("api_key_env"),
            env_cfg.get("api_key_env"),
            file_cfg.api_key_env,
        ),
        provider_base_url=_pick(
            cli_cfg.get("provider_base_url"),
            env_cfg.get("provider_base_url"),
            file_cfg.provider_base_url,
        ),
        max_context_tokens=_coerce_positive_int(
            _pick(
                cli_cfg.get("max_context_tokens"),
                env_cfg.get("max_context_tokens"),
                file_cfg.max_context_tokens,
            ),
            "max_context_tokens",
        ),
        compact_threshold=_coerce_threshold(
            _pick(
                cli_cfg.get("compact_threshold"),
                env_cfg.get("compact_threshold"),
                file_cfg.compact_threshold,
            )
        ),
        default_role=str(
            _pick(cli_cfg.get("default_role"), env_cfg.get("default_role"), file_cfg.default_role)
        ),
        permission_profile=_coerce_permission_profile(
            _pick(
                cli_cfg.get("permission_profile"),
                env_cfg.get("permission_profile"),
                file_cfg.permission_profile,
            )
        ),
        mcp_servers=file_cfg.mcp_servers,
    )

def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)

def _config_from_toml(data: dict[str, Any]) -> CodewrightConfig:
    llm = _table(data, "llm")
    agent = _table(data, "agent")
    workspace = _table(data, "workspace")
    api_key_env = _optional_str(llm.get("api_key_env"))
    api_key = _optional_str(llm.get("api_key"))
    if not api_key and api_key_env:
        api_key = os.environ.get(api_key_env)

    return CodewrightConfig(
        provider=_str_value(llm.get("provider"), "openai"),
        api_style=_coerce_api_style(llm.get("api_style") or DEFAULT_API_STYLE),
        model=_str_value(llm.get("model"), DEFAULT_MODEL),
        api_key=api_key or "",
        api_key_env=api_key_env,
        provider_base_url=_optional_str(llm.get("base_url")),
        max_context_tokens=_coerce_positive_int(
            agent.get("max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS),
            "agent.max_context_tokens",
        ),
        compact_threshold=_coerce_threshold(
            agent.get("compact_threshold", DEFAULT_COMPACT_THRESHOLD)
        ),
        default_role=_str_value(agent.get("default_role"), DEFAULT_ROLE),
        permission_profile=_coerce_permission_profile(
            workspace.get("permission_profile", DEFAULT_PERMISSION_PROFILE)
        ),
        mcp_servers=parse_mcp_servers(data),
    )

def _env_overrides(file_cfg: CodewrightConfig) -> dict[str, Any]:
    api_key_env = _env_str("CODEWRIGHT_API_KEY_ENV") or file_cfg.api_key_env
    api_key = _env_str("CODEWRIGHT_API_KEY")
    if not api_key:
        if api_key_env:
            api_key = os.environ.get(api_key_env)
        if not api_key:
            api_key = _env_str("OPENAI_API_KEY")

    return _drop_none(
        {
            "provider": _env_str("CODEWRIGHT_PROVIDER"),
            "api_style": _env_str("CODEWRIGHT_API_STYLE"),
            "model": _env_str("CODEWRIGHT_MODEL"),
            "api_key": api_key,
            "api_key_env": _env_str("CODEWRIGHT_API_KEY_ENV"),
            "provider_base_url": (
                _env_str("CODEWRIGHT_PROVIDER_BASE_URL")
                or _env_str("CODEWRIGHT_BASE_URL")
                or _env_str("OPENAI_BASE_URL")
            ),
            "max_context_tokens": _env_str("CODEWRIGHT_MAX_CONTEXT_TOKENS"),
            "compact_threshold": _env_str("CODEWRIGHT_COMPACT_THRESHOLD"),
            "default_role": _env_str("CODEWRIGHT_DEFAULT_ROLE"),
            "permission_profile": _env_str("CODEWRIGHT_PERMISSION_PROFILE"),
        }
    )

def _cli_overrides(overrides: CliOverrides) -> dict[str, Any]:
    return _drop_none(
        {
            "api_style": overrides.api_style,
            "model": overrides.model,
            "provider_base_url": overrides.provider_base_url,
            "max_context_tokens": overrides.max_context_tokens,
            "permission_profile": overrides.permission_profile,
        }
    )


def _table(data: dict[str, Any], name: str) -> dict[str, Any]:
    raw = data.get(name, {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return raw
def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}
def _env_str(name: str) -> str | None:
    return _optional_str(os.environ.get(name))
def _pick(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
def _str_value(value: Any, default: str) -> str:
    text = _optional_str(value)
    return text if text is not None else default
def _coerce_api_style(value: Any) -> ApiStyle:
    if value in ("chat_completions", "responses"):
        return value
    raise ValueError(
        f"api_style must be 'chat_completions' or 'responses', got {value!r}"
    )
def _coerce_permission_profile(value: Any) -> PermissionProfile:
    if isinstance(value, PermissionProfile):
        return value
    try:
        return PermissionProfile(str(value))
    except ValueError as exc:
        choices = ", ".join(p.value for p in PermissionProfile)
        raise ValueError(
            f"permission_profile must be one of: {choices}; got {value!r}"
        ) from exc
def _coerce_positive_int(value: Any, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out
def _coerce_threshold(value: Any) -> float:
    out = float(value)
    if not (0.0 < out <= 1.0):
        raise ValueError("compact_threshold must be in (0, 1]")
    return out




__all__ = [
    "DEFAULT_CONFIG_PATH",
    "CliOverrides",
    "CodewrightConfig",
    "load_config",
]
