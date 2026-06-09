from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

McpTransport = Literal["stdio", "streamable_http"]


@dataclass(frozen=True)
class McpServerConfig:

    name: str
    transport: McpTransport
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    bearer_token_env_var: str | None = None
    startup_timeout_sec: float = 30.0

    def bearer_token(self) -> str | None:
        if self.bearer_token_env_var is None:
            return None
        return os.environ.get(self.bearer_token_env_var)


def parse_mcp_servers(toml_data: dict) -> list[McpServerConfig]:

    raw_servers = toml_data.get("mcp_servers")
    if not raw_servers:
        return []
    if not isinstance(raw_servers, dict):
        raise ValueError(
            "[mcp_servers] must be a table of named server configs"
        )

    out: list[McpServerConfig] = []
    for name, entry in raw_servers.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"[mcp_servers.{name}] must be a TOML table"
            )
        out.append(_one(name, entry))
    return out


def load_mcp_servers_from_toml(toml_path: Path) -> list[McpServerConfig]:
    if not toml_path.exists():
        return []
    with open(toml_path, "rb") as fh:
        data = tomllib.load(fh)
    return parse_mcp_servers(data)


def _one(name: str, entry: dict) -> McpServerConfig:
    entry = _normalize_transport_entry(name, entry)
    transport_raw = entry.get("transport")
    if transport_raw is None:
        if "command" in entry:
            transport: McpTransport = "stdio"
        elif "url" in entry:
            transport = "streamable_http"
        else:
            raise ValueError(
                f"[mcp_servers.{name}] needs either `transport`, `command` (stdio), or `url` (streamable_http)"
            )
    elif transport_raw in ("stdio", "streamable_http"):
        transport = transport_raw
    else:
        raise ValueError(
            f"[mcp_servers.{name}] unknown transport={transport_raw!r}; "
            "expected stdio or streamable_http"
        )

    timeout = float(entry.get("startup_timeout_sec", 30.0))

    if transport == "stdio":
        command = entry.get("command")
        if not command:
            raise ValueError(f"[mcp_servers.{name}] stdio transport requires `command`")
        args_raw = entry.get("args", []) or []
        if not isinstance(args_raw, list):
            raise ValueError(f"[mcp_servers.{name}] `args` must be a list of strings")
        env_raw = entry.get("env", {}) or {}
        if not isinstance(env_raw, dict):
            raise ValueError(f"[mcp_servers.{name}] `env` must be a table of string->string")
        return McpServerConfig(
            name=name,
            transport="stdio",
            command=str(command),
            args=tuple(str(a) for a in args_raw),
            env={str(k): str(v) for k, v in env_raw.items()},
            startup_timeout_sec=timeout,
        )

    url = entry.get("url")
    if not url:
        raise ValueError(f"[mcp_servers.{name}] streamable_http requires `url`")
    bearer = entry.get("bearer_token_env_var")
    return McpServerConfig(
        name=name,
        transport="streamable_http",
        url=str(url),
        bearer_token_env_var=str(bearer) if bearer else None,
        startup_timeout_sec=timeout,
    )


def _normalize_transport_entry(name: str, entry: dict) -> dict:
    transport_raw = entry.get("transport")
    if not isinstance(transport_raw, dict):
        return entry

    normalized = dict(entry)
    transport_type = transport_raw.get("type")
    if transport_type is None:
        raise ValueError(f"[mcp_servers.{name}.transport] requires `type`")
    normalized["transport"] = transport_type
    for key, value in transport_raw.items():
        if key == "type":
            continue
        if key in normalized:
            raise ValueError(
                f"[mcp_servers.{name}] duplicate `{key}` in server and transport tables"
            )
        normalized[key] = value
    return normalized


__all__ = [
    "McpServerConfig",
    "McpTransport",
    "load_mcp_servers_from_toml",
    "parse_mcp_servers",
]
