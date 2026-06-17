# Codewright

A powerful and concise coding agent coding for the command line, written in Python.

Codewright : a streaming LLM loop, a polymorphic tool runtime, multi-agent
orchestration, session persistence, MCP tool servers, and an interactive TUI.

## Features

- **Interactive TUI** with streaming output, a status bar, and modal approvals.
- **Non-interactive mode** for one-shot prompts and scripting.
- **Semantic file tools** (`read_file`, `list_dir`, `find_files`, `search_text`)
  plus a stateful **`shell`** tool family (bash-anchored, with background jobs
  and output paging) and transactional **`apply_patch`** edits.
- **Permission profiles** that gate destructive shell commands and out-of-workspace access.
- **Multi-agent**: spawn child agents that run in their own sessions and report back.
- **Self-improving skills**: the agent distills reusable, test-verified knowledge
  into `<workspace>/skills/` and pulls it back in on later tasks.
- **MCP** tool servers over stdio and streamable-http.
- **Session persistence + resume** via append-only JSONL rollouts.
- **Pluggable providers**: OpenAI Chat Completions or Responses API, and any
  OpenAI-compatible endpoint (DeepSeek, Qwen/DashScope, Ollama, local gateways).

## Requirements

- Python **3.12** (`>=3.12,<3.13`)
- [`uv`](https://docs.astral.sh/uv/)
- An API key for an LLM provider (see [Configuration](#configuration))

## Installation

### As a global command (recommended)

Install Codewright as a standalone tool so `codewright` is available in any directory:

```bash
uv tool install --editable .      # run once from the repo root
uv tool update-shell              # if the uv tools bin dir is not on your PATH (then reopen the terminal)
```

`--editable` installs from source, so later code changes take effect without
reinstalling. Omit it for a pinned snapshot (re-run with `--reinstall` to update).
`pipx install --editable .` works too.

### From source (development)

```bash
uv sync                           # install deps incl. the dev group
uv run codewright --help          # run without a global install (inside the repo)
```

## Quick start

```bash
# Interactive TUI in the current directory (bare command == `codewright tui`):
codewright

# Interactive TUI in a specific workspace:
codewright --workspace path/to/project

# One non-interactive turn, prints the final message:
codewright run "explain what this project does" --workspace .

# List and resume saved sessions:
codewright list-sessions
codewright resume <session_id>                 # reopen in the TUI
codewright resume <session_id> --message "now add tests"   # one turn, then exit
```

## Configuration

Settings are read from `~/.codewright/config.toml`. Effective-value precedence
(highest wins): **CLI flags > environment variables > config file > built-in defaults**.

A model API key is the only thing you must supply — everything else has a
default. Provide a key via any one of:

| Source | How |
|---|---|
| `OPENAI_API_KEY` | export it in your shell (simplest) |
| `CODEWRIGHT_API_KEY` | export it in your shell |
| `[llm] api_key_env` | name of an env var holding the key (keeps the secret out of the file) |
| `[llm] api_key` | the key inline (discouraged; plaintext) |

A minimal `~/.codewright/config.toml`:

```toml
[llm]
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"
# base_url = "https://api.deepseek.com/v1"   # for OpenAI-compatible providers
```

See [`examples/config.example.toml`](examples/config.example.toml) for every
option (model/provider, context window, permission profile, shell path, skills
test runners, and MCP servers) with its default and explanation.

Useful environment overrides: `CODEWRIGHT_MODEL`, `CODEWRIGHT_API_KEY`,
`CODEWRIGHT_API_KEY_ENV`, `CODEWRIGHT_BASE_URL` (or `OPENAI_BASE_URL`),
`CODEWRIGHT_PERMISSION_PROFILE`, `CODEWRIGHT_DEFAULT_ROLE`, `CODEWRIGHT_SHELL_PATH`,
`CODEWRIGHT_MAX_CONTEXT_TOKENS`, `CODEWRIGHT_COMPACT_THRESHOLD`.

## Commands

| Command | Description |
|---|---|
| `codewright` | Launch the interactive TUI in the current directory (alias for `tui`). |
| `codewright tui` | Launch the interactive TUI. |
| `codewright run <prompt>` | Run a single non-interactive turn and print the final message. |
| `codewright resume <session_id>` | Resume a saved session (TUI, or `--message` for one turn). |
| `codewright list-sessions` | List saved sessions in the workspace. |

Common flags: `--workspace <dir>` (default: current dir), `--model`,
`--provider-base-url`, `--api-style {chat_completions,responses}`,
`--permission-profile {read_only,workspace_write,dangerous}`. `run` also accepts
`--max-context-tokens`, `--print-session-id`, and `--no-persist`. Run
`codewright <command> --help` for the full list.

### TUI keys

| Key | Action |
|---|---|
| `Ctrl-C` | Interrupt the current turn |
| `Ctrl-D` | Quit |
| `PageUp` / `PageDown` | Scroll history |
| `Ctrl-Home` / `Ctrl-End` | Jump to top / bottom |
| `F1` or `?` | Toggle status details |
| `y` / `s` / `n` / `a` | On an approval prompt: approve / approve for session / deny / abort |

## Permissions

The active permission profile decides which shell commands run without asking:

- **`read_only`** — no shell exec at all; only the read/search file tools.
- **`workspace_write`** (default) — exec is auto-allowed inside the workspace;
  destructive commands (`rm -rf`, `git reset --hard`, network access, …) and any
  cwd outside the workspace still prompt for approval.
- **`dangerous`** — nothing is auto-allowed; every exec prompts.

The workspace root is a hard boundary: reads and writes outside it require approval.

## Sessions

Each session is recorded as an append-only JSONL rollout under
`<workspace>/.codewright/sessions/<session_id>.jsonl`. `run` persists by default
(disable with `--no-persist`); the TUI always persists. Use `list-sessions` and
`resume` to pick up where you left off.

## Architecture

The engine sits behind two `asyncio.Queue`s: front ends submit **Op** instances
and consume **Event** instances, and never touch `Session` internals. A single
`submission_loop` drives `run_turn`, which streams the model, dispatches tool
calls through the executor, compacts history, and emits events.

## Development

```bash
uv run pytest                         # full test suite
uv run pytest tests/test_run_turn.py  # a single file
uv run ruff check src tests           # lint
uv run ruff check --fix src tests     # lint + autofix
uv run lint-imports                   # enforce the layering contracts (.importlinter)
```


## License

[MIT](LICENSE)
