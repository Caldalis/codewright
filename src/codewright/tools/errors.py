from __future__ import annotations


class RespondToModelError(Exception):
    """Recoverable: surface the message to the model as a failed tool result."""


class FatalToolError(Exception):
    """Engine-level: abort the entire turn."""
