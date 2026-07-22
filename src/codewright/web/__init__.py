"""Web front end: a WebSocket bridge exposing the Op/Event protocol to browsers.

Like ``tui/`` and ``cli/``, this package is a front end only — business layers
must never import it (enforced by ``.importlinter``).
"""

from codewright.web.server import create_app

__all__ = ["create_app"]
