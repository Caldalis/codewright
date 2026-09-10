#!/usr/bin/env bash
# Build a RELOCATABLE codewright runtime tarball.
#
# SWE-bench instance images ship the Python the target repo needs (often 3.6-3.11),
# but codewright requires 3.12. Rather than mutating each instance's environment --
# which would change what the graded tests run against -- we bake a standalone
# CPython 3.12 + codewright into /opt/cw once, and drop that tree into every
# instance container at the same absolute path.
#
# The tree is self-contained: the venv's interpreter lives at /opt/cw/python, so
# every absolute path inside it stays valid after untarring at /opt/cw.
#
# Build on the SAME architecture you will evaluate on (the runtime is native).
#
#   ./evals/build_runtime.sh [output_dir]        # default: evals/_runtime
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$REPO_ROOT/evals/_runtime}"
mkdir -p "$OUT_DIR"

echo ">> building codewright runtime (arch: $(uname -m))"
docker run --rm \
  -v "$REPO_ROOT":/src:ro \
  -v "$OUT_DIR":/out \
  ubuntu:22.04 bash -c '
set -euo pipefail
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq curl ca-certificates >/dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null 2>&1

mkdir -p /opt/cw
# Keep the standalone interpreter INSIDE /opt/cw so the tree relocates as a unit.
export UV_PYTHON_INSTALL_DIR=/opt/cw/python
uv python install 3.12 >/dev/null 2>&1
PY="$(uv python find 3.12)"
uv venv --python "$PY" /opt/cw/venv >/dev/null 2>&1
uv pip install --quiet --python /opt/cw/venv/bin/python /src

/opt/cw/venv/bin/codewright --help >/dev/null
tar czf /out/cw-runtime.tgz -C / opt/cw
'
echo ">> wrote $OUT_DIR/cw-runtime.tgz ($(du -h "$OUT_DIR/cw-runtime.tgz" | cut -f1))"
