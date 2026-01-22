#!/usr/bin/env bash
# PROV: FUSBAL.SCRIPTS.LINT.PIPELINE.01
# REQ: SYS-ARCH-15
# WHY: Run deterministic lint/format checks for the pipeline package in CI/local workflows.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python -m ruff check pipeline/src pipeline/tests
python -m black --check pipeline/src pipeline/tests

echo "[pipeline:lint] ok"