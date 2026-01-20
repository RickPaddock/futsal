#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

cd "${ROOT}"
echo "[greenfield] generate"
npm run -s generate

echo "[greenfield] start portal"
cd "${ROOT}/apps/portal"
npm run dev

