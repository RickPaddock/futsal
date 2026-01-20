#!/bin/zsh
# PROV: GREENFIELD.GOV.PORTAL_START.01
# REQ: SYS-ARCH-15
# WHY: Start the internal portal reliably by clearing common dev ports, then generating and launching Next.js.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

cd "${ROOT}"

echo "[greenfield] clear ports 3015-3020"
for port in {3015..3020}; do
  pids="$(lsof -ti tcp:${port} -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    continue
  fi
  echo "[greenfield] port ${port} in use by pid(s): ${pids}"
  kill -TERM ${pids} 2>/dev/null || true
done

sleep 0.6
for port in {3015..3020}; do
  pids="$(lsof -ti tcp:${port} -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    continue
  fi
  echo "[greenfield] force kill port ${port} pid(s): ${pids}"
  kill -KILL ${pids} 2>/dev/null || true
done

echo "[greenfield] generate"
npm run -s generate

echo "[greenfield] start portal"
cd "${ROOT}/apps/portal"
npm run dev
