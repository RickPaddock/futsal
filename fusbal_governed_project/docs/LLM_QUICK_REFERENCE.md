---
generated: true
source: spec/md/docs/LLM_QUICK_REFERENCE.mdt
source_sha256: sha256:96573c8ee3cb868db8adc42a9299121b1f5d909f99ad1e8e8f0fada8142bc055
---

# LLM quick reference

## Setup

- Install: `npm install`
- Generate: `npm run generate`
- Drift check: `npm run generate:check`
- Guardrails: `npm run guardrails`
- Intent audit: `npm run audit:intent -- --intent-id INT-001`
- Intent close: `npm run intent:close -- --intent-id INT-001 --closed-date YYYY-MM-DD --apply`

## Intent statuses

- `draft`: reserved ID; no tasks yet
- `todo`: tasks + requirements wired
- `closed`: audit passed + close applied

## Internal portal

- Start: `npm run portal:dev`
- URL: `http://127.0.0.1:3015/internal/intents`

## Evidence

`node tools/evidence/record_run.mjs --intent-id INT-001 --out status/audit/INT-001/runs/<run_id>/run.json -- npm run guardrails`

## Pipeline (Python, optional)

- Location: `pipeline/`
- Suggested setup: `python -m venv .venv && source .venv/bin/activate && pip install -e pipeline`
- Initialize a match bundle: `python -m fusbal_pipeline init --match-id MATCH_001 --out output/MATCH_001 --video path/to/video.mp4`
