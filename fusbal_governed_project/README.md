---
generated: true
source: spec/md/README.mdt
source_sha256: sha256:ca45916ac71f2d13200e3aa67eebbcd56eeed37ee8978602437166553f00b759
---

# Fusbal (Futsal Tracking)

This repository is **governed by generation + guardrails**.

Hard rule:
- All human-readable `.md` files are **machine-generated** from `spec/` sources.

## What this project is

Fusbal is a **trust-first, offline (post-processed) futsal match analysis** system:
- Track players and the ball from venue (or user) video.
- Project movement into pitch meters (BEV) when calibration is good.
- Infer conservative V1 events (shots/goals) and produce a confidence-heavy report.
- Treat sensors as optional `TrackSource`s that can only improve results.

## Commands

- Install dependencies: `npm install`
- Generate all derived outputs: `npm run generate`
- Check for drift: `npm run generate:check`
- Run guardrails: `npm run guardrails`
- Start internal portal: `npm run portal:dev`

## Where to edit

- Requirements (canonical): `spec/requirements/index.json`
- Intent definitions (canonical): `spec/intents/*.json`
- Markdown templates (sources): `spec/md/**.mdt`

## Key generated outputs

- Requirements view: `docs/requirements/requirements.md`
- Delivery plan (intents): `status/intents/<INT-ID>/intent.md`
- High-level delivery plan: `docs/delivery/HL_DELIVERY_PLAN.md`
- Output contract: `docs/data/OUTPUT_CONTRACT.md`
- Intent bundle(s): `status/intents/<INT-ID>/*`
- Portal feed: `status/portal/internal_intents.json`
