---
generated: true
source: spec/intents/INT-050.json
source_sha256: sha256:8985db3b2dcd122be149fa969bd2485ff4769556d0533a1a22cacdab52743ac3
intent_id: INT-050
title: Video ingest → plumbing proof → baseline real detections (UAT)
status: todo
created_date: 2026-01-22
close_gate:
  - "npm run generate"
  - "npm run guardrails"
  - "npm run audit:intent -- --intent-id INT-050"
---

# Intent: INT-050

- Given a provided local match video, ingest/iterate frames deterministically and export a valid V1 bundle (plumbing proof).
- Then add a baseline local ball detector that produces non-empty detections on the UAT clip while remaining trust-first (missing/unknown over wrong).
- UAT clip (preferred): `GoPro_Futsal_part2_trimmed_100mb.mp4` (provided locally; not committed to the repo).

## Work packages

### INT-050-001 — Plumbing proof (ingest → export bundle)

- TASK-VIDEO-INGEST-001 Deterministic frame decode + timebase extraction for local videos.
- TASK-RUN-VIDEO-001 CLI runner: ingest video → run detectors/trackers → export tracks/events bundle.

### INT-050-002 — Baseline real detections (UAT)

- TASK-BALL-DET-BASELINE-001 Baseline local ball detector producing non-empty detections on the provided clip with confidence/diagnostics.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `spec/md/docs/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

