---
generated: true
source: spec/intents/INT-050.json
source_sha256: sha256:be04c3de67972925825ff6bb51ffe97c2bb6bac5b35c7c6f7421b4b1773a46df
intent_id: INT-050
title: Video ingest → plumbing proof → baseline real detections (UAT)
status: closed
created_date: 2026-01-22
closed_date: 2026-01-22
close_gate:
  - "npm run generate"
  - "npm run guardrails"
  - "npm run audit:intent -- --intent-id INT-050"
---

# Intent: INT-050

- Given a provided local match video, ingest/iterate frames deterministically and export a valid V1 bundle (plumbing proof).
- Then add a baseline local ball detector that produces non-empty detections on the UAT clip while remaining trust-first (missing/unknown over wrong).
- UAT clip label (preferred): `GoPro_Futsal_part2_trimmed_100mb.mp4` (provided externally via CLI `--video` path; the filename is advisory only and must not be assumed repo-relative).

## Work packages

### INT-050-001 — Plumbing proof (ingest → export bundle)

- TASK-VIDEO-INGEST-001 Deterministic frame decode + timebase extraction for local videos.
- TASK-RUN-VIDEO-001 CLI runner: ingest video → run detectors/trackers → export tracks/events bundle.

### INT-050-002 — Baseline real detections (UAT)

- TASK-BALL-DET-BASELINE-001 Baseline local ball detector producing non-empty detections on the provided clip with confidence/diagnostics.
- TASK-BALL-DET-UAT-FIXTURE-SYNTH-001 Add a compact synthetic fixture spec + test to validate baseline present outputs without committing real video.

### INT-050-003 — Hardening + portal UX

- TASK-EVENTS-DETERMINISTIC-SORT-001 Sort events deterministically before writing events.json.
- TASK-VALIDATE-STRICT-AND-REPORT-001 Add validate --strict and emit validation_report.json from run-video.
- TASK-VIDEO-TOOLS-VERSION-DIAGNOSTICS-001 Capture and gate ffmpeg/ffprobe versions in diagnostics.
- TASK-PORTAL-READMODEL-CACHE-TEST-001 Add a unit test for portal audit-run listing cache behavior.
- TASK-RUN-VIDEO-STREAM-001 Stream run-video to reduce memory usage.
- TASK-VIDEO-TOOLS-TIMEOUT-001 Add video tooling timeouts and better stderr surfacing.
- TASK-RUN-VIDEO-NULL-ASSERT-001 Strengthen tests: detector disabled → zero present.
- TASK-DIAGNOSTICS-BALL-DETECTIONS-001 Persist per-frame ball detection diagnostics artifact.
- TASK-SHOTS-GOALS-CONFIG-001 Expose shots/goals inference thresholds via CLI flags.
- TASK-CONTRACT-BALL-FRAMEIDX-001 Require diagnostics.frame_index for all ball track records.
- TASK-VIDEO-TOOLS-NODE-DECOUPLE-001 Remove Node coupling from Python ingest tool resolution.
- TASK-BALL-DET-REGRESSION-001 Add a synthetic regression test for baseline detector present frames.
- TASK-CLI-EXIT-CODES-001 Add explicit exit codes for common CLI failures.
- TASK-PORTAL-AUDIT-RUNS-001 Show latest audit run summaries on the portal intent page.

## Runbooks (LLM navigation)

- Decision: `none`
- Templates: (none)
- Notes: No runbook changes required for this intent.

## Scope (paths)

- Allowed: `spec/`, `spec/md/docs/`, `pipeline/`, `status/audit/`
- Excluded: `docs/`, `status/intents/`, `status/portal/`, `status/wizard/`

