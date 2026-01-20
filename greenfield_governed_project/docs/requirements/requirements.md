---
generated: true
source: spec/requirements/index.json + spec/requirements/areas/core.json + spec/requirements/areas/greenfield.json + spec/requirements/areas/v1.json
source_sha256: sha256:29f64d4b5f67b28aeeccbdf199f6889970d031deab39d054b0f0562027a71906
---

# Requirements (generated)

Source: `spec/requirements/index.json`

## SYS-ARCH-15 — All code units MUST be traceable to requirements; shared utilities map to SYS-ARCH-15.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:req_tag_enforced_on_done`, `guardrails:repository_guardrails`
- Owner: `platform`
- Tags: `governance`, `traceability`

Acceptance:
- No code unit can be merged without REQ mapping.
- Shared/plumbing code uses SYS-ARCH-15 with a human WHY.

## AUD-REQ-10 — Generate + enforce Requirement ↔ Code provenance (100% policy with guardrails).

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:repository_guardrails`, `audit:intent`
- Owner: `platform`
- Tags: `governance`, `guardrails`

Acceptance:
- Deterministic provenance scanner exists and is enforced by guardrails.
- Derived reports exist and are never hand-edited.

## GREENFIELD-GOV-001 — All human-readable .md outputs are generated; no hand-edits allowed.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:markdown_generated_only`, `generate:check`
- Owner: `platform`
- Tags: `governance`, `generation`

Acceptance:
- All `.md` files contain generated frontmatter and fail guardrails if missing.
- Humans edit only canonical JSON sources and templates, then regenerate.

## GREENFIELD-GOV-002 — Generation is deterministic and drift is detectable.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:deterministic_generation`, `generate:check`
- Owner: `platform`
- Tags: `governance`, `generation`

Acceptance:
- `npm run generate:check` fails if generated outputs drift from canonical sources.
- `npm run guardrails` runs generate check before declaring success.

## GREENFIELD-GOV-003 — Intent status model is enforced (draft → todo → closed).

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:intent_status_model`
- Owner: `platform`
- Tags: `governance`, `intents`

Acceptance:
- `draft` intents have no tasks/work packages.
- `todo` intents have tasks wired.
- `closed` intents require `closed_date`.

## GREENFIELD-GOV-004 — Every non-draft intent must have task specs for all planned tasks.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:tasks_required_for_todo_intents`
- Owner: `platform`
- Tags: `governance`, `tasks`

Acceptance:
- For each `intent.task_ids_planned[]`, a matching `spec/tasks/<TASK_ID>.json` exists.
- Task specs must reference the same `intent_id`.

## GREENFIELD-GOV-005 — Every `todo` task has at least one deliverable.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:task_deliverables_required`
- Owner: `platform`
- Tags: `governance`, `tasks`

Acceptance:
- If a task is `status: todo`, `deliverables[]` is present and non-empty.
- Each deliverable has a stable id and human title.

## GREENFIELD-GOV-006 — New requirements are created and tracked via `REQ-*` subtasks.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:req_new_requirements_flow`
- Owner: `platform`
- Tags: `governance`, `requirements`

Acceptance:
- Any subtask id starting with `REQ-` must exist as a requirement entry.
- New requirements start with `tracking.implementation: todo`.

## GREENFIELD-GOV-007 — Closing an intent updates requirement implementation tracking only after code exists.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:close_updates_req_tracking`, `intent:close`
- Owner: `platform`
- Tags: `governance`, `close`

Acceptance:
- `npm run intent:close -- --apply` sets intent status to `closed` and writes `closed_date`.
- For any `REQ-*` created by the intent, `tracking.implementation` is set to `done` only when code contains `REQ: REQ-*` references.

## GREENFIELD-GOV-008 — Governance file size limits are enforced to prevent runaway specs.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:file_size_limits`
- Owner: `platform`
- Tags: `governance`, `scale`

Acceptance:
- File size limits are configured in `spec/project.json` and enforced by guardrails.

## GREENFIELD-GOV-009 — Requirements are split into multiple area files via an index entrypoint.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:requirements_split_supported`
- Owner: `platform`
- Tags: `governance`, `requirements`

Acceptance:
- `spec/requirements/index.json` lists `spec/requirements/areas/*.json` files.
- Generation and guardrails read from the index and treat the bundle as canonical.

## GREENFIELD-GOV-010 — Every requirement declares at least one guardrail/coverage mechanism to detect regressions.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:requirement_guardrail_coverage_required`
- Owner: `platform`
- Tags: `governance`, `requirements`

Acceptance:
- All requirement entries include a non-empty `guardrails[]` list.
- Guardrails enforce the presence of the `guardrails[]` list.

## GREENFIELD-PORTAL-001 — Internal portal task pages are human-readable (not raw JSON).

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `portal:ui_smoke`, `manual:portal_review`
- Owner: `platform`
- Tags: `portal`, `ux`, `governance`

Acceptance:
- `/internal/tasks/<TASK_ID>` renders task scope, acceptance, deliverables, and subtasks as structured sections.
- Raw canonical JSON remains accessible (e.g., via a collapsible section) for debugging and audits.
- Task pages link back to the parent intent when `intent_id` is present.

## FUSBAL-V1-TRUST-001 — Trust-first behavior: avoid identity swaps and ball/event hallucinations; prefer Unknown/missing over wrong.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:req_tag_enforced_on_done`
- Owner: `product`
- Tags: `v1`, `trust`

Acceptance:
- Player tracking prefers track breaks over ID swaps when ambiguous.
- Ball tracking explicitly supports missing/unknown state and does not fabricate positions.
- Event inference is high-precision (low-recall acceptable) and emits confidence with evidence links.

## FUSBAL-V1-OUT-001 — Produce a deterministic per-match output bundle (overlay, BEV, tracks, events, report) with diagnostics.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:req_tag_enforced_on_done`
- Owner: `platform`
- Tags: `v1`, `outputs`

Acceptance:
- A match run outputs a stable directory layout under `output/<match_id>/`.
- At minimum, an overlay video and a report describing confidence/known gaps are produced.
- BEV outputs are produced only when calibration quality passes thresholds; otherwise disabled with reasons.

## FUSBAL-V1-DATA-001 — All sources emit a canonical track record format suitable for fusion and reporting.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:req_tag_enforced_on_done`
- Owner: `platform`
- Tags: `v1`, `data`

Acceptance:
- Tracks are stored as `.jsonl` (one record per entity per sample) plus JSON metadata.
- Each sample includes timestamp, coordinate frame, uncertainty, and optional quality (0–1).

## FUSBAL-V1-CAL-001 — Calibration maps each camera to pitch coordinates with explicit quality gating and diagnostics.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:req_tag_enforced_on_done`
- Owner: `vision`
- Tags: `v1`, `calibration`

Acceptance:
- Calibration outputs homographies (or equivalent) and a quality score with failure reasons.
- If auto-calibration fails, a minimal manual fallback can produce a reusable calibration artifact.

## FUSBAL-V1-BEV-001 — When calibration quality is sufficient, generate bird’s-eye-view (BEV) tracks and video in pitch meters.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:req_tag_enforced_on_done`
- Owner: `vision`
- Tags: `v1`, `bev`

Acceptance:
- Players (and ball when available) are projected into a consistent pitch coordinate system in meters.
- BEV video renders movement with clear confidence cues and missing-state semantics.

## FUSBAL-V1-PLAYER-001 — Detect and track all players per match with conservative identity continuity.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:req_tag_enforced_on_done`
- Owner: `vision`
- Tags: `v1`, `players`

Acceptance:
- Per-frame detections carry confidence and are filtered to the pitch area when possible.
- Tracking produces track IDs and per-track confidence; ambiguous associations cause breaks, not swaps.

## FUSBAL-V1-TEAM-001 — Assign players to Team A/B via bib colors (plus smoothing) with an Unknown state.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:req_tag_enforced_on_done`
- Owner: `vision`
- Tags: `v1`, `teams`

Acceptance:
- Team label is output per player sample (A/B/Unknown) with confidence.
- Assignments are temporally smoothed and include diagnostics for ambiguous segments.

## FUSBAL-V1-BALL-001 — Track the ball when visible and explicitly represent missing/unknown spans without hallucination.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:req_tag_enforced_on_done`
- Owner: `vision`
- Tags: `v1`, `ball`

Acceptance:
- Ball detector/track output includes confidence and missing-state semantics.
- Long ball-missing spans are allowed; output must prefer unknown over wrong.

## FUSBAL-V1-EVENT-001 — Infer shots and goals conservatively in V1 (high precision) and emit confidence + evidence pointers.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:req_tag_enforced_on_done`
- Owner: `vision`
- Tags: `v1`, `events`

Acceptance:
- Events are emitted with timestamps and confidence; uncertain events are omitted or marked candidate.
- Goal events include a best-effort evidence window reference (time range / frame ids).

## FUSBAL-V1-SENSOR-001 — Sensors are optional inputs that only improve outputs when healthy; camera-only output remains sellable.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:req_tag_enforced_on_done`
- Owner: `platform`
- Tags: `v1`, `sensors`

Acceptance:
- The pipeline accepts sensor logs as additive TrackSources using the canonical track format.
- If sensor ingestion fails or is low quality, outputs fall back to camera-only without contract changes.

