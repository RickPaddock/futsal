---
generated: true
source: spec/requirements/index.json + spec/requirements/areas/core.json + spec/requirements/areas/greenfield.json + spec/requirements/areas/v1.json
source_sha256: sha256:feb34c62b6a85ba9b2633f4438b00371d6a2dec67080d4103fe933087541afb8
---

# Requirements (generated)

Source: `spec/requirements/index.json`

## SYS-ARCH-15 — All code units MUST be traceable to requirements; shared utilities map to SYS-ARCH-15.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `guardrails:req_tag_enforced_on_done`, `guardrails:repository_guardrails`
- Owner: `platform`
- Tags: `governance`, `traceability`

Acceptance:
- No code unit can be merged without REQ mapping.
- Shared/plumbing code uses SYS-ARCH-15 with a human WHY.

## AUD-REQ-10 — Generate + enforce Requirement ↔ Code provenance (100% policy with guardrails).

- Status: `canonical`
- Implementation: `done`
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
- New `REQ-*` requirements include `guardrails:req_tag_enforced_on_done` to prevent regressions.

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

## GREENFIELD-GOV-012 — Task specs are gold-standard and machine-checkable (scope, acceptance, file-level deliverables, and provable subtasks).

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:repository_guardrails`
- Owner: `platform`
- Tags: `governance`, `tasks`, `quality`

Acceptance:
- All `todo` tasks include non-empty `scope[]` and `acceptance[]`.
- All deliverables have file-level `paths[]` (no directory-only paths) and non-empty deliverable `acceptance[]`.
- All `todo` tasks include non-empty `subtasks[]` and every subtask includes `provenance_prefix` + non-empty `done_when[]`.

## GREENFIELD-GOV-013 — Every `REQ-*` requirement includes a regression guardrail (cannot be marked implemented without code references).

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `guardrails:repository_guardrails`, `guardrails:req_tag_enforced_on_done`
- Owner: `platform`
- Tags: `governance`, `requirements`, `regression`

Acceptance:
- Any requirement whose id starts with `REQ-` includes `guardrails:req_tag_enforced_on_done`.
- `npm run guardrails` fails if any `REQ-*` requirement omits the guardrail.
- `tracking.implementation: done` is rejected unless the repo contains a `REQ: <REQ_ID>` reference.

## GREENFIELD-GOV-014 — Intents are closed only via the close process (no manual status flips).

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `intent:close`, `manual:repo_review`
- Owner: `platform`
- Tags: `governance`, `intents`, `close`

Acceptance:
- Users do not manually set intent `status: closed` in JSON; they run `npm run intent:close -- --apply`.
- Close requires required audits and guardrails to have passed and to be evidenced under `status/audit/<INTENT_ID>/runs/<run_id>/`.
- Intent `closed_date` is set only by the close process.

## GREENFIELD-OPS-001 — Logging and error surfacing are usable for audits and operators.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `manual:ops_runbook`, `manual:portal_review`
- Owner: `platform`
- Tags: `ops`, `logging`, `observability`

Acceptance:
- Scripts print stable, searchable prefixes (e.g. `[generate]`, `[guardrails]`, `[audit]`, `[close]`).
- Portal actions that run automation (Refresh) surface errors and show actionable stdout/stderr.

## GREENFIELD-OPS-002 — Starting the internal portal clears common dev ports and starts reliably.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `manual:ops_runbook`
- Owner: `platform`
- Tags: `ops`, `portal`, `devex`

Acceptance:
- Starting the portal clears ports 3015–3020 (kills LISTEN pids) before launching Next dev server.
- Startup runs generation before starting the portal so data is current.

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

## GREENFIELD-PORTAL-002 — Internal portal provides copy-ready prompts for create/implement/audit/close flows.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `portal:ui_smoke`, `manual:portal_review`
- Owner: `platform`
- Tags: `portal`, `ux`, `workflows`

Acceptance:
- `/internal/intents` includes a 'Create prompt' action that shows the gold-standard intent creation prompt in an overlay.
- For intents with `status: todo`, actions 'Implement', 'Audit', and 'Close' appear and open overlays containing the gold-standard prompts.
- Prompts are prefilled with the intent id and include run_id/closed_date defaults suitable for pasting into an LLM.
- Prompts are sourced from canonical templates under `spec/prompts/*.prompt.txt` (not hardcoded ad-hoc text).
- No VS Code tasks/wizard flow is required to use the system; the portal is sufficient to obtain all prompts.

## GREENFIELD-PORTAL-003 — Portal surfaces intents in priority order with clickable work items.

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `portal:ui_smoke`, `manual:portal_review`
- Owner: `platform`
- Tags: `portal`, `ux`, `workflow`

Acceptance:
- `/internal/intents` lists intents ordered by status (todo first, then draft, then closed).
- Each intent links to its detail page; each planned task is clickable and human-readable.
- Intent list shows audit pass/fail state from evidence runs when available.

## GREENFIELD-PORTAL-004 — Portal refresh regenerates governed surfaces and reports failures.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `portal:ui_smoke`, `manual:portal_review`
- Owner: `platform`
- Tags: `portal`, `generation`, `ops`

Acceptance:
- Pressing Refresh triggers `npm run generate` and updates portal-visible surfaces.
- If generation fails, the portal shows actionable error output.

## GREENFIELD-PORTAL-005 — Portal lifecycle actions are derived from evidence-based readiness rules.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `portal:ui_smoke`, `manual:portal_review`
- Owner: `platform`
- Tags: `portal`, `workflows`, `governance`

Acceptance:
- Portal offers Implement when an intent has tasks and not all tasks are resolved.
- Portal offers Audit when all planned tasks are resolved.
- Portal offers Close only when all planned tasks are resolved, required per-task quality audits exist and pass, and close-gate evidence exists and passes.
- Intent list and intent detail pages use the same readiness computation and do not use manual toggles.

## GREENFIELD-PORTAL-006 — Portal validates and surfaces close-gate evidence status.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `portal:ui_smoke`, `manual:portal_review`
- Owner: `platform`
- Tags: `portal`, `evidence`, `close`

Acceptance:
- Portal checks each intent close gate command from generated scope and indicates satisfied/missing.
- A close gate is satisfied only when a `run.json` evidence record exists with `exit_code: 0` and a matching command.
- Portal surfaces the exact missing gate command(s) when Close is blocked.

## GREENFIELD-PORTAL-007 — Portal surfaces per-task quality audit status and blocks close when audits are missing/failing.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `portal:ui_smoke`, `manual:portal_review`
- Owner: `platform`
- Tags: `portal`, `audit`, `quality`, `close`

Acceptance:
- Portal displays per-task quality audit status for all tasks in `spec/intents/<INTENT_ID>.json:task_ids_planned[]`.
- Portal refuses to present Close as available when any planned task is missing the required per-task quality audit JSON or when any is failing.
- Portal distinguishes missing audits from missing close-gates when close is blocked.

## GREENFIELD-PORTAL-008 — Portal identifies missing planned tasks in the generated tracker feed.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `portal:ui_smoke`, `manual:portal_review`
- Owner: `platform`
- Tags: `portal`, `tasks`, `governance`

Acceptance:
- If an intent declares a planned task that is missing from the generated feed, portal surfaces it as missing.
- When planned tasks are missing from the feed, Close is considered impossible and the portal explains why.

## GREENFIELD-PORTAL-009 — Portal close failures explain missing audits vs missing gates (no generic errors).

- Status: `canonical`
- Implementation: `done`
- Guardrails: `portal:ui_smoke`, `manual:portal_review`
- Owner: `platform`
- Tags: `portal`, `ux`, `close`

Acceptance:
- When Close is blocked, portal lists missing per-task audit files separately from missing close-gate commands.
- Portal does not fail with a generic 'unknown' when structured missing information exists.

## GREENFIELD-PORTAL-010 — Portal root route (/) is a dashboard entrypoint.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `portal:ui_smoke`, `manual:portal_review`
- Owner: `platform`
- Tags: `portal`, `ux`

Acceptance:
- The portal renders a home page at `/` linking to governed workflow pages (e.g., `/internal/intents`).
- Deep internal routes are not the only entrypoint.

## GREENFIELD-PORTAL-011 — Portal serves prompt templates via a stable API and validates substitutions.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `portal:ui_smoke`, `manual:portal_review`
- Owner: `platform`
- Tags: `portal`, `workflows`, `prompts`

Acceptance:
- Portal exposes an API endpoint to fetch rendered prompts for create/implement/audit/close.
- The API fills stable variables including intent id, run id, and closed date (where applicable).
- If prompt loading or rendering fails, portal overlays show explicit errors.
- If any placeholders remain (e.g., `<...>`), the API returns an explicit error rather than silently returning an invalid prompt.

## GREENFIELD-PORTAL-012 — Portal refresh is same-origin hardened and writes evidence of generation runs.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `manual:portal_review`, `manual:security_review`
- Owner: `platform`
- Tags: `portal`, `generation`, `security`, `evidence`

Acceptance:
- Refresh rejects cross-origin requests and only accepts same-origin calls by default.
- Refresh writes a `run.json` evidence record for the generation run (intent-scoped when applicable).
- Refresh returns actionable stdout/stderr to the caller on failure.

## GREENFIELD-PORTAL-013 — Portal uses a shared read-model module for filesystem-backed domain logic.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `manual:repo_review`
- Owner: `platform`
- Tags: `portal`, `maintainability`

Acceptance:
- Intent/task pages and API routes share a single implementation for reading feed/spec/audit artefacts.
- Readiness and evidence validation logic lives in shared utilities (not duplicated across pages).

## GREENFIELD-PORTAL-014 — Portal audit run discovery supports nested run stages consistently.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `manual:repo_review`, `portal:ui_smoke`
- Owner: `platform`
- Tags: `portal`, `evidence`, `observability`

Acceptance:
- Portal discovers evidence runs under `status/audit/<INTENT_ID>/runs/<RUN_ID>/**/run.json` (nested stages).
- List and detail pages compute audit pass/fail from the same run discovery logic.

## GREENFIELD-PORTAL-015 — Portal validates route and API parameters used for filesystem access.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `manual:security_review`
- Owner: `platform`
- Tags: `portal`, `security`, `safety`

Acceptance:
- Portal validates intent ids and task ids against expected formats before using them in file paths.
- Invalid ids return 400/404 with explicit errors and do not read arbitrary filesystem paths.

## GREENFIELD-PORTAL-016 — Portal renders large artefacts as summaries with raw links (avoids giant JSON dumps by default).

- Status: `canonical`
- Implementation: `done`
- Guardrails: `manual:portal_review`
- Owner: `platform`
- Tags: `portal`, `ux`, `scale`

Acceptance:
- Intent pages show a concise summary for audit reports and quality reports, with links to view/download raw JSON.
- Portal remains usable when artefacts grow beyond small sizes.

## GREENFIELD-PORTAL-017 — Portal server-side rendering avoids unnecessary synchronous IO and scales with intent count.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `manual:repo_review`
- Owner: `platform`
- Tags: `portal`, `performance`

Acceptance:
- Portal limits run discovery and file reads to bounded work per request.
- Portal code uses shared helpers to avoid repeated reading of the same artefacts.

## GREENFIELD-PORTAL-018 — Prompt overlays allow choosing the run_id and reuse it across actions.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `manual:portal_review`
- Owner: `platform`
- Tags: `portal`, `ux`, `workflows`

Acceptance:
- Prompt overlays default to a generated UTC run id but allow the user to edit it.
- Rendered prompts use the chosen run id when filling templates.

## GREENFIELD-PORTAL-019 — Portal renders audit timestamps in UK-local format for intent/task views.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `manual:portal_review`
- Owner: `platform`
- Tags: `portal`, `ux`, `audits`

Acceptance:
- Intent detail page renders per-task audit timestamps in UK format (Europe/London) and links to the run report.
- Audit run timestamps in portal tables are consistently shown in UK format.

## GREENFIELD-PORTAL-020 — Portal provides an interactive preflight review prompt for intents before implementation.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `portal:ui_smoke`, `manual:portal_review`
- Owner: `platform`
- Tags: `portal`, `ux`, `workflows`, `prompts`

Acceptance:
- Intent list and intent detail pages expose a 'Preflight' action for non-closed intents.
- Preflight opens a copy-ready prompt overlay that instructs the LLM to review the intent and ask clarifying questions before implementation.
- Preflight prompt is sourced from `spec/prompts/*.prompt.txt` via the prompt API (not hardcoded).
- Prompt API rejects prompts with leftover placeholders (e.g., `<...>`).

## GREENFIELD-GOV-011 — All governed workflows are prompt-driven with explicit evidence capture (no undocumented manual steps).

- Status: `canonical`
- Implementation: `todo`
- Guardrails: `manual:prompt_review`, `manual:portal_review`
- Owner: `platform`
- Tags: `governance`, `workflows`, `evidence`

Acceptance:
- Create/Implement/Audit/Close each have a gold-standard prompt template (or portal-generated prompt) that fully specifies the steps and commands.
- Create prompts MUST add any new `REQ-*` requirements to `spec/requirements/**.json` immediately with `tracking.implementation: "todo"` and `guardrails:req_tag_enforced_on_done`.
- Audit prompts MUST run `npm run audit:intent` AND `npm run guardrails`, and write evidence JSON under `status/audit/<INTENT_ID>/runs/<run_id>/` (use a unique run_id per run).
- Audit prompts MUST enumerate all 10 improvement recommendations (titles + ROI/effort/risk) in the chat output; do not truncate to a top-N subset.
- Close prompts MUST only close after audits pass and guardrails pass, and requirement tracking moves to `done` only when the repo contains `REQ: <REQ_ID>` references.

## GREENFIELD-GOV-015 — Closing an intent requires per-task quality audit evidence for all planned tasks.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `intent:close`, `manual:prompt_review`
- Owner: `platform`
- Tags: `governance`, `close`, `quality`, `evidence`

Acceptance:
- For every task in `spec/intents/<INTENT_ID>.json` → `task_ids_planned[]`, a task quality audit report exists at `status/audit/<INTENT_ID>/runs/<run_id>/tasks/<TASK_ID>/quality_audit.json`.
- Each task quality audit report includes a gate status and blockers list and must pass (`gate.status: "pass"`, `gate.blockers: []`) before close can apply.
- Close dry-run (no apply) fails with actionable errors when any required task quality audit is missing or failing.

## GREENFIELD-GOV-016 — Each intent declares functional + non-functional quality areas and assigns planned tasks to both.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `audit:intent`, `manual:prompt_review`
- Owner: `platform`
- Tags: `governance`, `intents`, `quality`

Acceptance:
- Each `spec/intents/<INTENT_ID>.json` declares `quality_areas[]` including exactly one `area_id: "functional"` and one `area_id: "nonfunctional"`.
- Both areas include non-empty `task_ids[]` that are disjoint and whose union equals `task_ids_planned[]`.
- The non-functional area declares `categories_required` including: `correctness_safety`, `performance`, `security`, `maintainability`.

## GREENFIELD-GOV-017 — Per-task quality audit evidence covers functional + non-functional validation and is required to close.

- Status: `canonical`
- Implementation: `done`
- Guardrails: `intent:close`, `manual:prompt_review`
- Owner: `platform`
- Tags: `governance`, `audit`, `close`, `quality`, `evidence`

Acceptance:
- Audit runs produce per-task reports at `status/audit/<INTENT_ID>/runs/<run_id>/tasks/<TASK_ID>/quality_audit.json` that include both `functional` and `nonfunctional` sections.
- Per-task non-functional validation covers: correctness/safety, performance, security, maintainability.
- Close fails if any planned task is missing the per-task quality audit report, or if any report does not pass functional + non-functional gates.

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

