---
generated: true
source: spec/md/docs/runbooks/intent-and-task-workflow.mdt
source_sha256: sha256:0012c73f78f987c5bccc29f3990f35bf33f4efacfe04a8b856b133cd0c86d10d
---

# Intent and task workflow

This repo treats planning artefacts as canonical JSON and generates all human-readable `.md` and portal feeds.

## Canonical sources

- Intents: `spec/intents/*.json`
- Tasks: `spec/tasks/*.json`
- Requirements: `spec/requirements/index.json`

## Intent creation (INT-*)

1) Create an intent file at `spec/intents/INT-<NNN>.json`
2) Choose the intent status model:
   - `draft`: number reserved; no tasks yet (empty `task_ids_planned` and `work_packages`)
   - `todo`: tasks exist and are wired; requirements are updated if needed
   - `closed`: passed audit and close process completed successfully
2) Populate required fields:
   - `intent_id`, `title`, `status`, `created_date`
   - `requirements_in_scope[]`
   - `task_ids_planned[]`
   - `work_packages[]` with `items[]` lines starting with the task id
3) For every `task_ids_planned[]`, create a task spec at `spec/tasks/<TASK_ID>.json`
4) Run `npm run generate` then `npm run guardrails`

Intent templates:
- Draft: `spec/templates/intent.template.json`
- Todo: `spec/templates/intent.todo.template.json`

## Task creation (TASK-*)

Every task represents an actual deliverable.

Each task spec must include:

- `task_id`, `intent_id`, `title`, `status`
- `deliverables[]` (what gets shipped/changed; include paths when possible)
- `subtasks[]` (optional, but expected when the task spans multiple work areas)

Task template: `spec/templates/task.template.json`

## Subtasks (SUB-* and REQ-*)

- Use `SUB-*` for implementation subtasks inside a task.
- Use `REQ-*` subtasks when the deliverable is to create a new requirement:
  - `REQ-*` must exist in `spec/requirements/areas/*.json` with `tracking.implementation: "todo"`
  - Only set `tracking.implementation: "done"` after audit confirms code exists and references the requirement via `REQ: REQ-*`

## Regeneration + validation

1) `npm run generate`
2) `npm run guardrails`
