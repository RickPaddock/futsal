---
generated: true
source: spec/md/docs/runbooks/requirements-management.mdt
source_sha256: sha256:6189e5fe31f91ceb002341e1308e979f28e4f5a0ff812f4295f8ad7f6882225e
---

# Requirements management

## Canonical sources

- Requirements entrypoint: `spec/requirements/index.json`
- Requirements area files: `spec/requirements/areas/*.json` (listed by the index)

## Splitting strategy (keep file sizes down)

Split requirements by stable area:

- `spec/requirements/areas/core.json` — governance/platform invariants
- `spec/requirements/areas/v1.json` — V1 product requirements
- Add more area files as needed (e.g. `vision.json`, `pipeline.json`) and list them in `spec/requirements/index.json`.

## Requirement lifecycle fields

Each requirement has:

- `status`: `draft` | `canonical` | `deprecated` (definition lifecycle)
- `tracking.implementation`: `todo` | `done` (implementation tracking; only set to `done` after an audit confirms code exists)

## Adding new requirements (REQ-* flow)

New requirements are created from tasks/subtasks:

1) Create a subtask whose `subtask_id` starts with `REQ-` in a task spec: `spec/tasks/<TASK_ID>.json`
2) Add the same `REQ-*` entry to a requirements area file with:
   - `status: "draft"`
   - `tracking.implementation: "todo"`
3) Run `npm run guardrails` (fails if the `REQ-*` subtask exists but the requirement entry is missing)

When code exists and contains `REQ: REQ-....`, `tracking.implementation` may be updated to `done` as part of closing the intent.

## Templates

- Requirements index: `spec/templates/requirements_index.template.json`
- Requirements area: `spec/templates/requirements_area.template.json`

