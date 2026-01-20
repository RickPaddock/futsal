---
generated: true
source: spec/md/docs/runbooks/intent-close.mdt
source_sha256: sha256:7505316a0d6d675812856555fbea0424d57cde553573bb08f6b9c33da58167bd
---

# Intent close

Closing an intent updates canonical sources after an audit confirms:

- Task specs exist for all planned tasks
- Any new requirements (`REQ-*`) are tracked in requirements area files
- Code exists and references any `REQ-*` requirements via `REQ: REQ-*` tags

## Audit

- Run: `npm run audit:intent -- --intent-id INT-001`
- Optional report output: `npm run audit:intent -- --intent-id INT-001 --out status/audit/INT-001/runs/<run_id>/audit_report.json`

## Close (apply updates)

This updates:

- `spec/intents/<INTENT_ID>.json` (`status: "closed"`, plus `closed_date`)
- Any `REQ-*` requirements created by the intent (`tracking.implementation: "done"`)

Command:

`npm run intent:close -- --intent-id INT-001 --closed-date YYYY-MM-DD --apply`

## VSCode LLM prompt

Prompt template: `spec/prompts/intent_close.prompt.txt`
End-to-end close prompt: `spec/prompts/intent_close_end_to_end.prompt.txt`

## Status definitions (enforced)

- `draft`: intent exists, but no tasks yet (empty `task_ids_planned` / `work_packages`)
- `todo`: intent is fully specified with tasks; requirements are updated if needed
- `closed`: audit passed and close process applied successfully (`closed_date` required)
