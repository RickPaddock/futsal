---
generated: true
source: spec/md/docs/runbooks/single-source-of-truth.mdt
source_sha256: sha256:153765f96334a45d54da50b6b15fa1206c2b61c9c96a3159dcf6d67f4138216d
---

# Single source of truth

## Canonical sources

- Project config: `spec/project.json`
- Requirements (canonical): `spec/requirements/index.json`
- Intents: `spec/intents/*.json`
- Markdown template sources: `spec/md/**/*.mdt`

Notes:
- `spec/requirements/index.json` may be a single requirements file, or a requirements index that lists multiple area files.

## Derived surfaces

- Requirements view: `docs/requirements/requirements.md`
- Intent bundle(s): `status/intents/<INT-ID>/*`
- Portal feed: `status/portal/internal_intents.json`

## Workflow

1) Edit a canonical source.
2) Regenerate: `npm run generate`
3) Validate: `npm run guardrails`
4) Record evidence (recommended): see `docs/runbooks/evidence.md`
