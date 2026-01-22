---
generated: true
source: spec/md/docs/runbooks/internal-portal.mdt
source_sha256: sha256:b34a712bab0a0269fd9be01dc0aec5da499a0468dff39117f089fc4d24cb15e2
---

# Internal portal

The internal portal is a lightweight Next.js app that renders governance surfaces (intents, status) from generated JSON.

## Run

1) Generate: `npm run generate`
2) Start: `npm run portal:dev`
3) Open: `http://127.0.0.1:3015/internal/intents`

## Data flow

- Source intents: `spec/intents/*.json`
- Generated intent markdown: `status/intents/<INT-ID>/intent.md`
- Generated portal feed: `status/portal/internal_intents.json`

