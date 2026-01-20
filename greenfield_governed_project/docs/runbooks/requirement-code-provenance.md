---
generated: true
source: spec/md/docs/runbooks/requirement-code-provenance.mdt
source_sha256: sha256:923d27d98f11aae96d5e05351b7c7f99d024837491a3595079f7b1537e48bef8
---

# Requirement ↔ Code provenance (PROV / REQ / WHY)

Every in-scope code unit MUST declare:
- `PROV:` stable provenance identifier (survives refactors/renames)
- `REQ:` one or more requirement IDs it fulfills
- `WHY:` a single-line human explanation

If a unit is shared/plumbing, it MUST map to:
- `REQ: SYS-ARCH-15`

## Tag placement

### JS / TS

Put tags in a leading block comment before imports:

```js
/*
PROV: EXAMPLE.PROV.ID
REQ: AUD-REQ-10
WHY: Explains why this file exists.
*/
```

### Shell

Put tags after shebang:

```bash
#!/usr/bin/env bash
# PROV: EXAMPLE.PROV.ID
# REQ: SYS-ARCH-15
# WHY: Shared repo automation.
```

### Python

Put tags at the top of the file:

```py
# PROV: EXAMPLE.PROV.ID
# REQ: FUSBAL-V1-OUT-001, FUSBAL-V1-DATA-001
# WHY: Minimal CLI scaffolding for Fusbal match bundles.
```
