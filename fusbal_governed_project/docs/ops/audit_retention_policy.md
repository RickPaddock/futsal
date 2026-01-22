---
generated: true
source: spec/md/docs/ops/audit_retention_policy.mdt
source_sha256: sha256:209f9d5742d8d4bd3695b0af6a25af9b892598168c4a49321d99043125072b28
---

# Audit run retention policy

## Overview

This document describes the retention and archival strategy for audit evidence stored in `status/audit/`.

## Storage location

All audit runs for an intent are stored in:

```
status/audit//runs//
```

Where:
- `INTENT_ID`: Intent identifier (e.g., `INT-001`)
- `RUN_ID`: UTC timestamp in format `YYYYMMDD_HHMMSS` (e.g., `20260121_152501`)

## Retention guidelines

### Active intents (status: todo, draft)

- **Keep all runs**: Retain all audit evidence to track progress and regression testing
- **No automatic pruning**: Manual cleanup only when specifically required
- **Rationale**: Full history aids debugging and demonstrates continuous validation

### Closed intents (status: closed)

- **Keep final quality audit**: Always retain the quality audit from the closing run
- **Keep closing gate evidence**: Retain all evidence from the close gate run(s)
- **Optional archive**: Older intermediate runs may be archived after 90 days
- **Rationale**: Closing audit is permanent proof of completion; intermediate runs have diminishing value after closure

## Manual cleanup process

When storage becomes constrained:

1. **Identify candidates**: Look for closed intents with many intermediate runs
2. **Verify final audit**: Ensure `quality_audit.json` from closing run exists and is valid
3. **Archive or delete**: Move old intermediate runs to archive storage or delete
4. **Document decision**: Note cleanup in intent notes or project log

### Example cleanup command

```bash
# List runs for closed intent, excluding most recent
ls -d status/audit/INT-001/runs/* | head -n -1

# Review before deleting
# rm -rf status/audit/INT-001/runs/20260101_*
```

## Future automation (not yet implemented)

Potential future improvements:
- Automated archival script for closed intents (> 90 days)
- Compressed archive format for intermediate runs
- Retention policy enforcement in guardrails
- Evidence storage size monitoring and alerts

## Related requirements

- **GREENFIELD-OPS-003**: Audit run retention policy documentation
- **GREENFIELD-EVIDENCE-001**: Stdout/stderr capture in evidence runs

## See also

- [Evidence runbook](../runbooks/evidence.md): How to record evidence
- [Internal portal runbook](../runbooks/internal-portal.md): View audit runs in portal
