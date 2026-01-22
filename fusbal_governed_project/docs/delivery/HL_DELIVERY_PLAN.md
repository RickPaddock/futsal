---
generated: true
source: spec/md/docs/delivery/HL_DELIVERY_PLAN.mdt
source_sha256: sha256:fa2cf158fbdcadf814ca0cc3fcaa7a99d29d78d39a38a540bb5bc0a797c4bf66
---

# High-level delivery plan (V1)

This plan is implemented as **intents** in `spec/intents/*.json` and generated into `status/intents/<INT-ID>/intent.md`.

## V1 focus (trust-first)

- Accuracy and trust over feature breadth.
- Prefer Unknown/missing over wrong (ball + events).
- Prefer track breaks over identity swaps (players).

## Roadmap (ordered)

1) `INT-001` — Governance + generated surfaces (requirements, intents, portal, evidence).
2) `INT-010` — V1 output contract + pipeline scaffolding (match bundle + schemas + CLI).
3) `INT-020` — Calibration + BEV mapping MVP (quality-gated).
4) `INT-030` — Player tracking MVP (trust-first) + team assignment.
5) `INT-040` — Ball tracking + conservative shots/goals (V1).

## Definition of done (V1)

- A user can process a match video and receive `overlay.mp4` + a report that explains confidence/known gaps.
- When calibration quality passes, BEV outputs are produced; otherwise the system disables BEV with explicit reasons.
