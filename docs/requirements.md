# Futsal Player Tracking — MVP Requirements (Aligned)

Build an offline (post-processed) computer-vision system that takes a futsal match video and outputs **trustworthy** tracking, conservative events, and (when possible) a bird’s-eye tactical view. This is a **trust product**: explicit uncertainty and conservative behavior beat feature breadth.

## MVP decisions (locked)

1) **Single camera**: one static wide-angle camera on the sideline (like the sample GoPro clip).
2) **Bibs for MVP**: two bib colors are required for Team A/B labeling in MVP.
3) **Passes in V2**: pass detection and possession attribution are not MVP deliverables.
4) **Identity policy**: MVP targets a **99% identity correctness bar** (definition below) and prefers **track breaks over swaps**.
5) **BEV**: BEV outputs are produced only when calibration quality passes deterministic thresholds; otherwise BEV is explicitly disabled with reasons.

## Inputs (MVP)

- A single match video: static wide-angle sideline view (no handheld).
- Two bib colors provided by the venue/teams (Team A / Team B).
- Pitch dimensions (or a small set of supported pitch templates) available per venue.

## Outputs (MVP bundle)

Per match, produce an output bundle directory containing:

- `overlay.mp4`
  - Player boxes + track IDs.
  - Team label (A/B) + confidence (internally allow low-confidence/unknown handling; do not guess silently).
  - Ball marker when present + confidence; explicit missing/unknown semantics.
- `bev.mp4` (best-effort; only when calibration passes)
  - Top-down (pitch meters) rendering of player/ball movement.
- `tracks.jsonl`
  - Canonical time series (one record per entity per sample).
- `events.json`
  - Shots + goals only in MVP; conservative (high precision; low recall acceptable).
- `report.json` + `report.html`
  - Summary + confidence + known gaps + diagnostics pointers.
- `diagnostics/`
  - Calibration diagnostics and quality summaries (including BEV gating decision).

## Out of scope (V2+)

- Pass detection and possession attribution.
- Team separation without bibs (e.g., “inconsistent kits” without constraints).
- Jersey number OCR.
- Tactical formations / pressing metrics / pass networks / advanced event taxonomy.
- Real-time processing.

## Definitions (make requirements enforceable)

### Identity / track / segment
- **Identity**: a stable person-level label within one match (e.g., `player_07`).
- **Track**: time series emitted for one identity.
- **Segment**: a contiguous span of confident association for an identity.

### Break vs swap (trust-first)
- **Track break**: end the current segment because association is ambiguous; optionally start a new segment later (may be a new identity).
- **Identity swap**: continuing an identity label while it begins referring to a different person.

MVP rule: **breaks are allowed; swaps are not** (break rather than swap when ambiguous).

### 99% identity correctness (MVP target)
Identity correctness is measured on a labeled evaluation set as:

> Fraction of **player-visible frames** where the system’s identity assignment matches ground truth.

MVP target:
- **≥ 99% identity correctness** on the evaluation set.

Severity policy:
- Any detected **swap** is a severity-1 failure even if the aggregate percentage remains high.

### Ball state semantics
- **present**: a ball location is asserted with evidence and confidence.
- **missing**: the frame is evaluable but no acceptable ball position meets gating thresholds.
- **unknown**: the frame is not evaluable (e.g., decode error, missing timestamp). Unknown MUST include a reason.

### Shot / goal (MVP)
- **Shot**: conservative candidate inferred from ball motion consistent with a shot, only when evidence thresholds are met.
- **Goal**: conservative candidate/unknown inferred from a shot followed by a goal-like ball disappearance pattern.

MVP is permitted to emit **candidate/unknown only** until a later revision defines a “confirmed” bar.

### BEV (bird’s-eye view)
**BEV** (bird’s-eye view) is a top-down tactical representation where positions are mapped from image pixels into **pitch coordinates in meters** via camera→pitch calibration.

### Calibration quality thresholds (MVP gating)
BEV is enabled only when all thresholds pass (deterministic gating):
- `rms_reprojection_error_px <= 4.0`
- `inlier_ratio >= 0.60`
- `num_inliers >= 12`
- `marking_coverage_score_0_to_1 >= 0.50`

If gating fails, BEV artifacts MUST be marked missing and the report MUST explain why.

## Acceptance criteria (MVP)

- **Determinism**: same inputs/config produce stable outputs (byte-stable ordering for `tracks.jsonl`/`events.json`).
- **Trust-first**:
  - No ball/event hallucination: prefer missing/unknown over wrong.
  - No identity swaps: break instead when ambiguous.
- **BEV gating**:
  - If calibration passes, produce `bev.mp4` + pitch-frame tracks.
  - If calibration fails, do not produce BEV; report explicit failure reasons in diagnostics/report.
- **Usability**:
  - `overlay.mp4` is playable and actually contains markers when evidence exists.
  - `report.html` is non-placeholder and explains confidence/known gaps.

## Test footage (current)

We currently have test footage using a static GoPro wide-angle camera. Constraints:
- Large file size (11gb for 15 minutes)
- FishEye effect
- Near corners of pitch missing

![alt text](image.png)
