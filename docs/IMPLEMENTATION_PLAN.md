# Futsal Tracking System — Implementation Plan

Pass 1 is complete and verified on clip7. All remaining passes need a full rewrite to comply
with the new CLAUDE.md contract.

---

## Pass 2A — Mechanical Fragmentation

**File**: `src/skills/pass2a_fragmenter.py` (full rewrite)
**Input**: `pass1_raw.json`
**Output**: `pass2_fragments.json`, `pass2_validation.json`

### Split Triggers

- [x] **T1 — Track Collision**: same `track_id` produces >1 detection in the same frame → immediate split, reason `"track_collision"`
- [x] **T2 — Jersey Change**: jersey transitions X → Y (both non-None, X ≠ Y), must persist ≥ `JERSEY_CHANGE_PERSISTENCE` frames (15) → split, reason `"jersey_change"`
  - [x] NOT triggered by `None → X` (player turns around) or `X → None`
- [x] **T3 — Jersey Temporal Conflict**: same jersey number on two different tracks simultaneously → split the later-appearing occurrence, reason `"jersey_temporal_conflict"`
  - [x] Skip if jersey appears at very start of the fragment
- [x] **T4 — Team Assignment Discontinuity**: compare mean HSV of window `[t-30, t-1]` vs `[t+1, t+30]`
  - [x] Only trigger if HSV distance > `TEAM_SWITCH_HSV_THRESHOLD` AND both windows have ≥ `TEAM_SWITCH_MIN_SAMPLES` valid HSV samples AND both confidences ≥ `TEAM_SWITCH_CONFIDENCE`
  - [x] Single-frame changes must NEVER trigger this
  - [x] Use sub-clustering for non-bib team (compare against nearest sub-cluster centroid, not a global average)
  - [x] Reason: `"team_switch"`
- [x] **T5 — Impossible Motion Spike**: centroid displacement between adjacent frames > `MAX_PLAYER_SPEED * frame_dt` → immediate split, reason `"motion_spike"`

### Fragment Output

- [x] Assign `fragment_id` as sequential `F000001`, `F000002`, etc.
- [x] Compute per-fragment metadata:
  - [x] `track_id`, `start_frame`, `end_frame`, `detection_ids[]`
  - [x] `split_reason` (null for track's first fragment)
  - [x] `jersey_visible_ratio` (fraction of frames with non-None jersey_number)
  - [x] `dominant_jersey_number` (mode of observed jersey numbers, None if none seen)
  - [x] `occlusion_ratio` (fraction of frames where `jersey_roi_valid = False`)
  - [x] `mean_velocity` (mean centroid displacement per frame, pixels)
  - [x] `appearance_stability_score` (mean pairwise HSV similarity across fragment)
  - [x] `quality` (`HIGH` | `MEDIUM` | `LOW`)
- [x] Fragments shorter than `MIN_FRAGMENT_LENGTH` (15 frames) → `quality = "LOW"`, do NOT delete

### Expected Behaviour

- [x] Typical clip produces 20–30 fragments (significantly more → incorrect triggers)

### Validation (blocking — fail pipeline if any fail)

- [x] Fragment coverage = 100% of detections
- [x] No detection left unassigned
- [x] No fragment frame ranges overlap within the same track
- [x] Every split point has a recorded trigger reason
- [x] No frame assigned to more than one fragment on the same track

---

## Pass 2B — Fragment Quality Scoring

**File**: `src/skills/pass2b_fragment_scoring.py` (full rewrite)
**Input**: `pass2_fragments.json`
**Output**: `pass2b_scored_fragments.json`

- [x] Metadata-only pass — no splitting or merging
- [x] Confirm/compute `appearance_stability_score` (mean HSV pairwise similarity)
- [x] Confirm/compute `jersey_observability_score` (= `jersey_visible_ratio`)
- [x] Compute `motion_smoothness_score` (inverse of velocity variance)
- [x] Confirm `occlusion_ratio`
- [x] Assign quality tier:
  - [x] `HIGH`: appearance_stability ≥ 0.7 AND occlusion_ratio ≤ 0.2
  - [x] `MEDIUM`: appearance_stability ≥ 0.4
  - [x] `LOW`: otherwise

---

## Pass 2C — Ghost Generation

**File**: `src/skills/pass2c_ghost_generator.py` (full rewrite)
**Input**: `pass1_raw.json` + `pass2b_scored_fragments.json`
**Output**: `pass2_ghosts.json` (real fragments + ghost fragments, with `is_ghost` flag)

- [x] Build frame-by-frame `present_count` from Pass 1 detections
- [x] Initialise `level` = max player count across first 10 frames
- [x] Dynamic level: for each frame, if `present_count > level`, set `level = min(12, present_count)` — never decreases
- [x] For each frame: `missing_count = level - present_count`
- [x] For each missing player slot: create or extend a ghost fragment
  - [x] Track by `original_track_id` (most recently seen track for that slot), NOT by fragment_id
  - [x] No fixed max duration in Pass 2C (ghost persists until same-track reappearance or clip end)
  - [x] Ghost terminates when player track reappears OR clip ends
- [x] Tag ghosts: `is_ghost = True`, `quality = "GHOST"`, `exclude_from_clustering = True`

---

## Pass 3A — Identity Candidate Edges

**File**: `src/skills/pass3a_candidate_generator.py` (full rewrite)
**Input**: `pass2b_scored_fragments.json` (exclude `is_ghost = True`)
**Output**: `pass3a_candidates.json`

- [x] For each fragment pair (A, B) where `A.end_frame < B.start_frame`:
  - [x] Reject if `gap > MAX_IDENTITY_GAP` (300 frames)
  - [x] Reject if spatial distance > `MAX_PLAYER_SPEED * gap_frames`
  - [x] Reject if both fragments have non-None conflicting jersey numbers (hard conflict)
- [x] **Track continuity rule**: if same `track_id` AND `gap ≤ TRACK_CONTINUITY_GAP` (120 frames) AND spatial ok → MUST generate candidate with `track_continuity_score = 1.0`
- [x] Score each candidate:
  - [x] `track_continuity_score`: 1.0 if same track_id, else 0.0
  - [x] `velocity_consistency_score`: predicted vs actual entry position match
  - [x] `color_match_score`: HSV similarity between fragment appearances
  - [x] `jersey_match_score`: same number = high, unknown = neutral, conflict = rejected
  - [x] `temporal_gap_score`: penalty for longer gaps
  - [x] `overall = 0.50*track + 0.20*velocity + 0.15*color + 0.10*jersey + 0.05*temporal`
- [x] Discard candidates with `overall < MIN_CANDIDATE_SCORE` (0.35)
- [x] Pass 3A must NOT assign identity, team, or jersey — evidence edges only

---

## Pass 3B — Constraint Graph

**File**: `src/skills/pass3b_constraint_builder.py` (full rewrite)
**Input**: `pass3a_candidates.json` + `pass2b_scored_fragments.json`
**Output**: `pass3b_constraints.json`

- [x] `MUST_SAME`: `overall_candidate_score ≥ 0.85`
- [x] `CANNOT_SAME`: temporal overlap OR jersey conflict OR spatial impossibility
- [x] `SOFT_SAME`: `0.35 ≤ score < 0.85`
- [x] Before inserting MUST_SAME: verify it does not contradict any existing CANNOT_SAME
  - [x] If contradiction: FAIL FAST, write report, stop
- [x] Invariant: no fragment may MUST_SAME two fragments that CANNOT_SAME each other

---

## Pass 3C — Identity Commit

**File**: `src/skills/pass3c_identity_solver.py` (full rewrite)
**Input**: `pass3b_constraints.json` + `pass2b_scored_fragments.json`
**Output**: `pass3_identity_commit.json`, `pass3_validation.json`, `debug_metrics.json`

8-step algorithm (strict order):

- [x] **Step 1 — Build identity groups**: connected components via MUST_SAME edges
- [x] **Step 2 — Validate hard constraints**: no temporal overlaps, no CANNOT_SAME violations within group → FAIL FAST if any
- [x] **Step 3 — Jersey resolution**: per group, aggregate jersey observations weighted by confidence; conflicting numbers within a group → FAIL FAST
- [x] **Step 4 — Team assignment**: K-means (k=2) on HSV color embeddings; ghosts excluded (`exclude_from_clustering = True`)
  - [x] Lock immediately via `_locked_team` — single source of truth, never re-evaluated
- [x] **Step 5 — Lock identity labels**: `team_id` and `jersey_number` immutable after this point
- [x] **Step 6 — Enforce jersey exclusivity**: no two identities share the same jersey on the same team in the same frame
- [x] **Step 7 — Soft constraint optimisation**: apply SOFT_SAME merges only if ALL hold:
  - [x] No temporal overlap
  - [x] No CANNOT_SAME constraint
  - [x] Jersey numbers equal or both unknown
  - [x] Same `_locked_team`
  - [x] Jersey exclusivity preserved after merge
- [x] **Step 8 — Finalise**: assign `identity_id` (format `P07_team_a`); fragments inherit identity assignment
- [x] Fail conditions (write report, exit non-zero, do NOT write artifacts):
  - [x] Constraint contradictions
  - [x] Identity temporal overlap
  - [x] Team assignment fails
  - [x] Jersey exclusivity fails

---

## Ball Interpolation

**File**: `src/skills/ball_interpolator.py` (create)
**Input**: `pass1_raw.json` (ball_detections)
**Output**: `ball_interpolation.json`

- [ ] Detect gaps in ball detections
- [ ] Gaps ≤ 30 frames: linear interpolation → `state = "interpolated"`
- [ ] Gaps > 30 frames: `state = "out_of_play"` (no position field)
- [ ] Every frame must have a ball state entry (validates R5)
- [ ] Validate output

---

## Visualization

**File**: `src/skills/visualizer.py` (create)
**Input**: `pass3_identity_commit.json` + `ball_interpolation.json` ONLY
**Output**: `visualization.mp4`

- [ ] Bboxes colored by team (`team_a` = blue, `team_b` = red)
- [ ] Label: `<jersey> - <name>` if mapped (4=Spyros, 7=Rick, 10=Kiki); else jersey number; else identity_id
- [ ] Ghosts: dashed bboxes
- [ ] Ball: solid circle (real), dashed circle (interpolated), no circle (out_of_play)
- [ ] Deduplication: only suppress same `track_id` duplicates — different tracks may share screen space
- [ ] Visualizer CANNOT infer, fix, suppress, or merge entities — any inconsistency must be fixed upstream

---

## Data Models

**File**: `src/core/data_models.py`

- [x] `Fragment` — Pass 2A output fields listed above
- [x] `ScoredFragment` — extends Fragment with quality scores
- [x] `GhostFragment` — adds `is_ghost`, `exclude_from_clustering`, `original_track_id`
- [x] `CandidateEdge` — fragment_a_id, fragment_b_id, all score fields, overall_candidate_score
- [x] `Constraint` — type (MUST_SAME | CANNOT_SAME | SOFT_SAME), fragment_a_id, fragment_b_id
- [x] `CommittedIdentity` — identity_id, team_id, jersey_number, fragments[], is_ghost
- [ ] `BallState` — frame_idx, state (real | interpolated | out_of_play), centroid (optional)

---

## Constants

**File**: `src/core/constants.py`

- [x] `JERSEY_CHANGE_PERSISTENCE = 15`
- [x] `TEAM_SWITCH_WINDOW = 30`
- [x] `TEAM_SWITCH_HSV_THRESHOLD` (HSV distance for team switch detection)
- [x] `TEAM_SWITCH_CONFIDENCE` (min confidence per window)
- [x] `TEAM_SWITCH_MIN_SAMPLES` (min valid HSV observations per window)
- [x] `PASS2_MAX_PLAYER_SPEED` (pixels/frame for Pass 2A motion spike detection)
- [x] `MIN_FRAGMENT_LENGTH = 15`
- [x] `MAX_IDENTITY_GAP = 300`
- [x] `TRACK_CONTINUITY_GAP = 120`
- [x] `MIN_CANDIDATE_SCORE = 0.35`
- [x] `MUST_SAME_THRESHOLD = 0.85`
- [x] No fixed Pass 2C ghost duration constant (duration cap removed)

---

## Orchestrator / CLI

**File**: `src/main.py` or `src/orchestrator.py`

- [ ] Execute passes in strict order (1 → 2A → 2B → 2C → 3A → 3B → 3C → Ball → Viz)
- [ ] After each pass: validate output, FAIL FAST on error, do NOT write later artifacts
- [ ] Support `--passes 2a` style flag to run a subset of passes (for iterative development)

---

## Verification

After each pass, test on clip7:

- [x] `pass2_fragments.json`: 20–30 fragments, 100% detection coverage, no overlapping ranges
- [x] `pass2b_scored_fragments.json`: quality tiers (HIGH/MEDIUM/LOW) present on all fragments
- [x] `pass2_ghosts.json`: `is_ghost` flags correct, level tracks high-water mark
- [x] `pass3a_candidates.json`: same-track pairs always have a candidate edge
- [x] `pass3b_constraints.json`: no contradictions, MUST/CANNOT/SOFT edges present
- [x] `pass3_identity_commit.json`: 0 unknown teams, jersey exclusivity holds, correct player count
- [ ] `ball_interpolation.json`: state present at every frame
- [ ] `visualization.mp4`: correct team colours, labels, ghosts, ball tracking visible
