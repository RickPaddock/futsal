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
- [ ] Fix ghosts - they are still a bit floaty (see clip 2 around frame 1050) - leave until after analytics section is complete
---

## Ball Interpolation

**File**: `src/skills/ball_interpolator.py` (create)
**Input**: `pass1_raw.json` (ball_detections)
**Output**: `ball_interpolation.json`, `ball_interpolation_debug.mp4`

- [x] Detect gaps in ball detections
- [x] Collapse duplicate same-frame ball detections to the highest-confidence detection before interpolation
- [x] Gaps ≤ 30 frames: linear interpolation → `state = "interpolated"`
- [x] Gaps > 30 frames: `state = "unknown"` (no trusted position available)
- [x] Frames before first real detection and after last real detection: `state = "unknown"`
- [x] Every frame must have a ball state entry (validates R5)
- [x] Validate output
- [x] Normalize legacy `out_of_play` artifacts to `unknown` during model/schema validation
- [x] Suppress short false ball branches when they jump away from a supported trajectory and then jump back
- [x] Suppress detached low-confidence islands even when average edge speed alone would otherwise let them survive
- [x] Preserve locally supported true streaks while rejecting one-sided foot/shoe false positives

### Ball Debug Video

- [x] Render a dedicated ball interpolation debug video for visual inspection
- [x] Real detection frame: draw real bbox + solid centroid marker + `state=real`
- [x] Interpolated frame: draw no bbox, draw distinct interpolated centroid marker + `state=interpolated`
- [x] Unknown frame: draw no ball marker and display `state=unknown`
- [x] Overlay gap context: previous real frame, next real frame, gap length, fill mode
- [x] Show rejected raw detections in red for debugging false-capture filtering
- [x] Debug video is diagnostic only and MUST NOT infer, alter, suppress, or repair ball states

### Current Diagnostics

- [x] `src/skills/pass3_debug_visualizer.py` now loads `ball_interpolation.json` and overlays ball state in `pass3_debug.mp4`
- [x] `scripts/audit_ball_candidates.py` can rerun the ball model over a frame window and save all raw candidates for detector debugging
- [x] Audit video supports render-only reruns, separate render threshold, and capped candidate count to keep inspection usable

---

## 2D Bird's-Eye Pitch Projection

**File**: `src/skills/birds_eye_pitch.py` (create)
**Input**: `pass3_identity_commit.json` + `ball_interpolation.json` + existing clicked pitch calibration / homography inputs
**Output**: `birdseye_projection.json`, `birdseye_debug.mp4`

- [ ] Project committed player positions into court space using the existing clicked pitch calibration / fisheye-correction workflow
- [ ] Project ball positions into the same court space using the same homography
- [ ] Reuse the current clicked pitch setup as-is and leave `utils/click_points.py` unchanged
- [ ] If calibration is unavailable, fail this stage cleanly and report diagnostics instead of inventing court coordinates
- [ ] Every frame must include projected player positions for committed identities when projection is available
- [ ] Every frame must include projected ball position when ball state is `real` or `interpolated`
- [ ] Ghost-only player spans may be shown as estimated, but must be flagged as estimated and excluded from confirmed metric analytics

### 2D Pitch Render

- [ ] Render a top-down futsal pitch view suitable for both a debug video and a video inset
- [ ] Player markers must be circles colored by team
- [ ] If a player has a resolved jersey number, that number must be drawn inside the circle
- [ ] In our current named cases, jersey display must support 4 = Spyros, 7 = Rick, 10 = Kiki
- [ ] Ball marker must be shown on the projected pitch when ball state is `real` or `interpolated`
- [ ] Unknown ball frames must show no committed ball marker
- [ ] Projected pitch render is descriptive only and MUST NOT repair identity or ball artifacts

### Validation

- [ ] Projection must use committed identities only, never raw track IDs directly
- [ ] Team colors on the 2D pitch must match committed team assignment
- [ ] Jersey text inside player circles must only be shown when jersey identity is resolved
- [ ] Bird's-eye output must never alter upstream identity or ball states

---

## Visualization

**File**: `src/skills/visualizer.py` (create)
**Input**: `pass3_identity_commit.json` + `ball_interpolation.json` + `birdseye_projection.json`
**Output**: `visualization.mp4`

- [ ] Bboxes colored by team (`team_a` = blue, `team_b` = red)
- [ ] Label: `<jersey> - <name>` if mapped (4=Spyros, 7=Rick, 10=Kiki); else jersey number; else identity_id
- [ ] Ghosts: dashed bboxes
- [ ] Ball: solid circle (real), dashed circle (interpolated), no circle (unknown)
- [ ] Top-right corner inset must show the 2D bird's-eye pitch projection
- [ ] 2D inset must show team-colored player circles, resolved jersey numbers inside circles, and projected ball position
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
- [x] `BallState` / `BallPosition` — `frame_idx`, `state` (real | interpolated | unknown), `centroid` (optional), `bbox` (real only), `confidence`

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

- [ ] Execute passes in strict order (1 → 2A → 2B → 2C → 3A → 3B → 3C → Ball → BirdsEye → Viz → Analytics)
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
- [x] `ball_interpolation.json`: state present at every frame, short gaps interpolated, long gaps unknown
- [x] `ball_interpolation_debug.mp4`: real vs interpolated vs unknown visually distinguishable
- [ ] `birdseye_projection.json`: projected player + ball positions present when calibration is available
- [ ] `birdseye_debug.mp4`: top-down pitch shows team-colored player circles, jersey numbers when resolved, and projected ball
- [ ] `visualization.mp4`: correct team colours, labels, ghosts, ball tracking visible


---

## Analytics

**Purpose**: downstream match analytics built on stable player identity, jersey identity, and ball tracking.
This section exists only because the upstream goal is to make these analytics reliable.
Analytics MUST consume upstream artifacts only and MUST NOT alter, repair, or override identity or ball outputs.

**Files**: `src/skills/analytics_possession.py`, `src/skills/analytics_passes.py`, `src/skills/analytics_distance.py`, `src/skills/analytics_shots.py` (new)
**Input**: `pass1_raw.json` + `pass2_ghosts.json` + `pass3_identity_commit.json` + `ball_interpolation.json` + `birdseye_projection.json`
**Output**: `analytics_events.json`, `analytics_summary.json`, `player_distance_summary.json`

### Scope

- [ ] Track which identified player is in possession of the ball
- [ ] Track player-to-player passes
- [ ] Mark passes as successful or unsuccessful
- [ ] Track shot attempts and identify the shooter
- [ ] Track distance run for identified players
- [ ] Produce named summaries for jersey 4 = Spyros, 7 = Rick, 10 = Kiki

### Possession Model

- [ ] Possession MUST be derived from `ball_interpolation.json` + committed player identities, not from raw track IDs
- [ ] A player may only be considered in possession when:
  - [ ] ball state is `real` or `interpolated`
  - [ ] player identity is committed in `pass3_identity_commit.json`
  - [ ] player is the closest plausible controller of the ball
  - [ ] control persists for a minimum confirmation window
- [ ] Possession output must be frame-indexed and include confidence
- [ ] Ghost-only frames must NEVER create confirmed possession

### Pass Detection

- [ ] A pass starts when a player in confirmed possession releases the ball and loses control
- [ ] Passer = last confirmed controlling player before release
- [ ] Receiver = next confirmed controlling player after release, if any
- [ ] Successful pass:
  - [ ] receiver exists
  - [ ] receiver is on the same team as passer
  - [ ] receiver control is confirmed within a bounded receive window
- [ ] Unsuccessful pass:
  - [ ] opponent receives the ball, OR
  - [ ] ball becomes unknown for too long, OR
  - [ ] no receiver is confirmed inside the receive window
- [ ] Pass event fields:
  - [ ] `event_id`
  - [ ] `event_type = "pass"`
  - [ ] `start_frame`, `end_frame`
  - [ ] `passer_player_id`, `passer_jersey_number`
  - [ ] `receiver_player_id`, `receiver_jersey_number` (nullable)
  - [ ] `team_id`
  - [ ] `outcome = successful | unsuccessful | loose_ball`
  - [ ] `event_confidence`

### Shot Detection

- [ ] A shot attempt starts when a player in confirmed possession releases the ball with strong outbound motion
- [ ] Shooter = last confirmed controlling player before shot release
- [ ] Initial implementation only needs `shot_attempt`
- [ ] Goal / on-target / off-target classification can come later
- [ ] Shot event fields:
  - [ ] `event_id`
  - [ ] `event_type = "shot"`
  - [ ] `start_frame`, `end_frame`
  - [ ] `shooter_player_id`, `shooter_jersey_number`
  - [ ] `team_id`
  - [ ] `outcome = shot_attempt`
  - [ ] `event_confidence`

### Distance Run

- [ ] Distance must be computed from committed player identity positions, not raw track IDs
- [ ] Prefer court-space distance from `birdseye_projection.json` when calibration / homography is available
- [ ] If calibration is unavailable, report image-plane distance and explicitly mark units as `px`
- [ ] Distance must be accumulated from real observed player positions only
- [ ] Ghost-only spans must NOT contribute to confirmed distance run
- [ ] Distance summary fields:
  - [ ] `player_id`
  - [ ] `jersey_number`
  - [ ] `team_id`
  - [ ] `display_name` (Spyros / Rick / Kiki when mapped)
  - [ ] `distance_value`
  - [ ] `distance_unit = m | px`
  - [ ] `observed_frame_count`
  - [ ] `estimated_frame_count`
  - [ ] `distance_confidence`

### Named Player Reporting

- [ ] Report dedicated summaries for:
  - [ ] jersey 4 = Spyros
  - [ ] jersey 7 = Rick
  - [ ] jersey 10 = Kiki
- [ ] If a named jersey is not resolved in a clip, report that explicitly instead of fabricating analytics

### Validation

- [ ] Every analytics event must reference a valid committed identity
- [ ] A successful pass must never cross teams
- [ ] Distance output must always include explicit units
- [ ] Analytics must never modify or reinterpret upstream artifacts
- [ ] If possession is not stable enough, analytics should abstain rather than invent events

### Verification

- [ ] `analytics_events.json`: pass and shot events reference committed identities only
- [ ] `player_distance_summary.json`: named players present when jerseys resolve
- [ ] Successful passes always stay within team
- [ ] Distances clearly report `m` or `px`
- [ ] Analytics output is absent or abstains when upstream certainty is insufficient