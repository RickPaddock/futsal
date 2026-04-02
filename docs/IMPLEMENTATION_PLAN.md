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

### Projection Stability

- [x] Keep the raw image-derived player anchor available for audit even if the rendered 2D dot becomes stabilized
- [x] Add a stabilized court-space player position used for 2D display and downstream court-space analytics
- [x] Apply smoothing in court space, not image space, so homography noise and bbox jitter are handled after projection
- [x] Maintain stabilization state per committed player identity, never per transient track id alone
- [x] Clamp implausible per-frame court displacement so stationary or lightly occluded players do not wobble on the 2D pitch
- [x] Reduce measurement trust when bbox evidence becomes unstable due to partial occlusion, abrupt width or area shrink, or asymmetric bbox shifts
- [x] Preserve a clear raw-vs-stabilized audit trail so the debug render can show why the 2D dot moved or stayed stable

#### Stabilization Hierarchy

Apply these in order. Only move to the next level if the previous level is still not good enough on manual review of the bird's-eye debug render.

Current runtime status: the active Pass 4 2D path uses pre-homography foot-point stabilization with raw-vs-stabilized audit fields preserved in the artifact and debug render.

1. [x] Level 1: court-space smoothing on top of the current bbox-derived anchor
  - keep raw image anchors and raw projected positions for audit
  - smooth per committed player identity in court space
  - use trust heuristics and a plausible speed cap to suppress bbox jitter and partial-occlusion wobble
2. [x] Level 2: replace the bbox-derived anchor with a more stable ground-contact estimate using existing detections
  - derive a better floor-contact point from the lower body / lower bbox region, mask geometry, or another geometry-only estimate that does not require a new model
  - keep the same court-space stabilizer after the improved measurement source is introduced
3. [x] Level 3: add a footpoint / pose-keypoint measurement source
  - run a pose or keypoint model and use stable body cues such as head/shoulders/hips to infer the footpoint, falling back to direct ankles or feet only when they are clearly available
  - keep raw and stabilized outputs both auditable in the debug render
4. [ ] Level 4: only if the above still fails, escalate upstream measurement quality
  - consider segmentation-assisted footpoint recovery or stronger detector changes rather than stacking more smoothing onto a noisy measurement

#### Trajectory Segment Smoothing

- [x] Capture the current measurement path in audit-friendly form using raw and stabilized projected player positions
- [x] Add a second-stage court-space trajectory smoother that operates per committed player identity after the current height-based foot estimate and local stabilizer
- [x] Segment each player trajectory into stationary, coherent, transition, and reactive spans using existing bird's-eye JSON signals (`frame_idx`, `court_position`, `raw_court_position`, `stabilization_trust`, `is_ghost`, `is_estimated`)
- [x] End a smoothing span on frame gaps, trust cliffs, sharp direction changes, transitions into or out of near-stationary motion, and ghost or occlusion discontinuities
- [x] Fit straight-line motion first for coherent spans and only allow lightly curved fitting later if linear segments still look too rigid on visual review
- [x] Keep stationary spans near a held or median position instead of fitting motion through them
- [x] Preserve short reactive actions such as pivots, blocks, and abrupt stops by bypassing trajectory fitting on those spans
- [x] Blend any fitted path back toward the current measurement rather than replacing it outright
- [x] Enforce a hard maximum drift from the current measurement so the rendered 2D dot cannot become smooth but materially wrong
- [x] Add bird's-eye diagnostics for segment counts, segment types, frames modified, mean fitted-vs-measured delta, and max fitted-vs-measured delta
- [ ] Persist extra artifact fields only if diagnostics alone are insufficient for debugging trajectory smoothing decisions
- [ ] Rerender clip2 and manually verify that long A-to-B runs look calmer while pivots and defensive side-steps stay physically plausible

### 2D Pitch Render

- [ ] Render a top-down futsal pitch view suitable for both a debug video and a video inset
- [ ] Player markers must be circles colored by team
- [ ] If a player has a resolved jersey number, that number must be drawn inside the circle
- [ ] In our current named cases, jersey display must support 4 = Spyros, 7 = Rick, 10 = Kiki
- [ ] Ball marker must be shown on the projected pitch when ball state is `real` or `interpolated`
- [ ] Unknown ball frames must show no committed ball marker
- [x] Debug render must also show the source-frame player bboxes and the exact player anchor point used to generate the 2D projection
- [x] Debug render must show a real-ball bbox when available and an interpolated ball marker when the ball state is interpolated
- [x] Debug overlay palette should remain visually distinct and consistent with team identity; current bird's-eye debug convention is black for one team and orange for the other
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

- [x] Bboxes colored by team using the current debug convention (`team_a` = black, `team_b` = orange) unless a later visual design change is made deliberately
- [x] Label: `<jersey> - <name>` if mapped (4=Spyros, 7=Rick, 10=Kiki); else jersey number; else identity_id
- [x] Ghosts: dashed bboxes
- [x] Ball: solid circle (real), dashed circle (interpolated), no circle (unknown)
- [x] Top-right corner inset must show the 2D bird's-eye pitch projection
- [x] 2D inset must show team-colored player circles, resolved jersey numbers inside circles, and projected ball position
- [x] When stabilization is added, visualization must keep raw image overlays visible so bbox jitter can be compared against the stabilized 2D motion
- [x] Deduplication: only suppress same `track_id` duplicates — different tracks may share screen space
- [x] Visualizer CANNOT infer, fix, suppress, or merge entities — any inconsistency must be fixed upstream

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
Analytics should follow a deterministic two-step structure inspired by tracking-data event literature:
first assign frame-level possession, then derive passes and shots from possession changes plus futsal-specific rules.
Any ideas borrowed from 11-a-side football analytics must be adapted for futsal before use.

**Files**: `src/skills/analytics_possession.py`, `src/skills/analytics_passes.py`, `src/skills/analytics_distance.py`, `src/skills/analytics_shots.py` (new)
**Input**: `pass1_raw.json` + `pass2_ghosts.json` + `pass3_identity_commit.json` + `ball_interpolation.json` + `birdseye_projection.json`
**Output**: `analytics_possession.json`, `analytics_events.json`, `analytics_summary.json`, `player_distance_summary.json`

### Scope

- [ ] Track which identified player is in possession of the ball
- [ ] Track player-to-player passes
- [ ] Mark passes as successful or unsuccessful
- [ ] Track shot attempts and identify the shooter
- [ ] Track distance run for identified players
- [ ] Produce named summaries for jersey 4 = Spyros, 7 = Rick, 10 = Kiki
- [ ] Do NOT build goalkeeper-specific analytics; in futsal the rotating goalkeeper is treated as a normal committed player identity
- [ ] Do NOT import 11-a-side set-piece or restart logic directly; out-of-play and restart classification is deferred until futsal-specific rules are defined

### Possession Model

- [ ] Possession MUST be derived from `ball_interpolation.json` + committed player identities, not from raw track IDs
- [ ] Analytics must expose an explicit frame-indexed possession artifact in `analytics_possession.json`
- [ ] Internal possession reasoning may use any committed identity as context, but reported player analytics must be restricted to jersey-resolved identities
- [ ] A player may only be considered in possession when:
  - [ ] ball state is `real` or `interpolated`
  - [ ] player identity is committed in `pass3_identity_commit.json`
  - [ ] player is the closest plausible controller of the ball using court-space distance when available, with image-space fallback only when projection is unavailable
  - [ ] control persists for a minimum confirmation window before a handoff is committed
- [ ] Initial handoff confirmation window must be explicit and conservative (default target: 3 consecutive frames)
- [ ] Possession output must include a confidence score and an ambiguity state rather than forcing a binary owner in congested frames
- [ ] If multiple players are similarly plausible controllers, the frame must be marked ambiguous and possession must abstain
- [ ] Possession output must be frame-indexed and include confidence
- [x] Add an absolute control gate so extremely weak near-threshold candidates do not become confirmed possession just because they persist for 3 frames
- [x] Add a contested-control rule so close duels are allowed to stay ambiguous even when the current top-vs-second confidence margin is not small enough on its own
- [ ] Ghost-only frames must NEVER create confirmed possession
- [ ] No special goalkeeper possession rules are required; the rotating goalkeeper follows the same committed-identity possession rules as any other player

### Pass Detection

- [ ] A pass starts when a player in confirmed possession releases the ball and loses control
- [ ] Passer = last confirmed controlling player before release
- [ ] Receiver = next confirmed controlling player after release, if any
- [ ] Pass detection MUST consume `analytics_possession.json` rather than re-deriving possession ad hoc
- [ ] Initial implementation should populate `analytics_events.json` with pass events first; shot events can be added into the same artifact later
- [ ] Receive window must be explicit and bounded in time (initial target: about 2 seconds, parameterized by FPS)
- [ ] If the same player quickly regains confirmed control after release, treat it as recovery / failed control rather than a completed pass
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
  - [ ] `receiver_team_id` (nullable, required for unresolved-player audit display)
  - [ ] `team_id`
  - [ ] `outcome = successful | unsuccessful | loose_ball`
  - [ ] `event_confidence`
- [ ] Keep unresolved-player pass events in `analytics_events.json` for audit, but mark them as audit-only and label them as unknown player on video overlays
- [ ] Final player-attributed pass summaries must filter out those audit-only unresolved-player events
- [ ] Add an interception / deflection / instant-loss review path so the first downstream controller is not always treated as clean received possession when control was only momentary or chaotic

### Shot Detection

- [ ] A shot attempt starts when a player in confirmed possession releases the ball with strong outbound motion
- [ ] Shooter = last confirmed controlling player before shot release
- [ ] Shot detection MUST consume `analytics_possession.json` rather than re-deriving possession ad hoc
- [ ] Initial shot classification should prefer projected court-space ball velocity from `birdseye_projection.json` over raw image motion when calibration is available
- [ ] A first-pass implementation may use release speed only; goalward-direction and outcome classification can come later
- [ ] Initial implementation only needs `shot_attempt`
- [ ] Goal / on-target / off-target classification can come later
- [ ] Do NOT produce goalkeeper-specific save or keeper-distribution stats in the initial futsal analytics scope
- [ ] Shot event fields:
  - [ ] `event_id`
  - [ ] `event_type = "shot"`
  - [ ] `start_frame`, `end_frame`
  - [ ] `shooter_player_id`, `shooter_jersey_number`
  - [ ] `team_id`
  - [ ] `outcome = shot_attempt`
  - [ ] `event_confidence`
- [ ] Only emit player-attributed shot analytics when the shooter has a resolved jersey number; otherwise abstain from reported shooter attribution

### Distance Run

- [x] Distance must be computed from committed player identity positions, not raw track IDs
- [x] Prefer court-space distance from `birdseye_projection.json` when calibration / homography is available
- [ ] If calibration is unavailable, report image-plane distance and explicitly mark units as `px`
- [x] Distance must be accumulated from real observed player positions only
- [x] Ghost-only spans must NOT contribute to confirmed distance run
- [x] Rotating-goalkeeper minutes are treated the same as any other committed-player minutes; do NOT split distance into separate goalkeeper buckets
- [ ] Distance summary fields:
  - [x] `player_id`
  - [x] `jersey_number`
  - [x] `team_id`
  - [x] `display_name` (Spyros / Rick / Kiki when mapped)
  - [x] `distance_value`
  - [x] `distance_unit = m | px`
  - [x] `observed_frame_count`
  - [x] `estimated_frame_count`
  - [x] `distance_confidence`
- [x] Do NOT emit player distance summaries for identities without resolved jersey numbers

### Named Player Reporting

- [ ] Report dedicated summaries for:
  - [x] jersey 4 = Spyros
  - [x] jersey 7 = Rick
  - [x] jersey 10 = Kiki
- [x] If a named jersey is not resolved in a clip, report that explicitly instead of fabricating analytics
- [x] General player summaries must follow the same rule: only jersey-resolved players appear in reported analytics outputs

### Validation

- [ ] Every analytics event must reference a valid committed identity
- [ ] Every reported player-attributed analytics event must reference a valid committed identity with a resolved jersey number
- [ ] A successful pass must never cross teams
- [ ] Distance output must always include explicit units
- [ ] Analytics must never modify or reinterpret upstream artifacts
- [ ] If possession is not stable enough, analytics should abstain rather than invent events
- [ ] Pass and shot detection must consume the committed possession artifact rather than bypassing it with ad hoc nearest-player logic

### Analytics Visualization Consideration

- [ ] Consider `mplsoccer` as an optional post-pipeline reporting layer for static analytics visuals built from committed artifacts
- [ ] Do NOT use `mplsoccer` for frame-by-frame video rendering; keep the current OpenCV-based bird's-eye and visualization video pipeline for performance
- [ ] Restrict `mplsoccer` usage to descriptive outputs such as pass maps, touch maps, heatmaps, shot maps, team shape snapshots, and summary report figures
- [ ] Any `mplsoccer` output must consume upstream artifacts only and MUST NOT modify, repair, reinterpret, or override identity, ball, possession, or projection artifacts
- [ ] Because this repo targets futsal, any `mplsoccer` pitch usage must be validated against custom futsal court dimensions rather than assuming a standard 11-a-side pitch
- [ ] Analytics review should use a single unified `analytics_debug.mp4` overlay built from `analytics_possession.json`, `analytics_events.json`, and `birdseye_projection.json`
- [ ] The top-left analytics block should always show at least control, pass, and shot lines, with additional context lines when useful
- [ ] Audit text should render as bright yellow on black and sit low enough to avoid editor/video-software top banners

### Verification

- [ ] `analytics_possession.json`: frame-indexed possession exists, ambiguous frames abstain cleanly, and ghost-only frames never create confirmed possession
- [ ] `analytics_events.json`: all audit pass events are retained, while unresolved-player events are clearly marked as audit-only
- [ ] `analytics_debug.mp4`: control, pass, and shot lines are always visible; active pass events are visually auditable on video, including unknown-player events with team-only labels
- [x] `player_distance_summary.json`: only jersey-resolved players are reported, with named players present when jerseys resolve
- [ ] Successful passes always stay within team
- [ ] Distances clearly report `m` or `px`
- [ ] Analytics output is absent or abstains when upstream certainty is insufficient