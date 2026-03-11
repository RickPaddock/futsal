# Futsal Tracking System - Implementation Plan & Progress Tracker

**Status**: 🚧 In Progress
**Start Date**: 2026-02-17


#### Pass 2A Validation Contract Delta (from CLAUDE.md Section 2A) ⚠️ REQUIRED
[x] test 2a against ground truth and fix where nessesary ✅ T2 43/43 splits detected (±60f), T1 481/482 (1 miss = Pass 1 ByteTracker timing)

### Pass 2B: Fragment Quality Scoring
[ ] test 2b against ground truth and fix where nessesary. It should be validated for:
[ ] metadata correctness
[ ] quality scoring stability
[ ] runtime cost
[ ] fragment statistics

What to Measure in Pass 2B

Run on clips 2, 7, 11 and produce:
Fragment statistics
total fragments
avg fragment length
median fragment length
min / max
Quality scoring distribution
quality score histogram
jersey_visible_ratio distribution
occlusion_ratio distribution
appearance_stability_score distribution
Sanity checks

Confirm:
[ ] 100% fragments have metadata
[ ] no NaN scores
[ ] scores within expected ranges
[ ] Important Architectural Check Before Moving On

Verify this invariant - Pass 2B MUST NOT:
[ ] split fragments
[ ] merge fragments
[ ] delete fragments
[ ] change fragment_id
It should only attach metadata.


### Pass 2C: Ghost Generation
[ ] test 2c against ground truth and fix where nessesary

---

### Pass 3A: Identity Candidates
- [x] **[src/skills/pass3a_candidate_generator.py](../src/skills/pass3a_candidate_generator.py)** - Generate possibilities ✅
  - [x] For each fragment, generate:
    - [x] Possible teams with evidence scores (from HSV observability metadata)
    - [x] Possible jerseys with probabilities (aggregated from jersey detections/probabilities)
    - [x] Adjacent fragments (same-track continuity links in player evidence)
  - [x] **NO LOCKING** (just candidates, no decisions)
  - [x] Save `pass3_candidates.json`
- [ ] **Pass 3A debug video** (`--video-output 3a`) - Identity candidates
  - [ ] Show: Fragment bboxes with candidate teams/jerseys
  - [ ] Overlays: Evidence scores, probability distributions, adjacency links
  - [ ] Purpose: Verify candidate generation logic (are candidates reasonable?)

### Pass 3B: Constraint Graph
- [x] **[src/skills/pass3b_constraint_builder.py](../src/skills/pass3b_constraint_builder.py)** - Build constraint graph ✅
  - [x] **MUST_SAME constraints**: Track adjacency (same track_id) + ghost continuity
  - [x] **CANNOT_SAME constraints**: Jersey temporal exclusivity violations
  - [x] **SOFT_SAME constraints**: Track continuity preferences (track-gap weighted)
  - [x] Graph structure: fragment_id → constraint_ids
  - [x] **Note**: Team assignment happens in Pass 3C AFTER identity resolution
  - [x] Save `pass3_constraints.json`
- [ ] **Pass 3B debug video** (`--video-output 3b`) - Constraint graph
  - [ ] Show: Fragment bboxes with constraint edges overlaid
  - [ ] Overlays: MUST_SAME (green), CANNOT_SAME (red), SOFT_SAME (yellow)
  - [ ] Purpose: Verify constraint graph structure (are constraints correct?)

### Pass 3C: Identity Commit (Already completed in Priority 3)
✅ See Priority 3

### Pass 3 Runtime Regression (clip9)
- [x] Pass 3 runs end-to-end successfully on clip9 (exit code 0) ✅
- [x] Removed synthetic fallback jersey assignment in Pass 3C ✅
- [x] Overlap hard failures resolved (`PASS3C_PLAYER_OVERLAP_SAME_FRAME`) ✅
- [x] Jersey conflicts from fabricated `#1` removed; unresolved jerseys now explicit warnings (`PASS3C_JERSEY_UNRESOLVED`) ✅

---

## Priority 7: Ball & Visualization (Day 9)

### Ball Interpolation
- [ ] **[src/skills/ball_interpolator.py](../src/skills/ball_interpolator.py)** - Fill ball gaps
  - [ ] Detect gaps in ball detections
  - [ ] Gaps ≤ 30 frames: interpolate (linear or Kalman) → state = `interpolated`
  - [ ] Gaps > 30 frames: mark as "out of play" → state = `out_of_play` (NO position)
  - [ ] Flag interpolated frames
  - [ ] Validate: ball STATE exists at every frame (not position - state ∈ {real, interpolated, out_of_play})
  - [ ] Save `ball_interpolation.json`
  - [ ] Validate output
- [ ] **Ball interpolation debug video** (`--video-output ball`) - Ball tracking
  - [ ] Show: Ball positions (real=solid, interpolated=dashed, out_of_play=none)
  - [ ] Overlays: Ball state labels, gap lengths, interpolation method
  - [ ] Purpose: Verify ball interpolation logic (are gaps filled correctly?)

### Visualization
- [ ] **[src/skills/visualizer.py](../src/skills/visualizer.py)** - Final video output
  - [ ] Render video with:
    - [ ] Bboxes colored by player_id
    - [ ] Team colors (team_a, team_b)
    - [ ] Jersey numbers overlaid
    - [ ] Label rule: if jersey number known, show `<number> - <name>` when mapped (e.g., 4-Spyros, 7-Rick, 10-Kiki)
    - [ ] Label fallback: if jersey known but unmapped, show jersey number; if jersey unknown, show player_id
    - [ ] Ghosts rendered as dashed bboxes
    - [ ] Ball rendering by state:
      - [ ] `real`: solid circle
      - [ ] `interpolated`: dashed circle
      - [ ] `out_of_play`: no circle (ball not on court)
  - [ ] **Deduplication logic**:
    - [ ] ⚠️ CRITICAL: Only suppress if SAME track_id
    - [ ] Different tracks can occupy same space (ghost + occluder)
  - [ ] **Visualizer Isolation (CRITICAL - Prevents Truth Contamination)**:
    - [ ] Visualizer may ONLY consume `CommittedIdentity` + `ball_interpolation.json`
    - [ ] Visualizer CANNOT infer, fix, suppress, or merge entities
    - [ ] Any visual inconsistency MUST be solved upstream (Pass 1-3C)
    - [ ] No state leaks: Visualization logic cannot influence Pass 1-3C decisions
  - [ ] Save `visualization.mp4`

---

## Priority 7.5: Pitch Projection & Homography (Tactical Analysis)

### Homography Transformation
- [ ] **[src/geometry/homography.py](../src/geometry/homography.py)** - Camera to pitch transformation
  - [ ] **Reference implementation**: `https://github.com/RickPaddock/futsal/blob/rick_claude_2pass_mvp1/src/geometry/homography.py`
  - [ ] **Input**: `pass3_identity_commit.json`, `ball_interpolation.json` (camera coordinates)
  - [ ] **Calibration points** (13-point correspondence):
    - [ ] Source: Pixel coordinates from video frame
      - [ ]   court_length: 40.0
      - [ ]   court_width: 20.0
      - [ ]   goal_width: 3.0
      - [ ]   output_pixel_scale: 20  # Pixels per meter for 2D pitch (800x400 output)
      - [ ] Far Left Corner: [680, 595]
      - [ ] Left Red (PA Far): [843, 638] (3.5m penalty area depth)
      - [ ] Left Blue (PA Near): [147, 1075]
      - [ ] Far Right Corner: [3136, 611]
      - [ ] Right Red (PA Far): [2964, 650]
      - [ ] Right Blue (PA Near): [3644, 1100]
      - [ ] Center Black X (Far): [1900, 530]
      - [ ] Center Black X (Near): [1888, 1592]
      - [ ] Left Purple (Goal Far): [303, 772]
      - [ ] Left Orange (Goal Near): [128, 875]
      - [ ] Right Purple (Goal Far): [3511, 808]
      - [ ] Right Orange (Goal Near): [3692, 907]
      - [ ] Center Circle: [1891, 760]
    - [ ] Destination: Real-world pitch coordinates (meters)
      - [ ] Pitch dimensions: 40m x 20m (standard futsal)
      - [ ] Penalty area depth: 3.5m
      - [ ] Goal width: 3m (posts at y=8.5m and y=11.5m)
  - [ ] Compute homography matrix using cv2.findHomography() or DLT algorithm
  - [ ] Project all player centroid positions to pitch coordinates (x, y in meters)
  - [ ] Project all ball positions to pitch coordinates
  - [ ] Handle edge cases (players off-court, out of bounds)
  - [ ] Save `pitch_projection.json`
  - [ ] Validate output (positions within pitch bounds)

### Tactical Analysis Output
- [ ] **[src/skills/tactical_analyzer.py](../src/skills/tactical_analyzer.py)** - Compute tactical metrics
  - [ ] **Input**: `pitch_projection.json`, `pass3_identity_commit.json`
  - [ ] **Metrics to compute**:
    - [ ] Team centroids (average position per team)
    - [ ] Team spread (compactness in pitch space)
    - [ ] Player heat maps (time spent in each pitch zone)
    - [ ] Formation detection (4-0, 3-1, 2-2, etc.)
    - [ ] Player distances (pairwise distances in meters)
    - [ ] Offside positions (relative to ball and defenders)
    - [ ] Pass opportunities (player-to-player distances < threshold)
  - [ ] Save `tactical_metrics.json`

### Pitch Visualization
- [ ] **[src/skills/pitch_visualizer.py](../src/skills/pitch_visualizer.py)** - 2D overhead pitch view
  - [ ] **Input**: `pitch_projection.json`, `pass3_identity_commit.json`
  - [ ] Render 2D top-down pitch (40m x 20m)
    - [ ] Draw pitch lines (touchlines, goal lines, penalty areas, center circle)
    - [ ] Draw goal posts
    - [ ] Draw penalty spots
  - [ ] Render players as colored dots (team_a=blue, team_b=red)
  - [ ] Overlay jersey numbers on player dots
  - [ ] Render ball position (solid=real, dashed=interpolated)
  - [ ] Optional: Movement trails (last N seconds)
  - [ ] Optional: Formation lines connecting players
  - [ ] Save `pitch_visualization.mp4` (side-by-side with camera view, or separate)
- [ ] **Pitch debug video** (`--video-output pitch`) - Top-down tactical view
  - [ ] Show: 2D pitch with player/ball positions in real-world coordinates
  - [ ] Overlays: Team formations, player spacing, tactical metrics
  - [ ] Purpose: Verify homography transformation, tactical analysis

---

## Priority 8: Orchestration (Day 10)

### Pipeline Orchestrator
- [ ] **[src/orchestrator.py](../src/orchestrator.py)** - Pipeline execution
  - [ ] Execute passes in strict order:
    1. [ ] Pass 1: Raw Evidence
    2. [ ] Pass 2A: Mechanical Fragmentation
    3. [ ] Pass 2B: Fragment Quality Scoring
    4. [ ] Pass 2C: Ghost Generation
    5. [ ] Pass 3A: Identity Candidates
    6. [ ] Pass 3B: Constraint Graph
    7. [ ] Pass 3C: Identity Commit (LOCK POINT)
    8. [ ] Ball Interpolation
    9. [ ] Pitch Projection (Homography)
    10. [ ] Tactical Analysis
    11. [ ] Visualization (Camera View)
    12. [ ] Pitch Visualization (2D Top-Down View)
  - [ ] After each pass:
    - [ ] Load previous pass output
    - [ ] Run skill
    - [ ] Validate output
    - [ ] **FAIL-FAST** on validation error (do not write later artifacts)
    - [ ] Save JSON
  - [ ] Save `debug_metrics.json` (frame-by-frame debug info)
  - [ ] Return final output directory

### CLI Entry Point
- [ ] **[src/main.py](../src/main.py)** - Command-line interface
  - [ ] Parse arguments:
    - [ ] `--input <video_path>` - Single video
    - [ ] `--input-dir <folder>` - Batch process folder
    - [ ] `--output-dir <path>` - Override output location (default: videos/output/)
  - [ ] Create output directory: `videos/output/<clip_name>/`
  - [ ] Call orchestrator.run(video_path, output_dir)
  - [ ] Handle errors, log results
  - [ ] Display summary statistics

---

## Priority 9: Testing & Verification

### Regression Test
- [ ] **[tests/test_regression.py](../tests/test_regression.py)** - Permanent regression test
  - [ ] Test clip: `videos/input/GoPro_Futsal_part1_CLEANED_clip9.mp4`
  - [ ] Expected outcomes:
    - [ ] No huge bboxes (Track 14-style hallucinations filtered at Pass 1)
    - [ ] Zero `team = "unknown"` after Pass 3C
    - [ ] Zero jersey temporal conflicts
    - [ ] At most 12 concurrent players per frame
    - [ ] Ball present at every frame (real or interpolated)
    - [ ] InferenceSlicer-enabled ball detection improves or preserves recall vs full-frame baseline
  - [ ] Assert validation.passed == True for all passes
  - [ ] Assert specific metrics (from memory learnings)

### Unit Tests
- [ ] **[tests/core/test_data_models.py](../tests/core/test_data_models.py)** - Model validation
- [ ] **[tests/validation/test_global_rules.py](../tests/validation/test_global_rules.py)** - Rule enforcement
- [ ] **[tests/skills/test_pass1_extractor.py](../tests/skills/test_pass1_extractor.py)** - Mock inputs
- [ ] **[tests/skills/test_pass2a_fragmenter.py](../tests/skills/test_pass2a_fragmenter.py)** - Split logic
- [ ] **[tests/skills/test_pass2c_ghost_generator.py](../tests/skills/test_pass2c_ghost_generator.py)** - Ghost creation
- [ ] **[tests/skills/test_pass3c_identity_solver.py](../tests/skills/test_pass3c_identity_solver.py)** - Constraint satisfaction
- [ ] **[tests/detectors/test_ball_detector.py](../tests/detectors/test_ball_detector.py)** - Slicer vs full-frame behavior
  - [ ] Verifies slicer output format matches Pass 1 expectations
  - [ ] Verifies no duplicate/conflicting ball detections after overlap filtering

### End-to-End Test
- [ ] Run full pipeline on multiple clips
- [ ] Verify all 9 JSON artifacts created
- [ ] Verify all validation reports show `passed: true`
- [ ] Verify visualization quality:
  - [ ] Players colored by team
  - [ ] Jersey numbers visible
  - [ ] Name labels shown for mapped jerseys (4=Spyros, 7=Rick, 10=Kiki)
  - [ ] Ghosts shown as dashed during occlusions
  - [ ] Ball tracked throughout (solid/dashed)
  - [ ] No "unknown" grey boxes

---

## Success Criteria

- [ ] All 30 implementation files created and tested
- [ ] Pipeline runs without errors
- [ ] All validation reports pass (passed: true)
- [ ] Ball detector supports InferenceSlicer mode for small-object recall
- [ ] Visualization shows:
  - [ ] Correct team colors
  - [ ] Jersey numbers
  - [ ] Jersey-name labels for known mappings (4/7/10)
  - [ ] Ghosts during occlusions
  - [ ] Ball tracking (real + interpolated)
  - [ ] Zero "unknown" fragments
- [ ] Debug metrics show:
  - [ ] 0 unknown teams
  - [ ] 0 jersey temporal conflicts
  - [ ] Max 12 concurrent players
  - [ ] 100% frame coverage

