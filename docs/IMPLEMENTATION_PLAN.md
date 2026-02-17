# Futsal Tracking System - Implementation Plan & Progress Tracker

**Status**: 🚧 In Progress
**Start Date**: 2026-02-17
**Target Completion**: ~10 days

---

## Overview

Complete rebuild of the multi-pass futsal tracking system following strict contract-based architecture:
- Root-cause fixes only (no downstream patching)
- Pass immutability (Pass N cannot modify Pass N-1)
- Single identity commit point (Pass 3C lock)
- Fail-fast validation (execution halts on any violation)
- JSON is source of truth (auditable without video)

**Contract Document**: [CLAUDE.md](../CLAUDE.md)
**Full Plan Details**: [gentle-squishing-milner.md](../.claude/plans/gentle-squishing-milner.md)

---

## Priority 0: Master Contract (CRITICAL - DO THIS FIRST)

- [x] **Create [CLAUDE.md](../CLAUDE.md)** - Master implementation contract ✅
  - [x] Non-negotiable rules (R1-R5)
  - [x] Pass architecture and boundaries
  - [x] Entity model definitions (track_id, fragment_id, player_id, detection_id)
  - [x] Output directory contract
  - [x] Validation requirements
  - [x] Fail-fast policy

---

## Priority 1: Foundation & Core (Days 1-2)

### Core Data Structures
- [x] **[src/core/constants.py](../src/core/constants.py)** - All configuration values ✅
  - [x] Detection thresholds (PLAYER_CONF=0.5, JERSEY_CONF=0.3, BALL_CONF=0.3)
  - [x] Bbox filtering (MAX_HEIGHT=800px, MAX_WIDTH=600px, MAX_AREA_FRACTION=0.25)
  - [x] Fragment parameters (MIN_LENGTH=10, MAX_GAP=60, MERGE_CONSECUTIVE=True)
  - [x] HSV clustering (KMEANS_N_CLUSTERS=2, HSV_BINS=8)
  - [x] Ghost parameters (INITIAL_LEVEL_FRAMES=10, DYNAMIC_LEVEL_MAX=12)
  - [x] Ball interpolation (MAX_BALL_GAP_FRAMES=30)

- [x] **[src/core/types.py](../src/core/types.py)** - Enums and type aliases ✅
  - [x] `TeamID` enum (TEAM_A, TEAM_B, UNKNOWN)
  - [x] `FragmentQuality` enum (HIGH, MEDIUM, LOW, GHOST)
  - [x] `ConstraintType` enum (MUST_SAME, CANNOT_SAME, SOFT_SAME)

- [x] **[src/core/data_models.py](../src/core/data_models.py)** - Pydantic models ✅
  - [x] `Detection` (Pass 1 output)
  - [x] `Fragment` (Pass 2A output)
  - [x] `ScoredFragment` (Pass 2B output)
  - [x] `GhostFragment` (Pass 2C output)
  - [x] `IdentityCandidate` (Pass 3A output)
  - [x] `Constraint` (Pass 3B output)
  - [x] `CommittedIdentity` (Pass 3C output - LOCK POINT)
  - [x] Pass output models (Pass1Output, Pass2AOutput, etc.)

- [x] **[src/core/schemas.py](../src/core/schemas.py)** - JSON schema definitions ✅
  - [x] Schema definitions for all output artifacts
  - [x] Validation schemas for each pass

### Utility Functions
- [x] **[src/utils/geometry.py](../src/utils/geometry.py)** - Geometric operations ✅
  - [x] `iou(bbox1, bbox2)` - Intersection over Union
  - [x] `bbox_centroid(bbox)` - Calculate centroid
  - [x] `bbox_area(bbox)` - Calculate area
  - [x] `centroid_distance(c1, c2)` - Distance between centroids
  - [x] `filter_huge_bboxes()` - Multi-layer bbox defense
  - [x] `bbox_intersection()`, `bbox_union()` - Bbox operations
  - [x] `bbox_width()`, `bbox_height()`, `bbox_aspect_ratio()` - Bbox metrics
  - [x] `bbox_is_valid()`, `clip_bbox_to_frame()` - Bbox validation

- [x] **[src/utils/hsv_color.py](../src/utils/hsv_color.py)** - Color histogram extraction ✅
  - [x] `extract_hsv_histogram(frame, bbox)` - 8x8x8 bins, normalized
  - [x] `compare_hsv_histograms(hist1, hist2)` - Correlation metric
  - [x] `histogram_distance()` - Multiple distance metrics
  - [x] `is_histogram_valid()` - Validation helper
  - [x] `batch_extract_hsv_histograms()` - Batch extraction
  - [x] `average_histograms()`, `histogram_std()` - Aggregation helpers

- [x] **[src/utils/file_utils.py](../src/utils/file_utils.py)** - JSON I/O ✅
  - [x] `save_json(path, data, schema)` - Atomic save with validation
  - [x] `load_json(path, schema)` - Load with schema validation
  - [x] `get_output_dir(video_name)` - Create output folder structure
  - [x] `get_artifact_path()`, `list_artifacts()` - Artifact management
  - [x] `validate_artifact()` - Schema validation
  - [x] `pretty_print_json()`, `compact_json()` - JSON formatting

- [x] **[src/utils/logging_utils.py](../src/utils/logging_utils.py)** - Structured logging ✅
  - [x] `StructuredLogger` class - JSON-formatted logging
  - [x] Pass-level logging helpers (`log_split`, `log_constraint`, etc.)
  - [x] `get_logger()` - Factory function for pass loggers
  - [x] Pipeline summary logging

- [x] **[src/utils/video_io.py](../src/utils/video_io.py)** - Video I/O using PyAV ✅
  - [x] `VideoReader` class - Frame-by-frame reading
  - [x] `VideoWriter` class - Frame-by-frame writing
  - [x] `get_video_info()` - Metadata extraction
  - [x] `extract_frames()` - Frame extraction to images

---

## Priority 2: Validation FIRST (Day 3) ⚠️ CRITICAL

**⚠️ MUST BE COMPLETED BEFORE ANY PASS IMPLEMENTATION**

### Global Rules (Non-Negotiable)
- [x] **[src/validation/global_rules.py](../src/validation/global_rules.py)** - R1-R5 enforcement ✅
  - [x] `validate_r1_pass1_raw_truth()` - No team/identity in Pass 1, multi-layer bbox defense
  - [x] `validate_r2_no_unknown_teams()` - Every fragment has team after Pass 3C, max 6 per team
  - [x] `validate_r3_jersey_temporal_exclusivity()` - No jersey overlaps in time
  - [x] `validate_r4_player_continuity()` - Max 12 concurrent players (high water mark)
  - [x] `validate_r5_ball_never_disappears()` - Ball state at every frame (real/interpolated/out_of_play)

### Pass-Specific Validation
- [x] **[src/validation/pass1_rules.py](../src/validation/pass1_rules.py)** - Pass 1 validation ✅
  - [x] `validate_pass1_detections()` - Bbox validity, jersey confidence, HSV histogram
  - [x] `validate_pass1_ball_detections()` - Ball confidence, bbox validity, max 1 per frame
  - [x] `validate_pass1_frame_coverage()` - Frame index range, detection coverage
  - [x] `validate_pass1()` - All Pass 1 checks including R1

- [x] **[src/validation/pass2_rules.py](../src/validation/pass2_rules.py)** - Pass 2 validation ✅
  - [x] `validate_pass2a_fragments()` - No temporal overlaps, unique IDs, valid ranges
  - [x] `validate_pass2b_quality_scoring()` - Quality enum, score range, consistency metrics
  - [x] `validate_pass2c_ghosts()` - Ghost quality, source fields, estimated position, R4
  - [x] `validate_pass2()` - All Pass 2 checks

- [x] **[src/validation/pass3_rules.py](../src/validation/pass3_rules.py)** - Pass 3 validation ✅
  - [x] `validate_pass3a_candidates()` - Candidate evidence scores
  - [x] `validate_pass3b_constraints()` - Constraint types, fragment references, MUST/SOFT validity
  - [x] `validate_pass3c_identity_commit()` - No unresolved conflicts, R2, R3, player_id format
  - [x] `validate_pass3()` - All Pass 3 checks including identity lock point
  - [ ] **Add compactness-aware team validation (NEW)**
    - [ ] Assert exactly 2 resolved teams after Pass 3C
    - [ ] Assert compactness metrics exist (`cluster_compactness`, `compactness_ratio`)
    - [ ] Assert one cluster is compact OR both clusters compact (bib-vs-random tolerant)
    - [ ] FAIL-FAST if both clusters are diffuse and compactness difference < threshold

- [x] **[src/validation/ball_rules.py](../src/validation/ball_rules.py)** - Ball validation ✅
  - [x] `validate_ball_interpolation()` - R5, gap limits, speed plausibility
  - [x] `validate_ball_state_consistency()` - State consistency (real has bbox, out_of_play has no position)
  - [x] Ball STATE exists at every frame (not position - state ∈ {real, interpolated, out_of_play})
  - [x] Interpolation only for gaps ≤30 frames
  - [x] Out-of-play for gaps >30 frames
  - [x] Speed physical limits (max 100px/frame)

### Validator Orchestrator
- [x] **[src/validation/validator.py](../src/validation/validator.py)** - Main validator ✅
  - [x] `Validator` class - Main orchestrator
  - [x] `validate_pass1(data)` - Run all Pass 1 rules (including R1)
  - [x] `validate_pass2(data)` - Run all Pass 2 rules (including R4)
  - [x] `validate_pass3(data)` - Run all Pass 3 rules (including R2, R3)
  - [x] `validate_ball_interpolation(data)` - Run all ball rules (including R5)
  - [x] `validate_all()` - Run all validations
  - [x] `ValidationResult` - Pydantic model (passed, violations, warnings, timestamp)
  - [x] `ValidationError` exception - Raised on fail-fast
  - [x] `validate_and_raise()` - Convenience function for fail-fast
  - [x] **CRITICAL: Validation runs BEFORE writing JSON**
  - [x] **Failed pass writes NOTHING (not even partial artifacts)**
  - [x] **Validation JSON is the ONLY file written on failure**

---

## Priority 3: Pass 3C Identity Solver (Day 4) ⚠️ CRITICAL

**⚠️ MUST BE COMPLETED BEFORE UPSTREAM PASSES (Pass 1, 2A, 2B, 3A, 3B)**

- [x] **[src/skills/pass3c_identity_solver.py](../src/skills/pass3c_identity_solver.py)** - Constraint satisfaction solver ✅
  - [x] **Input**: Pass 3B constraints, Pass 2C fragments
  - [x] **Algorithm**:
    - [x] 1. Resolve identity using MUST_SAME constraints (track adjacency, ghost continuity)
    - [x] 2. Assign teams via K-means clustering on resolved identities (jersey HSV only)
    - [x] 3. Lock teams immediately via `_locked_team` (single source of truth)
    - [x] 4. Apply jersey inheritance (bidirectional with temporal exclusivity check)
    - [x] 5. Validate CANNOT_SAME constraints (temporal conflicts)
    - [x] 6. Optimize SOFT_SAME constraints (track continuity)
    - [x] 7. FAIL-FAST if unresolved conflicts
  - [x] **Helper: `_assign_and_lock_teams(fragments)`** - K-means team assignment, exclude ghosts
  - [x] **Helper: `_apply_jersey_inheritance(graph, fragments, assignments)`** - Bidirectional propagation
  - [x] **Helper: `_jersey_available(jersey, fragment, assignments)`** - Temporal exclusivity check
  - [x] **Output**: `CommittedIdentity` objects (player_id, team, jersey - all locked)

### Pass 3C Team Clustering Robustness (NEW - bibbed vs random)
- [ ] **Compact-vs-diffuse interpretation inside clustering (no downstream hacks)**
  - [ ] Run K-means (K=2) on **jersey HSV only**
  - [x] Gate K-means inputs by fragment quality metadata when available (`quality_score`, `hsv_consistency`)
  - [ ] Compute cluster compactness (intra-cluster variance or mean pairwise distance)
  - [ ] Identify more compact cluster as bibbed team evidence
  - [ ] Assign TEAM_A/TEAM_B deterministically from compactness interpretation, not raw label index
  - [ ] Lock assignments once resolved (no frame-by-frame oscillation)
  - [ ] Keep behavior symmetric for bib-vs-bib (both compact)
  - [ ] Keep behavior tolerant for bib-vs-random (one compact, one diffuse)
  - [ ] **Do NOT subcluster teams downstream** (team membership stays binary at Pass 3C)

- [ ] **Ambiguity fail-fast rule (contract-level)**
  - [ ] Define `COMPACTNESS_DIFF_MIN` threshold in constants
  - [ ] If both clusters are diffuse and compactness difference < threshold → FAIL-FAST
  - [ ] Write compactness diagnostics to `pass3_validation.json`

- [ ] **Debug metrics additions**
  - [ ] Add `cluster_compactness_a`, `cluster_compactness_b`, `compactness_ratio` to `debug_metrics.json`
  - [ ] Add `team_assignment_mode` (`compactness_guided_kmeans`) to solver log

---

## Priority 4: Detectors & Video I/O (Day 5)

### ML Model Wrappers
- [x] **[src/detectors/player_detector.py](../src/detectors/player_detector.py)** - YOLO wrapper ✅
  - [x] Load `models/PLAYER_MODEL_best_v1.pt`
  - [x] `detect(frame)` - Return bboxes with confidence
  - [x] Apply PLAYER_CONF_THRESHOLD (0.5)
  - [x] Multi-layer bbox defense (huge bbox filter)

- [x] **[src/detectors/ball_detector.py](../src/detectors/ball_detector.py)** - YOLO wrapper ✅
  - [x] Load `models/BALL_MODEL_best_v2.pt`
  - [x] `detect(frame)` - Return ball bboxes
  - [x] Apply BALL_CONF_THRESHOLD (0.3)

- [x] **[src/detectors/jersey_classifier.py](../src/detectors/jersey_classifier.py)** - YOLO wrapper ✅
  - [x] Load `models/JERSEY_MODEL_best_v1.pt`
  - [x] `classify(frame, bbox)` - Return jersey number + confidence
  - [x] Apply JERSEY_CONF_THRESHOLD (0.3, lower per memory learnings)
  - [x] `get_probabilities()` - Return probability distribution for Pass 3A

- [x] **[src/detectors/tracker.py](../src/detectors/tracker.py)** - ByteTrack wrapper ✅
  - [x] Initialize with TRACK_HIGH_THRESH (0.6), TRACK_LOW_THRESH (0.1)
  - [x] `update(detections)` - Return tracked detections with track_id
  - [x] Handle track lifecycle (birth, continuation, death)
  - [x] Fallback tracker if ByteTrack not available

### Video I/O
- [x] **[src/utils/video_io.py](../src/utils/video_io.py)** - Video reading/writing ✅ (completed in Priority 1)
  - [x] `VideoReader` class (using `av` library)
  - [x] `VideoWriter` class for visualization output
  - [x] Frame iteration, fps, width, height properties

---

## Priority 5: Pipeline Skills - Passes 1 & 2 (Days 6-7)

### Pass 1: Raw Evidence
- [x] **[src/skills/pass1_extractor.py](../src/skills/pass1_extractor.py)** - Raw evidence extraction ✅
  - [x] Run YOLO player detection
  - [x] **Filter huge bboxes** (multi-layer defense built into PlayerDetector):
    - [x] Layer 1: Absolute limits (800px height, 600px width)
    - [x] Layer 2: Relative limit (25% frame area)
  - [x] Run ByteTrack for temporary track_ids
  - [x] Run jersey classifier (conf ≥ 0.3)
  - [x] Extract HSV histograms (8x8x8 bins) from jersey ROI only (full-body HSV removed)
  - [x] Persist ROI geometry evidence (`jersey_roi_bbox`, `jersey_roi_valid`) for deterministic auditability
  - [x] Run ball detector
  - [x] Create detection_id: `{frame_idx}_{track_id}_{bbox_hash}`
  - [x] **NO team assignment, NO identity, NO player_id**
  - [x] Save `pass1_raw.json`
  - [x] Validate output (call validator.validate_pass1)
  - [x] FAIL-FAST if validation fails
- [x] **[test_pass1.py](../test_pass1.py)** - Test script for Pass 1 ✅

### Pass 2A: Mechanical Fragmentation
- [ ] **[src/skills/pass2a_fragmenter.py](../src/skills/pass2a_fragmenter.py)** - Track splitting
  - [ ] Group detections by track_id
  - [ ] Split triggers:
    - [ ] **Track overlap collision**: Same track_id produces >1 detection in same frame → immediate split (ByteTrack failure - root cause fix)
    - [ ] Appearance drift (HSV histogram change)
    - [ ] Jersey inconsistency:
      - [ ] ✅ Split: Jersey disappears (#4 → None) - track lost player
      - [ ] ✅ Split: Jersey changes (#7 → #4) - track jumped
      - [ ] ❌ NO split: Jersey first appearance (None → #4) - player turned around
    - [ ] Jersey temporal exclusivity (same jersey on different tracks simultaneously)
  - [ ] **Keep ALL fragments** (even < 10 frames)
  - [ ] Mark short fragments as `quality = "low"`
  - [ ] **Merge consecutive short fragments** on same track
  - [ ] Assign fragment_id: `F{counter:06d}`
  - [ ] Save `pass2_fragments.json`, log all splits
  - [ ] Validate output
  - [ ] FAIL-FAST if validation fails

### Pass 2B: Fragment Quality Scoring
- [ ] **[src/skills/pass2b_fragment_scoring.py](../src/skills/pass2b_fragment_scoring.py)** - Quality metadata
  - [ ] Compute quality scores (0-1):
    - [ ] Detection confidence stability (avg, min)
    - [ ] Bbox stability (low jitter)
    - [ ] Jersey consistency
    - [ ] HSV consistency
  - [ ] Assign quality: HIGH/MEDIUM/LOW (already GHOST from Pass 2C)
  - [ ] **Metadata only** (no identity decisions)
  - [ ] Save scored fragments
  - [ ] Validate output

### Pass 2C: Ghost Generation
- [ ] **[src/skills/pass2c_ghost_generator.py](../src/skills/pass2c_ghost_generator.py)** - Maintain player count
  - [ ] Initialize level from first 10 frames
  - [ ] **Dynamic level (high water mark)**:
    - [ ] Update if more players enter (never decrease)
    - [ ] Target level = high water mark up to 12 (futsal max)
    - [ ] Partial clips: May start with fewer players (e.g., 8 visible)
    - [ ] Off-screen starts: Level increases as players enter (10 → 11 → 12)
    - [ ] No substitutions in futsal: Once player enters, they don't leave (except briefly off-screen)
  - [ ] **Ghosts maintain identity continuity, NOT team symmetry**:
    - [ ] Team balance (6v6) enforced AFTER Pass 3C, NOT during ghost creation
  - [ ] Track players by `original_track_id` (NOT `fragment_id`)
  - [ ] Create ghosts when `tracked_count < level`
  - [ ] Ghost position: HOLD last known position (no interpolation)
  - [ ] Ghost duration: until reappearance or MAX_GAP (60 frames)
  - [ ] Mark ghosts: `is_ghost=True`, `quality="ghost"`
  - [ ] **Exclude ghosts from K-means clustering** (Pass 3C)
  - [ ] Save `pass2_ghosts.json`, log ghost creation
  - [ ] Validate output

---

## Priority 6: Pipeline Skills - Pass 3 (Days 8)

### Pass 3A: Identity Candidates
- [ ] **[src/skills/pass3a_candidate_generator.py](../src/skills/pass3a_candidate_generator.py)** - Generate possibilities
  - [ ] For each fragment, generate:
    - [ ] Possible teams with evidence scores (from HSV histograms)
    - [ ] Possible jerseys with probabilities (from jersey detections)
    - [ ] Adjacent fragments (track continuity links)
  - [ ] **NO LOCKING** (just candidates, no decisions)
  - [ ] Save `pass3_candidates.json`

### Pass 3B: Constraint Graph
- [ ] **[src/skills/pass3b_constraint_builder.py](../src/skills/pass3b_constraint_builder.py)** - Build constraint graph
  - [ ] **MUST_SAME constraints**: Track adjacency (same track_id) + ghost continuity
  - [ ] **CANNOT_SAME constraints**: Jersey temporal exclusivity violations
  - [ ] **SOFT_SAME constraints**: Track continuity preferences (velocity, appearance)
  - [ ] Graph structure: fragment_id → constraint_ids
  - [ ] **Note**: Team assignment happens in Pass 3C AFTER identity resolution
  - [ ] Save `pass3_constraints.json`

### Pass 3C: Identity Commit (Already completed in Priority 3)
✅ See Priority 3

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

### Visualization
- [ ] **[src/skills/visualizer.py](../src/skills/visualizer.py)** - Final video output
  - [ ] Render video with:
    - [ ] Bboxes colored by player_id
    - [ ] Team colors (team_a, team_b)
    - [ ] Jersey numbers overlaid
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
    9. [ ] Visualization
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
  - [ ] Assert validation.passed == True for all passes
  - [ ] Assert specific metrics (from memory learnings)

### Unit Tests
- [ ] **[tests/core/test_data_models.py](../tests/core/test_data_models.py)** - Model validation
- [ ] **[tests/validation/test_global_rules.py](../tests/validation/test_global_rules.py)** - Rule enforcement
- [ ] **[tests/skills/test_pass1_extractor.py](../tests/skills/test_pass1_extractor.py)** - Mock inputs
- [ ] **[tests/skills/test_pass2a_fragmenter.py](../tests/skills/test_pass2a_fragmenter.py)** - Split logic
- [ ] **[tests/skills/test_pass2c_ghost_generator.py](../tests/skills/test_pass2c_ghost_generator.py)** - Ghost creation
- [ ] **[tests/skills/test_pass3c_identity_solver.py](../tests/skills/test_pass3c_identity_solver.py)** - Constraint satisfaction

### End-to-End Test
- [ ] Run full pipeline on clip9
- [ ] Verify all 9 JSON artifacts created
- [ ] Verify all validation reports show `passed: true`
- [ ] Verify visualization quality:
  - [ ] Players colored by team
  - [ ] Jersey numbers visible
  - [ ] Ghosts shown as dashed during occlusions
  - [ ] Ball tracked throughout (solid/dashed)
  - [ ] No "unknown" grey boxes

---

## Success Criteria

- [ ] All 30 implementation files created and tested
- [ ] Pipeline runs on clip9 without errors
- [ ] All validation reports pass (passed: true)
- [ ] Visualization shows:
  - [ ] Correct team colors
  - [ ] Jersey numbers
  - [ ] Ghosts during occlusions
  - [ ] Ball tracking (real + interpolated)
  - [ ] Zero "unknown" fragments
- [ ] Debug metrics show:
  - [ ] 0 unknown teams
  - [ ] 0 jersey temporal conflicts
  - [ ] Max 12 concurrent players
  - [ ] 100% frame coverage

---

## Notes & Learnings

### Key Memory Learnings Incorporated
- ✅ Multi-layer bbox defense (Pass 1)
- ✅ Team assignment locking with `_locked_team` (Pass 3C)
- ✅ Fragment gap prevention (Pass 2A)
- ✅ Jersey temporal exclusivity (Pass 2A, Pass 3C)
- ✅ Bidirectional team/jersey inheritance (Pass 3C)
- ✅ Ghost exclusion from K-means (Pass 3C)
- ✅ Ghost visualization deduplication (visualizer)
- ✅ Dynamic level high water mark (Pass 2C)
- ✅ Track players by original_track_id, not fragment_id (Pass 2C)

### Critical Principles
- **Root-cause only**: Never patch downstream
- **Pass immutability**: Downstream cannot modify upstream
- **Single identity commit**: Pass 3C is the lock point
- **Fail-fast**: Halt on first violation
- **JSON is truth**: Everything auditable without video

---

**Last Updated**: 2026-02-17
**Estimated Completion**: Day 10 (~2026-02-27)
