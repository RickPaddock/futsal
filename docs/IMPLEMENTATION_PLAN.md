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

## Immediate Next: Ball Detection Accuracy (CRITICAL) ✅ COMPLETE

- [x] **Integrate `supervision.InferenceSlicer` in [src/detectors/ball_detector.py](../src/detectors/ball_detector.py)** ✅
  - [x] Add `use_inference_slicer` config flag and wire through Pass 1 extractor
  - [x] Lazy-init slicer on first frame using frame dimensions
  - [x] Use overlapping 2x2 tiling callback for small-object recall
  - [x] Merge tile detections with overlap filtering + NMS (`OverlapFilter.NON_MAX_SUPPRESSION`)
  - [x] Keep output schema unchanged (`Detection` objects only) to preserve pass immutability
  - [x] Validate on clip8 that ball recall improves without violating Pass 1/ball validators (+20% recall: 353→424)
  - [x] Fail-fast if slicer path produces malformed/duplicate detections in a frame

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
  - [x] `validate_r4_player_continuity()` - Presence completeness vs dynamic level (deficits are blocking)
  - [x] `validate_r5_ball_never_disappears()` - Ball state at every frame (real/interpolated/out_of_play)

### Pass-Specific Validation
- [x] **[src/validation/pass1_rules.py](../src/validation/pass1_rules.py)** - Pass 1 validation ✅
  - [x] `validate_pass1_detections()` - Structural bbox/frame validity (blocking) + perceptual diagnostics (warning)
  - [x] Pass 1 physical cap - keep best-12 detections per frame with explicit saturation warning
  - [x] `validate_pass1_ball_detections()` - Ball confidence, bbox validity, max 1 per frame
  - [x] `validate_pass1_frame_coverage()` - Frame index range, detection coverage
  - [x] `validate_pass1()` - All Pass 1 checks including R1

- [x] **[src/validation/pass2_rules.py](../src/validation/pass2_rules.py)** - Pass 2 validation ✅
  - [x] `validate_pass2a_fragments()` - No temporal overlaps, unique IDs, valid ranges
  - [x] `validate_pass2a_frame_coverage()` - Explicit per-track frame-span equality vs Pass 1
  - [x] `validate_pass2b_quality_scoring()` - Binary class (`real|occlusion_candidate`) enforcement + metadata diagnostics
  - [x] `validate_pass2c_ghosts()` - Ghost quality, source fields, same-track chain continuity, per-track lifespan audit, R4 deficits
  - [x] `PASS2C_PHYSICAL_MAX_EXCEEDED_DEFERRED` warning - identity collision signal deferred to Pass 3
  - [x] `validate_pass2()` - All Pass 2 checks

- [x] **[src/validation/pass3_rules.py](../src/validation/pass3_rules.py)** - Pass 3 validation ✅
  - [x] `validate_pass3a_candidates()` - Candidate evidence scores
  - [x] `validate_pass3b_constraints()` - Constraint types, fragment references, MUST/SOFT validity
  - [x] `validate_pass3c_identity_commit()` - No unresolved conflicts, R2, R3, player_id format
  - [x] `validate_pass3()` - All Pass 3 checks including identity lock point
  - [x] Physical reality checks (blocking): per-frame cap, ghost retirement, no same-player overlap, global R4 exact-level
  - [x] **Add compactness-aware team validation (NEW)** ✅
    - [x] Assert exactly 2 resolved teams after Pass 3C
    - [x] Assert compactness metrics exist (`cluster_compactness`, `compactness_ratio`)
    - [x] Assert one cluster is compact OR both clusters compact (bib-vs-random tolerant)
    - [x] FAIL-FAST if both clusters are diffuse and compactness difference < threshold

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
  - [x] Pass 2 warnings (non-blocking) are preserved separately from blocking errors
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
- [x] **Compact-vs-diffuse interpretation inside clustering (no downstream hacks)** ✅
  - [x] Run K-means (K=2) on **jersey HSV only**
  - [x] Gate K-means inputs by fragment quality metadata when available (`quality_score`, `hsv_consistency`)
  - [x] Compute cluster compactness (intra-cluster variance or mean pairwise distance)
  - [x] Identify more compact cluster as bibbed team evidence
  - [x] Assign TEAM_A/TEAM_B deterministically from compactness interpretation, not raw label index
  - [x] Lock assignments once resolved (no frame-by-frame oscillation)
  - [x] Keep behavior symmetric for bib-vs-bib (both compact)
  - [x] Keep behavior tolerant for bib-vs-random (one compact, one diffuse)
  - [x] **Do NOT subcluster teams downstream** (team membership stays binary at Pass 3C)

- [x] **Ambiguity fail-fast rule (contract-level)** ✅
  - [x] Define `COMPACTNESS_DIFF_MIN` threshold in constants
  - [x] If both clusters are diffuse and compactness difference < threshold → FAIL-FAST
  - [x] Write compactness diagnostics to `pass3_validation.json`

- [x] **Debug metrics additions** ✅
  - [x] Add `cluster_compactness_a`, `cluster_compactness_b`, `compactness_ratio` to `debug_metrics.json`
  - [x] Add `team_assignment_mode` (`compactness_guided_kmeans`) to solver log
- [ ] **Pass 3C debug video** (`--video-output 3c` or `--video-output 3`) - Identity commit (LOCK POINT)
  - [ ] Show: Fragment bboxes colored by final team (team_a=blue, team_b=red)
  - [ ] Overlays: player_id, jersey numbers, locked team assignments
  - [ ] Purpose: Verify final identity commit (are teams correct? jerseys correct? no unknowns?)

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
  - [x] **InferenceSlicer integration complete** ✅
    - [x] Add tile callback + overlap merge behavior aligned to reference branch
    - [x] Tile detections properly mapped to original frame coordinates
    - [x] Maintains compatibility with ball interpolation input contract
    - [x] Validated: +20% ball recall improvement (353→424 detections on clip8)

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
  - [x] **CRITICAL FIX: Fisheye bbox correction** ✅
    - [x] **Problem**: Fisheye lens → players at edges appear tilted → axis-aligned bbox cuts off jersey → incomplete crop → wrong HSV → false "appearance discontinuity" splits in Pass 2A
    - [x] **Root Cause**: Bad bbox → bad jersey crop → jersey COLOR change detected (HSV drift) → spurious TRIGGER 4 (Hard Appearance Discontinuity)
    - [x] **Solution**: Radial bbox expansion based on distance from frame center
    - [x] **Implementation**: [src/utils/geometry.py](../src/utils/geometry.py) `apply_fisheye_bbox_correction()`
    - [x] **Applied**: AFTER YOLO detection, BEFORE jersey classification/HSV extraction
    - [x] **Config**: `FISHEYE_CORRECTION_ENABLED=True`, `FISHEYE_EXPANSION_STRENGTH=0.15` (15% expansion at corners)
    - [x] **Benefit**: Reduces false jersey temporal exclusivity splits, improves Pass 2A fragment quality
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
- [x] **Pass 1 debug video** (`--video-output 1`) - Raw detections
  - [x] Show: YOLO bboxes, ByteTrack IDs, jersey numbers, confidence scores
  - [x] Purpose: Verify detection quality, track stability, jersey classification

### Pass 2A: Mechanical Fragmentation
- [x] **[src/skills/pass2a_fragmenter.py](../src/skills/pass2a_fragmenter.py)** - Track splitting ✅
  - [x] Group detections by track_id
  - [x] **Split triggers (Per CLAUDE.md Section 5 - EXHAUSTIVE list)**:
    - [x] **TRIGGER 1: Track Collision** - Same track_id produces >1 detection in same frame (ByteTrack failure)
    - [x] **TRIGGER 2: Jersey Change** - Jersey NUMBER changes (#7 → #4), NOT disappearance
      - [x] ❌ NO split: Jersey disappearance (#4 → None) - loss of observability
      - [x] ❌ NO split: Jersey first appearance (None → #4) - player turned around
    - [x] **TRIGGER 3: Jersey Temporal Exclusivity** - Same jersey on different tracks simultaneously ✅
      - [x] **Algorithm**: Use voting/consensus across fragment, NOT first appearance
      - [x] **Minimum observations**: Jersey must appear ≥3 times with conf ≥0.5 to be considered "owned"
      - [x] **Majority vote**: Fragment "owns" jersey if ≥50% of sampled observations show that jersey
      - [x] **Conflict detection**: Two fragments "own" same jersey + overlapping time → split the later one
      - [x] **No false positives**: Single misclassified frame does NOT cause split
    - [x] **TRIGGER 4: Hard Appearance Discontinuity** - ALL required: jersey visible both sides + large HSV + impossible motion
      - [x] ❌ NO split: Standalone appearance drift - lighting/angle change
      - [x] ❌ NO split: Standalone velocity spike - player running
  - [x] **Keep ALL fragments** (even < 10 frames)
  - [x] Mark short fragments as `quality = "low"`
  - [x] **Merge consecutive short fragments** on same track
  - [x] Assign fragment_id: `F{counter:06d}`
  - [x] Save `pass2_fragments.json`, log all splits
  - [x] Validate output
  - [x] FAIL-FAST if validation fails
- [x] **Pass 2 unified debug video** (`--video-output 2`) ✅
  - [x] **Combines all Pass 2 stages (2A fragments + 2B quality + 2C ghosts) in ONE video**
  - [x] Show: Fragment bboxes color-coded by quality (GREEN=high, YELLOW=medium, RED=low, GRAY=ghost)
  - [x] Ghosts: Dashed boxes with gray color
  - [x] Labels: Fragment ID, quality label (e.g., [HIGH]), quality score (Q=0.85), split reason
  - [x] Overlays: Frame stats (real count by quality H/M/L, ghost count, total players, splits)
  - [x] Purpose: Verify mechanical fragmentation, quality scoring, and ghost generation (complete Pass 2 view)


#### Pass 2A Validation Contract Delta (from CLAUDE.md Section 2A) ⚠️ REQUIRED

- [x] **KEEP (already correct)** ✅
  - [x] Fragment ID uniqueness
  - [x] `start_frame <= end_frame`
  - [x] No temporal overlap within the same `original_track_id`
  - [x] Validation is blocking (`severity="error"`)

- [x] **Fix coverage semantics: frame-based → detection-based** ✅
  - [x] Replace frame coverage checks with detection ID coverage checks
  - [x] Validate exact set equality: Pass 1 `detection_id` set == union of Pass 2A fragment `detection_ids`
  - [x] FAIL if any Pass 1 `detection_id` is missing from fragments
  - [x] FAIL if any fragment contains unknown `detection_id` not present in Pass 1
  - [x] FAIL if any `detection_id` appears in more than one fragment

- [x] **Enforce split metadata for non-initial fragments** ✅
  - [x] Add blocking rule: each non-initial fragment must include `split_reason`
  - [x] Add blocking rule: each non-initial fragment must include `split_trigger_frame`
  - [x] Add blocking rule: each non-initial fragment must include `split_rule_id` (e.g., `JERSEY_CHANGE`, `TRACK_COLLISION`)
  - [x] FAIL if split exists without logged trigger reason metadata

- [x] **Remove merge assumptions from Pass 2A validation** ✅
  - [x] Do not encourage or require merging short fragments
  - [x] Do not validate against short-fragment count/length as an error by itself
  - [x] Treat short fragments as valid evidence boundaries

- [x] **Tighten extra coverage behavior to blocking** ✅
  - [x] Change `PASS2A_EXTRA_COVERAGE` from warning to blocking error
  - [x] Exact coverage only; no fabricated fragment extent beyond Pass 1 detection evidence

- [x] **Add cross-track exclusivity validation** ✅
  - [x] Add global exclusivity check: no detection/frame evidence row can belong to more than one fragment total
  - [x] Ensure overlap detection is not limited to same-track comparisons

- [x] **Out of scope for Pass 2A validator (must NOT be added)** ✅
  - [x] Identity/team assignment checks
  - [x] Pass 2B quality scoring logic
  - [x] Pass 2C ghost logic
  - [x] Split-threshold policy checks (HSV/velocity tuning belongs to splitter implementation)

### Pass 2B: Fragment Quality Scoring
- [x] **[src/skills/pass2b_fragment_scoring.py](../src/skills/pass2b_fragment_scoring.py)** - Quality metadata ✅
  - [x] Compute quality scores (0-1):
    - [x] Detection confidence stability (avg, min)
    - [x] Bbox stability (low jitter)
    - [x] Jersey consistency
    - [x] HSV consistency
  - [x] Assign quality: HIGH/MEDIUM/LOW (already GHOST from Pass 2C)
  - [x] **Metadata only** (no identity decisions)
  - [x] Save scored fragments
  - [x] Validate output
- [x] **[test_pass2b.py](../test_pass2b.py)** - Test script for Pass 2B ✅
- [x] **Pass 2B visualization** - Integrated into unified `--video-output 2` ✅
  - [x] Quality-based color coding (GREEN=high, YELLOW=medium, RED=low)
  - [x] Quality scores and labels shown on each fragment
  - [x] See unified Pass 2 video above for complete visualization

### Pass 2C: Ghost Generation
- [x] **[src/skills/pass2c_ghost_generator.py](../src/skills/pass2c_ghost_generator.py)** - Maintain player count ✅
  - [x] Initialize level from first 10 frames
  - [x] **Dynamic level (high water mark)**:
    - [x] Update if more players enter (never decrease)
    - [x] Target level = high water mark up to 12 (futsal max)
    - [x] Partial clips: May start with fewer players (e.g., 8 visible)
    - [x] Off-screen starts: Level increases as players enter (10 → 11 → 12)
    - [x] No substitutions in futsal: Once player enters, they don't leave (except briefly off-screen)
  - [x] **Ghosts maintain identity continuity, NOT team symmetry**:
    - [x] Team balance (6v6) enforced AFTER Pass 3C, NOT during ghost creation
  - [x] Track players by `original_track_id` (NOT `fragment_id`)
  - [x] Create ghosts when `tracked_count < level`
  - [x] Ghost position: HOLD last known position (no interpolation)
  - [x] Ghost duration: until reappearance or MAX_GAP (60 frames)
  - [x] Mark ghosts: `is_ghost=True`, `quality="ghost"`
  - [x] **Exclude ghosts from K-means clustering** (Pass 3C responsibility, ghost flag available)
  - [x] Save `pass2_ghosts.json`, log ghost creation
  - [x] Validate output
- [ ] **Pass 2C debug video** (`--video-output 2c`) - Ghost generation
  - [ ] Show: Real fragments (solid) + ghost fragments (dashed)
  - [ ] Overlays: Player count ticker (tracked + ghosts = level), ghost reasons
  - [ ] Purpose: Verify dynamic level, ghost positioning, player count continuity

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
- [ ] **Pass 3A debug video** (`--video-output 3a`) - Identity candidates
  - [ ] Show: Fragment bboxes with candidate teams/jerseys
  - [ ] Overlays: Evidence scores, probability distributions, adjacency links
  - [ ] Purpose: Verify candidate generation logic (are candidates reasonable?)

### Pass 3B: Constraint Graph
- [ ] **[src/skills/pass3b_constraint_builder.py](../src/skills/pass3b_constraint_builder.py)** - Build constraint graph
  - [ ] **MUST_SAME constraints**: Track adjacency (same track_id) + ghost continuity
  - [ ] **CANNOT_SAME constraints**: Jersey temporal exclusivity violations
  - [ ] **SOFT_SAME constraints**: Track continuity preferences (velocity, appearance)
  - [ ] Graph structure: fragment_id → constraint_ids
  - [ ] **Note**: Team assignment happens in Pass 3C AFTER identity resolution
  - [ ] Save `pass3_constraints.json`
- [ ] **Pass 3B debug video** (`--video-output 3b`) - Constraint graph
  - [ ] Show: Fragment bboxes with constraint edges overlaid
  - [ ] Overlays: MUST_SAME (green), CANNOT_SAME (red), SOFT_SAME (yellow)
  - [ ] Purpose: Verify constraint graph structure (are constraints correct?)

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
- [ ] Run full pipeline on clip9
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
- [ ] Pipeline runs on clip9 without errors
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

---

## Notes & Learnings

### Key Memory Learnings Incorporated
- ✅ Multi-layer bbox defense (Pass 1)
- ✅ **Fisheye bbox correction** (Pass 1) - **NEW**: Radial expansion to prevent false splits
- ✅ Team assignment locking with `_locked_team` (Pass 3C)
- ✅ Fragment gap prevention (Pass 2A)
- ✅ Jersey temporal exclusivity (Pass 2A, Pass 3C)
- ✅ Bidirectional team/jersey inheritance (Pass 3C)
- ✅ Ghost exclusion from K-means (Pass 3C)
- ✅ Ghost visualization deduplication (visualizer)
- ✅ Dynamic level high water mark (Pass 2C)
- ✅ Track players by original_track_id, not fragment_id (Pass 2C)
- ✅ InferenceSlicer tiling for ball recall

### Critical Principles
- **Root-cause only**: Never patch downstream
- **Pass immutability**: Downstream cannot modify upstream
- **Single identity commit**: Pass 3C is the lock point
- **Fail-fast**: Halt on first violation
- **JSON is truth**: Everything auditable without video

---

## ONGOING: R4 Compliance & Contract Enforcement (CRITICAL FIX) 🚧

**Date Started**: 2026-02-19
**Status**: In Progress
**Plan Reference**: [robust-sniffing-graham.md](../.claude/plans/robust-sniffing-graham.md)

### Problem Identified
System violating R4 ("Players Never Disappear") due to:
1. **Contract Ambiguity**: CLAUDE.md confused fragments (identity) with ghosts (presence)
2. **Ghost Expiration Bug**: Ghosts expire after 60 frames without chaining
   - Example: Ghost G000003 expires at frame 150, no chain created for frames 151-160
   - Root cause: [pass2c_ghost_generator.py:515](../src/skills/pass2c_ghost_generator.py#L515) caps gap at 60 frames
3. **Lenient Validation**: 20% tolerance masks R4 violations
   - Allows frames 151-160 to have 11 players instead of 12

### Solution Overview
- **CLAUDE.md contract tightening** (4 changes)
  - Add section 0.5: Mandatory Pre-Change Gate (process enforcement)
  - Add section 3.5: Fragment vs Ghost Contract (explicit separation)
  - Tighten R4: Zero tolerance, explicit chaining rules
  - Update Pass 2C: Ghost chaining semantics

- **Validation strictness** (3 changes)
  - Replace 20% tolerance with zero tolerance (R4 invariant)
  - Add PRESENCE_UNIQUENESS rule (presence-based counting, NOT fragment ranges)
  - Add GHOST_CHAINING_CONTINUITY rule (zero gap between chains)

- **Ghost expiration fix** (2 changes)
  - Remove 60-frame cap in `_find_missing_players` (enable indefinite chaining)
  - Fix `player_states` update to track real detections only (preserve gap calculation)

### Implementation Progress

#### CLAUDE.md Contract Changes
- [x] **[CLAUDE.md](../CLAUDE.md) Section 0.5** - Mandatory Pre-Change Gate ✅
  - [x] 5-question pre-change checklist (source vs symptom, entities, invariants, failure mode, prevention)
  - [x] Process enforcement gate to prevent category errors

- [x] **[CLAUDE.md](../CLAUDE.md) Section 3.5** - Fragment vs Ghost Contract ✅
  - [x] Fragment Contract (Identity Layer): Fragments span gaps but do NOT imply presence
  - [x] Ghost Contract (Presence Layer): Maintain R4 invariant, enable chaining
  - [x] **CRITICAL**: "Fragments may span frames with no detections, but do NOT imply player presence; presence is satisfied exclusively by real detections or ghosts."

- [x] **[CLAUDE.md](../CLAUDE.md) R4 Definition** - Zero Tolerance ✅
  - [x] HARD INVARIANT: `tracked_count + ghost_count = level` at EVERY frame
  - [x] Ghost chaining: Chains continue INDEFINITELY until player reappears OR clip ends
  - [x] Validation: FAIL if ANY frame has count ≠ level (zero tolerance)

- [x] **[CLAUDE.md](../CLAUDE.md) Pass 2C Responsibilities** - Ghost Chaining Semantics ✅
  - [x] Ghost duration: 60 frames per ghost, then chain if player not reappeared
  - [x] Ghost chaining: When ghost expires, create new ghost if player still missing
  - [x] Ghost termination: ONLY when player reappears (new detection) OR clip ends

- [x] **[CLAUDE.md](../CLAUDE.md) Pass 2 Validation** - R4 Zero Tolerance ✅
  - [x] FAIL IF: Tracked + ghosts ≠ level (R4 zero tolerance)
  - [x] FAIL IF: Ghost expires without reappearance and no chain created

#### Validation Rule Changes
- [ ] **[src/validation/pass2_rules.py](../src/validation/pass2_rules.py) Lines 566-609** - Zero Tolerance
  - [ ] Replace 20% tolerance with exact equality check: `violation_frames = {f: c for f, c in frame_counts.items() if c != expected_level}`
  - [ ] Rule: `PASS2C_R4_INVARIANT_VIOLATED` (error severity)
  - [ ] Message: "R4 violated: {N} frames have count ≠ {level}. Per CLAUDE.md R4: 'At every frame: tracked_count + ghost_count = level' (ZERO TOLERANCE)."

- [ ] **[src/validation/pass2_rules.py](../src/validation/pass2_rules.py) After Line 612** - Presence Uniqueness
  - [ ] Add `validate_pass2c_presence_uniqueness(all_fragments, pass1_detections)`
  - [ ] **CRITICAL**: Presence counting based on ACTUAL PRESENCE (real detections + ghosts), NOT fragment time ranges
  - [ ] Build `frame_detections` map from Pass 1 to check actual detections
  - [ ] Only count presence if: (1) fragment is ghost, OR (2) fragment has detection at frame
  - [ ] FAIL if multiple reals or multiple ghosts for same track_id at same frame

- [ ] **[src/validation/pass2_rules.py](../src/validation/pass2_rules.py) After Presence Uniqueness** - Ghost Chaining
  - [ ] Add `validate_pass2c_ghost_chaining(all_fragments, pass1_detections)`
  - [ ] Check for gaps between consecutive ghosts on same track
  - [ ] FAIL if gap > 0 frames between ghost expiry and next ghost start
  - [ ] Rule: `PASS2C_GHOST_CHAIN_BROKEN` (error severity)

- [ ] **[src/validation/pass2_rules.py](../src/validation/pass2_rules.py) validate_pass2 Function** - Call New Rules
  - [ ] Add calls to `validate_pass2c_presence_uniqueness()`
  - [ ] Add calls to `validate_pass2c_ghost_chaining()`

#### Ghost Expiration Fix
- [ ] **[src/skills/pass2c_ghost_generator.py](../src/skills/pass2c_ghost_generator.py) Line 515** - Remove Cap
  - [ ] **Current**: `if gap > 0 and gap <= const.MAX_FRAGMENT_GAP:` (60-frame cap)
  - [ ] **Replace**: `if gap > 0:` (no cap - always consider missing)
  - [ ] **Rationale**: 60-frame cap prevents ghost chaining when gap > 60

- [ ] **[src/skills/pass2c_ghost_generator.py](../src/skills/pass2c_ghost_generator.py) Lines 249-263** - Real Detections Only
  - [ ] **Current**: `if last_bbox is not None:` (updates for ghosts too)
  - [ ] **Replace**: `if last_bbox is not None and not frag.is_ghost:` (real detections only)
  - [ ] **Rationale**: `last_frame` should track last REAL detection, not ghost frames

### Verification Plan
- [ ] **Unit Tests** - `tests/unit/test_ghost_chaining.py`
  - [ ] Test ghost expiration and chaining
  - [ ] Test multiple chained ghosts (200-frame gap)
  - [ ] Test ghost termination on reappearance
  - [ ] Test `_find_missing_players` with gap > 60 frames

- [ ] **Validation Tests** - `tests/unit/test_pass2_validation_strict.py`
  - [ ] Test R4 zero tolerance (1 frame violation = FAIL)
  - [ ] Test presence uniqueness rules
  - [ ] Test ghost chaining continuity

- [ ] **Regression Test** - `videos/input/GoPro_Futsal_part1_CLEANED_clip9.mp4`
  - [ ] Verify Ghost G000003 chains to Ghost G000004 at frames 149-150
  - [ ] Verify zero presence gaps (every frame = 12 players)
  - [ ] Verify validation catches frames 151-160 violation (before fix)
  - [ ] Verify validation passes (after fix)

### Expected Impact
- **Frames 151-160 bug FIXED**: Ghost chains automatically (no gaps)
- **All presence gaps caught**: Zero tolerance validation
- **Explicit contracts**: No future confusion about fragment vs ghost purposes
- **Mandatory gate**: Process prevents category errors from recurring

### Pre-Change Gate Answers (Per Section 0.5)

**1. Source vs symptom?**
- **Source fix**: Ghost expiration bug + contract ambiguity
- **Not symptom**: Not patching downstream (e.g., not fixing visualization to hide gaps)

**2. Entities involved?**
- **Ghosts** (presence layer): Enable chaining, no 60-frame cap
- **Fragments** (identity layer): Clarify that fragments do NOT imply presence
- **Validation**: Zero tolerance for R4

**3. Invariants affected?**
- **R4 (Players Never Disappear)**: Enforcement STRENGTHENED (20% tolerance → zero tolerance)

**4. Failure mode?**
- **Loud**: Validation errors at Pass 2C if ghost chaining broken or R4 violated

**5. Prevention mechanism?**
- **New validation rules**: PRESENCE_UNIQUENESS, GHOST_CHAINING_CONTINUITY, R4_INVARIANT_VIOLATED
- **Tightened contract**: Section 3.5 explicit fragment vs ghost separation
- **Process gate**: Section 0.5 mandatory pre-change checklist

---

**Last Updated**: 2026-02-19
**Estimated Completion**: Day 10 (~2026-02-27)
