# Futsal Player and Ball Tracking System - Technical Analysis
**Date:** February 11, 2026
**Methodology:** CLEAR (Clarify, Logic, Examples, Adapt, Results)

---

## C - CLARIFY

### Project Purpose

**Input:** Raw futsal match video from ceiling-mounted fisheye wide-angle camera (see [docs/image.png](image.png))

**Output:**
- Annotated video with player bounding boxes, team colors, jersey numbers, and track IDs
- JSON tracking data containing per-frame player positions, team assignments, jersey numbers, and ball positions
- Optional 2D pitch visualizations (birdseye view, Voronoi spatial dominance)

**Core Goal:** Maintain persistent player identity across difficult scenarios including:
- Dense player occlusions and clustering
- Players entering and exiting the frame
- Jersey visibility changes (players turning, brief occlusions)
- Fast movements and rapid position changes
- Challenging lighting and fisheye distortion

**Key Constraint:** Indoor futsal presents unique challenges compared to outdoor football:
- Faster gameplay with more frequent player interactions
- Smaller court means higher player density
- Overhead fisheye camera creates significant barrel distortion
- Team identification must handle bibbed vs. non-bibbed teams

---

### Missing or Ambiguous Information

#### Documentation Gaps

1. **Test Coverage Status**
   - A `tests/` directory exists but unclear what components are tested
   - No documented test execution procedures
   - No coverage reports or CI/CD integration visible

2. **Model Training Procedures**
   - Model weights present (~826 MB in `models/` directory)
   - Training datasets exist (`models/datasets/`)
   - No training scripts or procedures documented
   - No information on dataset annotation process or tools

3. **Performance Benchmarks**
   - No documented accuracy metrics on real futsal footage
   - No processing speed benchmarks
   - No memory usage profiling
   - No comparison to baseline tracking systems

4. **Deployment Instructions**
   - Inference-only system or can models be retrained?
   - GPU requirements unclear (VRAM, compute capability)
   - Expected processing speed per minute of video?
   - Can the system run on CPU-only hardware?

5. **Homography Calibration**
   - 13-point mapping mentioned in code
   - No calibration tool provided in repository
   - No documentation on calibration procedure
   - How to adapt to different camera angles/heights?

6. **SAM2 Segmentation Integration**
   - Marked as "optional" in code comments
   - When is SAM2 segmentation used vs. YOLO bounding boxes?
   - What performance impact does SAM2 have?
   - Is it required for certain scenarios?

7. **System Requirements**
   - GPU memory requirements
   - Minimum/recommended hardware specs
   - Python version compatibility
   - Operating system support

---

### Open Questions

#### Accuracy and Validation
- **Q1:** What accuracy metrics are achieved on real futsal footage?
  - IDF1 (identity F1 score) for persistent tracking?
  - MOTA (Multiple Object Tracking Accuracy)?
  - Team assignment accuracy rate?
  - Jersey number recognition accuracy?

- **Q2:** Are there ground truth annotations for evaluation?
  - Manually annotated test videos?
  - How many frames/clips annotated?
  - Inter-annotator agreement metrics?

- **Q3:** How many ID switches occur per minute of gameplay?
  - Acceptable threshold for production use?
  - Comparison to baseline systems (DeepSORT, BoT-SORT)?

#### Architecture Decisions
- **Q4:** How was the 3-pass architecture validated vs. alternatives?
  - Was end-to-end deep learning considered?
  - Why separate tracking from identity assignment?
  - Performance/accuracy trade-offs documented?

- **Q5:** Why ByteTrack specifically vs. other trackers?
  - Comparison to BoT-SORT, StrongSORT, Deep OC-SORT?
  - Does ByteTrack's IoU+Kalman approach have limitations for futsal?

- **Q6:** Why no Re-ID (re-identification) model?
  - Design decision or future work?
  - Are jersey numbers considered sufficient for re-identification?

#### Scalability and Robustness
- **Q7:** What's the maximum video length the system can handle?
  - Full 40-minute halves?
  - Memory constraints for long videos?
  - Batch processing supported?

- **Q8:** How does the system perform with different camera angles?
  - Only overhead fisheye tested?
  - Side-view or corner-mounted cameras supported?
  - Multiple camera fusion possible?

- **Q9:** What happens when jersey numbers are ambiguous or OCR fails?
  - Fallback to appearance-only tracking?
  - Manual annotation interface?

- **Q10:** How are edge cases like player substitutions handled?
  - New player entering gets new ID (expected)?
  - Can the system detect substitution events?

---

### Current Deployment Context (Proof of Concept)

**Team Configuration:**
- **Orange Team**: Orange bibbed players (numbered jerseys)
- **Black Team**: Random shirts (white/green/black, mostly non-numbered)
- **Design Constraint**: System MUST support any combination, but there will ALWAYS be at least 1 bibbed team

**Tracking Requirements:**
1. **Numbered Jersey Players (Priority: HIGH)**
   - MUST track consistently across occlusions and frame exits
   - Required for advanced analytics: pass calculations, distance traveled, player-specific metrics
   - Identity persistence is critical

2. **Non-Numbered Players (Priority: MEDIUM)**
   - MUST have correct team assignment (orange vs. black)
   - Track jumping is acceptable (identity persistence not required)
   - Only team-level statistics needed

3. **Occlusion Handling (Priority: HIGH)**
   - Players should remain tracked even when occluded
   - Rationale: "Players don't just disappear"
   - Potential solutions: SAM2 segmentation, position estimation during occlusion

**Equipment Notes:**
- Current ball: White (potentially problematic on light court)
- Question: Would darker ball improve tracking?

---

## L - LOGIC

### Project Structure

```
futsal/
├── src/                              # Main source code
│   ├── __main__.py                   # Entry point
│   ├── cli.py                        # Command-line interface (Click framework)
│   ├── detection/                    # Detection and tracking modules
│   │   ├── player_detector.py        # YOLOv8x player detection
│   │   ├── ball_detector.py          # YOLOv11 ball detection + InferenceSlicer
│   │   ├── jersey_classifier.py      # YOLO-based jersey digit classification
│   │   ├── tracking.py               # ByteTrack multi-object tracking
│   │   ├── team_clustering.py        # K-means HSV color clustering
│   │   └── segmentation_sam2.py      # SAM2 segmentation (optional)
│   ├── geometry/                     # Spatial transformations
│   │   ├── homography.py             # Pixel → court coordinate mapping
│   │   └── pitch_drawing.py          # 2D court visualization
│   ├── passes/                       # Core 3-pass pipeline
│   │   ├── pass0_setup.py            # Run folder creation
│   │   ├── pass1_collect.py          # Raw evidence collection (339 lines)
│   │   ├── pass2_geometry.py         # Track fragment splitting (1074 lines)
│   │   ├── pass3_identity.py         # Identity inference (1163 lines)
│   │   └── pass_visualize.py         # Video annotation (699 lines)
│   ├── utils/                        # Utilities
│   │   ├── data_models.py            # Pydantic data models
│   │   ├── video_io.py               # Video reading/writing (PyAV)
│   │   └── visualization.py          # Rendering utilities
│   └── viz/                          # Visualization tools
│       └── pitch_2d_positions.py     # 2D pitch position plots
├── config/
│   └── default.yaml                  # Configuration file (160+ parameters)
├── models/                            # Trained model weights (~826 MB)
│   ├── PLAYER_MODEL_best_v1.pt       # Player detection model
│   ├── BALL_MODEL_best_v2.pt         # Ball detection model
│   ├── JERSEY_MODEL_best_v1.pt       # Jersey classification model
│   └── datasets/                     # Training data
├── tests/                             # Test directory (status unclear)
├── docs/
│   └── image.png                     # Fisheye overhead camera perspective
├── output/                            # Generated outputs (gitignored)
└── ARCHIVE/                           # Previous versions
```

---

### End-to-End Processing Pipeline

The system uses a **3-Pass Architecture** that separates tracking from identity assignment:

#### Pass 0: Setup
**Purpose:** Initialize run directory structure

**Processing:**
- Creates timestamped output folder: `output/run_DDMMYY_HHMMSS/`
- Creates subdirectories: `pass1_raw/`, `pass2_identity/`, `pass3_final/`

**Output:** Empty directory structure ready for pipeline

**File:** [src/passes/pass0_setup.py](../src/passes/pass0_setup.py)

---

#### Pass 1: Raw Evidence Collection
**Purpose:** Detect players/ball and track using IoU+motion only (NO identity assignment)

**Processing:**
1. **Player Detection** ([src/detection/player_detector.py](../src/detection/player_detector.py))
   - YOLOv8x model (`models/PLAYER_MODEL_best_v1.pt`)
   - Confidence threshold: 0.35 (low to catch partial occlusions)
   - NMS IoU threshold: 0.35
   - Max detections: 12

2. **Ball Detection** ([src/detection/ball_detector.py](../src/detection/ball_detector.py))
   - YOLOv11 model (`models/BALL_MODEL_best_v2.pt`)
   - **InferenceSlicer for small objects:** 2×2 grid with 200px overlap
   - Confidence threshold: 0.3
   - Centroid-based filtering with 10-frame buffer
   - Anomaly detection: filters detections >500px from recent centroid
   - **Status:** Ball tracker currently disabled (line 243)

3. **Multi-Object Tracking** ([src/detection/tracking.py](../src/detection/tracking.py))
   - **ByteTrack algorithm** (IoU + Kalman filter + velocity consistency)
   - High confidence threshold: 0.6
   - Low confidence threshold: 0.1
   - Track buffer: 30 frames (keeps lost tracks for re-association)
   - Distance gating: `max_distance = base_dist + (frames_lost * 20)`, capped at 300px
   - **Critical:** Track IDs are temporary and disposable (identity assigned in Pass 3)

4. **Jersey Classification** ([src/detection/jersey_classifier.py](../src/detection/jersey_classifier.py))
   - YOLO classification model for digits 0-99
   - Extracts top 50% of bounding box (jersey region)
   - Outputs probabilities per frame (no voting yet)
   - Confidence threshold: 0.4

5. **Appearance Feature Extraction**
   - HSV color histograms: 96 bins (32 H × 32 S × 32 V)
   - Jersey region: 20%-55% vertical crop, 80% width
   - Masking filters: background, court floor (H 25-90°), skin tones
   - Sampled every 5 frames for storage efficiency

6. **Occlusion Metrics**
   - Confidence scores tracked per frame
   - Used in Pass 2 for divergence detection

**Output:** `pass1_raw/<clip>.json` containing:
- Temporary track IDs and bounding boxes
- Centroids and confidence scores
- HSV histograms (sampled)
- Jersey detection probabilities
- Occlusion metrics

**File:** [src/passes/pass1_collect.py](../src/passes/pass1_collect.py) (339 lines)

---

#### Pass 2: Geometry + Track Fragment Splitting
**Purpose:** Split tracks at divergence points where identity likely changed

**Processing:**

1. **Homography Transformation** ([src/geometry/homography.py](../src/geometry/homography.py))
   - 13-point calibration mapping
   - Pixel coordinates → court coordinates (meters)
   - Futsal court: 40m × 20m

2. **Divergence Detection** ([src/passes/pass2_geometry.py](../src/passes/pass2_geometry.py):542-839)

   Four signal types detect when a track "jumped" between players:

   a. **Velocity Spikes** (lines 577-602)
      - Threshold: >5.0 m/s between frames
      - Detects "teleporting" or rapid unrealistic movement

   b. **BBox Position Jumps** (lines 604-621)
      - Threshold: >2.0 m position jump
      - Catches spatial discontinuities

   c. **Occlusion Spikes** (lines 623-639)
      - Threshold: confidence drop >0.6
      - Sustained low confidence indicates lost tracking

   d. **Appearance Drift** (lines 641-738)
      - Chi-squared distance on HSV histograms
      - Threshold: 5.0
      - **Jersey-Aware Logic** (lines 678-738):
        - Only splits when jersey **changes** (#4 → #7) or **disappears** (#4 → None)
        - **Suppresses split** when jersey **first appears** (None → #4) - player turned around
        - **Suppresses split** when same jersey persists with appearance change - lighting/angle

3. **Jersey Inconsistency Detection** (lines 740-812)
   - Detects jersey state changes within rolling windows
   - **Split triggers:**
     - Jersey disappears: #4 → None (track lost player)
     - Jersey changes: #7 → #4 (track jumped to different player)
   - **NOT a split trigger:**
     - Jersey appears: None → #4 (player turning around)

4. **Jersey Temporal Exclusivity** ([src/passes/pass2_geometry.py](../src/passes/pass2_geometry.py):285-539)
   - **Purpose:** Prevent "two Spyros" bug (same jersey on multiple players simultaneously)
   - **Detection:** Identifies when same jersey appears on different tracks at same time
   - **Strategy:** Split the fragment where jersey appeared LATER (it "stole" the jersey)
   - **Iteration:** Runs up to 5 times until convergence
   - No threshold filtering - directly detects temporal conflicts

5. **Fragment Coverage Strategy** (lines 251-367)
   - **Keeps ALL fragments** regardless of length (prevents grey "unknown" gaps)
   - Marks short fragments (<10 frames) as `low_quality`
   - **Merges consecutive short fragments** on same track
   - Balances 100% temporal coverage with clustering quality

6. **Primary Fragment Assignment** (lines 967-995)
   - Computes appearance centroid BEFORE any splits
   - Fragment closest to pre-split appearance → `is_primary_fragment=True`
   - Other fragments → `is_primary_fragment=False` (identity jumps)

**Output:** `pass2_identity/<clip>_fragments.json` containing:
- Track fragments with start/end frames
- Divergence annotations (reason + metric values)
- Court coordinates for all positions
- Fragment quality indicators
- Primary fragment markers

**File:** [src/passes/pass2_geometry.py](../src/passes/pass2_geometry.py) (1074 lines)

---

#### Pass 3: Identity Inference
**Purpose:** Assign team colors and jersey numbers to fragments

**Processing:**

1. **Pileup Detection** ([src/passes/pass3_identity.py](../src/passes/pass3_identity.py):24-97)
   - Spatial radius: 1.5m
   - Marks unreliable time windows when ≥4 players clustered
   - Used for team size constraint validation (max 6 per team)

2. **K-Means Team Clustering** ([src/detection/team_clustering.py](../src/detection/team_clustering.py) + pass3:139-173)
   - **Input:** HSV histograms from all fragments
   - **Algorithm:** K-means with k=2 clusters
   - **Run once** at start (prevents label flipping between clips)
   - **Variance-Based Team Detection:**
     - Lower variance cluster = uniform team (bibbed) = TEAM_A
     - Higher variance cluster = diverse team = TEAM_B
   - Samples 50 histograms per fragment for clustering

3. **Bidirectional Team Inheritance** ([src/passes/pass3_identity.py](../src/passes/pass3_identity.py))
   - **Rationale:** A player's team can't change mid-game!
   - **Forward:** Fragment with team_a → next fragment (unknown) inherits team_a
   - **Backward:** Fragment with team_a ← previous fragment (unknown) inherits team_a
   - Handles fragments without valid HSV histograms (e.g., during occlusion)

4. **Jersey Number Inference** ([src/passes/pass3_identity.py](../src/passes/pass3_identity.py):283-454)
   - **Eligibility Criteria:**
     - Minimum detections: 3 jersey sightings
     - Lock threshold: Cumulative confidence ≥ 0.7
     - Coverage requirement: Detections span ≥15% of fragment duration
     - Minimum stability: 20 frames
   - **Single-Owner Invariant:** Each jersey can belong to AT MOST ONE fragment at any time
   - **Greedy Conflict Resolution:** Highest evidence fragment wins
   - **Conflict Reasons:**
     - `no_eligible_jersey`: Failed threshold checks
     - `conflict_lost`: Lost to higher-evidence fragment
     - `all_jerseys_conflict_lost`: All candidates had conflicts

5. **Bidirectional Jersey Inheritance** ([src/passes/pass3_identity.py](../src/passes/pass3_identity.py):457-648)
   - **Forward inheritance** (lines 573-605): Earlier fragment with jersey → later inherits
   - **Backward inheritance** (lines 607-640): Later fragment with jersey → earlier inherits
   - **CRITICAL CONSTRAINT:** Respects temporal exclusivity
     - Checks `_has_temporal_conflict()` before assigning (lines 594, 629)
     - Skips inheritance if jersey already in use elsewhere at overlapping times
     - Prevents "two Spyros" bugs

6. **Crop Export**
   - Selects best quality crops per team for visualization
   - Filters by jersey mask quality

**Output:** `pass3_final/<clip>_final.json` containing:
- Final team assignments (team_a, team_b, unknown)
- Jersey numbers with confidence scores
- Fragment status (reliable/unreliable)
- Player identity mappings

**File:** [src/passes/pass3_identity.py](../src/passes/pass3_identity.py) (1163 lines)

---

#### Visualization Pass
**Purpose:** Render annotated videos with tracking results

**Processing:**
- Loads Pass 3 output + original videos
- Renders bounding boxes colored by team
- Displays jersey numbers and player names
- Shows track IDs and fragment IDs
- Marks divergence points from Pass 2
- Optional 2D pitch views:
  - Birdseye view (BV): Player positions on court
  - Voronoi overlay: Spatial dominance regions
- Customizable annotation scale

**Output:** Annotated MP4 videos

**File:** [src/passes/pass_visualize.py](../src/passes/pass_visualize.py) (699 lines)

---

### Technology Stack

#### Detection Frameworks
- **YOLOv8x** (Ultralytics): Player detection
  - Model: `models/PLAYER_MODEL_best_v1.pt`
  - Input: Full video frames
  - Output: Bounding boxes with confidence scores

- **YOLOv11** (Ultralytics): Ball detection
  - Model: `models/BALL_MODEL_best_v2.pt`
  - InferenceSlicer: 2×2 tiled inference for small objects
  - Output: Ball bounding boxes

- **YOLO Classifier**: Jersey digit recognition
  - Model: `models/JERSEY_MODEL_best_v1.pt`
  - Input: Cropped player regions (top 50%)
  - Output: Class probabilities for digits 0-99

#### Tracking Algorithm
- **ByteTrack** (custom implementation)
  - IoU-based data association
  - Kalman filter motion prediction (constant velocity model)
  - Velocity consistency checks
  - Distance gating with adaptive thresholds
  - 3-stage association:
    1. High confidence (≥0.6) detections
    2. Low confidence (0.1-0.6) detections
    3. Lost track reactivation
  - Track buffer: 30 frames
  - Modified for Pass 1: NO team gating, NO appearance matching

#### Appearance & Clustering
- **HSV Color Histograms**
  - 96-bin feature vector (32 H × 32 S × 32 V)
  - Normalized and quantized
  - Chi-squared distance for comparison
  - Storage optimization: sampled every 5 frames

- **K-means Clustering** (scikit-learn)
  - k=2 for two teams
  - Fitted once on all fragments
  - Variance-based uniform team detection

#### Spatial Transformation
- **Homography Matrix** (OpenCV)
  - 13-point calibration
  - Maps pixel coords → court coords (meters)
  - 40m × 20m futsal court
  - 2D pitch scale: 20 pixels per meter (800×400 output)

#### Computer Vision Libraries
- **OpenCV**: Image processing, homography, histogram computation
- **Supervision**: Detection filtering, annotation rendering
- **Scikit-learn**: K-means clustering
- **NumPy/SciPy**: Linear assignment (Hungarian algorithm)
- **SAM2** (optional): Segmentation for occlusion recovery

#### Video I/O
- **PyAV**: Video reading/writing with codec support
- **Pillow**: Image processing

#### Utilities
- **Pydantic**: Data validation and type-safe models
- **YAML**: Configuration management
- **Click**: CLI framework
- **tqdm**: Progress bars

---

### Data Flow Summary

```
Raw Video (Fisheye Overhead)
         ↓
[Pass 1: Raw Evidence Collection]
- YOLOv8x player detection
- YOLOv11 ball detection (InferenceSlicer)
- ByteTrack (temporary IDs)
- Jersey classification (probabilities)
- HSV histogram extraction
         ↓
pass1_raw/<clip>.json (temporary track IDs, detections, histograms)
         ↓
[Pass 2: Geometry + Fragmentation]
- Homography transformation
- 4-type divergence detection
- Jersey temporal exclusivity
- Fragment splitting & merging
- Primary fragment assignment
         ↓
pass2_identity/<clip>_fragments.json (fragments with divergence markers)
         ↓
[Pass 3: Identity Inference]
- K-means team clustering (variance-based)
- Bidirectional team inheritance
- Jersey number inference (greedy conflict resolution)
- Bidirectional jersey inheritance (with temporal checks)
- Team size constraint validation
         ↓
pass3_final/<clip>_final.json (final identities: team + jersey)
         ↓
[Visualization]
- Annotate original video
- Optional 2D pitch views
         ↓
Annotated MP4 + tracking JSON
```

---

### Key Configuration Parameters

From [config/default.yaml](../config/default.yaml) (160+ parameters):

**Detection:**
- `player_confidence: 0.35` - Low to catch partial occlusions
- `ball_confidence: 0.3` - Catches partially visible balls
- `jersey_confidence: 0.4` - Jersey detection threshold
- `nms_iou: 0.35` - Non-max suppression
- `max_detections: 12` - Maximum players per frame

**Tracking (Pass 1):**
- `track_thresh_high: 0.5` - High confidence matching
- `track_thresh_low: 0.1` - Low confidence matching
- `track_buffer: 30` - Frames to keep lost tracks
- `match_threshold: 0.6` - IoU threshold for association
- `max_center_distance: 100` - Base distance gating (pixels)

**Pass 2 Divergence:**
- `appearance_threshold: 5.0` - Chi-squared distance
- `velocity_spike_threshold: 5.0` - m/s
- `bbox_jump_threshold: 2.0` - meters
- `occlusion_spike_threshold: 0.6` - Confidence drop
- `jersey_temporal_window: 150` - Frames for exclusivity check

**Pass 3 Identity:**
- `team_cap: 6` - Max concurrent players per team
- `kmeans_samples_per_fragment: 50` - Histogram sampling
- `jersey_lock_threshold: 1.5` - Cumulative confidence (note: code uses 0.7)
- `jersey_min_detections: 3` - Minimum sightings
- `jersey_min_coverage: 0.15` - 15% of fragment duration
- `jersey_min_stability: 20` - Minimum frames

**Homography:**
- `court_width: 40.0` - meters
- `court_height: 20.0` - meters
- `pixels_per_meter: 20` - 2D pitch scale

---

### CLI Commands

```bash
# Full pipeline (all passes)
python -m src.cli run-full --input-dir videos/input --output-dir output

# Individual passes
python -m src.cli pass1 --input-dir videos/input --output-dir output
python -m src.cli pass2 --run-dir output/run_060226_124530
python -m src.cli pass3 --run-dir output/run_060226_124530

# Visualization
python -m src.cli visualize --run-dir output/run_060226_124530 --2d BV

# 2D pitch views:
#   BV  = Birdseye view
#   VOI = Voronoi overlay
```

---

## E - EXAMPLES (Edge Case Analysis)

This section examines how the codebase handles five critical edge case categories common in futsal tracking, citing specific file locations and functions.

---

### 3.1 Team Allocation

#### How It Works

**Algorithm:** K-means clustering on HSV color histograms

**File:** [src/detection/team_clustering.py](../src/detection/team_clustering.py) + [src/passes/pass3_identity.py](../src/passes/pass3_identity.py):139-173

**Process:**
1. **Feature Extraction:**
   - Extract jersey region: 20%-55% vertical crop, 80% width of bounding box
   - Compute 96-bin HSV histogram (32 H × 32 S × 32 V)
   - Apply masking filters:
     - Background suppression
     - Court floor (H 25-90°)
     - Skin tones
     - Overexposed whites

2. **Clustering:**
   - K-means with k=2 (two teams)
   - Run ONCE on all fragments at start of Pass 3
   - Sample 50 histograms per fragment for efficiency

3. **Team Identification** (variance-based):
   ```python
   lower_variance_cluster = uniform team (bibbed) → TEAM_A
   higher_variance_cluster = diverse team → TEAM_B
   ```

4. **Assignment:**
   - Map each fragment to team_a or team_b based on cluster label
   - Fragments without valid histograms get "unknown" team

5. **Bidirectional Inheritance:**
   - **Forward:** Fragment with team_a → next fragment (unknown) inherits team_a
   - **Backward:** Fragment with unknown ← next fragment (team_a) inherits team_a
   - **Rationale:** Player's team can't change mid-game

6. **Validation:**
   - Team size constraint: max 6 concurrent players per team ([src/passes/pass3_identity.py](../src/passes/pass3_identity.py):24-97)
   - Pileup detection: spatial radius 1.5m, marks unreliable windows

#### Edge Cases HANDLED ✅

1. **Fragments with Missing/Invalid Histograms**
   - **Scenario:** Severe occlusion or player exiting frame
   - **Solution:** Bidirectional team inheritance from adjacent fragments on same track
   - **Code:** [src/passes/pass3_identity.py](../src/passes/pass3_identity.py) (team inheritance functions)

2. **Asymmetric Team Uniforms**
   - **Scenario:** One team bibbed (uniform), other team diverse clothing
   - **Solution:** Variance-based detection identifies uniform team as bibbed
   - **Code:** [src/detection/team_clustering.py](../src/detection/team_clustering.py):468-495

3. **Team Size Violations**
   - **Scenario:** Fragmentation bugs create impossible team sizes (>6 players)
   - **Solution:** Validation detects and logs violations for debugging
   - **Code:** [src/passes/pass3_identity.py](../src/passes/pass3_identity.py):24-97

#### Edge Cases PARTIALLY Handled ⚠️

1. **Similar Team Colors**
   - **Problem:** K-means may mis-cluster when both teams wear similar colors
   - **Current:** No fallback mechanism
   - **Risk:** Team labels may be incorrect or flicker
   - **Mitigation Needed:** Multi-cue assignment (color + spatial + jersey range)

2. **Lighting Changes Across Court**
   - **Problem:** Same jersey color appears different under different court lighting
   - **Current:** Single K-means clustering doesn't account for spatial lighting variation
   - **Risk:** Players may be mis-assigned to wrong team
   - **Mitigation Needed:** Spatial-aware clustering or lighting normalization

#### Edge Cases NOT Handled ❌

1. **Manual Team Annotation Fallback**
   - **Problem:** No way to correct K-means errors
   - **Missing:** GUI or config-based manual team annotation
   - **Impact:** Incorrect team assignments propagate through entire video

2. **Team Assignment Confidence Scores**
   - **Problem:** No quality metric exported for team assignments
   - **Missing:** Confidence scores based on histogram distance to cluster centers
   - **Impact:** Can't filter unreliable team assignments

#### Failure Modes

- **Failure:** K-means converges to suboptimal solution when colors are ambiguous
  - **Symptom:** Team A and Team B labels swapped
  - **Frequency:** Occasional, depends on color similarity
  - **Detection:** Manual review required

- **Failure:** Fragments during severe occlusion get "unknown" team
  - **Symptom:** Grey fragments in visualization between colored fragments
  - **Frequency:** Rare after bidirectional inheritance fix
  - **Detection:** Visualization shows gaps

- **Failure:** No ground truth validation
  - **Symptom:** No quantitative accuracy measurement
  - **Impact:** Can't optimize thresholds or validate improvements

---

### 3.2 Persistent Player Identity

#### How It Works

**Architecture:** 3-pass separation of tracking from identity

**Files:**
- [src/passes/pass2_geometry.py](../src/passes/pass2_geometry.py) (1074 lines) - Divergence detection
- [src/passes/pass3_identity.py](../src/passes/pass3_identity.py) (1163 lines) - Identity inference

**Mechanisms:**

**1. Appearance Drift Detection** ([src/passes/pass2_geometry.py](../src/passes/pass2_geometry.py):641-738)
- Chi-squared distance on HSV histograms
- Threshold: 5.0
- **Jersey-Aware Logic:**
  ```python
  # SPLITS when:
  if jersey_changes(prev=#4, curr=#7):  # Track jumped players
      split_track()
  if jersey_disappears(prev=#4, curr=None):  # Track lost player
      split_track()

  # SUPPRESSES SPLIT when:
  if jersey_first_appears(prev=None, curr=#4):  # Player turned around
      continue_track()  # Do NOT split
  if same_jersey_persists(prev=#4, curr=#4):  # Lighting change
      continue_track()  # Do NOT split
  ```

**2. Jersey-Aware Splitting** ([src/passes/pass2_geometry.py](../src/passes/pass2_geometry.py):740-812)
- Detects jersey state changes within rolling windows
- Only splits on jersey **change** or **disappearance**
- Does NOT split on jersey **first appearance**

**3. Jersey Temporal Exclusivity** ([src/passes/pass2_geometry.py](../src/passes/pass2_geometry.py):285-539)
- Detects same jersey on multiple tracks simultaneously
- Splits fragment where jersey appeared LATER (the "thief")
- Iterates up to 5 times until convergence
- Prevents "two Spyros" bug

**4. Primary Fragment Marking** ([src/passes/pass2_geometry.py](../src/passes/pass2_geometry.py):967-995)
- Computes appearance centroid before splits
- Fragment closest to centroid → `is_primary_fragment=True`
- Other fragments → `is_primary_fragment=False`

**5. Jersey Inheritance** ([src/passes/pass3_identity.py](../src/passes/pass3_identity.py):457-648)
- **Forward** (lines 573-605): Earlier fragment → later fragment
- **Backward** (lines 607-640): Later fragment → earlier fragment
- **Temporal Exclusivity Check** (lines 594, 629):
  ```python
  if _has_temporal_conflict(jersey, fragment):
      skip_inheritance()  # Jersey already in use elsewhere
  ```

#### Edge Cases HANDLED ✅

1. **Jersey First Appearance (Player Turning)**
   - **Scenario:** Player with back to camera turns around, jersey becomes visible (None → #4)
   - **Solution:** Appearance drift detection suppresses split on jersey first appearance
   - **Code:** [src/passes/pass2_geometry.py](../src/passes/pass2_geometry.py):678-738
   - **Why:** This is normal behavior, not a track jump

2. **Brief Occlusions**
   - **Scenario:** Jersey temporarily hidden (≤30 frames), then visible again
   - **Solution:** Jersey inheritance propagates number forward through gap
   - **Code:** [src/passes/pass3_identity.py](../src/passes/pass3_identity.py):573-605
   - **Result:** Identity maintained through occlusion

3. **"Two Spyros" Bug Prevention**
   - **Scenario:** Same jersey #4 appears on two tracks simultaneously
   - **Solution:** Temporal exclusivity detection splits the fragment that "stole" the jersey
   - **Code:** [src/passes/pass2_geometry.py](../src/passes/pass2_geometry.py):285-539
   - **Result:** Each jersey belongs to at most one player at any time

4. **Jersey Backward Inheritance**
   - **Scenario:** Player enters frame with back to camera, then turns to show jersey #10
   - **Solution:** Jersey #10 inherited backward to earlier frames of same track
   - **Code:** [src/passes/pass3_identity.py](../src/passes/pass3_identity.py):607-640
   - **Result:** Entire track gets jersey #10, not just frames after turning

#### Edge Cases PARTIALLY Handled ⚠️

1. **Long Occlusions (>30 frames)**
   - **Problem:** ByteTrack deletes lost tracks after 30 frames
   - **Current:** Track ends, new track created on re-appearance
   - **Code:** [src/detection/tracking.py](../src/detection/tracking.py):362-370
   - **Risk:** Player gets new ID after extended occlusion
   - **Mitigation Needed:** Re-ID embeddings for long-term re-association

2. **Multiple Players with Same Jersey (Different Teams)**
   - **Problem:** Both teams may have players wearing #4
   - **Current:** Team assignment happens separately from jersey assignment
   - **Risk:** Jersey inheritance may propagate wrong number if teams not yet assigned
   - **Mitigation:** Team assignment runs before jersey inference in Pass 3

3. **Jersey OCR Errors**
   - **Problem:** Misread jersey number (e.g., 8 detected as 3)
   - **Current:** Error propagates to entire fragment via jersey inference
   - **Code:** [src/passes/pass3_identity.py](../src/passes/pass3_identity.py):283-454
   - **Risk:** Entire fragment gets incorrect identity
   - **Mitigation Needed:** Cross-validation across fragments

#### Edge Cases NOT Handled ❌

1. **Re-ID After Track Deletion (>30 frames)**
   - **Problem:** No appearance embeddings for long-term re-identification
   - **Missing:** Deep Re-ID model (ResNet/OSNet) to match players after long gaps
   - **Impact:** Players get new IDs after substitutions or long off-screen periods

2. **Players Swapping Jerseys Mid-Game**
   - **Problem:** System assumes jersey numbers are stable
   - **Missing:** Detection of jersey swap events
   - **Impact:** System would fail if players exchange jerseys

3. **Identity Confidence Scores**
   - **Problem:** No quality metric for final player identity assignments
   - **Missing:** Aggregate confidence from jersey detection, team clustering, inheritance
   - **Impact:** Can't filter low-confidence identities

#### Failure Modes

- **ID Switch During Extended Occlusion**
  - **Trigger:** Player occluded or off-screen >30 frames
  - **Result:** New track ID assigned on re-appearance
  - **Frequency:** Common in fast gameplay with frequent frame exits
  - **Workaround:** None currently

- **Jersey OCR Error "Locked In"**
  - **Trigger:** Misread jersey number has highest cumulative confidence
  - **Result:** Entire fragment assigned wrong jersey number
  - **Frequency:** Depends on OCR accuracy (not documented)
  - **Workaround:** None currently

- **Fragments Without Jersey Numbers**
  - **Trigger:** Fragment too short (<3 jersey detections) or jersey never visible
  - **Result:** Fragment marked as `no_eligible_jersey`
  - **Frequency:** Common for players who never turn around
  - **Workaround:** Bidirectional inheritance from adjacent fragments

---

### 3.3 Occlusion Handling

#### How It Works

**Algorithm:** ByteTrack 3-stage association with Kalman filtering

**File:** [src/detection/tracking.py](../src/detection/tracking.py)

**Stage 1: High-Confidence Matching** (lines 209-233)
- Associates detections with conf ≥ 0.6 to active tracks
- Uses IoU + velocity consistency + distance gating
- Hungarian algorithm for optimal assignment
- Kalman predictions guide matching

**Stage 2: Low-Confidence Association** (lines 235-254)
- Associates detections with 0.1 ≤ conf < 0.6 to unmatched tracks
- **Critical:** Catches partially occluded players with reduced confidence
- Prevents track loss during partial occlusion

**Stage 3: Lost Track Reactivation** (lines 256-310)
- Re-associates high-conf detections with recently lost tracks
- Distance gating formula:
  ```python
  max_distance = max_center_distance + (frames_lost * 20)
  absolute_max = min(max_distance, 300)  # Capped at 300 pixels
  ```
- Accounts for time passed: longer lost = larger search radius
- Prevents "teleporting" by capping absolute max distance

**Kalman Filter** (lines 40-101)
- Constant velocity motion model
- Measurement noise scaled by bbox size:
  ```python
  noise_scale = max(w, h) / 100.0
  ```
- Larger boxes tolerate more pixel noise
- Predicts where tracks should appear next frame

**Track Buffer Management** (lines 362-370)
```python
track_buffer = 30  # Frames to keep lost tracks
for track in lost_tracks:
    if track.frames_since_last_detection > track_buffer:
        remove_track(track)
```

#### Edge Cases HANDLED ✅

1. **Partial Occlusions**
   - **Scenario:** Player half-occluded by another player, confidence drops to 0.3
   - **Solution:** Stage 2 low-confidence matching associates detection with existing track
   - **Code:** [src/detection/tracking.py](../src/detection/tracking.py):235-254
   - **Result:** Track maintained despite low confidence

2. **Brief Complete Occlusions (≤30 frames)**
   - **Scenario:** Player completely hidden for 20 frames
   - **Solution:** Track buffer keeps lost track; reactivated when player re-appears
   - **Code:** [src/detection/tracking.py](../src/detection/tracking.py):256-310
   - **Result:** Same track ID maintained through occlusion

3. **Kalman Prediction During Gaps**
   - **Scenario:** Player not detected for 5 frames
   - **Solution:** Kalman filter predicts position; used for re-association
   - **Code:** [src/detection/tracking.py](../src/detection/tracking.py):40-101
   - **Result:** Track reactivated near predicted position

4. **Distance Gating Prevents ID Swaps**
   - **Scenario:** Two players close together, one briefly occluded
   - **Solution:** Distance gating ensures track only reactivates within reasonable radius
   - **Code:** [src/detection/tracking.py](../src/detection/tracking.py):284-286
   - **Result:** Occluded player doesn't "jump" to nearby player's detection

#### Edge Cases PARTIALLY Handled ⚠️

1. **Extended Occlusions (>30 frames)**
   - **Problem:** Track deleted after 30 frames
   - **Current:** New track created when player re-appears
   - **Risk:** Player gets new ID, identity lost
   - **Mitigation Needed:** Longer track buffer or Re-ID-based re-association

2. **Dense Clusters (4+ players within 1m)**
   - **Problem:** IoU matching may fail when bboxes heavily overlap
   - **Current:** Low-confidence matching helps, but ID swaps still possible
   - **Risk:** Players swap IDs during dense clustering
   - **Mitigation Needed:** Appearance-based matching or segmentation (SAM2)

3. **Occlusion Reasoning**
   - **Problem:** No explicit reasoning about which player is in front
   - **Current:** Relies on detection confidence as proxy for occlusion severity
   - **Risk:** May lose track of player further from camera
   - **Mitigation Needed:** Depth estimation or explicit occlusion modeling

#### Edge Cases NOT Handled ❌

1. **Adaptive Track Buffer**
   - **Problem:** Fixed 30-frame buffer may be too short for some scenarios
   - **Missing:** Context-aware buffer (longer near frame edges, shorter in center)
   - **Impact:** Players frequently exiting frame lose IDs unnecessarily

2. **Appearance-Based Matching**
   - **Problem:** ByteTrack modified to use NO appearance features
   - **Missing:** HSV histogram matching during association (by design)
   - **Impact:** Relies purely on IoU + velocity, may fail in crowded scenes
   - **Rationale:** Appearance assignment deferred to Pass 3 (architectural decision)

3. **Segmentation-Based Tracking**
   - **Problem:** Bounding boxes overlap during occlusion
   - **Missing:** SAM2 integration unclear (marked "optional")
   - **Impact:** Can't separate overlapping players at pixel level

#### Failure Modes

- **Track Loss After 30 Frames**
  - **Trigger:** Player occluded or off-screen >30 frames
  - **Result:** Track deleted, new ID assigned on re-appearance
  - **Frequency:** Common in futsal (fast gameplay, small court)
  - **Example:** Player goes to sideline for substitution

- **ID Swap During Dense Clustering**
  - **Trigger:** Multiple players within 1m, bboxes overlap >80%
  - **Result:** IoU matching fails, tracks swap IDs
  - **Frequency:** Occasional during corner kicks or defensive clustering
  - **Example:** Two defenders marking an attacker

- **Kalman Drift During Long Occlusion**
  - **Trigger:** Player occluded 20-30 frames with unpredictable movement
  - **Result:** Kalman prediction drifts from actual position
  - **Frequency:** Rare (most occlusions <20 frames)
  - **Example:** Player falls, stays down, then gets up in different location

---

### 3.4 Players Entering/Leaving Frame

#### How It Works

**Algorithm:** Lost track re-association with lenient distance thresholds

**File:** [src/detection/tracking.py](../src/detection/tracking.py)

**Lost Track Re-Association** (lines 321-349)
```python
# When high-confidence detection appears:
for lost_track in recently_lost_tracks:
    distance = calculate_distance(detection, lost_track.predicted_position)
    max_allowed = 150 + (15 * lost_track.frames_lost)  # Lenient threshold
    max_allowed = min(max_allowed, 250)  # Capped at 250px

    if distance < max_allowed:
        reactivate_track(lost_track, detection)
        break
else:
    # No nearby lost track found
    if detection.confidence >= 0.7:
        create_new_track(detection)
```

**Track State Machine:**
```
tracked → (no match for N frames) → lost → (reactivated OR >30 frames) → removed
```

**Priority System:**
1. Active tracks get first priority for matching
2. Lost tracks get second priority for high-conf detections
3. New tracks created only if no nearby lost tracks exist

**Frame Exit Detection:**
- Implicit: Track becomes "lost" when no matching detection
- No explicit frame boundary detection

#### Edge Cases HANDLED ✅

1. **Brief Frame Exits (≤30 frames)**
   - **Scenario:** Player steps out of frame for 15 frames, then returns
   - **Solution:** Lost track re-association within lenient distance threshold
   - **Code:** [src/detection/tracking.py](../src/detection/tracking.py):321-349
   - **Result:** Same track ID maintained

2. **Priority for Recently-Lost Tracks**
   - **Scenario:** Player re-enters frame near existing active tracks
   - **Solution:** System checks lost tracks first before creating new track
   - **Code:** [src/detection/tracking.py](../src/detection/tracking.py):321-349
   - **Result:** Prevents duplicate IDs for same player

3. **Adaptive Distance Threshold**
   - **Scenario:** Player lost for 20 frames may have moved far from last position
   - **Solution:** Distance threshold increases with frames lost: `150 + 15*frames`
   - **Code:** [src/detection/tracking.py](../src/detection/tracking.py):321-349
   - **Result:** Accommodates longer-range movement during absence

4. **Distance Gating Prevents Incorrect Re-Association**
   - **Scenario:** Two players exit frame; one returns
   - **Solution:** Capped max distance (250px) prevents matching to wrong lost track
   - **Code:** [src/detection/tracking.py](../src/detection/tracking.py):321-349
   - **Result:** Players don't swap IDs when entering from different locations

#### Edge Cases PARTIALLY Handled ⚠️

1. **Players Exiting >30 Frames**
   - **Problem:** Lost tracks deleted after 30-frame buffer
   - **Current:** New track ID assigned on return
   - **Risk:** Player identity lost for substitutions or long sideline periods
   - **Mitigation Needed:** Trajectory-based re-association or Re-ID embeddings

2. **Two Players Entering at Same Location**
   - **Problem:** If two players exit nearby and return nearby, may swap IDs
   - **Current:** Distance-based matching can't distinguish
   - **Risk:** ID swaps when multiple players have similar entry trajectories
   - **Mitigation Needed:** Appearance-based disambiguation

#### Edge Cases NOT Handled ❌

1. **Trajectory-Based Re-Association**
   - **Problem:** No memory of exit trajectory or jersey number
   - **Missing:** Store (exit_location, velocity_vector, jersey_number) for deleted tracks
   - **Missing:** Match re-entering players by trajectory + jersey
   - **Impact:** Players re-entering after >30 frames get new IDs
   - **Example:** Player substitutes out and back in (40 frames later)

2. **Player Registry**
   - **Problem:** No global registry of players who left the court
   - **Missing:** Persistent storage of player appearances + jerseys
   - **Impact:** Can't recover identity after long absences
   - **Example:** Can't detect when same player returns after halftime

3. **Jersey-Based Re-Association**
   - **Problem:** Lost track re-association doesn't use jersey numbers
   - **Current:** Only uses spatial distance and time
   - **Missing:** Check if re-entering player has same jersey as lost track
   - **Impact:** May miss correct re-association if player moved far
   - **Note:** Jersey inheritance in Pass 3 partially compensates

#### Failure Modes

- **New ID After Substitution**
  - **Trigger:** Player exits for >30 frames (substitution or sideline)
  - **Result:** New track ID assigned on return
  - **Frequency:** Common (expected behavior, but may confuse analysis)
  - **Example:** Player #4 goes to sideline at frame 1000, returns at frame 1050
    - Track ends at ~frame 1030 (30-frame buffer)
    - New track created at frame 1050
    - Jersey inheritance (Pass 3) may recover jersey #4, but track ID different

- **ID Swap on Simultaneous Entry**
  - **Trigger:** Two players exit nearby, return nearby within 30 frames
  - **Result:** Lost tracks may re-associate to wrong detections
  - **Frequency:** Rare (requires specific timing and positioning)
  - **Example:** Two defenders exit bottom of frame, return moments later
    - If they swap positions during absence, IDs may swap

- **Lost Identity After Long Absence**
  - **Trigger:** Player absent >30 frames
  - **Result:** Track deleted, identity lost permanently
  - **Frequency:** Common for substitutions
  - **Example:** Player substituted out for 2 minutes
    - No way to recover identity when they return

---

### 3.5 Ball Tracking

#### How It Works

**Algorithm:** Dedicated ball tracker with centroid-based filtering

**File:** [src/detection/ball_detector.py](../src/detection/ball_detector.py)

**Ball Detection** (lines 19-69):
- Separate YOLOv11 model trained on futsal balls
- Model: `models/BALL_MODEL_best_v2.pt`
- Confidence threshold: 0.3 (low to catch partially visible balls)
- IoU threshold: 0.1 (allows dense NMS for small objects)

**InferenceSlicer for Small Objects** (lines 119-159):
- **Problem:** Ball is small object in high-res video, easily missed
- **Solution:** Tiled inference
  - 2×2 grid with 200px overlap
  - Tile size: `(frame_width//2 + 200) × (frame_height//2 + 200)` pixels
  - Per-tile inference with NMS aggregation
  - Roboflow-recommended approach for small object detection

**Centroid-Based Tracker** (lines 19-69):
- Maintains deque buffer of recent ball positions (10 frames)
- Computes historical centroid from buffer
- **Selection Strategy:**
  ```python
  if multiple_detections:
      select detection closest to historical_centroid
  ```
- **Anomaly Filtering:**
  ```python
  if distance_from_centroid > 500px:
      reject_as_false_positive()
  ```

**Current Status:**
- **Ball tracker DISABLED** (line 243)
- Using raw YOLO model output for validation
- Tracker exists but needs debugging before re-enable

#### Edge Cases HANDLED ✅

1. **Fast Ball Motion**
   - **Scenario:** Ball moving rapidly across frame (kick, pass)
   - **Solution:** InferenceSlicer breaks frame into tiles, each processed independently
   - **Code:** [src/detection/ball_detector.py](../src/detection/ball_detector.py):119-159
   - **Result:** Ball detected even with motion blur, as tiles are higher resolution

2. **Partial Ball Visibility**
   - **Scenario:** Ball partially occluded by player's foot or body
   - **Solution:** Low confidence threshold (0.3) catches marginal detections
   - **Code:** [src/detection/ball_detector.py](../src/detection/ball_detector.py)
   - **Result:** Ball detected even when <50% visible

3. **False Positive Filtering**
   - **Scenario:** White court markings detected as ball
   - **Solution:** Centroid proximity filter rejects detections >500px from recent centroid
   - **Code:** [src/detection/ball_detector.py](../src/detection/ball_detector.py):19-69
   - **Result:** Anomalous detections discarded

4. **Tile Overlap Prevents Edge Detection Failures**
   - **Scenario:** Ball falls exactly on tile boundary
   - **Solution:** 200px overlap ensures ball appears fully in at least one tile
   - **Code:** [src/detection/ball_detector.py](../src/detection/ball_detector.py):119-159
   - **Result:** Ball detected regardless of position in frame

#### Edge Cases PARTIALLY Handled ⚠️

1. **Ball Occluded by Players**
   - **Problem:** Ball completely hidden by dense player cluster
   - **Current:** Detection fails, centroid buffer maintains last known position
   - **Risk:** Ball tracking lost during extended occlusion (>10 frames buffer)
   - **Mitigation Needed:** Kalman filter or physics-based prediction

2. **Ball Exiting Frame**
   - **Problem:** No explicit handling of ball leaving/re-entering frame
   - **Current:** Creates new detection when ball returns
   - **Risk:** Ball ID may change (if tracker enabled)
   - **Mitigation:** Only one ball exists, so ID changes less critical

#### Edge Cases NOT Handled ❌

1. **Physics-Based Motion Prediction**
   - **Problem:** No Kalman filter or trajectory prediction for ball
   - **Missing:** Constant-acceleration motion model (unlike players)
   - **Impact:** Can't predict ball position during occlusion
   - **Comparison:** Players have Kalman filters; ball does not

2. **Ball-Player Interaction Modeling**
   - **Problem:** No explicit tracking of ball possession
   - **Missing:** When ball disappears, predict it's possessed by nearest player
   - **Missing:** Search near possessing player on ball re-appearance
   - **Impact:** Ball tracking lost during dribbling or close control

3. **Multiple Ball Detection Handling**
   - **Problem:** System assumes only one ball
   - **Missing:** Handling of spurious ball detections (advertising, logos)
   - **Impact:** May select wrong detection if multiple balls detected
   - **Note:** Centroid filtering mitigates this partially

4. **Ball Tracker Currently Disabled**
   - **Problem:** Code exists but disabled (line 243)
   - **Status:** Using raw model output, no frame-to-frame tracking
   - **Impact:** Ball ID may change every frame
   - **Action Needed:** Debug and re-enable tracker

#### Failure Modes

- **Ball Lost During Dense Clustering**
  - **Trigger:** Ball surrounded by 4+ players, completely occluded
  - **Result:** No detection for >10 frames, centroid buffer expires
  - **Frequency:** Common during corner kicks or goal-line scrambles
  - **Example:** Ball at player's feet during dribbling through defenders

- **False Positive on White Court Markings**
  - **Trigger:** White ball on white court lines or center circle
  - **Result:** Court marking detected as ball (low confidence)
  - **Frequency:** Occasional, depends on camera angle
  - **Mitigation:** Centroid filter rejects if far from last position

- **Ball Tracking Disabled**
  - **Trigger:** Current code configuration (line 243)
  - **Result:** Ball ID changes frame-to-frame
  - **Frequency:** Always (until tracker re-enabled)
  - **Impact:** Can't track ball possession or trajectory

- **No Motion Prediction**
  - **Trigger:** Ball occluded for 5-10 frames
  - **Result:** No prediction of where ball should re-appear
  - **Frequency:** Common during fast gameplay
  - **Example:** Ball kicked through cluster of players
    - Players tracked via Kalman
    - Ball not predicted, must be re-detected

---

### Edge Case Summary Table

| Category | ✅ Handled | ⚠️ Partial | ❌ Missing | Failure Mode |
|----------|-----------|-----------|-----------|--------------|
| **Team Allocation** | Variance detection, Bidirectional inheritance, Team size validation | Similar colors, Lighting variation | Manual annotation, Confidence scores | K-means suboptimal clustering |
| **Persistent Identity** | Jersey first appearance, Temporal exclusivity, Bidirectional inheritance | Long occlusions (>30f), OCR errors | Re-ID embeddings, Identity confidence | ID switches after 30 frames |
| **Occlusion** | 3-stage association, Kalman filter, Track buffer | Extended occlusions, Dense clusters | Adaptive buffer, Appearance matching | Track loss after 30 frames |
| **Enter/Exit Frame** | Lost track re-association, Adaptive distance, Priority system | Long exits (>30f), Multiple simultaneous | Trajectory matching, Player registry | New ID after substitution |
| **Ball Tracking** | InferenceSlicer, Partial visibility, Centroid filtering | Player occlusion, Frame exit | Kalman prediction, Ball-player interaction, **Tracker disabled** | Lost during clustering |

---

## A - ADAPT (Proposed Improvements)

This section proposes prioritized technical improvements to address identified gaps and failure modes, with specific implementation details and measurable outcomes.

---

### Priority 1: Tracking Robustness

#### P1.1: Add Re-ID Embeddings for Long Occlusions

**Problem:** Players occluded or off-screen >30 frames get new track IDs

**Solution:** Extract appearance embeddings during Pass 1, use for track re-association in Pass 2

**Implementation:**
- **Files to modify:**
  - [src/detection/player_detector.py](../src/detection/player_detector.py) - Add embedding extraction (ResNet50 or OSNet)
  - [src/passes/pass1_collect.py](../src/passes/pass1_collect.py) - Store embeddings in JSON (256-dim vectors)
  - [src/passes/pass2_geometry.py](../src/passes/pass2_geometry.py) - Add Re-ID-based fragment merging

- **Algorithm:**
  ```python
  for fragment_a, fragment_b in candidate_pairs:
      if temporal_gap(fragment_a, fragment_b) > 30_frames:
          similarity = cosine_similarity(emb_a, emb_b)
          if similarity >= reid_similarity_threshold:
              merge_fragments(fragment_a, fragment_b)
  ```

- **Config changes:** Add to `config/default.yaml`:
  ```yaml
  reid:
    model: "osnet_x1_0"  # or "resnet50"
    similarity_threshold: 0.7
    max_temporal_gap: 180  # frames (6 seconds)
  ```

**Expected Outcome:** Reduce ID switches by ≥40% on 10-minute test clip with heavy occlusion

**Validation:** Compare IDF1 scores before/after on annotated test videos

---

#### P1.2: Implement Trajectory-Based Track Association

**Problem:** Players exiting frame for >30 frames lose identity permanently

**Solution:** Store track trajectories with jersey numbers; match re-appearing players by (jersey + entry location + velocity)

**Implementation:**
- **Files to modify:**
  - [src/passes/pass2_geometry.py](../src/passes/pass2_geometry.py) - Add trajectory history storage
    - Store: (exit_frame, exit_position, exit_velocity, jersey_number)
  - [src/passes/pass3_identity.py](../src/passes/pass3_identity.py) - Implement trajectory matching
    - For fragments starting after gaps, search trajectory history
    - Match by: spatial proximity + velocity direction + jersey number

- **Algorithm:**
  ```python
  for new_fragment in fragments_after_gaps:
      for trajectory in trajectory_history:
          if frame_gap(trajectory, new_fragment) <= 180:
              score = (
                  spatial_score(trajectory.exit_pos, new_fragment.entry_pos) * 0.4 +
                  velocity_score(trajectory.exit_vel, new_fragment.entry_vel) * 0.3 +
                  jersey_score(trajectory.jersey, new_fragment.jersey) * 0.3
              )
              if score >= 0.7:
                  link_fragments(trajectory.fragment, new_fragment)
  ```

- **Config changes:**
  ```yaml
  trajectory_matching:
    max_gap_frames: 180  # 6 seconds
    spatial_weight: 0.4
    velocity_weight: 0.3
    jersey_weight: 0.3
    match_threshold: 0.7
  ```

**Expected Outcome:** Recover identity for ≥70% of players re-entering within 6 seconds

**Validation:** Manual annotation of 20 exit/entry events, measure recovery rate

---

#### P1.3: Add Adaptive Distance Gating

**Problem:** Fixed 300px max distance may be too restrictive for fast plays near goal

**Solution:** Scale max distance by court position (players near goal move faster)

**Implementation:**
- **Files to modify:**
  - [src/detection/tracking.py](../src/detection/tracking.py):280-290 - Replace fixed threshold

- **Algorithm:**
  ```python
  def get_adaptive_max_distance(court_position, frames_lost):
      # Distance from center line (0-20m)
      distance_from_center = abs(court_position.x - 20.0)

      # Players near goals (x=0 or x=40) move faster
      position_multiplier = 1.0 + 0.5 * (distance_from_center / 20.0)

      base = 100 + (frames_lost * 20)
      return min(base * position_multiplier, 400)  # Cap at 400px
  ```

- **Config changes:**
  ```yaml
  tracking:
    adaptive_distance: true
    position_multiplier_max: 1.5
    absolute_max_distance: 400
  ```

**Expected Outcome:** Reduce track loss during fast counter-attacks by ≥25%

**Validation:** Measure track continuity on clips with fast breaks from defense to attack

---

### Priority 2: Team Assignment Reliability

#### P2.1: Add Temporal Smoothing for Team Colors

**Problem:** K-means may flicker when team colors are ambiguous

**Solution:** Apply majority voting over 90-frame windows after K-means

**Implementation:**
- **Files to modify:**
  - [src/passes/pass3_identity.py](../src/passes/pass3_identity.py) - Add temporal smoothing

- **Algorithm:**
  ```python
  def smooth_team_assignments(fragments, window=90):
      for track_id in unique_tracks:
          track_fragments = get_fragments_by_track(track_id)

          for i, fragment in enumerate(track_fragments):
              # Get fragments within ±window frames
              nearby = get_fragments_in_window(fragment, window)

              # Majority vote
              votes = Counter([f.team for f in nearby])
              fragment.team_smoothed = votes.most_common(1)[0][0]
  ```

- **Config changes:**
  ```yaml
  team_assignment:
    temporal_smoothing: true
    smoothing_window_frames: 90
  ```

**Expected Outcome:** Reduce team assignment flicker by ≥90%

**Validation:** Count team changes per track before/after smoothing

---

#### P2.2: Implement Multi-Cue Team Assignment

**Problem:** Color-only assignment fails when teams wear similar colors

**Solution:** Combine color + court side + jersey number range

**Implementation:**
- **Files to modify:**
  - [src/detection/team_clustering.py](../src/detection/team_clustering.py) - Add multi-cue fusion
  - [src/passes/pass3_identity.py](../src/passes/pass3_identity.py) - Integrate cues

- **Algorithm:**
  ```python
  def assign_team_multi_cue(fragment):
      # Cue 1: Color clustering (existing)
      color_score_a = kmeans_probability(fragment.hsv, cluster_a)

      # Cue 2: Spatial prior (court side)
      side_score_a = 1.0 if fragment.mean_position.x < 20.0 else 0.0

      # Cue 3: Jersey number range (if configured)
      jersey_score_a = 1.0 if fragment.jersey in [1-10] else 0.0

      # Weighted fusion
      final_score = (
          color_score_a * 0.6 +
          side_score_a * 0.2 +
          jersey_score_a * 0.2
      )

      return "team_a" if final_score > 0.5 else "team_b"
  ```

- **Config changes:**
  ```yaml
  team_assignment:
    multi_cue: true
    use_spatial_prior: true
    spatial_weight: 0.2
    jersey_range_team_a: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    jersey_weight: 0.2
  ```

**Expected Outcome:** Improve team assignment accuracy by ≥15% on difficult color scenarios

**Validation:** Test on 3 matches with similar team colors, measure accuracy

---

#### P2.3: Add Manual Team Annotation Fallback

**Problem:** No way to correct K-means errors

**Solution:** GUI for manual annotation of 3-5 players per team, propagate via graph clustering

**Implementation:**
- **New files:**
  - `src/annotation/manual_team_tool.py` - OpenCV-based GUI
    - Display sample frames with detected players
    - User clicks to assign team labels (3-5 players per team)
    - Export annotations to JSON

  - `src/passes/pass3_identity.py` - Load and apply manual labels
    - Load manual annotations
    - Assign labels to annotated fragments
    - Propagate via spatial-temporal graph clustering

- **Algorithm:**
  ```python
  # Build similarity graph
  G = nx.Graph()
  for frag_i, frag_j in fragment_pairs:
      if temporally_adjacent(frag_i, frag_j):
          similarity = cosine_similarity(frag_i.hsv, frag_j.hsv)
          G.add_edge(frag_i, frag_j, weight=similarity)

  # Apply manual labels as seeds
  for frag in manually_labeled:
      frag.team = manual_labels[frag.id]
      frag.is_seed = True

  # Label propagation
  propagate_labels(G, seed_nodes=manually_labeled)
  ```

- **Config changes:**
  ```yaml
  team_assignment:
    manual_annotation_file: "annotations/manual_teams.json"
    use_manual_fallback: true
  ```

**Expected Outcome:** 100% team assignment accuracy when manual labels provided

**Validation:** User study with 3 non-expert annotators, measure accuracy

---

### Priority 3: Ball Tracking Accuracy

#### P3.1: Re-Enable Ball Tracker

**Problem:** Ball tracker exists but is disabled (line 243)

**Solution:** Debug and re-enable centroid-based ball tracker

**Implementation:**
- **Files to modify:**
  - [src/detection/ball_detector.py](../src/detection/ball_detector.py):243 - Remove disable flag
  - [src/passes/pass1_collect.py](../src/passes/pass1_collect.py) - Integrate ball tracker output

- **Testing required:**
  - Validate on clips with fast ball movement
  - Check for centroid drift during occlusion
  - Verify anomaly detection threshold (500px)

- **Debug checklist:**
  - Centroid calculation on empty buffer (initialization)
  - NMS aggregation across tiles
  - Persistence of ball ID across frames

**Expected Outcome:** Reduce ball ID switches by ≥60%

**Validation:** Manual review of 5 test clips (3 min each), count ID changes

---

#### P3.2: Add Kalman Filter for Ball Prediction

**Problem:** No motion prediction for ball (unlike players)

**Solution:** Implement constant-acceleration Kalman filter

**Implementation:**
- **Files to modify:**
  - [src/detection/ball_detector.py](../src/detection/ball_detector.py) - Add BallKalmanFilter class

- **Algorithm:**
  ```python
  class BallKalmanFilter:
      def __init__(self):
          # State: [x, y, vx, vy, ax, ay]
          self.state = np.zeros(6)

          # Constant acceleration model
          self.F = np.array([
              [1, 0, dt, 0, 0.5*dt**2, 0],
              [0, 1, 0, dt, 0, 0.5*dt**2],
              [0, 0, 1, 0, dt, 0],
              [0, 0, 0, 1, 0, dt],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]
          ])

      def predict(self):
          self.state = self.F @ self.state
          return self.state[:2]  # Predicted (x, y)
  ```

- **Config changes:**
  ```yaml
  ball_tracking:
    use_kalman: true
    process_noise: 50.0
    measurement_noise: 20.0
    max_prediction_frames: 10
  ```

**Expected Outcome:** Maintain ball tracking through 5-10 frame occlusions

**Validation:** Measure ball detection continuity rate on occluded clips

---

#### P3.3: Add Ball-Player Interaction Model

**Problem:** Ball occluded by players has no context

**Solution:** Track ball possession; predict ball near possessing player

**Implementation:**
- **Files to modify:**
  - [src/detection/ball_detector.py](../src/detection/ball_detector.py) - Add possession tracking
  - [src/passes/pass1_collect.py](../src/passes/pass1_collect.py) - Link ball to player in JSON

- **Algorithm:**
  ```python
  def update_ball_possession(ball_position, player_tracks):
      if ball_detected:
          nearest_player = min(player_tracks,
                              key=lambda p: distance(p.position, ball_position))

          if distance(nearest_player, ball_position) < 2.0:  # meters
              ball.possessing_player = nearest_player.id

      else:  # Ball not detected
          if ball.possessing_player:
              # Predict ball near possessing player
              search_region = expand_bbox(possessing_player.bbox, margin=50px)
              detections_in_region = filter_by_region(all_detections, search_region)
  ```

- **Config changes:**
  ```yaml
  ball_tracking:
    possession_distance_threshold: 2.0  # meters
    possession_search_margin: 50  # pixels
  ```

**Expected Outcome:** Reduce ball tracking loss by ≥30% during player contact

**Validation:** Measure ball detection rate during dribbling/possession phases

---

### Priority 4: Jersey Recognition

#### P4.1: Add Jersey Number Confidence Scores

**Problem:** No quality metric for final jersey assignments

**Solution:** Export cumulative confidence from Pass 3 to final JSON

**Implementation:**
- **Files to modify:**
  - [src/passes/pass3_identity.py](../src/passes/pass3_identity.py) - Add confidence export

- **Changes:**
  ```python
  # In output JSON:
  {
      "fragment_id": "F000021",
      "jersey_number": 4,
      "jersey_confidence": 0.87,  # NEW: cumulative confidence / frame_count
      "jersey_detection_count": 42,
      "jersey_coverage": 0.65  # fraction of fragment with detections
  }
  ```

- **Usage:** Filter low-confidence assignments in post-processing:
  ```python
  reliable_jerseys = [f for f in fragments if f.jersey_confidence >= 0.7]
  ```

**Expected Outcome:** Enable filtering of low-confidence jersey assignments

**Validation:** ROC curve analysis on manual annotations

---

#### P4.2: Implement Jersey Number Cross-Validation

**Problem:** Jersey OCR errors propagate to entire fragments

**Solution:** Compare jersey numbers across fragments, flag conflicts

**Implementation:**
- **Files to modify:**
  - [src/passes/pass3_identity.py](../src/passes/pass3_identity.py) - Add cross-validation

- **Algorithm:**
  ```python
  def cross_validate_jerseys(fragments):
      # Group by player trajectory (same track or spatially continuous)
      player_groups = cluster_fragments_by_trajectory(fragments)

      for group in player_groups:
          jersey_counts = Counter([f.jersey for f in group])

          if len(jersey_counts) > 1:
              # Conflict detected
              majority_jersey, majority_count = jersey_counts.most_common(1)[0]

              for fragment in group:
                  if fragment.jersey != majority_jersey:
                      fragment.jersey_conflict = True
                      fragment.jersey_conflict_reason = (
                          f"Minority jersey {fragment.jersey}, "
                          f"expected {majority_jersey}"
                      )
  ```

**Expected Outcome:** Detect ≥80% of jersey OCR errors automatically

**Validation:** Manual review of flagged vs. unflagged errors

---

### Priority 5: System-Level Improvements

#### P5.1: Add Evaluation Metrics Pipeline

**Problem:** No quantitative performance measurement

**Solution:** Implement standard tracking metrics (MOTA, IDF1)

**Implementation:**
- **New files:**
  - `src/evaluation/metrics.py` - Implement MOT metrics
    - MOTA: Multiple Object Tracking Accuracy
    - IDF1: ID F1 score
    - ID switches per minute
    - Team assignment accuracy
    - Jersey recognition accuracy

  - `src/evaluation/ground_truth_loader.py` - Load annotated test data
    - Parse annotation format (MOTChallenge, CVAT, custom JSON)
    - Align ground truth with predictions by frame

- **Metrics:**
  ```python
  def compute_mota(gt, pred):
      FN = false_negatives(gt, pred)
      FP = false_positives(gt, pred)
      IDSW = id_switches(gt, pred)
      GT = total_ground_truth_objects(gt)

      return 1 - (FN + FP + IDSW) / GT

  def compute_idf1(gt, pred):
      IDTP = correctly_identified_detections(gt, pred)
      IDFN = missed_identifications(gt, pred)
      IDFP = false_identifications(gt, pred)

      return 2 * IDTP / (2 * IDTP + IDFN + IDFP)
  ```

**Expected Outcome:** Quantitative benchmarks for all improvements

**Validation:** Establish baseline scores on 10-minute annotated test video

---

#### P5.2: Add Homography Calibration Tool

**Problem:** 13-point calibration mentioned but no tool provided

**Solution:** Interactive GUI for clicking court points

**Implementation:**
- **New files:**
  - `src/geometry/calibration_tool.py` - OpenCV-based GUI

- **Features:**
  ```python
  # Display sample frame from video
  # User clicks 13 court points:
  #   - 4 corners
  #   - 4 goal line intersections
  #   - 2 center circle points
  #   - 2 penalty area points
  #   - 1 center spot

  # Compute homography matrix
  H = cv2.findHomography(pixel_points, court_points)

  # Save to config file
  save_homography("config/homography.yaml", H)
  ```

- **CLI command:**
  ```bash
  python -m src.cli calibrate --video sample.mp4 --frame 100
  ```

**Expected Outcome:** Users can calibrate new cameras in <5 minutes

**Validation:** 3 non-expert users successfully calibrate test cameras

---

#### P5.3: Add Automated Testing Suite

**Problem:** `tests/` directory exists but unclear coverage

**Solution:** Add unit tests for critical functions

**Implementation:**
- **Files to create:**
  - `tests/test_pass2_geometry.py` - Test fragmentation logic
    - Divergence detection (4 signal types)
    - Jersey temporal exclusivity
    - Fragment merging

  - `tests/test_pass3_identity.py` - Test inheritance and conflicts
    - Bidirectional team inheritance
    - Bidirectional jersey inheritance
    - Temporal exclusivity checks
    - Single-owner invariant validation

  - `tests/test_tracking.py` - Test ByteTrack edge cases
    - 3-stage association
    - Distance gating
    - Track buffer management

- **Test examples:**
  ```python
  def test_jersey_first_appearance_no_split():
      """Jersey appearing (None → #4) should NOT split track"""
      track = create_test_track(jersey_timeline=[None, None, 4, 4, 4])
      fragments = detect_divergence(track)
      assert len(fragments) == 1  # No split

  def test_temporal_exclusivity():
      """Same jersey on two tracks simultaneously should split"""
      track_a = create_test_track(jersey=4, frames=100-200)
      track_b = create_test_track(jersey=4, frames=150-250)  # Overlaps
      fragments = detect_jersey_temporal_conflicts([track_a, track_b])
      assert len(fragments) == 3  # track_a, track_b_part1, track_b_part2
  ```

**Expected Outcome:** ≥80% code coverage on critical paths

**Validation:** 50+ unit tests, all pass on main branch

---

### Implementation Roadmap Summary

| Priority | Improvement | Files Modified | Expected Gain | Validation Method |
|----------|-------------|----------------|---------------|-------------------|
| P1.1 | Re-ID embeddings | player_detector.py, pass1_collect.py, pass2_geometry.py | -40% ID switches | IDF1 on test clips |
| P1.2 | Trajectory matching | pass2_geometry.py, pass3_identity.py | 70% recovery rate | Manual annotation |
| P1.3 | Adaptive distance | tracking.py | -25% track loss | Fast break clips |
| P2.1 | Temporal smoothing | pass3_identity.py | -90% team flicker | Track stability |
| P2.2 | Multi-cue teams | team_clustering.py, pass3_identity.py | +15% accuracy | Similar color matches |
| P2.3 | Manual annotation | NEW: manual_team_tool.py | 100% accuracy | User study |
| P3.1 | Re-enable ball tracker | ball_detector.py | -60% ID switches | Manual review |
| P3.2 | Ball Kalman filter | ball_detector.py | 5-10 frame occlusion | Continuity rate |
| P3.3 | Ball-player interaction | ball_detector.py, pass1_collect.py | -30% tracking loss | Possession phases |
| P4.1 | Jersey confidence | pass3_identity.py | Filtering capability | ROC curve |
| P4.2 | Jersey cross-validation | pass3_identity.py | 80% error detection | Manual review |
| P5.1 | Metrics pipeline | NEW: metrics.py, ground_truth_loader.py | Quantitative benchmarks | Baseline scores |
| P5.2 | Calibration GUI | NEW: calibration_tool.py | <5 min calibration | User study |
| P5.3 | Testing suite | NEW: test_*.py | 80% coverage | CI/CD integration |

---

## R - RESULTS (Status and Roadmap)

### 5.1 Current Status Summary

#### What Works Well ✅

1. **3-Pass Architecture**
   - Clean separation of concerns (tracking → geometry → identity)
   - Immutable data flow prevents cascading errors
   - Frame-explainable decisions (every split traceable)

2. **Jersey-Based Identity**
   - Jersey temporal exclusivity prevents "two Spyros" bugs
   - Bidirectional inheritance maintains identity continuity
   - Jersey-aware splitting suppresses false positives (player turning around)

3. **Fragment Coverage Strategy**
   - Keeps all fragments (no minimum length filtering)
   - Merges consecutive short fragments
   - Prevents grey "unknown" gaps in visualization

4. **Divergence Detection**
   - Multi-signal approach (velocity, position, occlusion, appearance)
   - Successfully identifies track jumps
   - Configurable thresholds per signal type

5. **Team Clustering**
   - Variance-based detection works for bibbed vs. non-bibbed teams
   - Bidirectional team inheritance handles occlusion gaps
   - Team size constraint validation catches fragmentation bugs

6. **Ball Detection**
   - InferenceSlicer improves small object detection
   - Tile overlap prevents edge detection failures
   - Low confidence threshold (0.3) catches partial visibility

#### What Is Partially Implemented ⚠️

1. **Ball Tracking**
   - Tracker exists but is **disabled** (line 243 in ball_detector.py)
   - Needs debugging before production use
   - No Kalman filter for ball motion prediction

2. **SAM2 Segmentation**
   - Marked as "optional" in code
   - Unclear when/how it's used
   - No documentation of performance impact

3. **Re-ID Capability**
   - System relies on jersey numbers + local appearance
   - No deep learning Re-ID embeddings
   - Can't recover identity after >30 frame gaps

4. **Team Assignment Confidence**
   - K-means clustering runs but no confidence scores exported
   - Can't filter unreliable team assignments
   - No multi-cue fusion (color only)

5. **Jersey OCR Validation**
   - Confidence scores computed but not exported
   - No cross-validation across fragments
   - OCR errors propagate to entire fragments

#### What Is Missing or Fragile ❌

1. **Long Occlusion Recovery**
   - No Re-ID embeddings for appearance-based re-association
   - Track buffer limited to 30 frames
   - Players get new IDs after extended occlusions

2. **Evaluation Metrics**
   - No ground truth annotations documented
   - No MOTA, IDF1, or standard tracking metrics
   - Can't quantify improvements or optimize thresholds

3. **Homography Calibration Tool**
   - 13-point calibration mentioned but no tool provided
   - No documentation of calibration procedure
   - Can't adapt to new camera angles without code changes

4. **Manual Annotation Fallback**
   - No way to correct K-means team assignment errors
   - No GUI for manual jersey/team labeling
   - Errors propagate through entire video

5. **Ball Motion Prediction**
   - No physics-based ball trajectory model
   - No ball-player interaction modeling
   - Ball tracking lost during possession/dribbling

6. **Automated Testing**
   - Test directory exists but coverage unclear
   - No CI/CD integration
   - Critical functions not tested

7. **Performance Benchmarks**
   - No documented processing speed
   - No memory usage profiling
   - No GPU requirements specified

---

### 5.2 Concrete Next Steps Roadmap

**Phase 1: Enable Existing Features (Weeks 1-2)**

1. **Re-enable ball tracker** 🎯 CRITICAL
   - Debug and re-enable centroid-based ball tracker
   - **Measurable outcome:** Ball ID switches reduced by ≥60% on test clips
   - **Validation:** Manual review of 5 test clips (3 min each)
   - **Files:** [src/detection/ball_detector.py](../src/detection/ball_detector.py):243, [src/passes/pass1_collect.py](../src/passes/pass1_collect.py)

2. **Add evaluation metrics pipeline** 📊
   - Implement MOTA, IDF1, team accuracy, jersey accuracy
   - **Measurable outcome:** Baseline scores established for 10-minute test video
   - **Validation:** Compute metrics on annotated test clips
   - **Files:** NEW: `src/evaluation/metrics.py`, `src/evaluation/ground_truth_loader.py`

3. **Export confidence scores** 📈
   - Add jersey confidence and team confidence to final JSON
   - **Measurable outcome:** All final outputs include confidence fields
   - **Validation:** JSON schema validation passes
   - **Files:** [src/passes/pass3_identity.py](../src/passes/pass3_identity.py)

**Phase 2: Improve Robustness (Weeks 3-5)**

4. **Add Re-ID embeddings** 🔍
   - Integrate OSNet or ResNet50 for appearance embeddings
   - **Measurable outcome:** ID switches reduced by ≥40% on heavy occlusion clips
   - **Validation:** IDF1 score improves by ≥15 points
   - **Files:** [src/detection/player_detector.py](../src/detection/player_detector.py), [src/passes/pass1_collect.py](../src/passes/pass1_collect.py), [src/passes/pass2_geometry.py](../src/passes/pass2_geometry.py)

5. **Implement trajectory-based re-association** 🎯
   - Store exit trajectories with jersey numbers
   - **Measurable outcome:** ≥70% of players re-entering within 6s maintain identity
   - **Validation:** Manual annotation of 20 exit/entry events
   - **Files:** [src/passes/pass2_geometry.py](../src/passes/pass2_geometry.py), [src/passes/pass3_identity.py](../src/passes/pass3_identity.py)

6. **Add Kalman filter for ball tracking** ⚽
   - Constant-acceleration model for ball motion
   - **Measurable outcome:** Ball tracking maintained through 5-10 frame occlusions
   - **Validation:** Ball detection continuity rate improves by ≥30%
   - **Files:** [src/detection/ball_detector.py](../src/detection/ball_detector.py)

**Phase 3: Team & Jersey Reliability (Weeks 6-7)**

7. **Add multi-cue team assignment** 🎨
   - Combine color + spatial + jersey range
   - **Measurable outcome:** Team assignment accuracy improves by ≥15% on difficult scenarios
   - **Validation:** Test on 3 matches with similar team colors
   - **Files:** [src/detection/team_clustering.py](../src/detection/team_clustering.py), [src/passes/pass3_identity.py](../src/passes/pass3_identity.py)

8. **Implement jersey cross-validation** ✅
   - Detect conflicts across fragments
   - **Measurable outcome:** ≥80% of jersey OCR errors flagged automatically
   - **Validation:** Manual review of flagged vs. unflagged errors
   - **Files:** [src/passes/pass3_identity.py](../src/passes/pass3_identity.py)

**Phase 4: Tooling & Testing (Week 8+)**

9. **Create homography calibration GUI** 🛠️
   - Interactive point-and-click calibration
   - **Measurable outcome:** Users can calibrate new cameras in <5 minutes
   - **Validation:** 3 non-expert users successfully calibrate test cameras
   - **Files:** NEW: `src/geometry/calibration_tool.py`

10. **Add automated test suite** 🧪
    - Unit tests for critical functions (divergence, inheritance, temporal exclusivity)
    - **Measurable outcome:** 50+ unit tests, ≥80% coverage on critical paths
    - **Validation:** All tests pass on main branch, CI/CD integration
    - **Files:** NEW: `tests/test_pass2_geometry.py`, `tests/test_pass3_identity.py`, `tests/test_tracking.py`

---

### 5.3 Recommended Evaluation Metrics

#### Tracking Performance

- **IDF1 (ID F1 Score)** 🎯 PRIMARY METRIC
  - Measures identity consistency across frames
  - Target: ≥85% on test set with moderate occlusion
  - Formula: `IDF1 = 2*IDTP / (2*IDTP + IDFN + IDFP)`

- **MOTA (Multiple Object Tracking Accuracy)**
  - Overall detection + tracking quality
  - Target: ≥90% on high-quality footage
  - Formula: `MOTA = 1 - (FN + FP + IDSW) / GT`

- **ID Switches per Minute**
  - Count of identity changes
  - Target: ≤2 switches per 5-minute clip
  - Critical for futsal analysis (need stable player IDs)

#### Team Assignment

- **Team Accuracy**
  - % of frames with correct team assignment
  - Target: ≥95% for distinct colors, ≥85% for similar colors
  - Requires ground truth team labels

- **Team Flicker Rate**
  - Team changes per fragment
  - Target: ≤1% of fragments have team changes
  - Measures temporal consistency

#### Jersey Recognition

- **Jersey Accuracy**
  - % of frames with correct jersey number
  - Target: ≥80% (bibbed team only)
  - Excludes frames where jersey not visible

- **Jersey Coverage**
  - % of players with assigned jersey numbers
  - Target: ≥90% of bibbed team players
  - Measures how often jerseys are detected

#### Ball Tracking

- **Ball Detection Rate**
  - % of frames with ball detected
  - Target: ≥70% (ball often occluded or out of view)
  - Baseline metric for small object detection

- **Ball ID Continuity**
  - % of frames with consistent ball ID
  - Target: ≥90% (once tracker re-enabled)
  - Measures tracking stability

#### System Performance

- **Processing Speed**
  - Frames per second
  - Target: Real-time (≥30 FPS) on RTX 3080
  - Critical for production deployment

- **Memory Usage**
  - Peak GPU/CPU memory
  - Target: ≤8GB GPU, ≤16GB RAM
  - Ensures deployability on standard hardware

---

### 5.4 Test Dataset Requirements

To validate improvements, create annotated test clips with:

1. **Heavy Occlusion Scenario** (10 min)
   - Dense player clustering
   - Frequent overlaps (4+ players within 2m)
   - Multiple simultaneous occlusions
   - Target: Stress-test tracking robustness

2. **Entry/Exit Scenario** (5 min)
   - Players frequently leaving and re-entering frame
   - Substitutions and sideline returns
   - Frame edge occlusions
   - Target: Test track re-association

3. **Similar Team Colors** (5 min)
   - Teams wearing similar color palettes
   - Challenging lighting conditions
   - Ambiguous team boundaries
   - Target: Test team assignment robustness

4. **Fast Play** (5 min)
   - Quick passes and counter-attacks
   - Rapid position changes
   - High player velocities
   - Target: Test adaptive distance gating

5. **Jersey Visibility** (5 min)
   - Players frequently turning (showing/hiding jerseys)
   - First-time jersey appearances
   - Brief jersey occlusions
   - Target: Test jersey inheritance logic

**Annotation Requirements:**
- Ground truth bounding boxes (every frame)
- Player identity labels (consistent IDs across clip)
- Team assignments (frame-level)
- Jersey numbers (when visible)
- Ball positions (when visible)

**Annotation Format:** MOTChallenge or CVAT compatible

---

### 5.5 Critical Risks and Mitigation

#### Production Risks (General)

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **ID switches during extended occlusion** | High | High | P1.1: Add Re-ID embeddings |
| **Team misassignment on similar colors** | High | Medium | P2.2: Multi-cue team assignment |
| **Jersey OCR errors propagate** | Medium | Medium | P4.2: Cross-validation |
| **Ball tracker disabled in production** | High | Current | P3.1: Re-enable and debug |
| **No quantitative benchmarks** | High | Current | P5.1: Metrics pipeline |
| **Manual calibration difficult** | Medium | Low | P5.2: Calibration GUI |
| **Insufficient test coverage** | Medium | Medium | P5.3: Automated tests |

#### Current PoC-Specific Risks

| Risk | Impact | Status | Immediate Fix | Priority |
|------|--------|--------|---------------|----------|
| **Team size constraint violated** | 🔴 Critical | Occurring | Hard constraint enforcement | P0 |
| **White/orange color confusion** | High | Occurring | Multi-cue team assignment + spatial priors | P1 |
| **Numbered players disappearing** | 🔴 Critical | Occurring | Increase buffer to 90 frames + ghost tracks | P0 |
| **White ball low contrast** | Medium | Occurring | Test darker ball (orange/yellow) | P1 |
| **Green jersey not detected** | Low | Occurring | Adjust masking filters (low priority) | P3 |
| **K-means performance bottleneck** | Medium | Occurring | Mini-batch K-means | P2 |

---

### 5.6 Deployment Considerations

**Hardware Requirements (to be documented):**
- GPU: NVIDIA RTX 3060 or higher (≥8GB VRAM)
- CPU: 8+ cores recommended
- RAM: 16GB minimum
- Storage: 10GB for models + outputs

**Performance Targets:**
- Real-time processing (≥30 FPS) for live analysis
- 5-10x real-time for batch processing
- <5 seconds initialization time

**Operational Monitoring:**
- Log ID switch frequency (alert if >5 per minute)
- Track team assignment flicker (alert if >10 changes per clip)
- Monitor ball detection rate (alert if <50%)

---

### 5.7 Current Deployment Issues & Observations

Based on current PoC testing (orange bibbed team vs. white/green/black shirt team):

#### Issue #1: Team Size Constraint Not Enforced 🔴 CRITICAL

**Symptom:**
- White shirt player correctly allocated to black team initially
- Fragment changes mid-track, player suddenly reassigned to orange team
- Orange team now has 7 players (exceeds max of 6)

**Root Cause Analysis:**
- White and orange colors are similar in HSV space
- K-means clustering may mis-assign white shirts to orange cluster
- Team size constraint validation ([src/passes/pass3_identity.py](../src/passes/pass3_identity.py):24-97) only **detects** violations but doesn't **prevent** them
- No enforcement mechanism to reject impossible team assignments

**Impact:** HIGH
- Breaks fundamental futsal rule (max 6 players per team)
- Makes team-based analytics unreliable
- Indicates fragmentation or clustering bugs

**Proposed Fix:**
- Add **hard constraint** in team assignment: reject assignments that violate team size
- When K-means produces invalid configuration, use spatial priors (court side) to resolve
- Log violations for debugging but enforce valid state

**Implementation:**
```python
def enforce_team_size_constraint(fragments, max_size=6):
    """Reassign fragments to respect team size constraint"""
    for frame_id in all_frames:
        team_a_count = count_concurrent(fragments, team="team_a", frame=frame_id)
        team_b_count = count_concurrent(fragments, team="team_b", frame=frame_id)

        if team_a_count > max_size:
            # Find least confident team_a assignment at this frame
            violators = get_lowest_confidence(fragments, team="team_a", frame=frame_id)
            reassign_to_team_b(violators[:team_a_count - max_size])
```

**File to modify:** [src/passes/pass3_identity.py](../src/passes/pass3_identity.py)

---

#### Issue #2: K-Means Clustering Performance 🟡 MEDIUM PRIORITY

**Symptom:**
- K-means clustering works well for team assignment
- Takes VERY LONG time to run in Pass 3
- Green bibbed player often doesn't cluster correctly (appears as "unknown")

**Root Cause Analysis:**
- K-means on large number of histograms (50 samples × N fragments) is slow
- Green shirt similar to green court background → masking filters remove jersey pixels
- Low jersey mask quality → histogram rejected → player not clustered

**Impact:** MEDIUM
- Performance bottleneck (acceptable for offline processing)
- Green jerseys problematic (low priority - can use better colored bibs)

**Proposed Fix (Performance):**
- Use Mini-Batch K-means instead of standard K-means (10-20x faster)
- Reduce samples per fragment from 50 to 20-30
- Cache cluster centers across clips in same match

**Proposed Fix (Green Jersey):**
- Adjust HSV masking to be less aggressive on green (court floor H 25-60° instead of 25-90°)
- Add separate "jersey detection" mask that's more permissive than "background removal" mask

**Implementation:**
```python
from sklearn.cluster import MiniBatchKMeans

# Replace:
kmeans = KMeans(n_clusters=2, random_state=42)

# With:
kmeans = MiniBatchKMeans(n_clusters=2, batch_size=100, random_state=42)
```

**File to modify:** [src/detection/team_clustering.py](../src/detection/team_clustering.py)

---

#### Issue #3: Ball Tracker Jumpy & Missing Fast Balls 🟡 MEDIUM PRIORITY

**Symptom:**
- 2D ball trajectory is very jumpy (not smooth)
- Fast-moving balls not always detected
- White ball on light court may have poor contrast

**Root Cause Analysis:**
- Ball tracker currently **disabled** (line 243 in ball_detector.py)
- Using raw YOLO detections without temporal smoothing
- White ball low contrast on light-colored court
- Fast motion creates blur → detection confidence drops below threshold

**Impact:** MEDIUM
- Ball tracking unreliable for possession analysis
- Can't compute accurate pass trajectories

**Proposed Fixes:**
1. **Re-enable ball tracker** (PRIORITY 1)
   - Debug centroid-based tracker
   - Add Kalman filter for smoothing (see P3.2)

2. **Improve detection robustness:**
   - Lower confidence threshold from 0.3 to 0.2 (catch more marginal detections)
   - Increase InferenceSlicer overlap from 200px to 300px (better small object coverage)
   - Add motion blur augmentation to training data

3. **Equipment recommendation:**
   - Test with **darker colored ball** (orange, yellow, or high-vis green)
   - Would improve contrast on light court
   - Likely to improve detection rate significantly

**Quick Test:**
- Try orange ball on same footage and compare detection rate

**Files to modify:**
- [src/detection/ball_detector.py](../src/detection/ball_detector.py):243 - Re-enable tracker
- [src/detection/ball_detector.py](../src/detection/ball_detector.py) - Add Kalman filter (see P3.2 implementation)

---

#### Issue #4: Occlusion Causing Players to Disappear 🔴 CRITICAL

**Symptom:**
- Players disappear from tracking during occlusion
- Reappear with new track IDs
- Rationale: "Players don't just disappear" - should maintain presence

**Root Cause Analysis:**
- ByteTrack deletes tracks after 30 frames without detection
- During dense occlusion (4+ players), YOLO may not detect all players
- No position estimation or segmentation fallback

**Impact:** CRITICAL (for numbered jersey players)
- Breaks identity persistence for numbered players
- Can't compute continuous player metrics (distance, possession time)
- Acceptable for non-numbered players, but HIGH priority fix overall

**Proposed Fixes:**

**Short-term (Immediate):**
1. **Increase track buffer** from 30 to 60 frames
   - Gives 2 seconds at 30 FPS for players to reappear
   - Low-risk, easy implementation
   - File: [src/detection/tracking.py](../src/detection/tracking.py):362-370

2. **Add "ghost tracks" during occlusion:**
   - When track lost, continue outputting last known position with `occluded=True` flag
   - Don't delete track for numbered jersey players (check Pass 3 output for jersey assignment)
   - Only delete after 90 frames (3 seconds)

**Medium-term (Weeks 3-5):**
3. **SAM2 segmentation integration:**
   - When multiple players cluster (spatial radius < 1.5m), use SAM2 to segment individuals
   - Extract per-player masks even during overlap
   - File: [src/detection/segmentation_sam2.py](../src/detection/segmentation_sam2.py) (currently marked "optional")

4. **Kalman prediction fallback:**
   - During occlusion, output Kalman-predicted positions
   - Mark as `predicted=True` in output JSON
   - Continue tracking based on predicted positions

**Implementation (Short-term):**
```python
# In tracking.py
TRACK_BUFFER_NUMBERED_PLAYERS = 90  # 3 seconds for numbered players
TRACK_BUFFER_REGULAR = 30  # 1 second for others

def should_remove_track(track, has_jersey_number):
    buffer = TRACK_BUFFER_NUMBERED_PLAYERS if has_jersey_number else TRACK_BUFFER_REGULAR
    return track.frames_since_last_detection > buffer
```

**File to modify:** [src/detection/tracking.py](../src/detection/tracking.py)

---

### Issue Summary & Priority Matrix

| Issue | Severity | Impact on PoC | Proposed Fix | Effort | Priority |
|-------|----------|---------------|--------------|--------|----------|
| #1: Team size constraint not enforced | 🔴 Critical | Violates futsal rules, breaks analytics | Hard constraint enforcement | Low | **P0 - Immediate** |
| #4: Players disappear during occlusion | 🔴 Critical | Breaks numbered player tracking | Increase buffer + ghost tracks | Low | **P0 - Immediate** |
| #3: Ball tracker jumpy/missing balls | 🟡 Medium | Limits pass analysis | Re-enable tracker + Kalman | Medium | **P1 - Week 1** |
| #2: K-means slow + green jersey issue | 🟡 Medium | Performance bottleneck | Mini-batch K-means, adjust masking | Medium | **P2 - Week 2** |

**Immediate Actions (This Week):**
1. Add team size constraint enforcement ([src/passes/pass3_identity.py](../src/passes/pass3_identity.py))
2. Increase track buffer to 60 frames for all, 90 for numbered players ([src/detection/tracking.py](../src/detection/tracking.py))
3. Add "ghost track" output during occlusion (continue tracking with `occluded=True` flag)

**Equipment Recommendations:**
- Test with darker colored ball (orange, yellow, high-vis green) for better contrast
- Ensure bibbed team uses high-contrast colors (avoid green bibs on green-tinted court)

---

## Summary

This futsal tracking system demonstrates a sophisticated multi-pass architecture with strong foundations in jersey-based identity management and divergence detection. The separation of tracking from identity assignment provides explainability and robustness.

**Key Strengths:**
- Jersey temporal exclusivity prevents identity conflicts
- Fragment coverage strategy eliminates visualization gaps
- Configurable multi-signal divergence detection
- Works well for bibbed vs. non-bibbed team scenarios

**Current PoC Configuration:**
- Orange bibbed players (numbered) vs. white/green/black shirts (mostly non-numbered)
- Priority: Consistent tracking for numbered jersey players (pass/distance analytics)
- Acceptable: Track jumping for non-numbered players (team assignment only)

**Critical Issues (Current Deployment):**
1. 🔴 **Team size constraint not enforced** - Players incorrectly switching teams mid-game, violating 6-player max
2. 🔴 **Players disappear during occlusion** - Breaks numbered player identity persistence (critical for analytics)
3. 🟡 **Ball tracker jumpy/missing fast balls** - White ball low contrast, tracker disabled
4. 🟡 **K-means clustering slow** - Performance bottleneck, green jerseys similar to background

**Immediate Priorities (This Week):**
1. **Enforce team size constraint** - Add hard limit preventing invalid team assignments
2. **Increase track buffer** - 60 frames general, 90 frames for numbered players
3. **Add ghost tracks** - Continue tracking during occlusion with `occluded=True` flag
4. **Test darker ball** - Orange/yellow ball for better contrast vs. white

**Short-term Priorities (Weeks 1-2):**
5. Re-enable ball tracker with Kalman smoothing
6. Switch to Mini-batch K-means (10-20x faster)
7. Adjust green jersey masking (court floor filter too aggressive)
8. Add evaluation metrics pipeline

**Medium-term Priorities (Weeks 3-5):**
9. Implement Re-ID embeddings for long occlusions
10. Integrate SAM2 segmentation for dense clustering
11. Add trajectory-based re-association

With immediate fixes (#1-4), the system will meet PoC requirements for consistent numbered player tracking and reliable team assignment. Medium-term improvements will achieve production-grade tracking suitable for professional futsal match analysis.

---

**Document Version:** 1.0
**Last Updated:** February 11, 2026
**Authors:** Claude Sonnet 4.5 (Analysis) + Rick (Domain Expert)
