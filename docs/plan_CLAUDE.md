# Futsal Player Tracking System - Implementation Plan

## Sources & Decision Rationale

This plan is based on:

1. **[docs/requirements.md](requirements.md)** - Project requirements document specifying:
   - Core goals (player tracking, ball tracking, events, stats)
   - Key constraint: "Accuracy > features" - this is a trust product
   - Dual tracking approach: all players for team stats, persistent IDs only for numbered players
   - Test footage constraints: GoPro wide-angle, fisheye, 11GB files

2. **User's prior experiment** (referenced in requirements.md):
   - Followed Roboflow basketball identification blog: https://blog.roboflow.com/identify-basketball-players/
   - Results: SAM model too slow (not viable), team clustering via color worked perfectly, jersey number assignment was accurate
   - **Critical finding: Ball tracking via YOLO/Roboflow is NOT accurate enough**

3. **Existing codebase**:
   - [utils/click_points.py](../utils/click_points.py) - Manual pitch calibration tool (8-9 points)
   - [utils/trim_compress_video.py](../utils/trim_compress_video.py) - FFmpeg video preprocessing
   - Empty `requirements.txt` and `models/` directory

4. **Technology choices rationale**:
   - **TrackNetV3 for ball** (not YOLO): User explicitly stated YOLO ball tracking failed. TrackNet is purpose-built for small fast-moving sports balls (tennis/badminton), uses 3-frame sequences to capture motion blur
   - **K-Means clustering for teams**: User confirmed this "worked perfectly" in Roboflow experiment
   - **ByteTrack for player MOT**: State-of-the-art tracker with good occlusion handling, widely used in sports analytics
   - **PaddleOCR for jerseys**: User's experiment showed jersey assignment was "accurate and persistent"

---

## Overview

Build a post-processed computer vision system that takes futsal match recordings and produces accurate player/ball tracking, team classification, event detection, and analytics.

**Key Principles** (from [requirements.md](requirements.md)):
- Accuracy > features (trust product)
- Low friction onboarding (calibrate once per venue)
- Track ALL players for team stats; maintain persistent IDs only for numbered jersey players

---

## Architecture

```
Video Input (11GB MP4)
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  STAGE 1: DETECTION & TRACKING                       │
│  - Player Detection (YOLOv8x) on RAW distorted frame │
│  - Ball Detection (TrackNetV3 + HSV ensemble)        │
│  - Ball ROI prediction (Kalman filter)               │
│  - Player Tracking (ByteTrack)                       │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  STAGE 2: IDENTIFICATION                             │
│  - Team Clustering (K-Means on HSV histograms)       │
│  - Jersey Number OCR (PaddleOCR + voting)            │
│  - Player Re-ID (embeddings, numbered players only)  │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  STAGE 3: GEOMETRY & EVENTS                          │
│  - Fisheye point undistortion (coords only, not img) │
│  - Homography (undistorted px → court meters)        │
│  - Possession Tracking (proximity state machine)     │
│  - Pass/Shot/Goal Detection                          │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  STAGE 4: ANALYTICS & OUTPUT                         │
│  - Heatmaps, Distance/Speed                          │
│  - Stats Aggregation                                 │
│  - Video Overlay, Bird's Eye View                    │
│  - Match Reports                                     │
└──────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Technology | Source/Rationale |
|-----------|------------|------------------|
| Player Detection | YOLOv8x / Roboflow | Roboflow blog approach worked; YOLOv8x is current best |
| **Ball Detection** | **TrackNetV3 + HSV ensemble** | User stated YOLO failed; ensemble for redundancy |
| **Ball Tracking** | **Kalman + ROI prediction** | Search predicted region only; faster + fewer false positives |
| Player Tracking | ByteTrack / supervision | Industry standard MOT; handles occlusions well |
| **Team Classification** | **Roboflow Sports `TeamClassifier`** | https://github.com/roboflow/sports - purpose-built for sports analytics |
| Jersey OCR | PaddleOCR | User's experiment: "accurate and persistent" |
| Video I/O | PyAV (FFmpeg) | Handles 11GB files via streaming (from requirements.md constraint) |
| **Court Mapping** | **Roboflow Sports `ViewTransformer`** | Homography with keypoint detection |
| **Path Smoothing** | **Roboflow Sports `clean_paths`** | Reduces track jitter |
| **Fisheye** | **Undistort points only** | Undistorting frames loses edges; math correction preserves all |
| Calibration | Manual points | Existing [click_points.py](../utils/click_points.py) provides foundation |

> **Note**: The [Roboflow Sports library](https://github.com/roboflow/sports) provides battle-tested implementations of `TeamClassifier`, `ViewTransformer`, `BallTracker`, and `clean_paths` that can replace custom implementations in Phase 2+.

---

## Project Structure

```
futsal/
├── config/
│   ├── venues/                    # Per-venue YAML configs
│   │   └── example_venue.yaml
│   └── default.yaml               # Default processing params
├── src/
│   ├── __init__.py
│   ├── pipeline.py                # Main orchestration
│   ├── detection/
│   │   ├── player_detector.py     # YOLOv8 wrapper
│   │   ├── ball_detector.py       # TrackNetV3 + Kalman
│   │   └── tracking.py            # ByteTrack integration
│   ├── identification/
│   │   ├── team_classifier.py     # K-Means clustering
│   │   ├── jersey_ocr.py          # Number recognition
│   │   └── player_reid.py         # Re-ID for numbered players
│   ├── geometry/
│   │   ├── fisheye.py             # Undistortion
│   │   ├── homography.py          # Court mapping
│   │   └── calibration.py         # Enhanced point picker
│   ├── events/
│   │   ├── possession.py          # Possession state machine
│   │   ├── passes.py              # Pass detection
│   │   ├── shots.py               # Shot detection
│   │   └── goals.py               # Goal detection
│   ├── analytics/
│   │   ├── heatmaps.py
│   │   ├── distance_speed.py
│   │   └── stats_compiler.py
│   ├── output/
│   │   ├── video_renderer.py      # Tracking overlay
│   │   ├── tactical_view.py       # Bird's eye animation
│   │   └── report_generator.py
│   └── utils/
│       ├── video_io.py            # FFmpeg wrapper
│       └── data_models.py         # Pydantic schemas
├── models/                        # Model weights (.pt files)
├── utils/                         # Existing utilities (click_points.py, etc.)
├── tests/
├── requirements.txt
└── main.py                        # CLI entry point
```

---

## Critical Implementation Details

### 1. Ball Tracking (Solving the YOLO Problem)

**Source**: User stated in [requirements.md](requirements.md): "Ball tracking via YOLO and other models found on the likes of Roboflow.com do not track the ball accurately and consistently."

**Solution**: Ensemble + ROI-based tracking:

```
Frame N: Ball detected at (500, 300)
              │
              ▼
        Kalman predicts Frame N+1 position → (510, 295)
              │
              ▼
        Search ROI (100px radius) around prediction
              │
       ┌──────┴──────┐
       ▼             ▼
   TrackNetV3    Color Filter (HSV orange)
   on ROI crop   on ROI crop
       │             │
       └──────┬──────┘
              ▼
        Fuse detections (confidence-weighted)
              │
              ▼
        Update Kalman with measurement
```

**Why ROI-based?**
- Faster: search 100x100px instead of full 1920x1080
- Fewer false positives: ignores random orange objects elsewhere
- Physics-aware: ball can't teleport between frames

**Ensemble strategy** (2 models):
1. **TrackNetV3** (primary) - 3-frame input captures motion blur
2. **HSV color filter** (backup) - cheap, catches obvious orange ball

**Adaptive ROI expansion**:
- Base radius: 80px
- Expand by 30% per missed frame (uncertainty grows)
- Cap at 300px
- After 30 consecutive misses: reset to full-frame search (ball likely out of play)

**Fusion logic**:
- If both detect within 50px: weighted average
- If only one detects: use it if confidence > 0.5, else use Kalman prediction
- If neither detects: use Kalman prediction, mark as interpolated

### 2. Player Identity Strategy

**Source**: [requirements.md](requirements.md) NOTE section: "We only want to track all players for positional analysis, but we only want to track players with a numbered jersey in terms of player stats and re-identification."

**Numbered Players (full re-ID):**
- Jersey number detected via OCR with voting (5+ confirmations)
- Visual embedding stored in gallery
- After track loss: match via embedding + OCR
- Persistent stats tracked

**Non-Numbered Players (track only):**
- Tracked for positional/team stats
- No re-identification after track loss
- New track ID assigned

### 3. Team Classification

**Source**: User stated: "The method of separating teams using clustering worked perfectly"

**Implementation**: K-Means on HSV color histograms extracted from player bounding boxes, k=2 for teams (optionally k=3 if referees need separation).

### 4. Event Detection Logic

**Possession:** Ball within 1.5m of player for 5+ frames
**Pass:** Same-team possession change, ball traveled 2m+, within 3 seconds
**Shot:** Ball velocity > 8 m/s toward goal, in attacking half
**Goal:** Ball crosses goal line within goal width (3m)

### 5. Fisheye Handling Strategy

**Source**: [requirements.md](requirements.md): "FishEye effect - any accurate tracking of players need to consider this. A player running in the 'fisheye area' of the pitch will be different to the centre of the camera."

**Problem**: Undistorting first loses pitch edges (corners already missing with GoPro wide-angle).

**Solution**: Keep distorted frames, apply correction only to coordinates.

```
┌─────────────────────────────────────────────────────────────┐
│  OPTION A: Undistort frames first (NOT recommended)         │
│                                                             │
│  Raw frame → Undistort → Detect → Track → Homography        │
│                 ↓                                           │
│            Loses edges! Already missing corners.            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  OPTION B: Undistort coordinates only (RECOMMENDED)         │
│                                                             │
│  Raw frame → Detect → Track → Undistort points → Homography │
│                                      ↓                      │
│                              No pixel loss!                 │
│                              Math corrects position.        │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
```python
def undistort_points(points_px, camera_matrix, dist_coeffs):
    """Convert distorted pixel coords to undistorted coords without losing frame edges"""
    points = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.fisheye.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)
    return undistorted.reshape(-1, 2)

# Pipeline:
# 1. Run detection on RAW distorted frame (full resolution, no cropping)
# 2. Get player/ball pixel positions
# 3. Undistort those points mathematically
# 4. Apply homography to undistorted points → court coordinates
```

**Why this works**:
- Detection models handle mild distortion fine (trained on varied data)
- Edge pixels still exist in raw frame - just geometrically warped
- Math correction preserves all positions, even in "fisheye area"
- Homography calibration done on undistorted points matches real court

**Calibration requirement**:
- Fisheye coefficients (`camera_matrix`, `dist_coeffs`) calibrated once per camera/lens
- Can use checkerboard calibration or estimate from known court lines

### 6. Venue Configuration

**Source**: [requirements.md](requirements.md): "Venue-first distribution beats individual sales (install once, many users)"

Calibrate once per venue, store in YAML:

```yaml
# config/venues/venue_name.yaml
venue:
  name: "Arena Name"

fisheye:
  camera_matrix: [[...]]
  distortion_coeffs: [...]

calibration:
  image_points_px: [[245, 89], [1715, 89], ...]  # From click_points.py
  court_points_m: [[0, 0], [40, 0], ...]         # Real-world coords
```

---

## Implementation Phases

**Configuration**: GPU (NVIDIA CUDA) | Fully automatic (no manual correction UI)

### Phase 1: Core Detection & Tracking (START HERE)

```
Step 1: Project Structure Setup
├── Create src/ directory structure
├── Create config/ directory with default.yaml
├── Populate requirements.txt
└── Create main.py CLI entry point

Step 2: Video I/O Module
├── src/utils/video_io.py
├── PyAV-based frame extraction (memory efficient for 11GB files)
├── Handle streaming to avoid memory issues
└── Frame batching for GPU inference

Step 3: Player Detection
├── src/detection/player_detector.py
├── YOLOv8x integration (GPU)
├── Confidence filtering + NMS
└── Return bbox + confidence per frame

Step 4: Player Tracking
├── src/detection/tracking.py
├── ByteTrack integration
├── Track ID persistence across frames
└── Handle occlusions gracefully

Step 5: Ball Detection
├── src/detection/ball_detector.py
├── TrackNetV3 integration (3-frame input)
├── Kalman filter for trajectory smoothing
├── Gap interpolation (up to 30 frames)
└── Fallback: fine-tune on futsal data if needed

Step 6: Debug Visualization
├── src/utils/visualization.py
├── Draw bboxes + track IDs on frames
├── Ball trajectory overlay
└── Export debug video for validation
```

### Phase 2: Identification & Geometry
- HSV histogram extraction + K-Means team clustering
- PaddleOCR jersey number recognition + voting
- Player re-ID gallery (numbered players only)
- Enhanced calibration wizard (extend click_points.py)
- Fisheye undistortion
- Homography transform (pixel → meters)

### Phase 3: Event Detection
- Possession state machine
- Pass detection algorithm
- Shot detection
- Goal detection
- Event linking (shot → goal, pass → shot)

### Phase 4: Analytics & Output
- Distance/speed calculation
- Heatmap generation
- Player + team stats aggregation
- Video overlay renderer
- Bird's eye tactical view
- Match report generator (HTML/JSON)

---

## Key Files to Modify/Create

| File | Purpose |
|------|---------|
| [requirements.txt](../requirements.txt) | Add all dependencies (currently empty) |
| src/pipeline.py | Main orchestration |
| src/detection/ball_detector.py | TrackNetV3 + Kalman (solves ball tracking) |
| src/identification/jersey_ocr.py | OCR with voting |
| src/events/possession.py | State machine |
| [utils/click_points.py](../utils/click_points.py) | Enhance to save YAML configs |
| config/default.yaml | Default processing params |

---

## Dependencies (requirements.txt)

```
# Core ML
torch>=2.0.0
ultralytics>=8.0.0
onnxruntime-gpu>=1.15.0

# Tracking
numpy>=1.24.0
scipy>=1.10.0
filterpy>=1.4.5
lap>=0.4.0

# Computer Vision
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0

# OCR
paddlepaddle-gpu>=2.5.0
paddleocr>=2.6.0

# Video
av>=10.0.0
decord>=0.6.0

# Data & Utils
pydantic>=2.0.0
pandas>=2.0.0
pyyaml>=6.0
tqdm>=4.65.0
click>=8.1.0

# Visualization
matplotlib>=3.7.0
Pillow>=9.5.0

# Reports
jinja2>=3.1.0
weasyprint>=59.0
```

---

## Verification Plan

1. **Detection accuracy**: Run on trimmed test clip, manually verify player/ball detections
2. **Tracking persistence**: Check track IDs maintained through occlusions
3. **Team classification**: Verify correct team assignment visually
4. **Jersey OCR**: Compare detected numbers to ground truth
5. **Homography**: Overlay court lines on bird's eye view, verify alignment
6. **Events**: Manually validate detected passes/shots/goals against video
7. **End-to-end**: Process full match, review generated report and overlay video

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Ball tracking still poor with TrackNet | Fine-tune on futsal data; add orange ball color filtering |
| OCR unreliable on motion blur | Voting mechanism across frames; select clearest frames |
| Homography inaccurate at edges | More calibration points; local correction at court edges |
| 11GB file memory issues | Stream processing via PyAV; chunk-based inference |
| Team clustering fails (similar colors) | Manual team color hint in config; spatial priors |

---

## References

- Roboflow Basketball Player Identification: https://blog.roboflow.com/identify-basketball-players/
- TrackNet (ball tracking): https://nol.cs.nctu.edu.tw:234/open-source/TrackNet
- ByteTrack: https://github.com/ifzhang/ByteTrack
- YOLOv8 (Ultralytics): https://github.com/ultralytics/ultralytics
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
