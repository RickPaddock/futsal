# Testing Guide - Futsal Tracking System

This guide explains how to test each pass of the pipeline as it's implemented.

---

## ✅ Pass 1: Raw Evidence Extraction (READY TO TEST!)

**Status**: ✅ Implemented and ready for testing

### Prerequisites

1. **Python Dependencies**
   ```bash
   pip install ultralytics numpy opencv-python scikit-learn pydantic tqdm av
   ```

2. **YOLO Models** (place in `models/` directory)
   - `PLAYER_MODEL_best_v1.pt` - Player detection
   - `BALL_MODEL_best_v2.pt` - Ball detection
   - `JERSEY_MODEL_best_v1.pt` - Jersey classification

3. **Test Video** (place in `videos/input/`)
   - `GoPro_Futsal_part1_CLEANED_clip9.mp4`

### Running the Test

```bash
# Run Pass 1 on test clip (first 100 frames)
python test_pass1.py
```

**What this does:**
- Loads the test video
- Runs YOLO player detection with multi-layer bbox defense
- Runs ByteTrack to assign temporary track_ids
- Classifies jersey numbers (conf ≥ 0.3)
- Extracts HSV color histograms
- Detects the ball
- Validates all outputs (R1 enforcement)
- Saves `pass1_raw.json` and `pass1_validation.json`

### Expected Output

```
Pass 1: Raw Evidence Extraction - Test
================================================================================

Input video: videos/input/GoPro_Futsal_part1_CLEANED_clip9.mp4
Output directory: videos/output/GoPro_Futsal_part1_CLEANED_clip9
Processing frames: 0 to 100

Pass 1: Extracting raw evidence: 100%|████████████| 100/100 [00:XX<00:00]

================================================================================
✅ Pass 1 Complete!
================================================================================

Player detections: ~800-1200
Ball detections: ~90-100

Output files:
  - videos/output/GoPro_Futsal_part1_CLEANED_clip9/pass1_raw.json
  - videos/output/GoPro_Futsal_part1_CLEANED_clip9/pass1_validation.json
```

### Inspecting the Results

**1. Check validation report:**
```bash
cat videos/output/GoPro_Futsal_part1_CLEANED_clip9/pass1_validation.json
```

Should show:
```json
{
  "passed": true,
  "violations": [],
  "warnings": [],
  "timestamp": "2026-02-17T..."
}
```

**2. Inspect detections (first 50 lines):**
```bash
cat videos/output/GoPro_Futsal_part1_CLEANED_clip9/pass1_raw.json | head -50
```

Should show detections with:
- ✅ `detection_id` (unique per detection)
- ✅ `frame_idx`, `track_id` (temporary)
- ✅ `bbox`, `confidence`
- ✅ `jersey_number`, `jersey_confidence` (or null)
- ✅ `hsv_histogram` (512-element array)
- ❌ NO `team` field (R1 enforcement)
- ❌ NO `player_id` field (R1 enforcement)

**3. Check metrics:**
```bash
# Count total detections
cat videos/output/GoPro_Futsal_part1_CLEANED_clip9/pass1_raw.json | grep '"detection_id"' | wc -l

# Count ball detections
cat videos/output/GoPro_Futsal_part1_CLEANED_clip9/pass1_raw.json | grep '"state": "real"' | wc -l

# Count jersey detections
cat videos/output/GoPro_Futsal_part1_CLEANED_clip9/pass1_raw.json | grep '"jersey_number":' | grep -v null | wc -l
```

### Troubleshooting

**Error: Video not found**
```
❌ Error: Video not found: videos/input/GoPro_Futsal_part1_CLEANED_clip9.mp4
```
- Ensure the video is at the correct path
- Check file name (case-sensitive on Linux/Mac)

**Error: Model not found**
```
FileNotFoundError: Player model not found: models/PLAYER_MODEL_best_v1.pt
```
- Place YOLO model files in `models/` directory
- Check model file names match exactly

**Error: ultralytics not found**
```
ImportError: ultralytics package not found
```
- Install: `pip install ultralytics`

**Validation failed: Huge bbox detected**
```
Pass 1 validation failed with 1 violations:
R1 violation: Huge bbox detected...
```
- This is EXPECTED behavior! The multi-layer defense is working
- The bbox was filtered at source (not included in output)
- Check the warning logs to see which bboxes were filtered

### What Gets Validated (R1)

Pass 1 validation checks:
- ✅ No `team` field in detections
- ✅ No `player_id` field in detections
- ✅ All bboxes within frame bounds
- ✅ All bboxes pass huge bbox defense (≤800px height, ≤600px width, ≤25% area)
- ✅ Jersey probabilities valid (0-1 range)
- ✅ HSV histograms valid (512 elements, normalized)
- ✅ Max 1 ball per frame

---

## 🚧 Pass 2: Mechanical Fragmentation (NOT YET IMPLEMENTED)

**Status**: ⏳ Pending implementation

**What it will do:**
- Split Pass 1 tracks into fragments
- Detect divergence points (jersey changes, appearance drift, etc.)
- Keep ALL fragments (mark short ones as low_quality)
- Merge consecutive short fragments

**Test command** (when implemented):
```bash
python test_pass2.py
```

---

## 🚧 Pass 3: Identity Resolution (NOT YET IMPLEMENTED)

**Status**: ⏳ Pending implementation (Pass 3C solver is ready)

**What it will do:**
- Generate identity candidates (Pass 3A)
- Build constraint graph (Pass 3B)
- Commit identities (Pass 3C) - 🔒 LOCK POINT

**Test command** (when implemented):
```bash
python test_pass3.py
```

---

## 🚧 Full Pipeline (NOT YET IMPLEMENTED)

**Status**: ⏳ Pending implementation

**Test command** (when implemented):
```bash
python src/main.py --input videos/input/GoPro_Futsal_part1_CLEANED_clip9.mp4
```

---

## Progress Tracking

| Pass | Status | Test Script | Output Artifacts |
|------|--------|-------------|------------------|
| **Pass 1** | ✅ Ready | `test_pass1.py` | `pass1_raw.json`, `pass1_validation.json` |
| Pass 2A | ⏳ Pending | - | `pass2_fragments.json` |
| Pass 2B | ⏳ Pending | - | (adds quality scores) |
| Pass 2C | ⏳ Pending | - | `pass2_ghosts.json`, `pass2_validation.json` |
| Pass 3A | ⏳ Pending | - | `pass3_candidates.json` |
| Pass 3B | ⏳ Pending | - | `pass3_constraints.json` |
| Pass 3C | ✅ Ready | - | `pass3_identity_commit.json`, `pass3_validation.json` |
| Ball | ⏳ Pending | - | `ball_interpolation.json` |
| Viz | ⏳ Pending | - | `visualization.mp4` |

---

**Last Updated**: 2026-02-17
