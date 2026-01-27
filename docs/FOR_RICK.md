# Ball Detection - Training Workflow

## Quick Summary

```
FULL 4K FRAMES → Label in Roboflow → Tile (with overlap) → Train YOLO
```

---

## Step 1: Frame Selection

**Goal:** ~500-1000 frames with ball visible

**Good frames:**
- Ball clearly visible (not blurred)
- Ball in different court positions (spread across frame)
- Mix of: stationary, rolling, in-air, near feet

**Avoid:**
- Ball off-screen
- Heavy motion blur
- Ball fully occluded

**Current dataset:** 718 frames from regular sampling (~every 0.5s). Adequate for testing on same video.

---

## Step 2: Label in Roboflow

- Upload **full 4K frames** (3840x2160)
- Draw bounding box around ball
- Export as **YOLOv11 format**
- Extract to: `models/datasets/BALL/<export_name>/`

**Why full resolution?** Roboflow uses normalized coordinates (0-1), which get correctly transformed when tiling.

---

## Step 3: Tile the Dataset

```bash
python utils/tile_dataset.py \
    --input "models/datasets/BALL/GoPro_BALL_v1.v12-original_size_v1.yolov11" \
    --output "models/datasets/BALL/GoPro_BALL_v1_tiled_overlap" \
    --keep-empty 0.0
```

**What this does:**
- Splits each 4K frame into 4 overlapping tiles (2120x1280 each)
- **200px overlap (~10%)** - optimal for small objects per [SAHI research](https://github.com/obss/sahi)
- `--keep-empty 0.0` = only saves tiles containing the ball (avoids 75% empty tiles)

**Output:** ~800-1000 tiles (vs 2872 if keeping empties)

---

## Step 4: Train

```bash
python utils/train_yolo.py --task ball
```

- Model: YOLOv11x (best for small objects)
- Input: 640x640 (tiles auto-resized)
- Output: `models/BALL_MODEL_best_v2.pt`

---

## Step 5: Update Config

Edit `config/default.yaml`:
```yaml
ball_detection:
  model: "models/BALL_MODEL_best_v2.pt"
  use_inference_slicer: true
```

---

## Key Points

| Question | Answer |
|----------|--------|
| Label on tiles or full res? | **Full resolution** |
| Overlap before or after training? | **Before** (tiling script adds it) |
| Keep empty tiles? | **No** (`--keep-empty 0.0`) |
| Why tile? | Ball is tiny in 4K; tiling makes it larger relative to 640x640 input |
| Why 200px overlap? | ~10% overlap is optimal per research (6-10% range) |
