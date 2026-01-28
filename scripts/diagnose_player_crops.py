"""
Diagnostic: sample crops, fit k-means, predict teams, and save per-player
crops so we can visually inspect what's going wrong.

Output:  videos/output/player_crops/<frame>_<det>/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import cv2
import numpy as np
import yaml

from src.utils.video_io import VideoReader
from src.detection.player_detector import PlayerDetector
from src.detection.team_clustering import TeamClustering
from src.utils.data_models import BoundingBox, TeamID

# ── Config ──────────────────────────────────────────────────────────
VIDEO   = Path("videos/input/GoPro_Futsal_part1_CLEANED.mp4")
OUT_DIR = Path("videos/output/player_crops")
CONFIG  = Path("config/default.yaml")

SAMPLE_FRAMES  = 10   # frames for k-means fitting (same as pipeline pre-pass)
PREDICT_FRAMES = 3    # extra frames to predict & save crops
FRAME_STEP     = 1    # step between frames


def main():
    import shutil
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    det_cfg = cfg["player_detection"]
    detector = PlayerDetector(
        model_path=det_cfg["model"],
        confidence_threshold=det_cfg["confidence_threshold"],
        iou_threshold=det_cfg["iou_threshold"],
        device=cfg["processing"]["device"],
        classes=det_cfg.get("classes", [0]),
        max_detections=det_cfg.get("max_detections", 12),
        input_scale=det_cfg.get("input_scale", 1.0),
    )

    team_cfg = cfg["team_classification"]
    tc = TeamClustering(
        bins=team_cfg.get("histogram_bins", 32),
        n_clusters=team_cfg.get("n_clusters", 2),
        min_samples=team_cfg.get("min_samples", 10),
        max_samples=team_cfg.get("max_samples", 200),
        force_palette=team_cfg.get("force_team_colors", {}),
        vividify=team_cfg.get("vividify", True),
    )

    reader = VideoReader(VIDEO)
    print(f"Video: {reader.width}x{reader.height} @ {reader.fps:.1f} fps, "
          f"{reader.total_frames} frames")

    # ── Phase 1: sample for k-means ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 1: Sampling {SAMPLE_FRAMES} frames for k-means")
    print(f"{'='*60}")

    sampled = 0
    low_quality_skipped = 0
    for frame_idx, frame in reader.frames(step=FRAME_STEP):
        if sampled >= tc.max_samples or frame_idx >= SAMPLE_FRAMES * FRAME_STEP:
            break
        detections = detector.detect(frame, frame_idx)
        for det in detections:
            hist, _, _, quality = tc.extract_features(frame, det.bbox)
            if quality >= tc.MIN_QUALITY:
                tc.add_sample(frame, det.bbox, frame_idx=frame_idx)
                sampled += 1
            else:
                low_quality_skipped += 1

    print(f"  Samples collected: {tc.sample_count}")
    print(f"  Low-quality skipped: {low_quality_skipped}")

    if not tc.ready:
        print(f"  NOT ENOUGH SAMPLES (need {tc.min_samples}). Aborting.")
        return

    tc.fit()
    print(f"  K-means fitted. Palette: {tc.describe_palette()}")
    print(f"  label_to_team: {tc.label_to_team}")

    # Show cluster sizes
    for label in range(tc.n_clusters):
        count = int(np.sum(tc.cluster_labels == label))
        team = tc.label_to_team.get(label, "?")
        print(f"  Cluster {label} ({team}): {count} samples")

    # ── Phase 2: predict and save crops ────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 2: Predicting teams and saving crops")
    print(f"{'='*60}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames_saved = 0
    for frame_idx, frame in reader.frames(step=30):
        if frames_saved >= PREDICT_FRAMES:
            break

        detections = detector.detect(frame, frame_idx)
        if len(detections) < 8:
            continue

        frames_saved += 1
        print(f"\n── Frame {frame_idx}: {len(detections)} detections ──")

        for det_i, det in enumerate(detections):
            bbox = det.bbox
            hist, mean_bgr, crop_bgr, quality = tc.extract_features(frame, bbox)
            predicted_team = tc.predict(hist) if quality >= tc.MIN_QUALITY else None
            team_str = predicted_team.value if predicted_team else "SKIP(low_q)"

            tag = f"f{frame_idx:05d}_d{det_i:02d}_{team_str}"
            crop_dir = OUT_DIR / tag
            crop_dir.mkdir(parents=True, exist_ok=True)

            # Crop region
            x1, y1, x2, y2 = tc._crop_jersey_region(bbox, frame.shape)
            if x2 <= x1 or y2 <= y1:
                continue
            roi_rgb = frame[y1:y2, x1:x2]
            roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)

            # Mask
            mask = tc._jersey_mask(roi_rgb)
            if mask is not None and np.any(mask):
                masked_rgb = cv2.bitwise_and(roi_rgb, roi_rgb, mask=mask)
                masked_bgr = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR)
            else:
                masked_bgr = np.zeros_like(roi_bgr)

            # Full bbox
            bx1, by1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
            bx2, by2 = min(frame.shape[1], int(bbox.x2)), min(frame.shape[0], int(bbox.y2))
            full_bbox_bgr = cv2.cvtColor(frame[by1:by2, bx1:bx2], cv2.COLOR_RGB2BGR)

            # HSV stats
            hsv_raw = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
            h_r, s_r, v_r = [np.mean(hsv_raw[:,:,c]) for c in range(3)]

            # Save
            cv2.imwrite(str(crop_dir / "1_full_bbox.jpg"), full_bbox_bgr)
            cv2.imwrite(str(crop_dir / "2_raw_crop.jpg"), roi_bgr)
            if mask is not None:
                cv2.imwrite(str(crop_dir / "3_mask.jpg"), mask)
            cv2.imwrite(str(crop_dir / "4_masked_crop.jpg"), masked_bgr)

            print(f"  [{tag}]  q={quality:.1%}  team={team_str:10s}  "
                  f"raw_HSV=({h_r:.0f},{s_r:.0f},{v_r:.0f})  "
                  f"mean_bgr=({mean_bgr[0]:.0f},{mean_bgr[1]:.0f},{mean_bgr[2]:.0f})")

    print(f"\nCrops saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
