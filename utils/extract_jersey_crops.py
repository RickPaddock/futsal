#!/usr/bin/env python3
"""Extract jersey-focused crops to build a digit classification dataset."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# Ensure repository root is importable when executed as a script.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.detection.player_detector import PlayerDetector, filter_detections_by_size
from src.utils.data_models import PlayerDetection
from src.utils.video_io import VideoReader


@dataclass(frozen=True)
class CropVariant:
    """Relative bbox slice expressed as start/end ratios of bbox height."""

    name: str
    y_start_ratio: float
    y_end_ratio: float


def compute_color_histogram(image_rgb: np.ndarray) -> np.ndarray:
    """Return normalized HSV histogram for coarse appearance matching."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 6], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_player_seeds(seed_specs: list[str]) -> Dict[str, np.ndarray]:
    """Load player seed images and precompute histograms."""
    seeds: Dict[str, np.ndarray] = {}
    for spec in seed_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --player-seed format '{spec}', expected label=path")
        label, path_str = spec.split("=", 1)
        label = label.strip()
        seed_path = Path(path_str).expanduser().resolve()
        if not seed_path.exists():
            raise FileNotFoundError(f"Player seed image not found: {seed_path}")
        seed_bgr = cv2.imread(str(seed_path))
        if seed_bgr is None:
            raise RuntimeError(f"Failed to read seed image: {seed_path}")
        seed_rgb = cv2.cvtColor(seed_bgr, cv2.COLOR_BGR2RGB)
        seeds[label] = compute_color_histogram(seed_rgb)
    return seeds


def classify_player(
    crop_rgb: np.ndarray,
    seeds: Dict[str, np.ndarray],
    threshold: float,
) -> Tuple[str, float]:
    """Assign crop to best matching player seed; returns (label, similarity)."""
    if not seeds:
        return "unknown", 0.0
    hist = compute_color_histogram(crop_rgb)
    best_label = "unknown"
    best_sim = 0.0
    for label, seed_hist in seeds.items():
        sim = cosine_similarity(hist, seed_hist)
        if sim > best_sim:
            best_sim = sim
            best_label = label
    if best_sim >= threshold:
        return best_label, best_sim
    return "unknown", best_sim


def is_orange_crop(
    crop_rgb: np.ndarray,
    hue_range: Tuple[float, float],
    sat_thresh: float,
    val_thresh: float,
) -> bool:
    """Heuristic to flag orange bibs using mean HSV values."""
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    mean_h = float(np.mean(hsv[:, :, 0]))
    mean_s = float(np.mean(hsv[:, :, 1]))
    mean_v = float(np.mean(hsv[:, :, 2]))
    low, high = hue_range
    if low <= mean_h <= high and mean_s >= sat_thresh and mean_v >= val_thresh:
        return True
    return False


def load_config(custom_config: Path | None) -> dict:
    root_dir = Path(__file__).resolve().parents[1]
    default_path = root_dir / "config" / "default.yaml"
    with open(default_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if custom_config:
        with open(custom_config, "r", encoding="utf-8") as fh:
            override = yaml.safe_load(fh) or {}
        config = deep_merge(config, override)

    return config


def deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_crop_variants(names: Iterable[str], top_ratio: float) -> list[CropVariant]:
    variants: list[CropVariant] = []
    for name in names:
        key = name.lower()
        if key == "top":
            variants.append(CropVariant("top", -0.05, min(1.05, top_ratio + 0.1)))
        elif key == "torso":
            variants.append(CropVariant("torso", -0.05, 0.85))
        elif key == "full":
            variants.append(CropVariant("full", -0.1, 1.05))
        else:
            raise ValueError(f"Unknown crop variant '{name}'")
    return variants


def passes_jersey_gate(detection: PlayerDetection, min_conf: float, min_aspect_ratio: float) -> bool:
    if detection.bbox.confidence < min_conf:
        return False
    width = detection.bbox.width
    height = detection.bbox.height
    if width <= 0 or height <= 0:
        return False
    aspect = height / max(width, 1e-6)
    return aspect >= min_aspect_ratio


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def should_keep_sample(
    center: tuple[float, float],
    frame_idx: int,
    recent: deque[tuple[int, tuple[float, float]]],
    min_frame_gap: int,
    min_center_distance: float,
) -> bool:
    for past_frame, past_center in recent:
        if frame_idx - past_frame <= min_frame_gap:
            if distance(center, past_center) < min_center_distance:
                return False
        else:
            break
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract jersey crops for digit labelling.")
    parser.add_argument("video", type=Path, help="Path to input video")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("videos/output/frames_training/JERSEY"),
        help="Directory for saved crops",
    )
    parser.add_argument("--config", "-c", type=Path, help="Optional custom config YAML")
    parser.add_argument("--frame-stride", type=int, default=5, help="Sample every Nth frame (after config gating)")
    parser.add_argument("--min-confidence", type=float, help="Minimum detection confidence (default from config)")
    parser.add_argument("--min-aspect", type=float, help="Minimum height/width ratio (default from config)")
    parser.add_argument("--max-crops", type=int, default=2000, help="Stop after saving this many crops")
    parser.add_argument("--min-frame-gap", type=int, default=8, help="Min frames between saved crops from nearby centers")
    parser.add_argument("--min-center-distance", type=float, default=70.0, help="Min pixel distance between saved crop centers within gap window")
    parser.add_argument("--crop-variants", nargs="+", default=["top"], help="Crop variants to export: top, torso, full")
    parser.add_argument("--crop-pad-y", type=float, default=0.12, help="Add this fraction of bbox height above/below each crop")
    parser.add_argument("--crop-pad-x", type=float, default=0.08, help="Add this fraction of bbox width to left/right of each crop")
    parser.add_argument("--min-crop-height", type=int, default=60, help="Skip crops shorter than this many pixels")
    parser.add_argument("--min-player-height", type=int, default=80, help="Filter detections shorter than this many pixels")
    parser.add_argument("--max-player-height", type=int, help="Filter detections taller than this many pixels (defaults to 90% of frame height)")
    parser.add_argument("--device", type=str, help="Override device for detector (cuda or cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Run detector but do not write files (prints counts only)")
    parser.set_defaults(clean_output=True)
    parser.add_argument(
        "--no-clean-output",
        dest="clean_output",
        action="store_false",
        help="Keep any existing crops/manifest instead of clearing the output directory",
    )
    parser.add_argument(
        "--ignore-orange",
        action="store_true",
        help="Skip crops whose average HSV color matches orange bib jerseys",
    )
    parser.add_argument(
        "--allow-orange",
        dest="ignore_orange",
        action="store_false",
        help="Keep orange bib crops (overrides --ignore-orange)",
    )
    parser.set_defaults(ignore_orange=True)
    parser.add_argument(
        "--orange-hue-range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=(5.0, 28.0),
        help="Hue range (0-180) considered orange for --ignore-orange",
    )
    parser.add_argument(
        "--orange-sat-thresh",
        type=float,
        default=90.0,
        help="Minimum saturation (0-255) to treat crop as orange",
    )
    parser.add_argument(
        "--orange-val-thresh",
        type=float,
        default=60.0,
        help="Minimum value/brightness (0-255) to treat crop as orange",
    )
    parser.add_argument(
        "--player-seed",
        action="append",
        default=[],
        help="Player label and seed image path in the form label=path; repeat for multiple players",
    )
    parser.add_argument(
        "--player-threshold",
        type=float,
        default=0.75,
        help="Minimum cosine similarity to assign crop to a player seed",
    )
    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    config = load_config(args.config)
    device = args.device or config.get("processing", {}).get("device", "cuda")
    player_cfg = config.get("player_detection", {})
    jersey_cfg = config.get("jersey_identification", {})

    min_conf = args.min_confidence if args.min_confidence is not None else float(jersey_cfg.get("gate_confidence", 0.75))
    min_aspect = args.min_aspect if args.min_aspect is not None else float(jersey_cfg.get("gate_min_aspect_ratio", 1.3))
    top_ratio = float(jersey_cfg.get("top_crop_ratio", 0.5))

    variants = build_crop_variants(args.crop_variants, top_ratio)
    player_seeds = load_player_seeds(args.player_seed)

    detector = PlayerDetector(
        model_path=player_cfg.get("model", "models/PLAYER_MODEL_best_v1.pt"),
        confidence_threshold=player_cfg.get("confidence_threshold", 0.3),
        iou_threshold=player_cfg.get("iou_threshold", 0.4),
        device=device,
        classes=player_cfg.get("classes", [0]),
        use_roboflow=player_cfg.get("use_roboflow", False),
        roboflow_model_id=player_cfg.get("roboflow_model_id"),
        max_detections=player_cfg.get("max_detections", 12),
        input_scale=player_cfg.get("input_scale", 1.0),
    )

    reader = VideoReader(args.video)
    output_dir = args.output
    crops_dir = output_dir / "crops"
    if not args.dry_run:
        if args.clean_output and output_dir.exists():
            shutil.rmtree(output_dir)
        crops_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    manifest_file = None
    manifest_writer = None
    if not args.dry_run:
        manifest_exists = manifest_path.exists()
        manifest_mode = "a" if (not args.clean_output and manifest_exists) else "w"
        manifest_file = open(manifest_path, manifest_mode, newline="", encoding="utf-8")
        manifest_writer = csv.writer(manifest_file)
        if manifest_mode == "w" or manifest_file.tell() == 0:
            manifest_writer.writerow([
                "filename",
                "frame_idx",
                "timestamp_sec",
                "variant",
                "confidence",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "player_label",
                "player_similarity",
            ])

    saved = 0
    total_examined = 0
    fps = reader.fps if reader.fps > 0 else 30.0
    recent_samples: deque[tuple[int, tuple[float, float]]] = deque(maxlen=1000)
    max_player_height = args.max_player_height or int(reader.height * 0.9)
    player_dirs: Dict[str, Path] = {}
    video_tag = Path(args.video).stem.replace(" ", "_")

    total_frames = reader.total_frames or 0
    tqdm_total = None
    if total_frames > 0:
        tqdm_total = max(1, total_frames // max(1, args.frame_stride))

    try:
        for frame_idx, frame in tqdm(reader.frames(step=args.frame_stride), total=tqdm_total, desc="Scanning"):
            detections = detector.detect(frame, frame_idx)
            detections = filter_detections_by_size(
                detections,
                min_height=args.min_player_height,
                max_height=max_player_height,
            )
            for det_idx, det in enumerate(detections):
                total_examined += 1
                if not passes_jersey_gate(det, min_conf, min_aspect):
                    continue

                center = det.bbox.center
                if not should_keep_sample(center, frame_idx, recent_samples, args.min_frame_gap, args.min_center_distance):
                    continue

                frame_bgr = None
                frame_height, frame_width = frame.shape[:2]
                saved_this_det = False

                for variant in variants:
                    height = det.bbox.height
                    width = det.bbox.width
                    pad_y = max(0.0, args.crop_pad_y)
                    pad_x = max(0.0, args.crop_pad_x)

                    y1 = det.bbox.y1 + (variant.y_start_ratio - pad_y) * height
                    y2 = det.bbox.y1 + (variant.y_end_ratio + pad_y) * height
                    x1 = det.bbox.x1 - pad_x * width
                    x2 = det.bbox.x2 + pad_x * width

                    x1_i = max(0, min(int(np.floor(x1)), frame_width - 1))
                    y1_i = max(0, min(int(np.floor(y1)), frame_height - 1))
                    x2_i = max(0, min(int(np.ceil(x2)), frame_width))
                    y2_i = max(0, min(int(np.ceil(y2)), frame_height))

                    if x2_i <= x1_i or y2_i <= y1_i:
                        continue

                    if (y2_i - y1_i) < args.min_crop_height:
                        continue

                    crop_rgb = frame[y1_i:y2_i, x1_i:x2_i]
                    if crop_rgb.size == 0:
                        continue

                    if (
                        args.ignore_orange
                        and is_orange_crop(
                            crop_rgb,
                            tuple(args.orange_hue_range),
                            args.orange_sat_thresh,
                            args.orange_val_thresh,
                        )
                    ):
                        continue

                    player_label = "unknown"
                    player_similarity = 0.0
                    if player_seeds:
                        player_label, player_similarity = classify_player(
                            crop_rgb,
                            player_seeds,
                            args.player_threshold,
                        )

                    if args.dry_run:
                        saved += 1
                        saved_this_det = True
                        continue

                    if frame_bgr is None:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    crop = frame_bgr[y1_i:y2_i, x1_i:x2_i]
                    base_name = f"{video_tag}_frame{frame_idx:06d}_det{det_idx:02d}_{variant.name}.jpg"
                    if player_label not in player_dirs:
                        player_dir = crops_dir / player_label
                        player_dir.mkdir(parents=True, exist_ok=True)
                        player_dirs[player_label] = player_dir
                    filepath = player_dirs[player_label] / base_name
                    if filepath.exists():
                        suffix = 1
                        stem = filepath.stem
                        ext = filepath.suffix
                        while filepath.exists():
                            filepath = player_dirs[player_label] / f"{stem}_{suffix}{ext}"
                            suffix += 1
                    cv2.imwrite(str(filepath), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    if manifest_writer is not None:
                        manifest_writer.writerow([
                            filepath.relative_to(crops_dir).as_posix(),
                            frame_idx,
                            frame_idx / fps,
                            variant.name,
                            f"{det.bbox.confidence:.4f}",
                            f"{det.bbox.x1:.1f}",
                            f"{det.bbox.y1:.1f}",
                            f"{det.bbox.x2:.1f}",
                            f"{det.bbox.y2:.1f}",
                            player_label,
                            f"{player_similarity:.4f}",
                        ])

                    saved += 1
                    saved_this_det = True

                    if saved >= args.max_crops:
                        break

                if saved_this_det:
                    recent_samples.appendleft((frame_idx, center))

                if saved >= args.max_crops:
                    break

            if saved >= args.max_crops:
                break
    finally:
        if manifest_file is not None:
            manifest_file.close()

    print(f"Examined detections: {total_examined}")
    print(f"Saved crops: {saved}")
    if not args.dry_run:
        print(f"Output directory: {crops_dir}")
        print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
