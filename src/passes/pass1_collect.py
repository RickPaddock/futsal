"""
Pass 1: Raw Evidence Collection

CRITICAL CONSTRAINTS:
- ❌ NO team assignment
- ❌ NO jersey assignment (store probabilities only)
- ❌ NO identity locking
- ✅ Tracking uses ONLY IoU + motion (ByteTracker adapted)
- ✅ Track IDs are TEMPORARY and disposable

For each clip, for each frame, for each detection, store:
- Players: temp_track_id, bbox, centroid, HSV histogram, jersey probabilities (per-frame),
  occlusion score, detection confidence
- Ball: position, confidence, speed vector

Output: pass1_raw/<clip_name>.json
"""

from pathlib import Path
import json
import numpy as np
import cv2
from tqdm import tqdm

try:
    import orjson  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    orjson = None

from src.utils.video_io import VideoReader
from src.detection.player_detector import PlayerDetector
from src.detection.ball_detector import BallDetector
from src.detection.jersey_classifier import JerseyClassifier
from src.detection.tracking import ByteTracker
from src.detection.team_clustering import TeamClustering


def extract_color_histogram(frame: np.ndarray, bbox: tuple[float, float, float, float], bins: int = 32) -> np.ndarray:
    """
    Extract HSV color histogram from bbox region.

    CRITICAL: Must use exact same bins/order as ARCHIVE (32 bins × 3 channels = 96 elements).

    Args:
        frame: RGB frame
        bbox: (x1, y1, x2, y2)
        bins: Number of bins per channel (default 32)

    Returns:
        96-element HSV histogram (32 bins × 3 channels)
    """
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = int(np.ceil(x2)), int(np.ceil(y2))

    if x2 <= x1 or y2 <= y1:
        return np.zeros(bins * 3, dtype=np.float32)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros(bins * 3, dtype=np.float32)

    # Convert RGB to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # Compute per-channel histograms
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

    # Normalize and concatenate
    hist = np.concatenate([
        cv2.normalize(hist_h, hist_h).flatten(),
        cv2.normalize(hist_s, hist_s).flatten(),
        cv2.normalize(hist_v, hist_v).flatten()
    ]).astype(np.float32)

    return hist


def _quantize_histogram(hist: np.ndarray) -> list[int]:
    """
    Quantize HSV histogram from float -> uint8 [0-255].
    """
    if hist.size == 0:
        return []
    hist = np.clip(hist, 0.0, 1.0)
    return (hist * 255.0).round().astype(np.uint8).tolist()


def _int_round(value: float) -> int:
    return int(round(float(value)))


def run_pass1(input_dir: Path | None, run_dir: Path, config: dict, video_files: list[Path] | None = None):
    """
    Run Pass 1: Raw Evidence Collection.

    Args:
        input_dir: Directory containing video clips (if video_files not provided)
        run_dir: Run output directory (output/run_DDMMYY_HHMMSS)
        config: Configuration dictionary
        video_files: Optional list of specific video files to process
    """
    run_dir = Path(run_dir)
    output_dir = run_dir / "pass1_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video clips (either from explicit list or by globbing directory)
    if video_files is None:
        if input_dir is None:
            print("Error: Must provide either input_dir or video_files")
            return
        input_dir = Path(input_dir)
        video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.MP4"))
        if not video_files:
            print(f"No video files found in {input_dir}")
            return
    else:
        # Convert to Path objects if needed
        video_files = [Path(f) for f in video_files]

    print(f"Found {len(video_files)} video clips")

    # Initialize detectors
    player_detector = PlayerDetector(
        model_path=config['models']['player'],
        confidence_threshold=config['detection']['player_confidence'],
        max_detections=config['detection']['max_detections'],
        input_scale=0.33,  # Scale 4K → ~720p for speed
    )
    ball_detector = BallDetector(
        model_path=config['models']['ball'],
        confidence_threshold=config['detection']['ball_confidence'],
    )
    jersey_classifier = JerseyClassifier(
        model_path=config['models']['jersey'],
    )

    # Initialize ByteTracker (IoU + motion ONLY, NO team gating)
    tracker = ByteTracker(
        track_high_thresh=config['tracking']['track_high_thresh'],
        track_low_thresh=config['tracking']['track_low_thresh'],
        new_track_thresh=config['tracking']['new_track_thresh'],
        track_buffer=config['tracking']['track_buffer'],
        match_thresh=config['tracking']['match_thresh'],
        max_center_distance=config['tracking']['max_center_distance'],
        velocity_weight=config['tracking']['velocity_weight'],
    )

    # Process each clip
    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")
        process_clip(
            video_path=video_path,
            output_dir=output_dir,
            player_detector=player_detector,
            ball_detector=ball_detector,
            jersey_classifier=jersey_classifier,
            tracker=tracker,
            config=config,
        )

    print(f"\nPass 1 complete! Output: {output_dir}")


def process_clip(
    video_path: Path,
    output_dir: Path,
    player_detector: PlayerDetector,
    ball_detector: BallDetector,
    jersey_classifier: JerseyClassifier,
    tracker: ByteTracker,
    config: dict,
):
    """
    Process a single video clip for Pass 1.

    Args:
        video_path: Path to video file
        output_dir: Output directory for JSON
        player_detector: Player detector instance
        ball_detector: Ball detector instance
        jersey_classifier: Jersey classifier instance
        tracker: ByteTracker instance
        config: Configuration dictionary
    """
    # Reset tracker for new clip
    tracker.reset()

    # Jersey-histogram extractor (jersey crop + mask)
    jersey_hist_extractor = TeamClustering(bins=32)
    jersey_hist_quality_threshold = float(
        config.get("pass1", {}).get("jersey_hist_quality_threshold", TeamClustering.MIN_QUALITY)
    )

    # Open video
    reader = VideoReader(str(video_path))

    # Prepare output structure
    output_data = {
        "clip_name": video_path.name,
        "fps": reader.fps,
        "total_frames": reader.total_frames,
        "width": reader.width,
        "height": reader.height,
        "tracks": {}
    }

    # Process frames
    frame_stride = config.get('jersey', {}).get('frame_stride', 5)
    hsv_sample_stride = config.get('pass1', {}).get('hsv_sample_stride', 5)

    for frame_idx, frame in tqdm(reader.frames(), total=reader.total_frames, desc=f"{video_path.name}"):
        # Detect players
        player_detections = player_detector.detect(frame, frame_idx)

        # Detect ball
        ball_detections = ball_detector.detect_frame(frame, frame_idx)

        # Prepare detections for tracker (convert to numpy array)
        if player_detections:
            det_array = np.array([
                [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.bbox.confidence]
                for d in player_detections
            ], dtype=np.float32)
        else:
            det_array = np.empty((0, 5), dtype=np.float32)

        # Update tracker (IoU + motion ONLY, NO team gating)
        tracks = tracker.update(det_array, frame_idx)

        # Build track-centric player data
        for track in tracks:
            bbox = track.tlbr
            bbox_tuple = (bbox[0], bbox[1], bbox[2], bbox[3])

            track_id = str(track.track_id)
            track_data = output_data["tracks"].setdefault(track_id, {
                "frames": [],
                "bboxes": [],
                "centroids": [],
                "confidences": [],
                "occlusion_scores": [],
                "hsv_histograms": [],
                "team_probs": [],
                "jersey_decisions": [],
                "is_interpolated": []
            })

            frame_idx_int = int(frame_idx)
            bbox_list = [
                _int_round(bbox[0]),
                _int_round(bbox[1]),
                _int_round(bbox[2]),
                _int_round(bbox[3]),
            ]
            centroid_list = [
                _int_round((bbox[0] + bbox[2]) / 2),
                _int_round((bbox[1] + bbox[3]) / 2),
            ]

            # Compressed jersey decision (id + confidence) sampled every N frames
            jersey_id = None
            jersey_conf = 0.0
            if frame_idx_int % frame_stride == 0:
                from src.utils.data_models import BoundingBox as BBox
                bbox_obj = BBox(
                    x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                    confidence=track.score
                )
                jersey_probs = jersey_classifier.get_probabilities(frame, bbox_obj) or {}
                if jersey_probs:
                    jersey_id, jersey_conf = max(jersey_probs.items(), key=lambda kv: kv[1])

            # Compute occlusion score (1.0 - confidence as proxy)
            occlusion_score = 1.0 - track.score

            # Extract HSV histogram with ADAPTIVE SAMPLING
            # Sample every N frames normally, BUT also sample during high occlusion (close contact)
            # This densifies appearance data exactly when identity swaps are most likely
            adaptive_occlusion_threshold = config.get('pass1', {}).get('adaptive_occlusion_threshold', 0.3)
            should_sample = (frame_idx_int % hsv_sample_stride == 0) or (occlusion_score > adaptive_occlusion_threshold)

            hsv_hist_quantized = None
            if should_sample:
                hsv_hist, _, _, quality = jersey_hist_extractor.extract_features(frame, bbox_obj)
                if quality >= jersey_hist_quality_threshold and hsv_hist.size == 96:
                    hsv_hist_quantized = _quantize_histogram(hsv_hist)

            track_data["frames"].append(frame_idx_int)
            track_data["bboxes"].append(bbox_list)
            track_data["centroids"].append(centroid_list)
            track_data["confidences"].append(float(track.score))
            track_data["occlusion_scores"].append(float(occlusion_score))
            track_data["hsv_histograms"].append(hsv_hist_quantized)
            track_data["team_probs"].append(None)
            track_data["jersey_decisions"].append([jersey_id, float(jersey_conf)])
            track_data["is_interpolated"].append(False)  # TODO: detect Kalman predictions

        # Ball data
        if ball_detections:
            # Take highest confidence ball
            best_ball = max(ball_detections, key=lambda b: b.bbox.confidence)
            ball_data = output_data.setdefault("ball", {
                "frames": [],
                "positions": [],
                "confidences": [],
                "speed_vectors": []
            })
            ball_data["frames"].append(int(frame_idx))
            ball_data["positions"].append([
                _int_round((best_ball.bbox.x1 + best_ball.bbox.x2) / 2),
                _int_round((best_ball.bbox.y1 + best_ball.bbox.y2) / 2)
            ])
            ball_data["confidences"].append(float(best_ball.bbox.confidence))
            ball_data["speed_vectors"].append([0.0, 0.0])  # TODO: compute from temporal tracking

    # Save JSON
    pretty_json = config.get("pass1", {}).get("pretty_json", True)
    output_path = output_dir / f"{video_path.stem}.json"
    if orjson is not None:
        with open(output_path, "wb") as f:
            if pretty_json:
                f.write(orjson.dumps(output_data, option=orjson.OPT_INDENT_2))
            else:
                f.write(orjson.dumps(output_data))
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty_json:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            else:
                json.dump(output_data, f, separators=(",", ":"), ensure_ascii=False)

    print(f"  Saved: {output_path}")
    total_detections = sum(len(t["frames"]) for t in output_data["tracks"].values())
    print(f"  Tracks: {len(output_data['tracks'])}")
    print(f"  Total player detections: {total_detections}")
