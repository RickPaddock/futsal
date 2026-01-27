"""
Extract high-value frames from futsal video for training detectors.

Two modes:
- PLAYER mode: Frames with occlusions, congestion, detection failures
- BALL mode: Diverse frames across the video (for small/blurry ball detection)

Uses batch processing - stops early once enough high-quality frames are found.

Usage:
    python utils/extract_training_frames.py --mode player
    python utils/extract_training_frames.py --mode ball

Requirements:
    pip install ultralytics opencv-python numpy
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from dataclasses import dataclass
from ultralytics import YOLO


# ============================================================================
# CONFIGURATION (edit these values directly)
# ============================================================================

INPUT_VIDEO = "videos/input/GoPro_Futsal_part1_CLEANED.mp4"
OUTPUT_FOLDER_PLAYER = "videos/output/frames_training_players"
OUTPUT_FOLDER_BALL = "videos/output/frames_training/BALL"
YOLO_MODEL = "models/yolo11x.pt"  # Will download if not present (for player mode)

# Frame sampling - PLAYER mode (for complex player scenarios)
PLAYER_FRAME_SKIP = 2  # Process every Nth frame (1 = all frames, 2 = every other)
PLAYER_TARGET_FRAMES = 750  # Target number of frames to extract (700-800 range)
PLAYER_MIN_TEMPORAL_GAP_SEC = 0.5  # Minimum seconds between selected frames

# Frame sampling - BALL mode (for diverse ball appearances)
BALL_FRAME_SKIP = 15  # Sample less frequently for diversity
BALL_TARGET_FRAMES = 1500  # Need more frames due to ball being small/blurry
BALL_MIN_TEMPORAL_GAP_SEC = 0.5  # Larger gaps for more diversity

# Batch processing - PLAYER mode
PLAYER_BATCH_SIZE = 1000  # Frames to process per batch (before checking if done)
PLAYER_MIN_SCORE_THRESHOLD = 15.0  # Minimum score to consider a frame "high-value"
PLAYER_EARLY_STOP_MULTIPLIER = 1.5  # Stop when candidates >= target * multiplier

# Expected players on court - PLAYER mode only
EXPECTED_PLAYERS = 12  # 5v5 + 2 goalkeepers + refs (adjustable)

# Scoring weights - PLAYER mode only
W_UNDERCOUNT = 10.0  # Weight for missing players (detected < expected)
W_MEAN_IOU = 5.0  # Weight for average IoU between boxes (crowding)
W_LARGE_BOX = 3.0  # Weight for unusually large boxes (merged detections)
W_COUNT_CHANGE = 2.0  # Weight for rapid detection count changes

# Detection thresholds - PLAYER mode only
CONFIDENCE_THRESHOLD = 0.3  # YOLO confidence threshold
PERSON_CLASS_ID = 0  # COCO class ID for 'person'
LARGE_BOX_AREA_RATIO = 0.02  # Box area / frame area threshold for "large"

# Easy frame sampling - PLAYER mode only
EASY_FRAME_RATIO = 0.05  # Fraction of easy frames to include for balance


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FrameData:
    """Stores detection data and score for a single frame."""
    frame_idx: int
    timestamp: float
    detection_count: int
    mean_iou: float
    large_box_ratio: float
    score: float

    def __lt__(self, other):
        """For heap comparison - lower score = lower priority."""
        return self.score < other.score


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def compute_mean_pairwise_iou(boxes: np.ndarray) -> float:
    """Compute mean IoU across all pairs of boxes."""
    n = len(boxes)
    if n < 2:
        return 0.0

    total_iou = 0.0
    pair_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            total_iou += compute_iou(boxes[i], boxes[j])
            pair_count += 1

    return total_iou / pair_count if pair_count > 0 else 0.0


def compute_large_box_ratio(boxes: np.ndarray, frame_area: float) -> float:
    """Compute ratio of boxes that are unusually large."""
    if len(boxes) == 0:
        return 0.0

    large_count = 0
    for box in boxes:
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        if box_area / frame_area > LARGE_BOX_AREA_RATIO:
            large_count += 1

    return large_count / len(boxes)


def compute_frame_score(
    detection_count: int,
    mean_iou: float,
    large_box_ratio: float,
    prev_count: int | None
) -> float:
    """
    Compute difficulty score for a frame.
    Higher score = more valuable for training.
    """
    # Undercount penalty
    undercount = max(0, EXPECTED_PLAYERS - detection_count)
    undercount_score = W_UNDERCOUNT * undercount

    # Crowding score (high IoU = overlapping boxes)
    crowding_score = W_MEAN_IOU * mean_iou

    # Large box score (potential merged detections)
    large_box_score = W_LARGE_BOX * large_box_ratio

    # Count change score
    count_change_score = 0.0
    if prev_count is not None:
        count_change_score = W_COUNT_CHANGE * abs(detection_count - prev_count)

    return undercount_score + crowding_score + large_box_score + count_change_score


def format_timestamp(seconds: float) -> str:
    """Format timestamp for filename."""
    return f"{seconds:.1f}".replace(".", "_")


def can_select_frame(timestamp: float, selected_times: list[float], min_gap: float) -> bool:
    """Check if frame can be selected given temporal spacing constraint."""
    for t in selected_times:
        if abs(timestamp - t) < min_gap:
            return False
    return True


def select_frames_with_spacing(
    frame_data: list[FrameData],
    target_count: int,
    min_gap_sec: float
) -> list[FrameData]:
    """Select top frames while enforcing minimum temporal spacing."""
    sorted_frames = sorted(frame_data, key=lambda x: x.score, reverse=True)

    selected = []
    selected_times = []

    for frame in sorted_frames:
        if len(selected) >= target_count:
            break

        if can_select_frame(frame.timestamp, selected_times, min_gap_sec):
            selected.append(frame)
            selected_times.append(frame.timestamp)

    return selected


def add_easy_frames(
    all_frames: list[FrameData],
    selected_frames: list[FrameData],
    easy_count: int,
    min_gap_sec: float
) -> list[FrameData]:
    """Add some easy frames (low score) for training balance."""
    sorted_frames = sorted(all_frames, key=lambda x: x.score)

    selected_times = [f.timestamp for f in selected_frames]
    added = []

    for frame in sorted_frames:
        if len(added) >= easy_count:
            break

        if can_select_frame(frame.timestamp, selected_times, min_gap_sec):
            added.append(frame)
            selected_times.append(frame.timestamp)

    return selected_frames + added


# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================

def extract_ball_frames(input_video: str, output_folder: str):
    """Extract diverse frames for ball training (simple uniform sampling)."""
    
    input_path = Path(input_video)
    output_path = Path(output_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"=== BALL TRACKING MODE ===")
    print(f"Opening video: {input_path}")
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {frame_width}x{frame_height}, {fps:.2f} fps, {total_frames} frames")
    print(f"Extracting every {BALL_FRAME_SKIP} frames with {BALL_MIN_TEMPORAL_GAP_SEC}s minimum gap")
    print(f"Target: {BALL_TARGET_FRAMES} frames")

    frame_idx = 0
    saved_count = 0
    last_saved_time = -BALL_MIN_TEMPORAL_GAP_SEC  # Allow first frame

    while saved_count < BALL_TARGET_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        # Skip frames according to BALL_FRAME_SKIP and temporal gap
        if (frame_idx % BALL_FRAME_SKIP == 0 and 
            timestamp - last_saved_time >= BALL_MIN_TEMPORAL_GAP_SEC):
            
            ts_str = format_timestamp(timestamp)
            filename = f"ball_frame_{frame_idx:06d}_t{ts_str}s.jpg"
            filepath = output_path / filename

            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
            last_saved_time = timestamp

            if saved_count % 100 == 0:
                progress = 100 * frame_idx / total_frames
                print(f"  Saved {saved_count}/{BALL_TARGET_FRAMES} frames ({progress:.1f}% through video)")

        frame_idx += 1

    cap.release()

    print(f"\nDone! Saved {saved_count} frames to {output_path}")
    print(f"\nNext steps for ball tracking:")
    print(f"  1. Upload frames to Roboflow")
    print(f"  2. Annotate the ball in each frame (white, may be small/blurry)")
    print(f"  3. Train a YOLO model with ball class")
    print(f"  4. Since ball is small, consider using a larger YOLO model (yolo11x)")


def extract_player_frames(input_video: str, output_folder: str):
    """Main function to extract high-value training frames using batch processing."""

    # Resolve paths
    input_path = Path(input_video)
    output_path = Path(output_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"=== PLAYER TRACKING MODE ===")
    print(f"Loading YOLO model: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)

    print(f"Opening video: {input_path}")
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_width * frame_height

    print(f"Video info: {frame_width}x{frame_height}, {fps:.2f} fps, {total_frames} frames")
    print(f"Processing every {PLAYER_FRAME_SKIP} frame(s) in batches of {PLAYER_BATCH_SIZE}...")
    print(f"Target: {PLAYER_TARGET_FRAMES} frames, early stop at {int(PLAYER_TARGET_FRAMES * PLAYER_EARLY_STOP_MULTIPLIER)} candidates")

    # Collect all candidate frames (using heap to track top scores efficiently)
    all_frame_data: list[FrameData] = []
    high_value_count = 0  # Frames above PLAYER_MIN_SCORE_THRESHOLD

    prev_count = None
    frame_idx = 0
    batch_count = 0
    early_stopped = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames according to PLAYER_FRAME_SKIP
        if frame_idx % PLAYER_FRAME_SKIP != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps

        # Run YOLO inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Extract person detections
        boxes = []
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                if int(cls) == PERSON_CLASS_ID:
                    boxes.append(box.cpu().numpy())

        boxes = np.array(boxes) if boxes else np.array([]).reshape(0, 4)
        detection_count = len(boxes)

        # Compute metrics
        mean_iou = compute_mean_pairwise_iou(boxes)
        large_box_ratio = compute_large_box_ratio(boxes, frame_area)

        # Compute score
        score = compute_frame_score(
            detection_count, mean_iou, large_box_ratio, prev_count
        )

        frame_data = FrameData(
            frame_idx=frame_idx,
            timestamp=timestamp,
            detection_count=detection_count,
            mean_iou=mean_iou,
            large_box_ratio=large_box_ratio,
            score=score
        )
        all_frame_data.append(frame_data)

        if score >= PLAYER_MIN_SCORE_THRESHOLD:
            high_value_count += 1

        prev_count = detection_count
        batch_count += 1

        # Check batch completion
        if batch_count >= PLAYER_BATCH_SIZE:
            progress = 100 * frame_idx / total_frames
            print(f"  Batch complete: frame {frame_idx}/{total_frames} ({progress:.1f}%) - "
                  f"high-value candidates: {high_value_count}")

            # Early stopping check
            required_candidates = int(PLAYER_TARGET_FRAMES * PLAYER_EARLY_STOP_MULTIPLIER)
            if high_value_count >= required_candidates:
                print(f"\n  Early stop: Found {high_value_count} high-value candidates "
                      f"(>= {required_candidates} required)")
                early_stopped = True
                break

            batch_count = 0

        frame_idx += 1

    cap.release()

    if early_stopped:
        print(f"Processed {len(all_frame_data)} frames before early stop")
    else:
        print(f"\nProcessed entire video: {len(all_frame_data)} frames")

    # Select frames
    hard_frame_count = int(PLAYER_TARGET_FRAMES * (1 - EASY_FRAME_RATIO))
    easy_frame_count = PLAYER_TARGET_FRAMES - hard_frame_count

    print(f"Selecting {hard_frame_count} hard frames and {easy_frame_count} easy frames...")

    selected_frames = select_frames_with_spacing(
        all_frame_data, hard_frame_count, PLAYER_MIN_TEMPORAL_GAP_SEC
    )

    selected_frames = add_easy_frames(
        all_frame_data, selected_frames, easy_frame_count, PLAYER_MIN_TEMPORAL_GAP_SEC
    )

    print(f"Selected {len(selected_frames)} frames total")

    # Save selected frames
    print(f"Saving frames to: {output_path}")

    cap = cv2.VideoCapture(str(input_path))

    # Sort by frame index for efficient sequential access
    selected_frames.sort(key=lambda x: x.frame_idx)
    selected_indices = {f.frame_idx: f for f in selected_frames}
    max_frame_needed = max(f.frame_idx for f in selected_frames)

    frame_idx = 0
    saved_count = 0

    while frame_idx <= max_frame_needed:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in selected_indices:
            fd = selected_indices[frame_idx]

            ts_str = format_timestamp(fd.timestamp)
            filename = f"frame_{fd.frame_idx:06d}_t{ts_str}s_count{fd.detection_count:02d}.jpg"
            filepath = output_path / filename

            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

            if saved_count % 100 == 0:
                print(f"  Saved {saved_count}/{len(selected_frames)} frames")

        frame_idx += 1

    cap.release()

    print(f"\nDone! Saved {saved_count} frames to {output_path}")

    # Print statistics
    if selected_frames:
        scores = [f.score for f in selected_frames]
        counts = [f.detection_count for f in selected_frames]

        print(f"\nStatistics:")
        print(f"  Score range: {min(scores):.2f} - {max(scores):.2f}")
        print(f"  Mean score: {np.mean(scores):.2f}")
        print(f"  Detection count range: {min(counts)} - {max(counts)}")
        print(f"  Frames with undercount (<{EXPECTED_PLAYERS}): "
              f"{sum(1 for c in counts if c < EXPECTED_PLAYERS)}")
        if early_stopped:
            print(f"  Note: Early stopped - only processed {100*len(all_frame_data)*PLAYER_FRAME_SKIP/total_frames:.1f}% of video")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract training frames for player or ball detection"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["player", "ball"],
        default="player",
        help="Extraction mode: 'player' for complex player scenarios, 'ball' for diverse ball frames"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_VIDEO,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output folder (defaults based on mode)"
    )
    
    args = parser.parse_args()
    
    # Set default output folder based on mode
    if args.output is None:
        args.output = OUTPUT_FOLDER_BALL if args.mode == "ball" else OUTPUT_FOLDER_PLAYER
    
    if args.mode == "ball":
        extract_ball_frames(args.input, args.output)
    else:
        extract_player_frames(args.input, args.output)
