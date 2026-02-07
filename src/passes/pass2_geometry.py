"""
Pass 2: Divergence Detection + Track Fragment Splitting

CRITICAL CONSTRAINTS:
- ✅ Pass 1 data is IMMUTABLE (read-only)
- ✅ All decisions must be frame-explainable (track which signal triggered)
- ✅ Split tracks into fragments at divergence points
- ✅ Apply homography (pixel → court meters)
- ❌ NO fragment merging (that's Pass 3's job)
- ❌ NO identity assignment (that's Pass 3's job)

Divergence Signals (4 types):
1. Track crossings: Spatial proximity < threshold (players swap IDs)
2. Velocity spikes: Sudden motion change > threshold (teleporting)
3. BBox jumps: Large position jump between frames (tracking error)
4. Occlusion spikes: Confidence drop > threshold (lost/found player)

Output: pass2_identity/<clip>_fragments.json
"""

from pathlib import Path
import json
import numpy as np
from typing import Any

try:
    import orjson  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    orjson = None

from src.geometry.homography import create_homography_from_config, CourtHomography


def _dequantize_histogram(hist_quantized: list[int]) -> np.ndarray:
    """
    De-quantize HSV histogram from uint8 [0-255] back to float32 [0.0-1.0].

    Args:
        hist_quantized: Quantized histogram (96 elements as uint8 list)

    Returns:
        Float32 histogram (96 elements)
    """
    if not hist_quantized or len(hist_quantized) == 0:
        return np.zeros(96, dtype=np.float32)
    return np.array(hist_quantized, dtype=np.float32) / 255.0


def run_pass2(run_dir: Path, config: dict):
    """
    Run Pass 2: Divergence Detection + Track Fragment Splitting.

    Args:
        run_dir: Run output directory (output/run_DDMMYY_HHMMSS)
        config: Configuration dictionary
    """
    run_dir = Path(run_dir)
    pass1_dir = run_dir / "pass1_raw"
    pass2_dir = run_dir / "pass2_identity"
    pass2_dir.mkdir(parents=True, exist_ok=True)

    # Load homography
    homography = create_homography_from_config(config)
    if homography is None:
        print("ERROR: Homography not configured. Pass 2 requires calibration.")
        return

    # Get Pass 1 JSON files
    pass1_files = list(pass1_dir.glob("*.json"))
    if not pass1_files:
        print(f"No Pass 1 JSON files found in {pass1_dir}")
        return

    print(f"Found {len(pass1_files)} Pass 1 JSON files")

    # Process each clip
    for pass1_file in pass1_files:
        print(f"\nProcessing: {pass1_file.name}")
        process_clip_pass2(
            pass1_file=pass1_file,
            output_dir=pass2_dir,
            homography=homography,
            config=config,
        )

    print(f"\nPass 2 complete! Output: {pass2_dir}")


def process_clip_pass2(
    pass1_file: Path,
    output_dir: Path,
    homography: CourtHomography,
    config: dict,
):
    """
    Process a single clip for Pass 2.

    Args:
        pass1_file: Path to Pass 1 JSON file
        output_dir: Output directory for Pass 2 JSON
        homography: CourtHomography instance
        config: Configuration dictionary
    """
    # Load Pass 1 data (IMMUTABLE - read-only)
    with open(pass1_file, 'r', encoding='utf-8') as f:
        pass1_data = json.load(f)

    clip_name = pass1_data.get("clip_name", pass1_file.stem)
    fps = pass1_data.get("fps", 30.0)

    # Get divergence detection config
    div_cfg = config.get("divergence", {})
    crossing_threshold = div_cfg.get("crossing_threshold_meters", 1.0)
    velocity_threshold = div_cfg.get("velocity_spike_threshold", 5.0)
    velocity_window = div_cfg.get("velocity_window_frames", 3)
    bbox_jump_threshold = div_cfg.get("bbox_jump_threshold_meters", 2.0)
    occlusion_threshold = div_cfg.get("occlusion_threshold", 0.6)
    occlusion_window = div_cfg.get("occlusion_window_frames", 5)
    min_fragment_length = div_cfg.get("min_fragment_length", 10)

    # Prepare output structure
    output_data = {
        "clip_name": clip_name,
        "fps": fps,
        "fragments": []
    }

    # Process each track from Pass 1
    tracks = pass1_data.get("tracks", {})
    fragment_id_counter = 0

    for track_id, track_data in tracks.items():
        frames = track_data.get("frames", [])
        bboxes = track_data.get("bboxes", [])
        centroids = track_data.get("centroids", [])
        confidences = track_data.get("confidences", [])
        occlusion_scores = track_data.get("occlusion_scores", [])
        hsv_histograms = track_data.get("hsv_histograms", [])
        jersey_decisions = track_data.get("jersey_decisions", [])

        if len(frames) < min_fragment_length:
            # Skip tracks that are too short
            continue

        # Convert pixel centroids to court coordinates
        court_positions = []
        for centroid in centroids:
            court_x, court_y = homography.pixel_to_court(centroid[0], centroid[1])
            court_positions.append([court_x, court_y])

        # Detect divergence points (frame indices where track should split)
        divergence_points = detect_divergences(
            frames=frames,
            court_positions=court_positions,
            confidences=confidences,
            occlusion_scores=occlusion_scores,
            fps=fps,
            crossing_threshold=crossing_threshold,
            velocity_threshold=velocity_threshold,
            velocity_window=velocity_window,
            bbox_jump_threshold=bbox_jump_threshold,
            occlusion_threshold=occlusion_threshold,
            occlusion_window=occlusion_window,
        )

        # Split track into fragments at divergence points
        fragments = split_track_into_fragments(
            track_id=track_id,
            frames=frames,
            bboxes=bboxes,
            centroids=centroids,
            court_positions=court_positions,
            confidences=confidences,
            occlusion_scores=occlusion_scores,
            hsv_histograms=hsv_histograms,
            jersey_decisions=jersey_decisions,
            divergence_points=divergence_points,
            min_fragment_length=min_fragment_length,
        )

        # Add fragments to output
        for fragment in fragments:
            fragment["fragment_id"] = f"frag_{fragment_id_counter:06d}"
            fragment_id_counter += 1
            output_data["fragments"].append(fragment)

    # Save JSON
    output_path = output_dir / f"{pass1_file.stem}_fragments.json"
    pretty_json = config.get("pass2", {}).get("pretty_json", True)

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
    print(f"  Fragments: {len(output_data['fragments'])}")


def detect_divergences(
    frames: list[int],
    court_positions: list[list[float]],
    confidences: list[float],
    occlusion_scores: list[float],
    fps: float,
    crossing_threshold: float,
    velocity_threshold: float,
    velocity_window: int,
    bbox_jump_threshold: float,
    occlusion_threshold: float,
    occlusion_window: int,
) -> list[dict[str, Any]]:
    """
    Detect divergence points (frame indices where track should split).

    Returns:
        List of divergence dicts with:
        - frame_idx: Frame where divergence occurred
        - reason: Type of divergence (crossing, velocity_spike, bbox_jump, occlusion_spike)
        - metric: Numeric value that triggered the detection
    """
    divergences = []

    # Velocity spike detection
    for i in range(velocity_window, len(frames)):
        # Compute velocity over window
        dt = (frames[i] - frames[i - velocity_window]) / fps
        if dt <= 0:
            continue

        dx = court_positions[i][0] - court_positions[i - velocity_window][0]
        dy = court_positions[i][1] - court_positions[i - velocity_window][1]
        velocity = np.sqrt(dx**2 + dy**2) / dt

        # Check previous velocity
        if i >= 2 * velocity_window:
            dt_prev = (frames[i - velocity_window] - frames[i - 2 * velocity_window]) / fps
            if dt_prev > 0:
                dx_prev = court_positions[i - velocity_window][0] - court_positions[i - 2 * velocity_window][0]
                dy_prev = court_positions[i - velocity_window][1] - court_positions[i - 2 * velocity_window][1]
                velocity_prev = np.sqrt(dx_prev**2 + dy_prev**2) / dt_prev

                velocity_change = abs(velocity - velocity_prev)
                if velocity_change > velocity_threshold:
                    divergences.append({
                        "frame_idx": frames[i],
                        "reason": "velocity_spike",
                        "metric": float(velocity_change),
                    })

    # BBox jump detection (frame-to-frame)
    for i in range(1, len(frames)):
        # Check if frames are consecutive (or nearly consecutive)
        frame_gap = frames[i] - frames[i - 1]
        if frame_gap > 5:
            # Large frame gap = interpolation, not a jump
            continue

        dx = court_positions[i][0] - court_positions[i - 1][0]
        dy = court_positions[i][1] - court_positions[i - 1][1]
        jump_distance = np.sqrt(dx**2 + dy**2)

        if jump_distance > bbox_jump_threshold:
            divergences.append({
                "frame_idx": frames[i],
                "reason": "bbox_jump",
                "metric": float(jump_distance),
            })

    # Occlusion spike detection (sustained confidence drop)
    for i in range(occlusion_window, len(frames)):
        # Check if occlusion score is high over a window
        window_occlusion = occlusion_scores[i - occlusion_window:i]
        mean_occlusion = np.mean(window_occlusion)

        if mean_occlusion > occlusion_threshold:
            # Check if previous window was low occlusion (spike)
            if i >= 2 * occlusion_window:
                prev_window = occlusion_scores[i - 2 * occlusion_window:i - occlusion_window]
                prev_mean = np.mean(prev_window)
                if prev_mean < occlusion_threshold * 0.5:
                    divergences.append({
                        "frame_idx": frames[i],
                        "reason": "occlusion_spike",
                        "metric": float(mean_occlusion),
                    })

    # Sort divergences by frame index
    divergences.sort(key=lambda d: d["frame_idx"])

    return divergences


def split_track_into_fragments(
    track_id: str,
    frames: list[int],
    bboxes: list[list[int]],
    centroids: list[list[int]],
    court_positions: list[list[float]],
    confidences: list[float],
    occlusion_scores: list[float],
    hsv_histograms: list[list[int] | None],
    jersey_decisions: list[list[Any]],
    divergence_points: list[dict[str, Any]],
    min_fragment_length: int,
) -> list[dict[str, Any]]:
    """
    Split a track into fragments at divergence points.

    Args:
        track_id: Original track ID from Pass 1
        frames: Frame indices
        bboxes: Bounding boxes
        centroids: Pixel centroids
        court_positions: Court coordinates (meters)
        confidences: Detection confidences
        occlusion_scores: Occlusion scores
        hsv_histograms: HSV histograms (quantized, sampled)
        jersey_decisions: Jersey decisions [number, confidence]
        divergence_points: List of divergence dicts
        min_fragment_length: Minimum frames for a valid fragment

    Returns:
        List of fragment dicts
    """
    fragments = []

    # Extract divergence frame indices
    split_frames = [d["frame_idx"] for d in divergence_points]

    # Split track at divergence points
    start_idx = 0
    for split_frame in split_frames:
        # Find index in frames list
        try:
            split_idx = frames.index(split_frame)
        except ValueError:
            continue

        # Create fragment from start_idx to split_idx
        if split_idx - start_idx >= min_fragment_length:
            fragment = create_fragment(
                track_id=track_id,
                frames=frames[start_idx:split_idx],
                bboxes=bboxes[start_idx:split_idx],
                centroids=centroids[start_idx:split_idx],
                court_positions=court_positions[start_idx:split_idx],
                confidences=confidences[start_idx:split_idx],
                occlusion_scores=occlusion_scores[start_idx:split_idx],
                hsv_histograms=hsv_histograms[start_idx:split_idx],
                jersey_decisions=jersey_decisions[start_idx:split_idx],
            )
            # Annotate why fragment ended
            fragment["split_reason"] = next(
                (d for d in divergence_points if d["frame_idx"] == split_frame), None
            )
            fragments.append(fragment)

        start_idx = split_idx

    # Create final fragment (from last split to end)
    if len(frames) - start_idx >= min_fragment_length:
        fragment = create_fragment(
            track_id=track_id,
            frames=frames[start_idx:],
            bboxes=bboxes[start_idx:],
            centroids=centroids[start_idx:],
            court_positions=court_positions[start_idx:],
            confidences=confidences[start_idx:],
            occlusion_scores=occlusion_scores[start_idx:],
            hsv_histograms=hsv_histograms[start_idx:],
            jersey_decisions=jersey_decisions[start_idx:],
        )
        fragment["split_reason"] = None  # No divergence (end of track)
        fragments.append(fragment)

    return fragments


def create_fragment(
    track_id: str,
    frames: list[int],
    bboxes: list[list[int]],
    centroids: list[list[int]],
    court_positions: list[list[float]],
    confidences: list[float],
    occlusion_scores: list[float],
    hsv_histograms: list[list[int] | None],
    jersey_decisions: list[list[Any]],
) -> dict[str, Any]:
    """
    Create a fragment dict with aggregated statistics.

    Args:
        track_id: Original track ID from Pass 1
        frames: Frame indices for this fragment
        bboxes: Bounding boxes
        centroids: Pixel centroids
        court_positions: Court coordinates (meters)
        confidences: Detection confidences
        occlusion_scores: Occlusion scores
        hsv_histograms: HSV histograms (quantized, sampled)
        jersey_decisions: Jersey decisions [number, confidence]

    Returns:
        Fragment dict
    """
    # Aggregate jersey probabilities
    jersey_prob_timeline = {}
    for decision in jersey_decisions:
        if decision and len(decision) == 2:
            jersey_id, jersey_conf = decision
            if jersey_id is not None and jersey_conf > 0.0:
                if jersey_id not in jersey_prob_timeline:
                    jersey_prob_timeline[jersey_id] = {"total": 0.0, "count": 0}
                jersey_prob_timeline[jersey_id]["total"] += jersey_conf
                jersey_prob_timeline[jersey_id]["count"] += 1

    # Compute mean HSV histogram (for Pass 3 team clustering)
    valid_histograms = []
    for hist_quantized in hsv_histograms:
        if hist_quantized is not None:
            hist_float = _dequantize_histogram(hist_quantized)
            valid_histograms.append(hist_float)

    if valid_histograms:
        mean_hsv_histogram = np.mean(valid_histograms, axis=0).tolist()
    else:
        mean_hsv_histogram = np.zeros(96, dtype=np.float32).tolist()

    # Compute visibility quality (1 - mean occlusion)
    visibility_quality = 1.0 - np.mean(occlusion_scores)

    # Compute spatial footprint statistics
    court_positions_array = np.array(court_positions)
    mean_position = court_positions_array.mean(axis=0).tolist()
    spatial_variance = float(np.var(court_positions_array))

    fragment = {
        "original_track_id": track_id,
        "start_frame": frames[0],
        "end_frame": frames[-1],
        "frame_count": len(frames),
        "mean_hsv_histogram": mean_hsv_histogram,
        "jersey_prob_timeline": jersey_prob_timeline,
        "spatial_footprint": {
            "court_positions": court_positions,
            "mean_position": mean_position,
            "spatial_variance": spatial_variance,
        },
        "visibility_quality": float(visibility_quality),
        "mean_confidence": float(np.mean(confidences)),
    }

    return fragment
