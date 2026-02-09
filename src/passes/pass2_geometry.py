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


def compute_appearance_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compute appearance distance between two HSV histograms using chi-squared distance.

    This is the CORE appearance-consistency metric for divergence detection.
    Chi-squared is robust for histogram comparison and handles normalization well.

    Args:
        hist1: First HSV histogram (96 elements, float32, normalized)
        hist2: Second HSV histogram (96 elements, float32, normalized)

    Returns:
        Distance in [0, ∞), where:
        - 0.0 = identical appearance
        - ~0.3 = same player, different lighting/angle
        - ~0.5+ = different player (identity swap)
        - inf = invalid histograms
    """
    if hist1 is None or hist2 is None or hist1.size == 0 or hist2.size == 0:
        return float('inf')

    if hist1.shape != hist2.shape or hist1.size != 96:
        return float('inf')

    # Chi-squared distance: Σ (h1[i] - h2[i])² / (h1[i] + h2[i] + eps)
    # Robust to histogram normalization and handles zero bins gracefully
    eps = 1e-10
    numerator = (hist1 - hist2) ** 2
    denominator = hist1 + hist2 + eps
    chi_squared = np.sum(numerator / denominator)

    return float(chi_squared)


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
        # NOW INCLUDES APPEARANCE DRIFT DETECTION (critical for identity swap fix)
        divergence_points = detect_divergences(
            frames=frames,
            court_positions=court_positions,
            confidences=confidences,
            occlusion_scores=occlusion_scores,
            hsv_histograms=hsv_histograms,
            fps=fps,
            crossing_threshold=crossing_threshold,
            velocity_threshold=velocity_threshold,
            velocity_window=velocity_window,
            bbox_jump_threshold=bbox_jump_threshold,
            occlusion_threshold=occlusion_threshold,
            occlusion_window=occlusion_window,
            appearance_threshold=div_cfg.get("appearance_threshold", 0.5),
            appearance_window=div_cfg.get("appearance_window", 3),
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
    hsv_histograms: list[list[int] | None],
    fps: float,
    crossing_threshold: float,
    velocity_threshold: float,
    velocity_window: int,
    bbox_jump_threshold: float,
    occlusion_threshold: float,
    occlusion_window: int,
    appearance_threshold: float,
    appearance_window: int,
) -> list[dict[str, Any]]:
    """
    Detect divergence points (frame indices where track should split).

    CRITICAL: Appearance drift is now a FIRST-CLASS divergence signal.
    This fixes identity swaps during close contact (T5 orange→white bug).

    Returns:
        List of divergence dicts with:
        - frame_idx: Frame where divergence occurred
        - reason: Type of divergence (velocity_spike, bbox_jump, occlusion_spike, appearance_drift)
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

    # ========================================================================
    # APPEARANCE DRIFT DETECTION (CRITICAL FOR IDENTITY SWAP BUG FIX)
    # ========================================================================
    # Strategy: Build rolling appearance centroid and detect when new samples
    # drift too far from the established appearance baseline.
    #
    # This catches cases like T5 (orange→white) where motion is smooth but
    # appearance suddenly changes during close contact.
    # ========================================================================

    # Build list of valid histogram indices (skip None entries from sparse sampling)
    valid_hist_indices = [i for i, h in enumerate(hsv_histograms) if h is not None and len(h) == 96]

    if len(valid_hist_indices) >= appearance_window + 1:
        # Slide window through valid histograms
        for window_start_idx in range(len(valid_hist_indices) - appearance_window):
            # Get indices of first half of window (baseline appearance)
            baseline_end_idx = window_start_idx + max(1, appearance_window // 2)
            baseline_indices = valid_hist_indices[window_start_idx:baseline_end_idx]

            # Compute baseline appearance centroid
            baseline_hists = [_dequantize_histogram(hsv_histograms[i]) for i in baseline_indices]
            baseline_centroid = np.mean(baseline_hists, axis=0)

            # Check each histogram in second half of window against baseline
            test_start_idx = baseline_end_idx
            test_end_idx = min(window_start_idx + appearance_window + 1, len(valid_hist_indices))

            for test_idx in range(test_start_idx, test_end_idx):
                track_idx = valid_hist_indices[test_idx]
                test_hist = _dequantize_histogram(hsv_histograms[track_idx])

                # Compute appearance distance
                distance = compute_appearance_distance(baseline_centroid, test_hist)

                # Trigger split if distance exceeds threshold
                if distance > appearance_threshold:
                    divergences.append({
                        "frame_idx": frames[track_idx],
                        "reason": "appearance_drift",
                        "metric": float(distance),
                    })

    # Sort divergences by frame index and deduplicate
    divergences.sort(key=lambda d: d["frame_idx"])

    # Deduplicate: if multiple divergences at same frame, keep one with highest metric
    if divergences:
        deduped = []
        last_frame = None
        last_div = None

        for div in divergences:
            if div["frame_idx"] != last_frame:
                if last_div is not None:
                    deduped.append(last_div)
                last_frame = div["frame_idx"]
                last_div = div
            else:
                # Same frame - keep divergence with higher metric
                if div["metric"] > last_div["metric"]:
                    last_div = div

        if last_div is not None:
            deduped.append(last_div)

        divergences = deduped

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

    CRITICAL: Primary fragment assignment ensures the original track ID
    stays with the fragment that maintains appearance consistency.
    This fixes the T5 identity swap bug.

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
        List of fragment dicts with "is_primary_fragment" flag
    """
    fragments = []

    # Extract divergence frame indices
    split_frames = [d["frame_idx"] for d in divergence_points]

    # ========================================================================
    # COMPUTE PRE-SPLIT APPEARANCE CENTROID
    # ========================================================================
    # This is the ground truth appearance BEFORE any identity swap.
    # The fragment closest to this centroid gets flagged as primary
    # (keeps original track identity), others are secondary (ID jumps).
    # ========================================================================
    pre_split_histograms = []
    first_split_frame = split_frames[0] if split_frames else float('inf')

    for i, hist in enumerate(hsv_histograms):
        if hist is not None and len(hist) == 96 and frames[i] < first_split_frame:
            pre_split_histograms.append(_dequantize_histogram(hist))

    pre_split_centroid = None
    if pre_split_histograms:
        pre_split_centroid = np.mean(pre_split_histograms, axis=0)

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

            # Compute appearance distance to pre-split centroid
            if pre_split_centroid is not None and fragment.get("mean_hsv_histogram"):
                fragment_hist = np.array(fragment["mean_hsv_histogram"], dtype=np.float32)
                fragment["appearance_distance_to_origin"] = compute_appearance_distance(
                    pre_split_centroid, fragment_hist
                )
            else:
                fragment["appearance_distance_to_origin"] = float('inf')

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

        # Compute appearance distance to pre-split centroid
        if pre_split_centroid is not None and fragment.get("mean_hsv_histogram"):
            fragment_hist = np.array(fragment["mean_hsv_histogram"], dtype=np.float32)
            fragment["appearance_distance_to_origin"] = compute_appearance_distance(
                pre_split_centroid, fragment_hist
            )
        else:
            fragment["appearance_distance_to_origin"] = float('inf')

        fragments.append(fragment)

    # ========================================================================
    # ASSIGN PRIMARY FRAGMENT FLAG
    # ========================================================================
    # The fragment with MINIMUM appearance distance to pre-split centroid
    # is the PRIMARY fragment (keeps original track identity).
    # All others are SECONDARY (identity jumps/swaps).
    #
    # For T5 case:
    #   - Fragment with orange appearance → is_primary_fragment=True
    #   - Fragment with white appearance → is_primary_fragment=False
    # ========================================================================
    if fragments and pre_split_centroid is not None:
        min_distance = float('inf')
        primary_idx = 0

        for i, frag in enumerate(fragments):
            dist = frag.get("appearance_distance_to_origin", float('inf'))
            if dist < min_distance:
                min_distance = dist
                primary_idx = i

        # Mark primary and secondary fragments
        for i, frag in enumerate(fragments):
            frag["is_primary_fragment"] = (i == primary_idx)
    else:
        # No pre-split centroid (no valid histograms) - mark first as primary
        for i, frag in enumerate(fragments):
            frag["is_primary_fragment"] = (i == 0)

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
