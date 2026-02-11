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
        # NOW INCLUDES APPEARANCE DRIFT + JERSEY INCONSISTENCY DETECTION
        divergence_points = detect_divergences(
            frames=frames,
            court_positions=court_positions,
            confidences=confidences,
            occlusion_scores=occlusion_scores,
            hsv_histograms=hsv_histograms,
            jersey_decisions=jersey_decisions,
            fps=fps,
            crossing_threshold=crossing_threshold,
            velocity_threshold=velocity_threshold,
            velocity_window=velocity_window,
            bbox_jump_threshold=bbox_jump_threshold,
            occlusion_threshold=occlusion_threshold,
            occlusion_window=occlusion_window,
            appearance_threshold=div_cfg.get("appearance_threshold", 0.5),
            appearance_window=div_cfg.get("appearance_window", 3),
            jersey_enabled=div_cfg.get("jersey_inconsistency_enabled", True),
            jersey_appear_threshold=div_cfg.get("jersey_appear_threshold", 0.5),
            jersey_disappear_threshold=div_cfg.get("jersey_disappear_threshold", 0.3),
            jersey_window=div_cfg.get("jersey_consistency_window", 10),
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

    # ========================================================================
    # JERSEY TEMPORAL EXCLUSIVITY POST-PROCESSING (ITERATIVE)
    # ========================================================================
    # Detect when a jersey disappears from one fragment and appears on another.
    # This catches track jumps that appearance-based detection may miss
    # (e.g., black #4 → black no-number jump at frame ~700).
    #
    # Run iteratively because a single fragment may jump between multiple people:
    # Person 1 (no jersey) → Person 2 (no jersey) → Person 3 (jersey #4)
    # Each jump needs to be detected and split separately.
    # ========================================================================
    jersey_temporal_cfg = div_cfg.get("jersey_temporal_exclusivity", {})
    if jersey_temporal_cfg.get("enabled", True):
        max_iterations = 5  # Safety limit to prevent infinite loops
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            fragments_before = output_data["fragments"]
            output_data["fragments"] = detect_jersey_temporal_conflicts(
                fragments=output_data["fragments"],
                pass1_tracks=tracks,
                homography=homography,
                min_fragment_length=min_fragment_length,
                appear_threshold=jersey_temporal_cfg.get("appear_threshold", 0.5),
                temporal_window=jersey_temporal_cfg.get("temporal_window_frames", 50),
            )
            # Stop if no new splits were made
            if len(output_data["fragments"]) == len(fragments_before):
                print(f"  Jersey temporal exclusivity: converged after {iteration} iteration(s)")
                break
        else:
            print(f"  Jersey temporal exclusivity: stopped after {max_iterations} iterations (max reached)")


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


def detect_jersey_temporal_conflicts(
    fragments: list[dict[str, Any]],
    pass1_tracks: dict[str, Any],
    homography: CourtHomography,
    min_fragment_length: int,
    appear_threshold: float,
    temporal_window: int,
) -> list[dict[str, Any]]:
    """
    Post-process fragments to detect jersey temporal exclusivity violations.

    This function addresses the scenario where:
    1. Jersey #X disappears from Fragment A at frame F
    2. Jersey #X appears on Fragment B at frame F+delta (delta < temporal_window)
    3. Split Fragment B at the jersey appearance frame

    This catches track jumps that appearance-based detection may miss when
    both players have similar appearance (e.g., black shirt → black shirt #4).

    Args:
        fragments: List of fragment dicts
        pass1_tracks: Original Pass 1 track data (for re-splitting)
        homography: CourtHomography instance
        min_fragment_length: Minimum frames for a valid fragment
        appear_threshold: Min confidence for jersey appearance
        temporal_window: Max frame gap to consider jersey jump

    Returns:
        Updated list of fragments with temporal conflicts resolved
    """
    # Build jersey timeline: jersey_id -> [(fragment_idx, start_frame, end_frame, max_conf)]
    jersey_timeline = {}

    for frag_idx, fragment in enumerate(fragments):
        jersey_probs = fragment.get("jersey_prob_timeline", {})
        start_frame = fragment.get("start_frame", 0)
        end_frame = fragment.get("end_frame", 0)
        original_track_id = fragment.get("original_track_id")

        for jersey_id, stats in jersey_probs.items():
            count = stats.get("count", 0)
            total_conf = stats.get("total", 0.0)
            avg_conf = total_conf / count if count > 0 else 0.0

            if avg_conf >= appear_threshold:
                # Find FIRST frame where this jersey appears in this fragment
                jersey_first_frame = start_frame  # Default to fragment start
                if original_track_id in pass1_tracks:
                    track_data = pass1_tracks[original_track_id]
                    frames = track_data.get("frames", [])
                    jersey_decisions = track_data.get("jersey_decisions", [])
                    for frame_idx, decision in zip(frames, jersey_decisions):
                        if start_frame <= frame_idx <= end_frame:
                            if decision and len(decision) == 2:
                                jid, conf = decision
                                if jid == jersey_id and conf >= appear_threshold:
                                    jersey_first_frame = frame_idx
                                    break

                if jersey_id not in jersey_timeline:
                    jersey_timeline[jersey_id] = []
                jersey_timeline[jersey_id].append({
                    "fragment_idx": frag_idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "jersey_first_frame": jersey_first_frame,  # CRITICAL: actual appearance frame
                    "max_conf": avg_conf,
                    "fragment_id": fragment.get("fragment_id", "unknown"),
                    "track_id": fragment.get("original_track_id", "unknown"),
                })

    # Sort timeline by JERSEY FIRST APPEARANCE FRAME (not fragment start)
    for jersey_id in jersey_timeline:
        jersey_timeline[jersey_id].sort(key=lambda x: x["jersey_first_frame"])

    # Debug: Print jersey timeline for jersey #4
    if 4 in jersey_timeline:
        print(f"    [DEBUG] Jersey #4 timeline (sorted by first appearance):")
        for entry in jersey_timeline[4]:
            print(f"      - {entry['fragment_id']} (track {entry['track_id']}): "
                  f"fragment frames {entry['start_frame']}-{entry['end_frame']}, "
                  f"jersey first appears at frame {entry['jersey_first_frame']}, avg_conf={entry['max_conf']:.2f}")

    # ========================================================================
    # DETECT TEMPORAL OVERLAPS: Same jersey on different tracks simultaneously
    # ========================================================================
    # More robust than threshold-based filtering: directly detect when two
    # fragments from different tracks claim the same jersey during overlapping time.
    # Split the fragment that "stole" the jersey (had it appear later).
    # ========================================================================
    fragments_to_split = []  # List of (fragment_idx, split_frame, reason)

    for jersey_id, timeline in jersey_timeline.items():
        # Check all pairs of fragments (not just sequential)
        for i in range(len(timeline)):
            for j in range(i + 1, len(timeline)):
                frag_a = timeline[i]
                frag_b = timeline[j]

                # Skip if same track (same player, brief occlusion is normal)
                if frag_a["track_id"] == frag_b["track_id"]:
                    continue

                # Check if fragments overlap in time
                overlap_start = max(frag_a["start_frame"], frag_b["start_frame"])
                overlap_end = min(frag_a["end_frame"], frag_b["end_frame"])

                if overlap_start <= overlap_end:
                    # OVERLAP DETECTED: Both fragments claim same jersey during overlap period
                    overlap_duration = overlap_end - overlap_start + 1

                    # Decide which fragment to keep vs split:
                    # PRIMARY: Highest average confidence wins (most reliable detection)
                    # SECONDARY: If confidence tied, earliest appearance wins (tiebreak only)
                    # GUARDRAIL: Don't let low-confidence block high-confidence
                    CONF_STRONG = 0.7

                    conf_a = frag_a["max_conf"]
                    conf_b = frag_b["max_conf"]

                    # Determine winner based on confidence
                    if conf_a > conf_b:
                        winner = frag_a
                        loser = frag_b
                    elif conf_b > conf_a:
                        winner = frag_b
                        loser = frag_a
                    else:
                        # Confidence tied - use earliest appearance as tiebreak
                        if frag_a["jersey_first_frame"] < frag_b["jersey_first_frame"]:
                            winner = frag_a
                            loser = frag_b
                        else:
                            winner = frag_b
                            loser = frag_a

                    # GUARDRAIL: Don't let low-confidence winner block high-confidence loser
                    # If winner is weak but loser is strong, swap them
                    if winner["max_conf"] < CONF_STRONG and loser["max_conf"] >= CONF_STRONG:
                        winner, loser = loser, winner

                    # Split the loser at its jersey appearance
                    split_target = loser
                    split_frame = loser["jersey_first_frame"]
                    prior_frag = winner

                    # Only split if jersey appears AFTER fragment starts (not at the very beginning)
                    # This prevents splitting when jersey is visible from the start
                    if split_frame > split_target["start_frame"]:
                        # Get track data to verify split frame
                        fragment_to_split = fragments[split_target["fragment_idx"]]
                        original_track_id = fragment_to_split.get("original_track_id")

                        if original_track_id in pass1_tracks:
                            fragments_to_split.append({
                                "fragment_idx": split_target["fragment_idx"],
                                "split_frame": split_frame,
                                "reason": "jersey_overlap_conflict",
                                "jersey_id": jersey_id,
                                "prior_fragment_idx": prior_frag["fragment_idx"],
                                "overlap_duration": overlap_duration,
                                "loser_conf": loser["max_conf"],
                                "winner_conf": winner["max_conf"],
                            })

    # If no conflicts detected, return fragments unchanged
    if not fragments_to_split:
        return fragments

    # Debug: Print detected conflicts
    print(f"    Detected {len(fragments_to_split)} jersey overlap conflict(s):")
    for split_info in fragments_to_split:
        frag_idx = split_info["fragment_idx"]
        fragment = fragments[frag_idx]
        prior_frag = fragments[split_info["prior_fragment_idx"]]
        loser_conf = split_info.get("loser_conf", 0.0)
        winner_conf = split_info.get("winner_conf", 0.0)
        print(f"      - Fragment {fragment.get('fragment_id')} (track {fragment.get('original_track_id')}): "
              f"jersey #{split_info['jersey_id']} conflicts with {prior_frag.get('fragment_id')} "
              f"(overlap={split_info['overlap_duration']} frames, split at {split_info['split_frame']}, "
              f"loser_conf={loser_conf:.2f}, winner_conf={winner_conf:.2f})")

    # Apply splits (process in reverse order to maintain indices)
    fragments_to_split.sort(key=lambda x: x["fragment_idx"], reverse=True)
    new_fragments = list(fragments)

    for split_info in fragments_to_split:
        frag_idx = split_info["fragment_idx"]
        split_frame = split_info["split_frame"]

        # Get original fragment
        original_fragment = new_fragments[frag_idx]
        original_track_id = original_fragment.get("original_track_id")

        # Get original Pass 1 track data
        if original_track_id not in pass1_tracks:
            continue

        track_data = pass1_tracks[original_track_id]
        frames = track_data.get("frames", [])
        bboxes = track_data.get("bboxes", [])
        centroids = track_data.get("centroids", [])
        confidences = track_data.get("confidences", [])
        occlusion_scores = track_data.get("occlusion_scores", [])
        hsv_histograms = track_data.get("hsv_histograms", [])
        jersey_decisions = track_data.get("jersey_decisions", [])

        # Find split index in original track data
        start_frame = original_fragment.get("start_frame")
        end_frame = original_fragment.get("end_frame")

        try:
            start_idx = frames.index(start_frame)
            end_idx = frames.index(end_frame) + 1
            split_idx_in_track = frames.index(split_frame)
        except ValueError:
            continue

        # Ensure split is within fragment bounds
        if split_idx_in_track <= start_idx or split_idx_in_track >= end_idx:
            continue

        # Compute relative split index within fragment
        split_idx = split_idx_in_track - start_idx

        # Convert pixel centroids to court coordinates
        court_positions = []
        for centroid in centroids[start_idx:end_idx]:
            court_x, court_y = homography.pixel_to_court(centroid[0], centroid[1])
            court_positions.append([court_x, court_y])

        # Create two new fragments
        fragment_before = create_fragment(
            track_id=original_track_id,
            frames=frames[start_idx:start_idx + split_idx],
            bboxes=bboxes[start_idx:start_idx + split_idx],
            centroids=centroids[start_idx:start_idx + split_idx],
            court_positions=court_positions[:split_idx],
            confidences=confidences[start_idx:start_idx + split_idx],
            occlusion_scores=occlusion_scores[start_idx:start_idx + split_idx],
            hsv_histograms=hsv_histograms[start_idx:start_idx + split_idx],
            jersey_decisions=jersey_decisions[start_idx:start_idx + split_idx],
        )
        fragment_before["split_reason"] = {
            "frame_idx": split_frame,
            "reason": split_info["reason"],
            "metric": split_info["jersey_id"],
        }
        fragment_before["fragment_id"] = original_fragment.get("fragment_id", "frag_unknown")
        fragment_before["is_primary_fragment"] = original_fragment.get("is_primary_fragment", False)

        fragment_after = create_fragment(
            track_id=original_track_id,
            frames=frames[start_idx + split_idx:end_idx],
            bboxes=bboxes[start_idx + split_idx:end_idx],
            centroids=centroids[start_idx + split_idx:end_idx],
            court_positions=court_positions[split_idx:],
            confidences=confidences[start_idx + split_idx:end_idx],
            occlusion_scores=occlusion_scores[start_idx + split_idx:end_idx],
            hsv_histograms=hsv_histograms[start_idx + split_idx:end_idx],
            jersey_decisions=jersey_decisions[start_idx + split_idx:end_idx],
        )
        fragment_after["split_reason"] = None
        fragment_after["fragment_id"] = f"{original_fragment.get('fragment_id', 'frag_unknown')}_split"
        fragment_after["is_primary_fragment"] = False  # After split is secondary (jersey jump)

        # Replace original fragment with two new fragments
        # CRITICAL: Keep BOTH fragments to prevent gaps (even if short)
        # Fragment gaps are worse than having short fragments
        replacement_fragments = []

        # Mark short fragments as low_quality
        before_length = len(fragment_before.get("spatial_footprint", {}).get("court_positions", []))
        if before_length < min_fragment_length:
            fragment_before["low_quality"] = True
            fragment_before["low_quality_reason"] = f"short_fragment_{before_length}_frames"
        replacement_fragments.append(fragment_before)

        after_length = len(fragment_after.get("spatial_footprint", {}).get("court_positions", []))
        if after_length < min_fragment_length:
            fragment_after["low_quality"] = True
            fragment_after["low_quality_reason"] = f"short_fragment_{after_length}_frames"
        replacement_fragments.append(fragment_after)

        # Replace in list (always replace with both fragments)
        new_fragments[frag_idx:frag_idx+1] = replacement_fragments

    return new_fragments


def detect_divergences(
    frames: list[int],
    court_positions: list[list[float]],
    confidences: list[float],
    occlusion_scores: list[float],
    hsv_histograms: list[list[int] | None],
    jersey_decisions: list[list[Any]],
    fps: float,
    crossing_threshold: float,
    velocity_threshold: float,
    velocity_window: int,
    bbox_jump_threshold: float,
    occlusion_threshold: float,
    occlusion_window: int,
    appearance_threshold: float,
    appearance_window: int,
    jersey_enabled: bool,
    jersey_appear_threshold: float,
    jersey_disappear_threshold: float,
    jersey_window: int,
) -> list[dict[str, Any]]:
    """
    Detect divergence points (frame indices where track should split).

    CRITICAL: Appearance drift + jersey inconsistency are FIRST-CLASS divergence signals.
    This fixes identity swaps during close contact (T5 orange→white, black #4 transitions).

    Returns:
        List of divergence dicts with:
        - frame_idx: Frame where divergence occurred
        - reason: Type of divergence (velocity_spike, bbox_jump, occlusion_spike, appearance_drift, jersey_inconsistency)
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
                    # ================================================================
                    # JERSEY-AWARE APPEARANCE DRIFT SPLITTING
                    # ================================================================
                    # Only split if jersey was ALREADY visible and then changes/disappears,
                    # indicating the track jumped to a different player.
                    #
                    # SUPPRESS split when:
                    # 1. No jersey → Jersey appears: Player just turned around
                    # 2. Same jersey in both windows: Lighting/angle change
                    #
                    # ALLOW split when:
                    # 3. Jersey changes (e.g., #4 → #7): Track jumped to different player
                    # 4. Jersey disappears (e.g., #4 → none): Track lost the player
                    # ================================================================

                    # Check baseline: does it have a jersey?
                    # Use lower threshold (0.3) to catch cases where jersey is visible but not super clear
                    baseline_has_jersey = False
                    baseline_jersey_id = None
                    for baseline_idx in baseline_indices:
                        if baseline_idx < len(jersey_decisions):
                            dec = jersey_decisions[baseline_idx]
                            if dec and len(dec) == 2:
                                jid, conf = dec
                                if conf >= 0.3:  # Jersey visible in baseline (relaxed threshold)
                                    baseline_has_jersey = True
                                    baseline_jersey_id = jid
                                    break

                    # Check test: does it have a jersey?
                    test_has_jersey = False
                    test_jersey_id = None
                    if track_idx < len(jersey_decisions):
                        dec = jersey_decisions[track_idx]
                        if dec and len(dec) == 2:
                            jid, conf = dec
                            if conf >= 0.3:  # Jersey visible in test (relaxed threshold)
                                test_has_jersey = True
                                test_jersey_id = jid

                    # Case 1: Jersey first appearance (player turning around)
                    # Baseline: no jersey → Test: jersey #X
                    # This is NOT a track jump, just player becoming visible
                    if not baseline_has_jersey and test_has_jersey:
                        continue  # SUPPRESS: Player turned around

                    # Case 2: Same jersey in both windows (lighting/angle change)
                    # Baseline: jersey #X → Test: jersey #X
                    # This is NOT a track jump, just appearance change
                    if baseline_has_jersey and test_has_jersey and baseline_jersey_id == test_jersey_id:
                        continue  # SUPPRESS: Same player, lighting change

                    # Case 3: Jersey changes OR Case 4: Jersey disappears
                    # - Baseline: jersey #4 → Test: jersey #7 (TRACK JUMPED to different player)
                    # - Baseline: jersey #4 → Test: no jersey (TRACK LOST the player)
                    # These indicate track identity swap, so ALLOW split
                    divergences.append({
                        "frame_idx": frames[track_idx],
                        "reason": "appearance_drift",
                        "metric": float(distance),
                    })

    # ========================================================================
    # JERSEY NUMBER INCONSISTENCY DETECTION (CRITICAL FOR INTRA-TEAM SWAPS)
    # ========================================================================
    # Strategy: Detect when jersey number appears/disappears/changes within a track.
    #
    # This catches cases like:
    # - Black shirt (no number) → Black shirt #4 (identity swap within team)
    # - Jersey #7 → Jersey #4 (number swap)
    # - Jersey #4 → No number (player switches)
    #
    # Appearance drift alone can miss these if both players have similar base colors.
    # ========================================================================
    if jersey_enabled and len(jersey_decisions) >= jersey_window:
        # Slide window through jersey decisions
        for i in range(jersey_window, len(jersey_decisions)):
            # Get baseline window (earlier frames)
            baseline_start = max(0, i - jersey_window)
            baseline_decisions = jersey_decisions[baseline_start:i]

            # Get test window (current frame neighborhood)
            test_start = i
            test_end = min(i + jersey_window, len(jersey_decisions))
            test_decisions = jersey_decisions[test_start:test_end]

            # Compute dominant jersey state in each window
            # State can be: None (no jersey), or jersey_id (e.g., 4, 7, 10)
            def get_dominant_jersey(decisions: list) -> tuple[Any, float]:
                """Return (jersey_id, avg_confidence) for dominant jersey in window."""
                jersey_votes = {}  # jersey_id -> list of confidences
                for dec in decisions:
                    if dec is None or len(dec) < 2:
                        jersey_votes[None] = jersey_votes.get(None, []) + [0.0]
                    else:
                        jid, conf = dec[0], dec[1]
                        if conf < jersey_disappear_threshold:
                            jersey_votes[None] = jersey_votes.get(None, []) + [0.0]
                        else:
                            jersey_votes[jid] = jersey_votes.get(jid, []) + [conf]

                # Find dominant jersey (most votes with highest avg confidence)
                if not jersey_votes:
                    return None, 0.0

                dominant_jid = max(jersey_votes.items(), key=lambda kv: len(kv[1]))[0]
                avg_conf = float(np.mean(jersey_votes[dominant_jid]))
                return dominant_jid, avg_conf

            baseline_jersey, baseline_conf = get_dominant_jersey(baseline_decisions)
            test_jersey, test_conf = get_dominant_jersey(test_decisions)

            # Trigger split if jersey state changed significantly
            # NOTE: We do NOT split when jersey first appears (None → #X)
            # because that's a normal case (player turns around).
            # We ONLY split when jersey disappears or changes number.
            if baseline_jersey != test_jersey:
                # Case 1: Jersey disappeared (#4 → None)
                # This indicates the track lost the player
                if baseline_jersey is not None and test_jersey is None and baseline_conf > jersey_appear_threshold:
                    divergences.append({
                        "frame_idx": frames[i],
                        "reason": "jersey_inconsistency",
                        "metric": float(baseline_conf),
                    })

                # Case 2: Jersey number changed (#7 → #4)
                # This indicates the track jumped to a different player
                elif baseline_jersey is not None and test_jersey is not None:
                    if baseline_conf > jersey_appear_threshold and test_conf > jersey_appear_threshold:
                        divergences.append({
                            "frame_idx": frames[i],
                            "reason": "jersey_inconsistency",
                            "metric": float(max(baseline_conf, test_conf)),
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
        # CRITICAL: Use lower threshold (5 frames) to prevent gaps while avoiding 1-2 frame noise
        # Fragments < 5 frames lack meaningful data (HSV, jersey) and cause unknown assignments
        fragment_length = split_idx - start_idx
        if fragment_length >= 5:  # Lower than min_fragment_length to keep meaningful fragments
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

            # Mark short fragments as low_quality
            if fragment_length < min_fragment_length:
                fragment["low_quality"] = True
                fragment["low_quality_reason"] = f"short_fragment_{fragment_length}_frames"

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
    # CRITICAL: Use lower threshold (5 frames) to prevent gaps while avoiding 1-2 frame noise
    # Fragments < 5 frames lack meaningful data (HSV, jersey) and cause unknown assignments
    fragment_length = len(frames) - start_idx
    if fragment_length >= 5:  # Lower than min_fragment_length to keep meaningful fragments
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

        # Mark short fragments as low_quality
        if fragment_length < min_fragment_length:
            fragment["low_quality"] = True
            fragment["low_quality_reason"] = f"short_fragment_{fragment_length}_frames"

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
        appearance_var = float(np.var(valid_histograms))
    else:
        mean_hsv_histogram = np.zeros(96, dtype=np.float32).tolist()
        appearance_var = 0.0

    # Compute jersey coverage (fraction of fragment with jersey detections >= 0.3 conf)
    jersey_frames = sum(1 for d in jersey_decisions if d and len(d) == 2 and d[1] >= 0.3)
    jersey_coverage = jersey_frames / len(frames) if frames else 0.0

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
        "appearance_var": appearance_var,
        "jersey_coverage": jersey_coverage,
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
