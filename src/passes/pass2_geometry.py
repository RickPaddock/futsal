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


def _bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    """Compute IoU between two xyxy boxes."""
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return float(inter_area / denom)


def build_overlap_contamination_index(
    tracks: dict[str, Any],
    proximity_threshold_px: float,
    iou_threshold: float,
) -> dict[str, set[int]]:
    """
    Build per-track contamination frame sets.

    A frame is marked contaminated for two tracks when they are spatially close
    enough that appearance/jersey evidence is considered identity-unsafe.
    """
    contamination_by_track: dict[str, set[int]] = {str(track_id): set() for track_id in tracks.keys()}
    frame_observations: dict[int, list[tuple[str, list[int]]]] = {}

    for track_id, track_data in tracks.items():
        frames = track_data.get("frames", [])
        bboxes = track_data.get("bboxes", [])
        for frame_idx, bbox in zip(frames, bboxes):
            if not bbox or len(bbox) != 4:
                continue
            frame_observations.setdefault(int(frame_idx), []).append((str(track_id), bbox))

    proximity_threshold_sq = proximity_threshold_px * proximity_threshold_px

    for frame_id, observations in frame_observations.items():
        if len(observations) < 2:
            continue

        for i in range(len(observations)):
            track_a, bbox_a = observations[i]
            ax = 0.5 * (bbox_a[0] + bbox_a[2])
            ay = 0.5 * (bbox_a[1] + bbox_a[3])

            for j in range(i + 1, len(observations)):
                track_b, bbox_b = observations[j]
                bx = 0.5 * (bbox_b[0] + bbox_b[2])
                by = 0.5 * (bbox_b[1] + bbox_b[3])

                dx = ax - bx
                dy = ay - by
                close_enough = (dx * dx + dy * dy) <= proximity_threshold_sq
                overlapping = _bbox_iou(bbox_a, bbox_b) >= iou_threshold

                if close_enough or overlapping:
                    contamination_by_track.setdefault(track_a, set()).add(int(frame_id))
                    contamination_by_track.setdefault(track_b, set()).add(int(frame_id))

    return contamination_by_track


def export_overlap_audit(
    tracks: dict[str, Any],
    contamination_by_track: dict[str, set[int]],
    output_path: Path,
    target_track_id: str,
    frame_start: int,
    frame_end: int,
) -> None:
    """Export per-frame nearest-track distance/IoU audit for contamination calibration."""
    target_track = tracks.get(target_track_id)
    if not target_track:
        print(f"  [OVERLAP_AUDIT] Track {target_track_id} not found")
        return

    target_frames = target_track.get("frames", [])
    target_bboxes = target_track.get("bboxes", [])
    if not target_frames or not target_bboxes:
        print(f"  [OVERLAP_AUDIT] Track {target_track_id} has no frame/bbox data")
        return

    frame_observations: dict[int, list[tuple[str, list[int]]]] = {}
    for track_id, track_data in tracks.items():
        frames = track_data.get("frames", [])
        bboxes = track_data.get("bboxes", [])
        for frame_idx, bbox in zip(frames, bboxes):
            if not bbox or len(bbox) != 4:
                continue
            frame_idx_int = int(frame_idx)
            if frame_idx_int < frame_start or frame_idx_int > frame_end:
                continue
            frame_observations.setdefault(frame_idx_int, []).append((str(track_id), bbox))

    per_frame = []
    nearest_distances = []
    max_ious = []
    contaminated_count = 0

    contamination_frames = contamination_by_track.get(target_track_id, set())

    for frame_idx, bbox in zip(target_frames, target_bboxes):
        frame_idx_int = int(frame_idx)
        if frame_idx_int < frame_start or frame_idx_int > frame_end:
            continue
        if not bbox or len(bbox) != 4:
            continue

        ax = 0.5 * (bbox[0] + bbox[2])
        ay = 0.5 * (bbox[1] + bbox[3])

        nearest_track_id = None
        nearest_distance_px = None
        max_iou = 0.0
        max_iou_track_id = None

        for other_track_id, other_bbox in frame_observations.get(frame_idx_int, []):
            if other_track_id == target_track_id:
                continue
            if not other_bbox or len(other_bbox) != 4:
                continue

            bx = 0.5 * (other_bbox[0] + other_bbox[2])
            by = 0.5 * (other_bbox[1] + other_bbox[3])
            distance_px = float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))
            iou_val = _bbox_iou(bbox, other_bbox)

            if nearest_distance_px is None or distance_px < nearest_distance_px:
                nearest_distance_px = distance_px
                nearest_track_id = other_track_id

            if iou_val > max_iou:
                max_iou = float(iou_val)
                max_iou_track_id = other_track_id

        contaminated = frame_idx_int in contamination_frames
        if contaminated:
            contaminated_count += 1

        if nearest_distance_px is not None:
            nearest_distances.append(nearest_distance_px)
        max_ious.append(max_iou)

        per_frame.append({
            "frame": frame_idx_int,
            "nearest_track_id": nearest_track_id,
            "nearest_distance_px": nearest_distance_px,
            "max_iou": max_iou,
            "max_iou_track_id": max_iou_track_id,
            "contaminated_current_rule": contaminated,
        })

    if not per_frame:
        print(f"  [OVERLAP_AUDIT] No frames in range {frame_start}-{frame_end} for track {target_track_id}")
        return

    nearest_arr = np.array(nearest_distances, dtype=np.float32) if nearest_distances else np.array([], dtype=np.float32)
    iou_arr = np.array(max_ious, dtype=np.float32) if max_ious else np.array([], dtype=np.float32)

    summary = {
        "track_id": target_track_id,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frames_analyzed": len(per_frame),
        "contaminated_frames_current_rule": int(contaminated_count),
        "contaminated_ratio_current_rule": float(contaminated_count / max(1, len(per_frame))),
    }

    if nearest_arr.size > 0:
        summary["nearest_distance_px_percentiles"] = {
            "p05": float(np.percentile(nearest_arr, 5)),
            "p25": float(np.percentile(nearest_arr, 25)),
            "p50": float(np.percentile(nearest_arr, 50)),
            "p75": float(np.percentile(nearest_arr, 75)),
            "p95": float(np.percentile(nearest_arr, 95)),
        }

    if iou_arr.size > 0:
        summary["max_iou_percentiles"] = {
            "p05": float(np.percentile(iou_arr, 5)),
            "p25": float(np.percentile(iou_arr, 25)),
            "p50": float(np.percentile(iou_arr, 50)),
            "p75": float(np.percentile(iou_arr, 75)),
            "p95": float(np.percentile(iou_arr, 95)),
        }

    audit_payload = {
        "summary": summary,
        "per_frame": per_frame,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, ensure_ascii=False, indent=2)

    print(
        f"  [OVERLAP_AUDIT] Saved: {output_path.name} | frames={summary['frames_analyzed']} "
        f"contaminated={summary['contaminated_frames_current_rule']}"
    )


def _build_signature(
    indices: list[int],
    hsv_histograms: list[list[int] | None],
    jersey_decisions: list[list[Any]],
    jersey_conf_threshold: float,
) -> dict[str, Any]:
    """Create a compact appearance/jersey signature from a set of reliable indices."""
    histograms = []
    jersey_counts: dict[Any, int] = {}
    jersey_conf_sum: dict[Any, float] = {}
    jersey_visible_count = 0

    for idx in indices:
        if 0 <= idx < len(hsv_histograms):
            hist = hsv_histograms[idx]
            if hist is not None and len(hist) == 96:
                histograms.append(_dequantize_histogram(hist))

        if 0 <= idx < len(jersey_decisions):
            decision = jersey_decisions[idx]
            if decision and len(decision) == 2:
                jersey_id, jersey_conf = decision
                if jersey_conf >= jersey_conf_threshold:
                    jersey_visible_count += 1
                    jersey_counts[jersey_id] = jersey_counts.get(jersey_id, 0) + 1
                    jersey_conf_sum[jersey_id] = jersey_conf_sum.get(jersey_id, 0.0) + float(jersey_conf)

    mean_hist = np.mean(histograms, axis=0) if histograms else None
    dominant_hsv_mode = int(np.argmax(mean_hist)) if mean_hist is not None else None

    dominant_jersey = None
    dominant_jersey_conf = 0.0
    if jersey_counts:
        dominant_jersey = max(jersey_counts.items(), key=lambda kv: kv[1])[0]
        total_conf = jersey_conf_sum.get(dominant_jersey, 0.0)
        dominant_jersey_conf = total_conf / max(1, jersey_counts[dominant_jersey])

    bib_presence_ratio = jersey_visible_count / max(1, len(indices))

    return {
        "mean_hist": mean_hist,
        "dominant_hsv_mode": dominant_hsv_mode,
        "dominant_jersey": dominant_jersey,
        "dominant_jersey_conf": float(dominant_jersey_conf),
        "bib_presence_ratio": float(bib_presence_ratio),
        "sample_count": len(indices),
    }


def detect_overlap_identity_contradictions(
    fragments: list[dict[str, Any]],
    pass1_tracks: dict[str, Any],
    homography: CourtHomography,
    contamination_by_track: dict[str, set[int]],
    min_fragment_length: int,
    jersey_conf_threshold: float,
    signature_window_frames: int,
    appearance_contradiction_threshold: float,
    bib_presence_threshold: float,
    bib_absence_threshold: float,
    debug_enabled: bool = False,
    debug_track_id: str | None = None,
    debug_frame_start: int | None = None,
    debug_frame_end: int | None = None,
) -> list[dict[str, Any]]:
    """
    Retroactively split fragments when overlap contamination caused identity contradiction.

    A split is considered identity-invalid when the split boundary falls inside a
    contamination window and stable pre/post reliable signatures contradict.
    """
    if not fragments:
        return fragments

    def _debug_track(track_id_val: str, frame_val: int | None) -> bool:
        if not debug_enabled:
            return False
        if debug_track_id is not None and str(track_id_val) != str(debug_track_id):
            return False
        if frame_val is not None and debug_frame_start is not None and frame_val < debug_frame_start:
            return False
        if frame_val is not None and debug_frame_end is not None and frame_val > debug_frame_end:
            return False
        return True

    fragments_to_split: list[dict[str, Any]] = []

    for frag_idx, fragment in enumerate(fragments):
        track_id = str(fragment.get("original_track_id", ""))
        if not track_id or track_id not in pass1_tracks:
            continue

        contamination_frames = contamination_by_track.get(track_id, set())
        if not contamination_frames:
            continue

        track_data = pass1_tracks[track_id]
        frames = track_data.get("frames", [])
        if not frames:
            continue

        start_frame = fragment.get("start_frame")
        end_frame = fragment.get("end_frame")
        if start_frame is None or end_frame is None:
            continue

        try:
            start_idx = frames.index(start_frame)
            end_idx = frames.index(end_frame) + 1
        except ValueError:
            continue

        frag_indices = list(range(start_idx, end_idx))
        contaminated_indices = [idx for idx in frag_indices if int(frames[idx]) in contamination_frames]
        if not contaminated_indices:
            continue

        contamination_segments: list[tuple[int, int]] = []
        seg_start = contaminated_indices[0]
        seg_prev = contaminated_indices[0]
        for idx in contaminated_indices[1:]:
            if idx == seg_prev + 1:
                seg_prev = idx
                continue
            contamination_segments.append((seg_start, seg_prev))
            seg_start = idx
            seg_prev = idx
        contamination_segments.append((seg_start, seg_prev))

        hsv_histograms = track_data.get("hsv_histograms", [])
        jersey_decisions = track_data.get("jersey_decisions", [])
        split_recorded = False
        for segment_start, segment_end in contamination_segments:
            boundary_frame = int(frames[segment_end])

            pre_reliable = [idx for idx in frag_indices if idx < segment_start and int(frames[idx]) not in contamination_frames]
            post_reliable = [idx for idx in frag_indices if idx > segment_end and int(frames[idx]) not in contamination_frames]

            effective_window = min(signature_window_frames, len(pre_reliable), len(post_reliable))
            if effective_window < 3:
                if _debug_track(track_id, boundary_frame):
                    print(
                        f"[OVERLAP_DEBUG] track={track_id} frame={boundary_frame} "
                        f"skip=insufficient_reliable_windows pre={len(pre_reliable)} post={len(post_reliable)} req>=3"
                    )
                continue

            pre_window = pre_reliable[-effective_window:]
            post_window = post_reliable[:effective_window]

            pre_signature = _build_signature(pre_window, hsv_histograms, jersey_decisions, jersey_conf_threshold)
            post_signature = _build_signature(post_window, hsv_histograms, jersey_decisions, jersey_conf_threshold)

            jersey_mismatch = (
                pre_signature["dominant_jersey"] is not None
                and post_signature["dominant_jersey"] is not None
                and pre_signature["dominant_jersey"] != post_signature["dominant_jersey"]
            )

            bib_flip = (
                (pre_signature["bib_presence_ratio"] >= bib_presence_threshold and post_signature["bib_presence_ratio"] <= bib_absence_threshold)
                or (post_signature["bib_presence_ratio"] >= bib_presence_threshold and pre_signature["bib_presence_ratio"] <= bib_absence_threshold)
            )

            appearance_distance = 0.0
            hsv_mode_change = False
            if pre_signature["mean_hist"] is not None and post_signature["mean_hist"] is not None:
                appearance_distance = compute_appearance_distance(pre_signature["mean_hist"], post_signature["mean_hist"])
                hsv_mode_change = appearance_distance >= appearance_contradiction_threshold

            if not (jersey_mismatch or bib_flip or hsv_mode_change):
                if _debug_track(track_id, boundary_frame):
                    print(
                        f"[OVERLAP_DEBUG] track={track_id} frame={boundary_frame} skip=no_contradiction "
                        f"dist={appearance_distance:.3f} jersey_mismatch={jersey_mismatch} bib_flip={bib_flip} hsv_mode_change={hsv_mode_change}"
                    )
                continue

            split_idx_in_track = post_window[0]
            if split_idx_in_track <= start_idx or split_idx_in_track >= end_idx:
                if _debug_track(track_id, boundary_frame):
                    print(
                        f"[OVERLAP_DEBUG] track={track_id} frame={boundary_frame} skip=invalid_retro_split_index idx={split_idx_in_track}"
                    )
                continue

            split_frame = int(frames[split_idx_in_track])
            contradiction_flags = []
            if jersey_mismatch:
                contradiction_flags.append("jersey_mismatch")
            if bib_flip:
                contradiction_flags.append("bib_flip")
            if hsv_mode_change:
                contradiction_flags.append("hsv_mode_change")

            if _debug_track(track_id, boundary_frame):
                print(
                    f"[OVERLAP_DEBUG] track={track_id} frame={boundary_frame} trigger=overlap_identity_contradiction "
                    f"retro_split={split_frame} flags={contradiction_flags} dist={appearance_distance:.3f}"
                )

            fragments_to_split.append({
                "fragment_idx": frag_idx,
                "split_frame": split_frame,
                "appearance_distance": float(appearance_distance),
                "contradictions": contradiction_flags,
                "existing_split_frame": boundary_frame,
                "pre_signature": {
                    "dominant_hsv_mode": pre_signature["dominant_hsv_mode"],
                    "dominant_jersey": pre_signature["dominant_jersey"],
                    "dominant_jersey_conf": pre_signature["dominant_jersey_conf"],
                    "bib_presence_ratio": pre_signature["bib_presence_ratio"],
                    "sample_count": pre_signature["sample_count"],
                },
                "post_signature": {
                    "dominant_hsv_mode": post_signature["dominant_hsv_mode"],
                    "dominant_jersey": post_signature["dominant_jersey"],
                    "dominant_jersey_conf": post_signature["dominant_jersey_conf"],
                    "bib_presence_ratio": post_signature["bib_presence_ratio"],
                    "sample_count": post_signature["sample_count"],
                },
            })
            split_recorded = True
            break

        if split_recorded:
            continue

    if not fragments_to_split:
        return fragments

    print(f"    Detected {len(fragments_to_split)} overlap identity contradiction(s)")

    fragments_to_split.sort(key=lambda item: item["fragment_idx"], reverse=True)
    new_fragments = list(fragments)

    for split_info in fragments_to_split:
        frag_idx = split_info["fragment_idx"]
        split_frame = split_info["split_frame"]
        original_fragment = new_fragments[frag_idx]
        original_track_id = str(original_fragment.get("original_track_id", ""))
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

        start_frame = original_fragment.get("start_frame")
        end_frame = original_fragment.get("end_frame")
        try:
            start_idx = frames.index(start_frame)
            end_idx = frames.index(end_frame) + 1
            split_idx_in_track = frames.index(split_frame)
        except ValueError:
            continue

        if split_idx_in_track <= start_idx or split_idx_in_track >= end_idx:
            continue

        split_idx = split_idx_in_track - start_idx
        court_positions = []
        for centroid in centroids[start_idx:end_idx]:
            court_x, court_y = homography.pixel_to_court(centroid[0], centroid[1])
            court_positions.append([court_x, court_y])

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
            "reason": "overlap_identity_contradiction",
            "metric": split_info["appearance_distance"],
            "severity": "hard",
            "existing_split_frame": split_info["existing_split_frame"],
            "contradictions": split_info["contradictions"],
            "pre_signature": split_info["pre_signature"],
            "post_signature": split_info["post_signature"],
        }
        fragment_before["fragment_id"] = original_fragment.get("fragment_id", "frag_unknown")
        fragment_before["parent_fragment_id"] = original_fragment.get("parent_fragment_id")
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
        fragment_after["split_reason"] = original_fragment.get("split_reason")
        original_frag_id = original_fragment.get("fragment_id", "frag_unknown")
        fragment_after["fragment_id"] = f"{original_frag_id}_split"
        fragment_after["parent_fragment_id"] = original_frag_id
        fragment_after["is_primary_fragment"] = False

        before_len = len(fragment_before.get("spatial_footprint", {}).get("court_positions", []))
        after_len = len(fragment_after.get("spatial_footprint", {}).get("court_positions", []))
        min_identity_fragment_length = 3

        if before_len < min_identity_fragment_length or after_len < min_identity_fragment_length:
            continue

        replacement = []
        if before_len >= min_fragment_length or before_len >= min_identity_fragment_length:
            replacement.append(fragment_before)
        if after_len >= min_fragment_length or after_len >= min_identity_fragment_length:
            replacement.append(fragment_after)
        if len(replacement) != 2:
            continue

        new_fragments[frag_idx:frag_idx + 1] = replacement

    return new_fragments


def create_ghost_fragments(
    fragments: list[dict[str, Any]],
    pass1_tracks: dict[str, Any],
    homography: Any,
    level_init_frames: int = 30,
) -> list[dict[str, Any]]:
    """
    Create ghost fragments to maintain player count invariant.

    CRITICAL: Pass 1 is truth. Fragments are metadata only.
    Ghost presence is determined by Pass 1 frame lists, not fragment spans.

    Level is DYNAMIC: increases as more players enter the field, never decreases.
    - Start with 10 players → level = 10
    - 11th player enters → level = 11
    - 12th player enters → level = 12 (capped)
    - Level NEVER goes below the high water mark
    """
    # Step 1: Initialize level as rolling high water mark
    # Start with a conservative estimate from first few frames
    initial_level = 0
    for frame in range(1, min(level_init_frames + 1, 11)):  # First 10 frames
        present_count = len({
            track_id
            for track_id, track in pass1_tracks.items()
            if frame in track.get("frames", [])
        })
        initial_level = max(initial_level, present_count)

    level = min(12, initial_level)  # Cap at 12 for futsal
    print(f"  Ghost tracking: initial level = {level} (from first 10 frames)")

    # Step 2: Find all frames that need processing
    all_frames_set = set()
    for track in pass1_tracks.values():
        all_frames_set.update(track.get("frames", []))

    if not all_frames_set:
        return fragments

    all_frames = sorted(all_frames_set)
    max_frame = max(all_frames)

    # Step 3: For each frame, check if we need ghosts
    # Level is DYNAMIC: updates as more players enter
    ghost_fragments = []
    active_ghosts = {}  # track_id -> {start_frame, positions, ...}
    ghost_counter = 0

    for frame in range(1, max_frame + 1):
        # Compute present players from Pass 1 ONLY
        present_players = {
            track_id
            for track_id, track in pass1_tracks.items()
            if frame in track.get("frames", [])
        }

        present_count = len(present_players)

        # DYNAMIC LEVEL: Update level if more players enter (never decrease)
        # This handles players entering from off-screen
        if present_count > level:
            level = min(12, present_count)  # Cap at 12 for futsal

        # Close ghosts for reappeared players
        for track_id in list(active_ghosts.keys()):
            if track_id in present_players:
                # Player reappeared - close ghost
                ghost = active_ghosts.pop(track_id)
                ghost["end_frame"] = frame - 1
                ghost["frame_count"] = ghost["end_frame"] - ghost["start_frame"] + 1
                ghost_fragments.append(ghost)

        # Create ghosts for missing players
        missing_count = level - present_count
        if missing_count > 0:
            # Find which players are missing
            all_known_players = set(pass1_tracks.keys())
            missing_players = all_known_players - present_players

            # Prioritize players that were recently seen
            recently_seen = []
            for track_id in missing_players:
                track_frames = pass1_tracks[track_id].get("frames", [])
                if track_frames and max(track_frames) >= frame - 60:  # Within last 2 seconds
                    last_seen = max(f for f in track_frames if f < frame) if any(f < frame for f in track_frames) else 0
                    recently_seen.append((track_id, last_seen))

            recently_seen.sort(key=lambda x: x[1], reverse=True)

            for track_id, last_seen in recently_seen[:missing_count]:
                if track_id not in active_ghosts:
                    # Create new ghost
                    # Get last known bbox from Pass 1
                    track_data = pass1_tracks[track_id]
                    track_frames = track_data.get("frames", [])
                    if last_seen in track_frames:
                        idx = track_frames.index(last_seen)
                        last_bbox = track_data["bboxes"][idx]
                        last_centroid = track_data["centroids"][idx]

                        # Find source fragment for team/jersey
                        source_frag = None
                        for frag in fragments:
                            if (frag["original_track_id"] == track_id and
                                frag["start_frame"] <= last_seen <= frag["end_frame"]):
                                source_frag = frag
                                break

                        ghost = {
                            "fragment_id": f"ghost_{ghost_counter:06d}",
                            "original_track_id": track_id,
                            "start_frame": frame,
                            "end_frame": frame,  # Updated as ghost persists
                            "frame_count": 0,
                            "is_ghost": True,
                            # CRITICAL: Exclude ghosts from all Pass 3 clustering/voting
                            "exclude_from_clustering": True,
                            "exclude_from_team_vote": True,
                            "exclude_from_identity_vote": True,
                            "exclude_from_stats": True,
                            "pixel_bboxes": [last_bbox],
                            "pixel_centroids": [last_centroid],
                            "mean_hsv_histogram": source_frag.get("mean_hsv_histogram") if source_frag else None,
                            "jersey_prob_timeline": source_frag.get("jersey_prob_timeline", {}) if source_frag else {},
                            "spatial_footprint": {
                                "court_positions": [],
                                "mean_position": [0, 0],
                                "spatial_variance": 0.0,
                            },
                            "visibility_quality": 0.0,
                            "mean_confidence": 0.0,
                        }
                        active_ghosts[track_id] = ghost
                        ghost_counter += 1
                else:
                    # Update existing ghost - hold position
                    ghost = active_ghosts[track_id]
                    ghost["pixel_bboxes"].append(ghost["pixel_bboxes"][-1])
                    ghost["pixel_centroids"].append(ghost["pixel_centroids"][-1])
                    ghost["end_frame"] = frame

    # Close remaining ghosts (clip ended)
    for ghost in active_ghosts.values():
        ghost["frame_count"] = ghost["end_frame"] - ghost["start_frame"] + 1
        ghost_fragments.append(ghost)

    print(f"  Ghost fragments created: {len(ghost_fragments)}")
    print(f"  Final level (high water mark): {level} players")
    return fragments + ghost_fragments


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

    tracks = pass1_data.get("tracks", {})
    overlap_cfg = div_cfg.get("overlap_contamination", {})
    overlap_enabled = overlap_cfg.get("enabled", True)
    overlap_contamination_by_track = (
        build_overlap_contamination_index(
            tracks=tracks,
            proximity_threshold_px=float(overlap_cfg.get("proximity_threshold_pixels", 70.0)),
            iou_threshold=float(overlap_cfg.get("iou_threshold", 0.1)),
        )
        if overlap_enabled
        else {str(track_id): set() for track_id in tracks.keys()}
    )

    # Process each track from Pass 1
    fragment_id_counter = 0

    overlap_audit_cfg = overlap_cfg.get("audit", {})
    if overlap_enabled and overlap_audit_cfg.get("enabled", False):
        audit_track_id = str(overlap_audit_cfg.get("track_id", "2"))
        audit_frame_start = int(overlap_audit_cfg.get("frame_start", 820))
        audit_frame_end = int(overlap_audit_cfg.get("frame_end", 930))
        audit_path = output_dir / f"{pass1_file.stem}_overlap_audit_track_{audit_track_id}.json"
        export_overlap_audit(
            tracks=tracks,
            contamination_by_track=overlap_contamination_by_track,
            output_path=audit_path,
            target_track_id=audit_track_id,
            frame_start=audit_frame_start,
            frame_end=audit_frame_end,
        )

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

        contamination_frames = overlap_contamination_by_track.get(str(track_id), set())
        reliable_mask = [int(frame_idx) not in contamination_frames for frame_idx in frames]

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
            appearance_min_consecutive=div_cfg.get("appearance_min_consecutive", 2),
            appearance_threshold_strong=div_cfg.get("appearance_threshold_strong", 6.0),
            debug_track_id=div_cfg.get("appearance_debug", {}).get("track_id"),
            debug_frame_start=div_cfg.get("appearance_debug", {}).get("frame_start"),
            debug_frame_end=div_cfg.get("appearance_debug", {}).get("frame_end"),
            debug_enabled=div_cfg.get("appearance_debug", {}).get("enabled", False),
            track_id=track_id,
            jersey_enabled=div_cfg.get("jersey_inconsistency_enabled", True),
            jersey_appear_threshold=div_cfg.get("jersey_appear_threshold", 0.5),
            jersey_disappear_threshold=div_cfg.get("jersey_disappear_threshold", 0.3),
            jersey_window=div_cfg.get("jersey_consistency_window", 10),
            reliable_mask=reliable_mask,
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

    if overlap_enabled:
        overlap_before = output_data["fragments"]
        overlap_debug_cfg = overlap_cfg.get("debug", {})
        output_data["fragments"] = detect_overlap_identity_contradictions(
            fragments=output_data["fragments"],
            pass1_tracks=tracks,
            homography=homography,
            contamination_by_track=overlap_contamination_by_track,
            min_fragment_length=min_fragment_length,
            jersey_conf_threshold=float(div_cfg.get("jersey_appear_threshold", 0.5)),
            signature_window_frames=int(overlap_cfg.get("signature_window_frames", 6)),
            appearance_contradiction_threshold=float(
                overlap_cfg.get("appearance_contradiction_threshold", div_cfg.get("appearance_threshold_strong", 6.0))
            ),
            bib_presence_threshold=float(overlap_cfg.get("bib_presence_threshold", 0.5)),
            bib_absence_threshold=float(overlap_cfg.get("bib_absence_threshold", 0.2)),
            debug_enabled=bool(overlap_debug_cfg.get("enabled", False)),
            debug_track_id=overlap_debug_cfg.get("track_id"),
            debug_frame_start=overlap_debug_cfg.get("frame_start"),
            debug_frame_end=overlap_debug_cfg.get("frame_end"),
        )
        if len(output_data["fragments"]) != len(overlap_before):
            print(f"  Overlap contradiction splitting: {len(overlap_before)} -> {len(output_data['fragments'])} fragments")

    jersey_conf_threshold = float(div_cfg.get("jersey_appear_threshold", 0.5))
    for fragment in output_data["fragments"]:
        fragment_id = fragment.get("fragment_id", "frag_unknown")
        jersey_prob_timeline = fragment.get("jersey_prob_timeline", {}) or {}
        strong_jerseys = []
        for jersey_id, stats in jersey_prob_timeline.items():
            count = int(stats.get("count", 0))
            if count <= 0:
                continue
            avg_conf = float(stats.get("total", 0.0)) / float(count)
            if count >= 2 and avg_conf >= jersey_conf_threshold:
                strong_jerseys.append((jersey_id, avg_conf, count))

        if len(strong_jerseys) > 1:
            strong_jerseys.sort(key=lambda item: (item[1], item[2]), reverse=True)
            raise AssertionError(
                f"Pass2 fragment purity violation: {fragment_id} has multiple strong jerseys "
                f"{[(jid, round(conf, 3), cnt) for jid, conf, cnt in strong_jerseys]}"
            )

    # Create ghost fragments (Pass 1-driven, fragment-agnostic)
    ghost_cfg = config.get("ghost_tracking", {})
    if ghost_cfg.get("enabled", True):
        level_init_frames = int(ghost_cfg.get("level_init_frames", 30))
        output_data["fragments"] = create_ghost_fragments(
            fragments=output_data["fragments"],
            pass1_tracks=tracks,
            homography=homography,
            level_init_frames=level_init_frames,
        )

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

                    # Decide which fragment to split:
                    # Split the one where the jersey appears LATER (it "stole" the jersey)
                    if frag_a["jersey_first_frame"] < frag_b["jersey_first_frame"]:
                        # Fragment A had jersey first, split Fragment B at its jersey appearance
                        split_target = frag_b
                        split_frame = frag_b["jersey_first_frame"]
                        prior_frag = frag_a
                    else:
                        # Fragment B had jersey first, split Fragment A at its jersey appearance
                        split_target = frag_a
                        split_frame = frag_a["jersey_first_frame"]
                        prior_frag = frag_b

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
        print(f"      - Fragment {fragment.get('fragment_id')} (track {fragment.get('original_track_id')}): "
              f"jersey #{split_info['jersey_id']} conflicts with {prior_frag.get('fragment_id')} "
              f"(overlap={split_info['overlap_duration']} frames, split at {split_info['split_frame']})")

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
        fragment_before["parent_fragment_id"] = original_fragment.get("parent_fragment_id")
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
        original_frag_id = original_fragment.get("fragment_id", "frag_unknown")
        fragment_after["fragment_id"] = f"{original_frag_id}_split"
        fragment_after["parent_fragment_id"] = original_frag_id  # NEW: Point to the original fragment
        fragment_after["is_primary_fragment"] = False  # After split is secondary (jersey jump)

        # Replace original fragment with two new fragments
        # For jersey temporal exclusivity splits (identity jumps):
        # - Allow short fragments (minimum 3 frames) because they represent critical transitions
        # For other splits:
        # - Require min_fragment_length for both pieces
        is_jersey_split = split_info.get("reason") == "jersey_temporal_exclusivity"
        min_length_for_jersey_split = max(3, min_fragment_length // 3)  # Allow 1/3 normal length for jersey splits

        replacement_fragments = []
        before_len = len(fragment_before.get("spatial_footprint", {}).get("court_positions", []))
        after_len = len(fragment_after.get("spatial_footprint", {}).get("court_positions", []))
        
        # Keep before part if it meets minimum
        if before_len >= min_fragment_length:
            replacement_fragments.append(fragment_before)
        elif is_jersey_split and before_len >= min_length_for_jersey_split:
            replacement_fragments.append(fragment_before)
        
        # Keep after part if it meets minimum
        if after_len >= min_fragment_length:
            replacement_fragments.append(fragment_after)
        elif is_jersey_split and after_len >= min_length_for_jersey_split:
            replacement_fragments.append(fragment_after)

        # If both fragments are too short (even for jersey splits), keep original
        if not replacement_fragments:
            continue

        # Replace in list
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
    appearance_min_consecutive: int,
    appearance_threshold_strong: float,
    debug_track_id: str | None,
    debug_frame_start: int | None,
    debug_frame_end: int | None,
    debug_enabled: bool,
    track_id: str,
    jersey_enabled: bool,
    jersey_appear_threshold: float,
    jersey_disappear_threshold: float,
    jersey_window: int,
    reliable_mask: list[bool] | None = None,
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
                        "severity": "soft",
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
                "severity": "soft",
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
                        "severity": "soft",
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
    valid_hist_indices = [
        i
        for i, h in enumerate(hsv_histograms)
        if h is not None and len(h) == 96 and (reliable_mask is None or (i < len(reliable_mask) and reliable_mask[i]))
    ]

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

            consecutive_drift_hits = 0

            for test_idx in range(test_start_idx, test_end_idx):
                track_idx = valid_hist_indices[test_idx]
                test_hist = _dequantize_histogram(hsv_histograms[track_idx])

                # Compute appearance distance
                distance = compute_appearance_distance(baseline_centroid, test_hist)

                # Trigger split only after consecutive drift hits, unless a strong jump is detected
                if distance >= appearance_threshold_strong:
                    consecutive_drift_hits = max(1, appearance_min_consecutive)
                elif distance > appearance_threshold:
                    consecutive_drift_hits += 1
                else:
                    consecutive_drift_hits = 0

                if debug_enabled and debug_track_id is not None:
                    if str(track_id) == str(debug_track_id):
                        frame_val = frames[track_idx]
                        if (debug_frame_start is None or frame_val >= debug_frame_start) and (
                            debug_frame_end is None or frame_val <= debug_frame_end
                        ):
                            print(
                                f"[APPEAR_DEBUG] track={track_id} frame={frame_val} dist={distance:.3f} cons={consecutive_drift_hits} "
                                f"thr={appearance_threshold} strong={appearance_threshold_strong}"
                            )

                if consecutive_drift_hits >= max(1, appearance_min_consecutive):
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
                        if debug_enabled and debug_track_id is not None and str(track_id) == str(debug_track_id):
                            frame_val = frames[track_idx]
                            if (debug_frame_start is None or frame_val >= debug_frame_start) and (
                                debug_frame_end is None or frame_val <= debug_frame_end
                            ):
                                print("[APPEAR_DEBUG] suppress: jersey first appearance")
                        continue  # SUPPRESS: Player turned around

                    # Case 2: Same jersey in both windows (lighting/angle change)
                    # Baseline: jersey #X → Test: jersey #X
                    # This is NOT a track jump, just appearance change
                    if baseline_has_jersey and test_has_jersey and baseline_jersey_id == test_jersey_id:
                        if debug_enabled and debug_track_id is not None and str(track_id) == str(debug_track_id):
                            frame_val = frames[track_idx]
                            if (debug_frame_start is None or frame_val >= debug_frame_start) and (
                                debug_frame_end is None or frame_val <= debug_frame_end
                            ):
                                print("[APPEAR_DEBUG] suppress: same jersey in baseline/test")
                        continue  # SUPPRESS: Same player, lighting change

                    # Case 3: Jersey changes OR Case 4: Jersey disappears
                    # - Baseline: jersey #4 → Test: jersey #7 (TRACK JUMPED to different player)
                    # - Baseline: jersey #4 → Test: no jersey (TRACK LOST the player)
                    # These indicate track identity swap, so ALLOW split
                    divergences.append({
                        "frame_idx": frames[track_idx],
                        "reason": "appearance_drift",
                        "metric": float(distance),
                        "severity": "hard" if distance >= appearance_threshold_strong else "soft",
                    })
                    if debug_enabled and debug_track_id is not None and str(track_id) == str(debug_track_id):
                        frame_val = frames[track_idx]
                        if (debug_frame_start is None or frame_val >= debug_frame_start) and (
                            debug_frame_end is None or frame_val <= debug_frame_end
                        ):
                            print("[APPEAR_DEBUG] split: appearance_drift")
                    break

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
                        "severity": "hard",
                    })

                # Case 2: Jersey number changed (#7 → #4)
                # This indicates the track jumped to a different player
                elif baseline_jersey is not None and test_jersey is not None:
                    if baseline_conf > jersey_appear_threshold and test_conf > jersey_appear_threshold:
                        divergences.append({
                            "frame_idx": frames[i],
                            "reason": "jersey_inconsistency",
                            "metric": float(max(baseline_conf, test_conf)),
                            "severity": "hard",
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
        # DEBUG
        if track_id == "2" and 820 <= split_frame <= 870:
            print(f"[SPLIT_TRACK_DEBUG] track_id={track_id} splitting at frame {split_frame}, current start_idx={start_idx} (frame {frames[start_idx] if start_idx < len(frames) else 'EOF'})")
        # Find index in frames list
        try:
            split_idx = frames.index(split_frame)
        except ValueError:
            continue

        # Create fragment from start_idx to split_idx
        # Allow short fragments if they're created by identity-jump divergences
        divergence_for_split = next(
            (d for d in divergence_points if d["frame_idx"] == split_frame), None
        )
        identity_jump_reason = divergence_for_split.get("reason") if divergence_for_split else None
        is_identity_jump = identity_jump_reason in {"appearance_drift", "jersey_inconsistency", "jersey_temporal_exclusivity", "velocity_spike"}
        
        fragment_length = split_idx - start_idx
        min_length_for_fragment = 3 if is_identity_jump else min_fragment_length
        
        if fragment_length >= min_length_for_fragment:
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
            fragment["split_reason"] = divergence_for_split

            # Compute appearance distance to pre-split centroid
            if pre_split_centroid is not None and fragment.get("mean_hsv_histogram"):
                fragment_hist = np.array(fragment["mean_hsv_histogram"], dtype=np.float32)
                fragment["appearance_distance_to_origin"] = compute_appearance_distance(
                    pre_split_centroid, fragment_hist
                )
            else:
                fragment["appearance_distance_to_origin"] = float('inf')

            fragments.append(fragment)
            # DEBUG
            if track_id == "2" and 820 <= frames[start_idx] <= 870:
                print(f"[SPLIT_TRACK_DEBUG] Created fragment frames {frames[start_idx]}-{frames[split_idx-1]} (length={fragment_length})")

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
