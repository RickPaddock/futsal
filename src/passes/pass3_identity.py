"""
Pass 3: Identity Inference (Offline)

Output: pass3_final/<clip>.json
"""

from pathlib import Path
import json
import numpy as np
from typing import Any
import cv2
from sklearn.cluster import KMeans

from src.utils.video_io import VideoReader
from src.utils.data_models import BoundingBox
from src.detection.team_clustering import TeamClustering

try:
    import orjson
except ImportError:
    orjson = None


def run_pass3(run_dir: Path, config: dict):
    run_dir = Path(run_dir)
    pass2_dir = run_dir / "pass2_identity"
    pass3_dir = run_dir / "pass3_final"
    pass3_dir.mkdir(parents=True, exist_ok=True)

    pass2_files = list(pass2_dir.glob("*_fragments.json"))
    if not pass2_files:
        print(f"No Pass 2 fragment JSON files found in {pass2_dir}")
        return

    print(f"Found {len(pass2_files)} Pass 2 fragment files")

    for pass2_file in pass2_files:
        print(f"Processing: {pass2_file.name}")
        process_clip_pass3(pass2_file=pass2_file, output_dir=pass3_dir, config=config, run_dir=run_dir)

    print(f"Pass 3 complete! Output: {pass3_dir}")


def process_clip_pass3(pass2_file: Path, output_dir: Path, config: dict, run_dir: Path):
    with open(pass2_file, "r", encoding="utf-8") as f:
        pass2_data = json.load(f)

    clip_name = pass2_data.get("clip_name", pass2_file.stem)
    fragments = pass2_data.get("fragments", [])

    if len(fragments) == 0:
        print(f"  No fragments found")
        return

    team_cfg = config.get("team_clustering", {})
    jersey_cfg = config.get("jersey", {})
    n_clusters = team_cfg.get("n_clusters", 2)
    jersey_lock_threshold = jersey_cfg.get("lock_threshold", 0.7)
    jersey_min_detections = jersey_cfg.get("min_detections", 3)
    jersey_min_coverage = jersey_cfg.get("min_coverage_pct", 0.15)
    jersey_frame_stride = max(1, int(jersey_cfg.get("frame_stride", 1)))

    fragment_histograms = []
    valid_fragment_indices = []
    for i, fragment in enumerate(fragments):
        hist = fragment.get("mean_hsv_histogram")
        if hist and len(hist) == 96:
            fragment_histograms.append(hist)
            valid_fragment_indices.append(i)

    if len(fragment_histograms) < n_clusters:
        print(f"  Not enough valid fragments")
        return

    X = np.array(fragment_histograms, dtype=np.float32)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    cluster_variances = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_hist = X[cluster_mask]
        if len(cluster_hist) > 0:
            variance = np.var(cluster_hist, axis=0).mean()
            cluster_variances.append((cluster_id, variance))

    cluster_variances.sort(key=lambda x: x[1])
    bibbed_cluster_id = cluster_variances[0][0]

    print(f"  K-Means: {len(fragments)} fragments to 2 teams")
    print(f"  TEAM_A (bibbed): cluster {bibbed_cluster_id}")

    for i, cluster_id in zip(valid_fragment_indices, cluster_labels):
        if cluster_id == bibbed_cluster_id:
            fragments[i]["team"] = "team_a"
        else:
            fragments[i]["team"] = "team_b"


    # Get detailed jersey assignments (returns dict with assignment details)
    jersey_assignments_detailed = _infer_jersey_numbers_detailed(
        fragments,
        lock_threshold=jersey_lock_threshold,
        min_detections=jersey_min_detections,
        min_coverage_pct=jersey_min_coverage,
        frame_stride=jersey_frame_stride,
    )

    # Display assignment summary with statistics
    _print_assignment_summary(fragments, jersey_assignments_detailed)

    identities = []
    for fragment in fragments:
        team = fragment.get("team", "unknown")
        fragment_id = fragment.get("fragment_id")

        # Get assignment details
        assignment = jersey_assignments_detailed.get(fragment_id, {})
        jersey_number = assignment.get("jersey_number")
        confidence = assignment.get("confidence", 0.0)
        reason = assignment.get("reason", "no_assignment")

        if jersey_number is not None:
            player_id = f"{team.upper()}_{jersey_number}" if team in ("team_a", "team_b") else f"UNKNOWN_{jersey_number}"
        else:
            if team == "team_a":
                player_id = "TEAM_A_UNKNOWN"
            elif team == "team_b":
                player_id = "TEAM_B_UNKNOWN"
            else:
                player_id = "UNKNOWN"

        identity = {
            "fragment_id": fragment_id,
            "player_id": player_id,
            "team": team,
            "jersey_number": jersey_number,
            "confidence": confidence,
            "assignment_reason": reason,  # Add reason for explainability
            "start_frame": fragment.get("start_frame"),
            "end_frame": fragment.get("end_frame"),
        }
        identities.append(identity)

    output_data = {
        "clip_name": clip_name,
        "identities": identities,
        "team_summary": {
            "team_a_count": len([i for i in identities if i["team"] == "team_a"]),
            "team_b_count": len([i for i in identities if i["team"] == "team_b"]),
            "team_a_jerseys": list(set(i["jersey_number"] for i in identities if i["jersey_number"] is not None)),
        }
    }

    output_stem = pass2_file.stem.replace('_fragments', '')
    output_path = output_dir / f"{output_stem}.json"
    pretty_json = config.get("pass3", {}).get("pretty_json", True)

    if orjson is not None:
        with open(output_path, "wb") as f:
            if pretty_json:
                f.write(orjson.dumps(output_data, option=orjson.OPT_INDENT_2))
            else:
                f.write(orjson.dumps(output_data))
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            if pretty_json:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            else:
                json.dump(output_data, f, separators=(",", ":"), ensure_ascii=False)

    print(f"  Saved: {output_path}")
    print(f"  Identities: {len(identities)}")
    print(f"  TEAM_A: {output_data['team_summary']['team_a_count']} fragments")
    print(f"  TEAM_B: {output_data['team_summary']['team_b_count']} fragments")

    _export_team_crops(
        run_dir=run_dir,
        clip_name=clip_name,
        output_dir=output_dir,
        output_stem=output_stem,
        fragments=fragments,
        config=config,
    )


def _infer_jersey_numbers_detailed(
    fragments: list[dict[str, Any]],
    lock_threshold: float = 0.7,
    min_detections: int = 3,
    min_coverage_pct: float = 0.15,
    frame_stride: int = 1,
) -> dict[str, dict[str, Any]]:
    """
    Assign jersey numbers to fragments with strict single-owner enforcement.

    Core Invariant: At any frame, a jersey number can belong to AT MOST ONE fragment.

    Args:
        fragments: List of fragment dicts from Pass 2
        lock_threshold: Minimum cumulative confidence for assignment
        min_detections: Minimum number of jersey detections required
        min_coverage_pct: Detections must span at least this fraction of fragment duration

    Returns:
        Dict mapping fragment_id -> {
            "jersey_number": int or None,
            "confidence": float,
            "reason": str (explainable assignment reason)
        }
    """
    MIN_DETECTIONS = min_detections
    MIN_COVERAGE_PCT = min_coverage_pct

    # Step 1: Build fragment metadata and eligibility
    fragment_metadata = []
    for fragment in fragments:
        fragment_id = fragment.get("fragment_id")
        start_frame = fragment.get("start_frame")
        end_frame = fragment.get("end_frame")
        team = fragment.get("team", "unknown")
        jersey_timeline = fragment.get("jersey_prob_timeline", {})
        frame_count = fragment.get("frame_count", end_frame - start_frame + 1)
        expected_samples = max(1, int(np.ceil(frame_count / max(frame_stride, 1))))

        # Check eligibility for each jersey number
        eligible_jerseys = {}
        for jersey_id, stats in jersey_timeline.items():
            jersey_num = int(jersey_id)
            total_conf = stats.get("total", 0.0)
            count = stats.get("count", 0)

            # Eligibility checks
            if count < MIN_DETECTIONS:
                continue
            if total_conf < lock_threshold:
                continue

            # Check temporal coverage (detections span enough of sampled frames)
            coverage = count / expected_samples
            if coverage < MIN_COVERAGE_PCT:
                continue

            # Compute evidence score (for conflict resolution)
            evidence_score = total_conf  # Can be enhanced with count × mean_confidence

            eligible_jerseys[jersey_num] = {
                "total_confidence": total_conf,
                "count": count,
                "coverage": coverage,
                "evidence_score": evidence_score,
            }

        fragment_metadata.append({
            "fragment_id": fragment_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "team": team,
            "eligible_jerseys": eligible_jerseys,
        })

    # Step 2: Build temporal overlap index (which fragments overlap in time)
    def fragments_overlap(f1_meta, f2_meta) -> bool:
        """Check if two fragments overlap in time."""
        return not (f1_meta["end_frame"] < f2_meta["start_frame"] or
                   f2_meta["end_frame"] < f1_meta["start_frame"])

    # Step 3: Build conflict groups (fragments competing for same jersey)
    # jersey_num -> list of (fragment_id, evidence_score, start_frame, end_frame, jersey_stats)
    jersey_candidates = {}

    for frag_meta in fragment_metadata:
        fragment_id = frag_meta["fragment_id"]
        eligible_jerseys = frag_meta["eligible_jerseys"]

        for jersey_num, jersey_stats in eligible_jerseys.items():
            if jersey_num not in jersey_candidates:
                jersey_candidates[jersey_num] = []

            jersey_candidates[jersey_num].append({
                "fragment_id": fragment_id,
                "evidence_score": jersey_stats["evidence_score"],
                "start_frame": frag_meta["start_frame"],
                "end_frame": frag_meta["end_frame"],
                "jersey_stats": jersey_stats,
            })

    # Step 4: Resolve conflicts using greedy algorithm
    # Process jerseys one at a time, assign to non-overlapping fragments with highest evidence
    assignments = {}
    assigned_fragments = set()  # Track which fragments have been assigned

    for jersey_num, candidates in jersey_candidates.items():
        # Sort candidates by evidence score (descending)
        candidates_sorted = sorted(candidates, key=lambda c: c["evidence_score"], reverse=True)

        # Greedily assign to non-overlapping fragments
        assigned_to_jersey = []
        for candidate in candidates_sorted:
            fragment_id = candidate["fragment_id"]

            # Skip if fragment already assigned a jersey
            if fragment_id in assigned_fragments:
                continue

            # Check if this fragment overlaps with any already assigned to this jersey
            overlaps = False
            for assigned_candidate in assigned_to_jersey:
                # Check temporal overlap
                if not (candidate["end_frame"] < assigned_candidate["start_frame"] or
                       assigned_candidate["end_frame"] < candidate["start_frame"]):
                    overlaps = True
                    break

            if not overlaps:
                # Assign this jersey to this fragment
                assignments[fragment_id] = {
                    "jersey_number": jersey_num,
                    "confidence": candidate["jersey_stats"]["total_confidence"],
                    "reason": f"assigned (conf={candidate['jersey_stats']['total_confidence']:.2f}, count={candidate['jersey_stats']['count']})",
                }
                assigned_fragments.add(fragment_id)
                assigned_to_jersey.append(candidate)
            else:
                # Fragment lost conflict due to temporal overlap with higher-evidence fragment
                if fragment_id not in assignments:  # Only set if not already assigned
                    assignments[fragment_id] = {
                        "jersey_number": None,
                        "confidence": 0.0,
                        "reason": f"conflict_lost_jersey_{jersey_num} (temporal_overlap_with_higher_evidence)",
                    }

    # Step 5: Fill in null assignments for fragments with no eligible jerseys
    for frag_meta in fragment_metadata:
        fragment_id = frag_meta["fragment_id"]
        team = frag_meta["team"]

        if fragment_id not in assignments:
            if not frag_meta["eligible_jerseys"]:
                assignments[fragment_id] = {
                    "jersey_number": None,
                    "confidence": 0.0,
                    "reason": "no_eligible_jersey (failed_eligibility_checks)",
                }
            else:
                # Had eligible jerseys but lost all conflicts
                assignments[fragment_id] = {
                    "jersey_number": None,
                    "confidence": 0.0,
                    "reason": "all_jerseys_conflict_lost",
                }

    # Step 6: Validate single-owner invariant (defensive check)
    _validate_single_owner_invariant(assignments, fragment_metadata)

    # Return detailed assignments with reasons
    return assignments


def _print_assignment_summary(
    fragments: list[dict[str, Any]],
    assignments: dict[str, dict[str, Any]]
) -> None:
    """Print a summary of jersey assignments with statistics."""
    # Count assignments by status
    assigned_count = 0
    team_a_count = 0
    team_b_count = 0
    conflict_lost_count = 0
    no_eligible_count = 0

    jersey_assignments = {}  # jersey_num -> count

    for fragment in fragments:
        fragment_id = fragment.get("fragment_id")
        team = fragment.get("team", "unknown")
        assignment = assignments.get(fragment_id, {})
        jersey_num = assignment.get("jersey_number")
        reason = assignment.get("reason", "unknown")

        if team == "team_a":
            team_a_count += 1
        elif team == "team_b":
            team_b_count += 1

        if jersey_num is not None:
            assigned_count += 1
            jersey_assignments[jersey_num] = jersey_assignments.get(jersey_num, 0) + 1
        elif "conflict_lost" in reason:
            conflict_lost_count += 1
        elif "no_eligible" in reason:
            no_eligible_count += 1

    print(f"  Jersey Assignment Summary:")
    print(f"    Total fragments: {len(fragments)} (team_a: {team_a_count}, team_b: {team_b_count})")
    print(f"    Assigned: {assigned_count}/{team_a_count} team_a fragments")
    print(f"    Conflict lost: {conflict_lost_count}")
    print(f"    No eligible jersey: {no_eligible_count}")

    if jersey_assignments:
        print(f"    Jersey distribution: {dict(sorted(jersey_assignments.items()))}")

    # Display per-fragment details (only for team_a assigned jerseys)
    print(f"  Assigned Jerseys:")
    for fragment in fragments:
        fragment_id = fragment.get("fragment_id")
        team = fragment.get("team", "unknown")
        assignment = assignments.get(fragment_id, {})
        jersey_num = assignment.get("jersey_number")
        reason = assignment.get("reason", "unknown")

        if team == "team_a" and jersey_num is not None:
            start_frame = fragment.get("start_frame")
            end_frame = fragment.get("end_frame")
            print(f"    {fragment_id}: jersey_{jersey_num} [{start_frame:5d}-{end_frame:5d}] - {reason}")


def _validate_single_owner_invariant(
    assignments: dict[str, dict[str, Any]],
    fragment_metadata: list[dict[str, Any]]
) -> None:
    """
    Validate that the single-owner invariant holds:
    At any frame, a jersey number belongs to at most one fragment.

    Raises:
        AssertionError: If invariant is violated
    """
    # Build jersey ownership timeline
    jersey_timeline = {}  # jersey_num -> list of (start, end, fragment_id)

    for frag_meta in fragment_metadata:
        fragment_id = frag_meta["fragment_id"]
        assignment = assignments.get(fragment_id)
        if not assignment or assignment["jersey_number"] is None:
            continue

        jersey_num = assignment["jersey_number"]
        start_frame = frag_meta["start_frame"]
        end_frame = frag_meta["end_frame"]

        if jersey_num not in jersey_timeline:
            jersey_timeline[jersey_num] = []

        # Check for overlaps with existing assignments
        for existing_start, existing_end, existing_frag in jersey_timeline[jersey_num]:
            # Check temporal overlap
            if not (end_frame < existing_start or existing_end < start_frame):
                raise AssertionError(
                    f"INVARIANT VIOLATION: Jersey {jersey_num} assigned to overlapping fragments "
                    f"{fragment_id} [{start_frame}-{end_frame}] and "
                    f"{existing_frag} [{existing_start}-{existing_end}]"
                )

        jersey_timeline[jersey_num].append((start_frame, end_frame, fragment_id))

    # If we reach here, invariant holds
    pass


def _is_bib_like_crop(crop_bgr: np.ndarray) -> bool:
    """
    Detect if crop is bib-like (high saturation, low hue variance).

    Args:
        crop_bgr: Crop image in BGR format

    Returns:
        True if crop has characteristics of a uniform bib (high saturation, low hue variance)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False

    # Convert to HSV
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Need enough valid pixels to compute stats
    mask_valid = s > 0
    if np.count_nonzero(mask_valid) < 10:
        return False

    # Compute saturation and hue statistics
    mean_sat = np.mean(s[mask_valid])
    hue_variance = np.var(h[mask_valid])

    # Bib-like: high saturation (vivid color) + low hue variance (uniform color)
    # Relaxed for futsal lighting: S>100 (was 120), hue_var<20 (was 15)
    # Green bibs: S>100, hue_var<20
    # Orange bibs: S>100, hue_var<20
    return (mean_sat > 100) and (hue_variance < 20)


def _crop_jersey_region(
    bbox: list[int],
    frame_shape: tuple[int, int, int],
    top_skip: float = 0.25,  # Increased from 0.2 to skip more of head/shoulders
    bottom_cut: float = 0.55,
    width_shrink: float = 0.8,
) -> tuple[int, int, int, int]:
    """Crop upper-torso jersey region from bbox (25%-55% vertical, 80% width)."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    cx = (x1 + x2) / 2.0

    new_w = width * width_shrink
    ny1 = int(max(0, y1 + height * top_skip))
    ny2 = int(min(frame_shape[0], y1 + height * bottom_cut))
    nx1 = int(max(0, cx - new_w / 2.0))
    nx2 = int(min(frame_shape[1], cx + new_w / 2.0))
    return nx1, ny1, nx2, ny2


def _is_valid_bbox(frame_shape: tuple[int, int, int], bbox: BoundingBox) -> bool:
    w = bbox.width
    h = bbox.height
    if w <= 0 or h <= 0:
        return False

    ar = w / h
    min_w, min_h = 20, 35
    max_ar = 3.5
    min_ar = 0.3
    max_h = frame_shape[0] * 0.8

    if w < min_w or h < min_h:
        return False
    if ar > max_ar or ar < min_ar:
        return False
    if h > max_h:
        return False
    return True


def _build_frame_index(tracks: dict[str, Any]) -> dict[int, list[tuple[str, list[int]]]]:
    frame_index: dict[int, list[tuple[str, list[int]]]] = {}
    for track_id, track_data in tracks.items():
        frames = track_data.get("frames", [])
        bboxes = track_data.get("bboxes", [])
        for frame_idx, bbox in zip(frames, bboxes):
            frame_index.setdefault(int(frame_idx), []).append((str(track_id), bbox))
    return frame_index


def _is_isolated_sample(
    bbox: BoundingBox,
    frame_shape: tuple[int, int, int],
    other_bboxes: list[list[int]],
    proximity_thresh: float,
    iou_thresh: float = 0.010,
) -> tuple[bool, str]:
    """
    Check if sample is isolated from other players.

    Args:
        bbox: Bounding box to check
        frame_shape: Frame dimensions (h, w, c)
        other_bboxes: List of other player bboxes [x1, y1, x2, y2]
        proximity_thresh: Minimum distance to other players (pixels)
        iou_thresh: Maximum IoU overlap with other players

    Returns:
        (is_isolated, rejection_reason): (True, "") if isolated, (False, reason) otherwise
    """
    if not other_bboxes:
        return True, ""

    crop_region = _crop_jersey_region([bbox.x1, bbox.y1, bbox.x2, bbox.y2], frame_shape)

    def iou(box_a, box_b):
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])
        inter_w = max(0, x_b - x_a)
        inter_h = max(0, y_b - y_a)
        inter_area = inter_w * inter_h
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area

    bbox_center = bbox.center
    for other in other_bboxes:
        other_box = BoundingBox(x1=other[0], y1=other[1], x2=other[2], y2=other[3])
        dx = bbox_center[0] - other_box.center[0]
        dy = bbox_center[1] - other_box.center[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < proximity_thresh:
            return False, f"proximity_too_close (dist={dist:.1f}px < {proximity_thresh:.1f}px)"

        iou_val = iou(crop_region, other)
        if iou_val > iou_thresh:
            return False, f"iou_overlap (iou={iou_val:.4f} > {iou_thresh:.3f})"

        cx, cy = other_box.center
        if crop_region[0] <= cx <= crop_region[2] and crop_region[1] <= cy <= crop_region[3]:
            return False, "center_containment (other player center inside crop)"

    return True, ""


def _select_sample_indices(total: int, max_samples: int) -> list[int]:
    if total <= 0 or max_samples <= 0:
        return []
    if total <= max_samples:
        return list(range(total))
    return np.linspace(0, total - 1, max_samples, dtype=int).tolist()


def _full_bbox_crop(frame: np.ndarray, bbox: list[int]) -> np.ndarray | None:
    """Crop full bounding box (for fallback only)."""
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(int(x2), frame.shape[1]), min(int(y2), frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None
    crop_rgb = frame[y1:y2, x1:x2]
    if crop_rgb.size == 0:
        return None
    return cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)


def _jersey_region_crop(frame: np.ndarray, bbox: list[int]) -> np.ndarray | None:
    """Crop jersey region (torso only) from bounding box."""
    # Apply jersey region calculation (top 20%-55%, width 80%)
    crop_region = _crop_jersey_region(bbox, frame.shape)
    x1, y1, x2, y2 = crop_region

    # Clamp to frame boundaries
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(int(x2), frame.shape[1]), min(int(y2), frame.shape[0])

    if x2 <= x1 or y2 <= y1:
        return None

    crop_rgb = frame[y1:y2, x1:x2]
    if crop_rgb.size == 0:
        return None

    return cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)




def _export_team_crops(
    run_dir: Path,
    clip_name: str,
    output_dir: Path,
    output_stem: str,
    fragments: list[dict[str, Any]],
    config: dict,
) -> None:
    pass3_cfg = config.get("pass3", {})
    save_team_crops = pass3_cfg.get("save_team_crops", True)
    if not save_team_crops:
        return

    pass1_file = run_dir / "pass1_raw" / f"{Path(clip_name).stem}.json"
    if not pass1_file.exists():
        print(f"  [CROPS] Pass 1 file not found: {pass1_file}")
        return

    input_dir = Path(pass3_cfg.get("crops_input_dir", "videos/input"))
    video_path = input_dir / clip_name
    if not video_path.exists():
        print(f"  [CROPS] Video not found: {video_path}")
        return

    crops_root = output_dir / f"{output_stem}_team_crops"
    team_a_dir = crops_root / "teamA"
    team_b_dir = crops_root / "teamB"
    team_a_dir.mkdir(parents=True, exist_ok=True)
    team_b_dir.mkdir(parents=True, exist_ok=True)

    # Crop export configuration (all from pass3)
    crops_per_fragment = int(pass3_cfg.get("crops_per_fragment", 6))
    min_per_team = int(pass3_cfg.get("min_crops_per_team", 15))
    crop_mode = pass3_cfg.get("crop_mode", "jersey")
    proximity_thresh = float(pass3_cfg.get("proximity_threshold_px", 24.0))

    # Quality thresholds (futsal-tuned)
    quality_threshold_bib = float(pass3_cfg.get("quality_threshold_bib", 0.30))
    quality_threshold_standard = float(pass3_cfg.get("quality_threshold_standard", 0.45))
    quality_threshold_high_confidence = float(pass3_cfg.get("quality_threshold_high_confidence", 0.70))

    # Isolation checks
    iou_threshold = float(pass3_cfg.get("iou_threshold", 0.010))

    # Candidate selection
    candidate_multiplier = int(pass3_cfg.get("candidate_multiplier", 3))
    fallback_promotion_threshold = int(pass3_cfg.get("fallback_promotion_threshold", 3))

    # Team clustering configuration (for histogram extraction only)
    team_cfg = config.get("team_clustering", {})
    bins = int(team_cfg.get("bins", 32))
    clustering = TeamClustering(bins=bins)

    with open(pass1_file, "r", encoding="utf-8") as f:
        pass1_data = json.load(f)

    tracks = pass1_data.get("tracks", {})
    frame_index = _build_frame_index(tracks)
    reader = VideoReader(video_path)
    frame_cache: dict[int, np.ndarray] = {}

    team_fragments = {
        "team_a": [f for f in fragments if f.get("team") == "team_a"],
        "team_b": [f for f in fragments if f.get("team") == "team_b"],
    }
    per_team_target = {}
    for team_key, team_list in team_fragments.items():
        if not team_list:
            continue
        per_fragment_target = int(np.ceil(min_per_team / len(team_list)))
        per_team_target[team_key] = max(crops_per_fragment, per_fragment_target)

    for fragment in fragments:
        team = fragment.get("team")
        if team not in ("team_a", "team_b"):
            continue

        track_id = str(fragment.get("original_track_id"))
        track_data = tracks.get(track_id)
        if not track_data:
            continue

        frames = track_data.get("frames", [])
        bboxes = track_data.get("bboxes", [])
        hsv_histograms = track_data.get("hsv_histograms", [])
        if not frames or not bboxes or len(frames) != len(bboxes):
            continue

        start_frame = fragment.get("start_frame", frames[0])
        end_frame = fragment.get("end_frame", frames[-1])

        samples = [
            (f_idx, bbox, hist)
            for f_idx, bbox, hist in zip(frames, bboxes, hsv_histograms or [None] * len(frames))
            if start_frame <= f_idx <= end_frame
        ]
        if not samples:
            continue

        if crop_mode == "histogram_source":
            samples = [s for s in samples if s[2] is not None]
            if not samples:
                continue

        target_per_fragment = per_team_target.get(team, crops_per_fragment)
        sample_indices = _select_sample_indices(
            len(samples),
            max(target_per_fragment * candidate_multiplier, target_per_fragment)
        )
        fragment_id = fragment.get("fragment_id", "frag_unknown")
        out_dir = team_a_dir if team == "team_a" else team_b_dir

        candidates = []
        fallback_samples = []
        rejection_stats = {}  # Track rejection reasons for logging

        for sample_pos in sample_indices:
            frame_idx, bbox, hist = samples[sample_pos]
            if frame_idx in frame_cache:
                frame = frame_cache[frame_idx]
            else:
                try:
                    frame = reader.get_frame(int(frame_idx))
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"  [CROPS] Failed frame {frame_idx}: {exc}")
                    continue
                frame_cache[frame_idx] = frame

            bbox_obj = BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])

            # Always use jersey region crop (torso only) to avoid full-body crops
            jersey_crop = _jersey_region_crop(frame, bbox)

            # Reject samples where jersey region crop fails (prevents full-body crops)
            if jersey_crop is None or jersey_crop.size == 0:
                rejection_stats["jersey_crop_failed"] = rejection_stats.get("jersey_crop_failed", 0) + 1
                continue

            # Keep full bbox as emergency fallback only
            fallback_crop = _full_bbox_crop(frame, bbox)
            fallback_samples.append((frame_idx, jersey_crop, sample_pos))

            if crop_mode == "histogram_source":
                quality = 1.0
                candidates.append((quality, frame_idx, jersey_crop, sample_pos, "histogram_source"))
                continue

            # Bbox validation
            if not _is_valid_bbox(frame.shape, bbox_obj):
                rejection_stats["bbox_invalid"] = rejection_stats.get("bbox_invalid", 0) + 1
                continue

            # Isolation check (with rejection reason)
            others = [b for tid, b in frame_index.get(int(frame_idx), []) if tid != track_id]
            is_isolated, rejection_reason = _is_isolated_sample(bbox_obj, frame.shape, others, proximity_thresh, iou_threshold)
            if not is_isolated:
                rejection_key = rejection_reason.split(" ")[0]  # Extract first word (proximity_too_close, iou_overlap, etc.)
                rejection_stats[rejection_key] = rejection_stats.get(rejection_key, 0) + 1
                continue

            # Extract features (jersey mode)
            if crop_mode == "jersey":
                hist, _, crop_bgr, quality = clustering.extract_features(frame, bbox_obj)

                # HIGH-CONFIDENCE IMMEDIATE ACCEPT: Quality > threshold → skip other checks
                if quality > quality_threshold_high_confidence:
                    candidates.append((quality, frame_idx, crop_bgr, sample_pos, "high_quality_accept"))
                    continue

                # Adaptive quality threshold based on bib-like characteristics
                # Use raw jersey_crop (torso only) for bib detection, not masked crop_bgr
                is_bib = _is_bib_like_crop(jersey_crop)
                adaptive_threshold = quality_threshold_bib if is_bib else quality_threshold_standard

                if quality < adaptive_threshold:
                    rejection_reason = f"quality_too_low_q{quality:.2f}_t{adaptive_threshold:.2f}"
                    rejection_stats[rejection_reason] = rejection_stats.get(rejection_reason, 0) + 1
                    continue

                if crop_bgr is None or crop_bgr.size == 0:
                    rejection_stats["empty_crop"] = rejection_stats.get("empty_crop", 0) + 1
                    continue

                # Standard accept with adaptive threshold
                accept_reason = "bib_accept" if is_bib else "standard_accept"
                candidates.append((quality, frame_idx, crop_bgr, sample_pos, accept_reason))
            else:
                # Full mode - use jersey region crop (not full bbox)
                quality = 1.0
                candidates.append((quality, frame_idx, jersey_crop, sample_pos, "full_mode"))

        # Sort candidates by quality (descending)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # FALLBACK PROMOTION: If we have few primary candidates but many fallbacks, promote best fallback
        if len(candidates) < target_per_fragment and len(fallback_samples) >= fallback_promotion_threshold:
            fallback_samples.sort(key=lambda x: x[0])  # Sort by frame_idx
            best_fallback = fallback_samples[0]
            frame_idx, crop_bgr, sample_pos = best_fallback
            # Add to candidates with quality 0.6 (between thresholds)
            candidates.append((0.6, frame_idx, crop_bgr, sample_pos, "promoted_fallback"))
            print(f"    [PROMO] {fragment_id}: Promoted fallback from frame {frame_idx}")

        # Save top N candidates
        saved_count = 0
        for quality, frame_idx, crop_bgr, sample_pos, accept_reason in candidates[:target_per_fragment]:
            filename = f"{fragment_id}_f{frame_idx}_tid{track_id}_{sample_pos}.jpg"
            cv2.imwrite(str(out_dir / filename), crop_bgr)
            saved_count += 1

        # Fallback: save one fallback crop if no primary candidates
        if not candidates and fallback_samples:
            fallback_samples.sort(key=lambda x: x[0])
            frame_idx, crop_bgr, sample_pos = fallback_samples[0]
            filename = f"{fragment_id}_f{frame_idx}_tid{track_id}_{sample_pos}_fallback.jpg"
            cv2.imwrite(str(out_dir / filename), crop_bgr)
            saved_count = 1

        # Log rejection statistics
        if rejection_stats:
            rejection_summary = ", ".join([f"{k}:{v}" for k, v in sorted(rejection_stats.items())])
            print(f"    [REJECT] {fragment_id}: {rejection_summary} (saved {saved_count})")

    print(f"  [CROPS] Saved team crops to: {crops_root}")
