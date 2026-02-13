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


def validate_team_size_constraint(fragments: list[dict], max_per_team: int = 6) -> None:
    """
    Enforce hard team size constraint: max 6 concurrent players per team.

    This is a SAFETY NET to catch fragmentation bugs. If violated, it indicates
    that appearance-based splitting failed to trigger correctly (or team clustering
    is wrong). This should NEVER happen in correct implementation.

    Args:
        fragments: List of fragment dicts with team assignments
        max_per_team: Maximum concurrent players per team (default: 6 for futsal)

    Raises:
        No exceptions - violations are logged as warnings for debugging
    """
    team_timelines = {"team_a": [], "team_b": []}

    # Build timeline for each team
    for fragment in fragments:
        team = fragment.get("team")
        if team in team_timelines:
            team_timelines[team].append({
                "fragment_id": fragment["fragment_id"],
                "start": fragment["start_frame"],
                "end": fragment["end_frame"],
            })

    # Check for temporal overlaps exceeding max_per_team
    violations_found = False
    for team, timeline in team_timelines.items():
        if not timeline:
            continue

        # Get all frames covered by this team
        all_frames = set()
        for frag in timeline:
            all_frames.update(range(frag["start"], frag["end"] + 1))

        # For each frame, count concurrent fragments
        max_concurrent = 0
        worst_frame = None

        for frame in sorted(all_frames):
            concurrent = [
                frag for frag in timeline
                if frag["start"] <= frame <= frag["end"]
            ]

            if len(concurrent) > max_concurrent:
                max_concurrent = len(concurrent)
                worst_frame = frame

            if len(concurrent) > max_per_team:
                if not violations_found:
                    print(f"\n[WARNING] TEAM SIZE CONSTRAINT VIOLATION DETECTED!")
                    violations_found = True

                print(f"    {team.upper()}: {len(concurrent)} concurrent players at frame {frame} (max={max_per_team})")
                print(f"       Fragments: {[f['fragment_id'] for f in concurrent]}")

                # Mark fragments for review
                for frag_info in concurrent:
                    for fragment in fragments:
                        if fragment["fragment_id"] == frag_info["fragment_id"]:
                            fragment["team_constraint_violation"] = True

        if max_concurrent > 0:
            print(f"  [{team.upper()}] Max concurrent players: {max_concurrent} at frame {worst_frame}")

    if violations_found:
        print(f"\n[WARNING] This indicates an appearance-based splitting bug or team clustering error.")
        print(f"    Review fragments marked with 'team_constraint_violation': true")
    else:
        print(f"  [OK] Team size constraint satisfied (max {max_per_team} per team)")


def enforce_team_size_constraint(fragments: list[dict], max_per_team: int = 6) -> int:
    """
    Enforce hard team size cap by demoting weakest overflow fragments to unknown.

    Rules:
    - Never auto-flip team_a <-> team_b.
    - Demote entire fragment to unknown when it contributes to over-cap states.
    - Deterministic weakest-first ordering.

    Returns:
        Number of fragments demoted to unknown.
    """
    if max_per_team <= 0:
        return 0

    recent_entry_window_frames = 180

    fragment_by_id = {
        f.get("fragment_id"): f
        for f in fragments
        if f.get("fragment_id")
    }

    def _weakness_key(fragment_obj: dict) -> tuple:
        label_source = str(fragment_obj.get("label_source") or "")
        source_penalty = 1 if "kmeans" in label_source else 0
        start_frame = int(fragment_obj.get("start_frame", 0) or 0)
        identity_jump = bool(fragment_obj.get("identity_jump", False))
        visibility_quality = float(fragment_obj.get("visibility_quality", 0.0) or 0.0)
        mean_confidence = float(fragment_obj.get("mean_confidence", 0.0) or 0.0)
        frame_count = int(fragment_obj.get("frame_count", 0) or 0)
        frag_id = str(fragment_obj.get("fragment_id") or "")
        # Smaller tuple => weaker fragment, demoted first.
        return (
            source_penalty,
            -start_frame,
            frame_count,
            0 if identity_jump else 1,
            visibility_quality,
            mean_confidence,
            frag_id,
        )

    demoted_ids: set[str] = set()
    changed = True
    safety_iter = 0

    while changed and safety_iter < 5000:
        safety_iter += 1
        changed = False

        for team in ("team_a", "team_b"):
            team_fragments = [
                f for f in fragments
                if f.get("team") == team and f.get("fragment_id") not in demoted_ids
            ]
            if not team_fragments:
                continue

            all_frames: set[int] = set()
            for fragment in team_fragments:
                start = fragment.get("start_frame")
                end = fragment.get("end_frame")
                if start is None or end is None:
                    continue
                all_frames.update(range(int(start), int(end) + 1))

            for frame in sorted(all_frames):
                concurrent = [
                    f for f in team_fragments
                    if f.get("start_frame") is not None
                    and f.get("end_frame") is not None
                    and int(f["start_frame"]) <= frame <= int(f["end_frame"])
                    and f.get("fragment_id") not in demoted_ids
                ]

                over = len(concurrent) - max_per_team
                if over <= 0:
                    continue

                recent_or_jump = [
                    f for f in concurrent
                    if (
                        bool(f.get("identity_jump", False))
                        or (
                            f.get("start_frame") is not None
                            and int(f.get("start_frame")) >= int(frame - recent_entry_window_frames)
                        )
                    )
                ]

                candidate_pool = recent_or_jump if len(recent_or_jump) >= over else concurrent
                candidates = sorted(candidate_pool, key=_weakness_key)
                to_demote = candidates[:over]

                for fragment in to_demote:
                    frag_id = fragment.get("fragment_id")
                    if not frag_id or frag_id in demoted_ids:
                        continue
                    demoted_ids.add(frag_id)
                    fragment["team"] = "unknown"
                    fragment["team_confidence"] = "team_cap_enforced"
                    fragment["label_source"] = "team_cap_enforced"
                    fragment["team_constraint_violation"] = True
                    fragment["violation_reason"] = f"over_capacity_{team}_frame_{frame}"
                    fragment["over_capacity_frame_count"] = int(fragment.get("over_capacity_frame_count", 0) or 0) + 1
                    changed = True

                if changed:
                    break

            if changed:
                break

    if demoted_ids:
        print(f"  [TEAM_CAP_ENFORCE] Demoted {len(demoted_ids)} fragment(s) to unknown to satisfy cap={max_per_team}")

    return len(demoted_ids)


def recover_unknown_team_assignments(
    fragments: list[dict[str, Any]],
    team_cap: int = 6,
    min_margin: float = 0.12,
    min_ratio: float = 1.04,
    max_anchor_gap_frames: int = 5,
) -> int:
    """
    Reassign unknown fragments to a team when both conditions hold:
    1) Evidence supports the team (color centroid margin and/or same-track anchors).
    2) Team has spare capacity for every frame in fragment lifespan.
    """
    if team_cap <= 0:
        return 0

    teams = ("team_a", "team_b")

    fragments_by_track: dict[str, list[dict[str, Any]]] = {}
    for fragment in fragments:
        track_id = fragment.get("original_track_id")
        if track_id is None:
            continue
        fragments_by_track.setdefault(str(track_id), []).append(fragment)
    for track_frags in fragments_by_track.values():
        track_frags.sort(key=lambda f: f.get("start_frame", -1))

    team_hist: dict[str, list[np.ndarray]] = {"team_a": [], "team_b": []}
    for fragment in fragments:
        team = fragment.get("team")
        hist = fragment.get("mean_hsv_histogram")
        if team in team_hist and hist and len(hist) == 96:
            team_hist[team].append(np.array(hist, dtype=np.float32))

    if not team_hist["team_a"] or not team_hist["team_b"]:
        return 0

    centroid_a = np.mean(np.stack(team_hist["team_a"], axis=0), axis=0)
    centroid_b = np.mean(np.stack(team_hist["team_b"], axis=0), axis=0)

    occupancy: dict[str, dict[int, int]] = {"team_a": {}, "team_b": {}}
    for fragment in fragments:
        team = fragment.get("team")
        if team not in teams:
            continue
        start = fragment.get("start_frame")
        end = fragment.get("end_frame")
        if start is None or end is None:
            continue
        for frame in range(int(start), int(end) + 1):
            occupancy[team][frame] = occupancy[team].get(frame, 0) + 1

    def _fit_capacity(fragment_obj: dict[str, Any], team: str) -> bool:
        start = fragment_obj.get("start_frame")
        end = fragment_obj.get("end_frame")
        if start is None or end is None:
            return False
        for frame in range(int(start), int(end) + 1):
            if occupancy[team].get(frame, 0) >= team_cap:
                return False
        return True

    def _track_anchor_team(fragment_obj: dict[str, Any]) -> str | None:
        track_id = fragment_obj.get("original_track_id")
        if track_id is None:
            return None
        track_frags = fragments_by_track.get(str(track_id), [])
        if not track_frags:
            return None

        frag_id = fragment_obj.get("fragment_id")
        idx = -1
        for i, f in enumerate(track_frags):
            if f.get("fragment_id") == frag_id:
                idx = i
                break
        if idx < 0:
            return None

        current_start = fragment_obj.get("start_frame")
        current_end = fragment_obj.get("end_frame")
        if current_start is None or current_end is None:
            return None

        prev_team = None
        for prev in reversed(track_frags[:idx]):
            prev_team_val = prev.get("team")
            prev_end = prev.get("end_frame")
            if prev_end is None:
                continue
            if int(current_start) - int(prev_end) - 1 > max_anchor_gap_frames:
                break
            if prev_team_val in teams:
                prev_team = prev_team_val
                break

        next_team = None
        for nxt in track_frags[idx + 1:]:
            next_team_val = nxt.get("team")
            next_start = nxt.get("start_frame")
            if next_start is None:
                continue
            if int(next_start) - int(current_end) - 1 > max_anchor_gap_frames:
                break
            if next_team_val in teams:
                next_team = next_team_val
                break

        if prev_team and next_team and prev_team == next_team:
            return prev_team
        return prev_team or next_team

    candidates: list[tuple[float, dict[str, Any], str, str]] = []
    for fragment in fragments:
        if fragment.get("team") != "unknown":
            continue

        hist = fragment.get("mean_hsv_histogram")
        if not hist or len(hist) != 96:
            continue

        vector = np.array(hist, dtype=np.float32)
        dist_a = float(np.linalg.norm(vector - centroid_a))
        dist_b = float(np.linalg.norm(vector - centroid_b))

        if dist_a <= dist_b:
            preferred_team = "team_a"
            preferred_dist = dist_a
            other_dist = dist_b
        else:
            preferred_team = "team_b"
            preferred_dist = dist_b
            other_dist = dist_a

        margin = other_dist - preferred_dist
        ratio = other_dist / max(preferred_dist, 1e-6)
        anchor_team = _track_anchor_team(fragment)

        evidence_ok = bool(anchor_team in teams) or (margin >= min_margin and ratio >= min_ratio)
        if not evidence_ok:
            continue

        candidate_team = anchor_team if anchor_team in teams else preferred_team
        if not _fit_capacity(fragment, candidate_team):
            continue

        score = margin + (0.35 if anchor_team == candidate_team else 0.0)
        reason = "unknown_recovered_anchor" if anchor_team == candidate_team else "unknown_recovered_color"
        candidates.append((score, fragment, candidate_team, reason))

    candidates.sort(key=lambda item: item[0], reverse=True)

    reassigned = 0
    for _, fragment, team, reason in candidates:
        if fragment.get("team") != "unknown":
            continue
        if not _fit_capacity(fragment, team):
            continue

        fragment["team"] = team
        fragment["team_confidence"] = "unknown_recovered"
        fragment["label_source"] = reason
        fragment["unknown_recovery_reason"] = reason

        start = fragment.get("start_frame")
        end = fragment.get("end_frame")
        if start is not None and end is not None:
            for frame in range(int(start), int(end) + 1):
                occupancy[team][frame] = occupancy[team].get(frame, 0) + 1

        reassigned += 1

    if reassigned > 0:
        print(f"  [UNKNOWN_RECOVERY] Reassigned {reassigned} unknown fragment(s) to teams")

    return reassigned


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


def _calculate_team_variances(fragments: list[dict]) -> dict[str, float]:
    """
    Calculate intra-team appearance variance for each team independently.
    
    Metric: Mean Euclidean distance to team centroid across all histogram dimensions.
    
    Args:
        fragments: List of fragment dicts with team and mean_hsv_histogram
        
    Returns:
        {team_id: variance_float}
    
    Invariant:
        - Reads fragment.team (READ-ONLY)
        - Does not modify any fragments
        - Returns numeric values only
    """
    team_variances = {}
    
    for team_id in ["team_a", "team_b"]:
        # Collect histograms for this team
        team_histograms = []
        for fragment in fragments:
            if fragment.get("team") == team_id:
                hist = fragment.get("mean_hsv_histogram")
                if hist and len(hist) == 96:  # Valid 96-dim HSV histogram
                    team_histograms.append(np.array(hist, dtype=np.float32))
        
        if len(team_histograms) < 2:
            # Not enough samples - assign neutral variance
            team_variances[team_id] = 0.0
            continue
        
        # Compute team centroid
        X = np.array(team_histograms)
        centroid = np.mean(X, axis=0)
        
        # Compute mean distance to centroid
        distances = np.linalg.norm(X - centroid, axis=1)
        variance = float(np.mean(distances))
        
        team_variances[team_id] = variance
    
    return team_variances


def _detect_multi_appearance_teams(
    team_variances: dict[str, float],
    ratio_threshold: float = 1.5
) -> set[str]:
    """
    Detect which teams have high appearance variance (multi-appearance).
    
    Decision rule (relative variance only):
        If var(team_i) > ratio_threshold * min(other_team_vars)
        → mark team_i as multi-appearance
    
    Args:
        team_variances: {team_id: float}
        ratio_threshold: Ratio multiplier (default 1.5)
        
    Returns:
        Set of team_ids marked as multi-appearance
        
    Invariant:
        - Read-only on input
        - Returns team identifiers only
        - No modifications to fragments
    """
    multi_appearance = set()
    
    variances = {k: v for k, v in team_variances.items() if v > 0.0}
    if len(variances) < 2:
        return multi_appearance
    
    min_var = min(variances.values())
    threshold = ratio_threshold * min_var
    
    for team_id, var in variances.items():
        if var > threshold:
            multi_appearance.add(team_id)
    
    return multi_appearance


def _subcluster_appearance_modes(
    fragments: list[dict],
    team_id: str,
    max_modes: int = 3
) -> None:
    """
    Sub-cluster fragments within one team to discover appearance modes.
    
    METADATA-ONLY: Adds appearance_mode_id and appearance_mode_confidence to each fragment.
    NEVER modifies frag.team or any temporal logic.
    
    Args:
        fragments: Full fragment list (filters by team internally)
        team_id: Team to sub-cluster (e.g., "team_a")
        max_modes: Hard cap on k for MiniBatchKMeans (default 3)
        
    Side effects (metadata only):
        fragment["appearance_mode_id"] = int (if multi-appearance)
        fragment["appearance_mode_confidence"] = float
        
    Invariant:
        - Reads fragment.team (READ-ONLY)
        - NEVER modifies fragment.team
        - NEVER modifies timelines, divergence, or size constraints
        - Output is informational, no feedback loops
    """
    from sklearn.cluster import MiniBatchKMeans
    
    # Collect histograms for this team
    team_fragments = [f for f in fragments if f.get("team") == team_id]
    team_histograms = []
    valid_indices = []
    
    for i, frag in enumerate(team_fragments):
        hist = frag.get("mean_hsv_histogram")
        if hist and len(hist) == 96:
            team_histograms.append(np.array(hist, dtype=np.float32))
            valid_indices.append(i)
    
    if len(team_histograms) < 2:
        # Not enough samples for sub-clustering
        return
    
    # Determine k: min of max_modes and number of fragments
    k = min(max_modes, len(team_histograms))
    if k < 2:
        return
    
    # Run MiniBatchKMeans to find appearance modes
    X = np.array(team_histograms)
    try:
        mbk = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=max(3, len(X) // 2))
        mode_labels = mbk.fit_predict(X)
        centroids = mbk.cluster_centers_
    except Exception as e:
        # If clustering fails, skip metadata (don't crash)
        return
    
    # Assign mode_id and confidence to valid fragments
    for array_idx, frag_idx in enumerate(valid_indices):
        frag = team_fragments[frag_idx]
        mode_id = int(mode_labels[array_idx])
        
        # Compute confidence as 1 / distance to nearest centroid
        hist = X[array_idx]
        dist_to_centroid = np.linalg.norm(hist - centroids[mode_id])
        # Normalize distance to approximate confidence (lower dist = higher conf)
        mode_conf = max(0.0, 1.0 / (1.0 + dist_to_centroid))
        
        frag["appearance_mode_id"] = mode_id
        frag["appearance_mode_confidence"] = float(mode_conf)


def _mark_identity_jumps(fragments: list[dict]) -> dict[str, dict]:
    """
    Mark fragments that were created by identity-jump splits.

    A fragment is considered an identity-jump if its parent fragment ended
    due to an identity jump (appearance drift or jersey conflict splits).

    Returns:
        fragment_lookup: {fragment_id: fragment}
    """
    jump_reasons = {
        "appearance_drift",
        "identity_jump",
        "jersey_inconsistency",
        "jersey_temporal_exclusivity",
        "overlap_identity_contradiction",
        "velocity_spike",
    }

    fragment_lookup = {f.get("fragment_id"): f for f in fragments if f.get("fragment_id")}
    print(f"[IDENTITY_JUMP_DEBUG] Total fragments: {len(fragments)}")

    # Build per-track ordering for post-split assignment
    fragments_by_track: dict[str, list[dict]] = {}
    for fragment in fragments:
        track_id = fragment.get("original_track_id")
        if not track_id:
            continue
        fragments_by_track.setdefault(track_id, []).append(fragment)

    for track_id, track_frags in fragments_by_track.items():
        track_frags.sort(key=lambda f: f.get("start_frame", -1))

    def _is_hard_split(split_reason: Any) -> bool:
        if not isinstance(split_reason, dict):
            return False
        reason_val = split_reason.get("reason")
        severity = split_reason.get("severity")
        return reason_val in jump_reasons and severity == "hard"

    # Mark identity-jump fragments (post-split fragments, not pre-split)
    identity_jump_ids: set[str] = set()
    identity_jump_sources: dict[str, dict[str, Any]] = {}

    for track_id, track_frags in fragments_by_track.items():
        for idx, fragment in enumerate(track_frags):
            split_reason = fragment.get("split_reason")
            if not _is_hard_split(split_reason):
                continue

            split_frame = split_reason.get("frame_idx") if isinstance(split_reason, dict) else None
            reason_val = split_reason.get("reason") if isinstance(split_reason, dict) else None
            next_fragment = None
            for candidate in track_frags[idx + 1:]:
                candidate_start = candidate.get("start_frame")
                if candidate_start is None:
                    continue
                if split_frame is None or candidate_start >= split_frame:
                    next_fragment = candidate
                    break

            if next_fragment:
                next_id = next_fragment.get("fragment_id")
                if next_id:
                    identity_jump_ids.add(next_id)
                    identity_jump_sources[next_id] = {
                        "reason": reason_val,
                        "split_frame": split_frame,
                        "source_fragment_id": fragment.get("fragment_id"),
                    }
                    print(
                        f"[IDENTITY_JUMP_MARK] Fragment {next_id}: post-split from {reason_val} at frame {split_frame}"
                    )

    for fragment in fragments:
        fragment_id = fragment.get("fragment_id")
        if not fragment_id:
            continue

        parent_id = fragment.get("parent_fragment_id")
        if not parent_id:
            base_id = fragment_id
            while base_id.endswith("_split"):
                base_id = base_id[: -len("_split")]
            if base_id != fragment_id and base_id in fragment_lookup:
                parent_id = base_id

        fragment["parent_fragment_id"] = parent_id

        identity_jump = fragment_id in identity_jump_ids
        source_info = identity_jump_sources.get(fragment_id)

        # Also check if parent was created by a split (has parent split_reason)
        if not identity_jump and parent_id:
            parent = fragment_lookup.get(parent_id)
            if parent:
                parent_reason = parent.get("split_reason")
                parent_reason_val = parent_reason.get("reason") if isinstance(parent_reason, dict) else None
                if _is_hard_split(parent_reason):
                    identity_jump = True
                    source_info = {
                        "reason": parent_reason_val,
                        "split_frame": parent_reason.get("frame_idx") if isinstance(parent_reason, dict) else None,
                        "source_fragment_id": parent_id,
                    }
                    print(f"[IDENTITY_JUMP_MARK] Fragment {fragment_id}: parent={parent_id}, parent reason={parent_reason_val}")
            else:
                # Parent not found - but check if this fragment itself is a _split
                # This happens when parent got dropped due to being too short
                if "_split" in fragment_id and parent_id and "frag_" in parent_id:
                    # Only mark if this fragment itself has a hard split reason
                    if _is_hard_split(fragment.get("split_reason")):
                        identity_jump = True
                        self_reason = fragment.get("split_reason")
                        self_reason_val = self_reason.get("reason") if isinstance(self_reason, dict) else None
                        source_info = {
                            "reason": self_reason_val,
                            "split_frame": self_reason.get("frame_idx") if isinstance(self_reason, dict) else None,
                            "source_fragment_id": parent_id,
                        }
                        print(
                            f"[IDENTITY_JUMP_MARK] Fragment {fragment_id}: orphaned after-split, parent={parent_id} not found, marking identity_jump"
                        )

        fragment["identity_jump"] = identity_jump
        fragment["identity_jump_source_reason"] = source_info.get("reason") if source_info else None
        fragment["identity_jump_source_frame"] = source_info.get("split_frame") if source_info else None
        fragment["identity_jump_source_fragment"] = source_info.get("source_fragment_id") if source_info else None

    return fragment_lookup


def _apply_identity_jump_locks(
    fragments: list[dict],
    assignments: dict[str, dict[str, Any]],
    fragment_lookup: dict[str, dict]
) -> dict[str, dict[str, Any]]:
    """
    Enforce identity-jump constraints: jersey can only come from parent fragment.

    If parent jersey is unavailable or conflicts, mark as unknown.
    """
    # Build fragment metadata lookup (fragment_id -> {start_frame, end_frame, team})
    fragment_metadata = {}
    for fragment in fragments:
        fragment_id = fragment.get("fragment_id")
        fragment_metadata[fragment_id] = {
            "start_frame": fragment.get("start_frame"),
            "end_frame": fragment.get("end_frame"),
            "team": fragment.get("team", "unknown"),
        }

    def _has_temporal_conflict(jersey_number: int, target_fragment_id: str) -> bool:
        target_meta = fragment_metadata.get(target_fragment_id)
        if not target_meta:
            return False

        target_start = target_meta["start_frame"]
        target_end = target_meta["end_frame"]
        target_team = target_meta.get("team", "unknown")

        for frag_id, assignment in assignments.items():
            if frag_id == target_fragment_id:
                continue
            assigned_jersey = assignment.get("jersey_number")
            if assigned_jersey != jersey_number:
                continue

            other_meta = fragment_metadata.get(frag_id)
            if not other_meta:
                continue

            other_team = other_meta.get("team", "unknown")
            if (
                target_team in ("team_a", "team_b")
                and other_team in ("team_a", "team_b")
                and target_team != other_team
            ):
                continue

            other_start = other_meta["start_frame"]
            other_end = other_meta["end_frame"]
            if target_start <= other_end and target_end >= other_start:
                return True

        return False

    for fragment in fragments:
        if not fragment.get("identity_jump"):
            continue

        frag_id = fragment.get("fragment_id")
        parent_id = fragment.get("parent_fragment_id")
        parent_assignment = assignments.get(parent_id) if parent_id else None

        if parent_assignment and parent_assignment.get("jersey_number") is not None:
            parent_jersey = parent_assignment.get("jersey_number")
            if not _has_temporal_conflict(parent_jersey, frag_id):
                assignments[frag_id] = {
                    "jersey_number": parent_jersey,
                    "confidence": parent_assignment.get("confidence", 0.0),
                    "reason": f"identity_jump_parent_lock ({parent_id})",
                }
                continue

        assignments[frag_id] = {
            "jersey_number": None,
            "confidence": 0.0,
            "reason": "identity_jump_unknown",
        }

    return assignments


def _repair_short_transition_bridges(
    fragments: list[dict],
    assignments: dict[str, dict[str, Any]],
    max_bridge_frames: int = 120,
) -> None:
    """
    Forward-correct short transition fragments from the immediate successor in the same track.

    Purpose:
    - Reduce delayed team correction after overlap/crossing splits.
    - Prevent brief jersey disappearance when the next stable fragment has strong identity.

    Rules:
    - Candidate must be identity_jump or continuity_locked.
    - Candidate duration <= max_bridge_frames.
    - Next fragment must be contiguous and have strong team source (kmeans*).
    - Team is forwarded when candidate team differs from next team.
    - Jersey is forwarded only if next has jersey and no temporal conflict.
    """
    fragment_lookup = {
        f.get("fragment_id"): f for f in fragments if f.get("fragment_id")
    }

    fragments_by_track: dict[str, list[dict]] = {}
    for fragment in fragments:
        track_id = fragment.get("original_track_id")
        if track_id is None:
            continue
        fragments_by_track.setdefault(str(track_id), []).append(fragment)

    for track_frags in fragments_by_track.values():
        track_frags.sort(key=lambda f: f.get("start_frame", -1))

    def _has_jersey_conflict(target_fragment_id: str, jersey_number: int | None) -> bool:
        if jersey_number is None:
            return False

        target_fragment = fragment_lookup.get(target_fragment_id)
        if not target_fragment:
            return False

        target_start = target_fragment.get("start_frame")
        target_end = target_fragment.get("end_frame")
        target_team = target_fragment.get("team", "unknown")
        if target_start is None or target_end is None:
            return False

        for other_fragment in fragments:
            other_id = other_fragment.get("fragment_id")
            if not other_id or other_id == target_fragment_id:
                continue

            other_assignment = assignments.get(other_id, {})
            if other_assignment.get("jersey_number") != jersey_number:
                continue

            other_team = other_fragment.get("team", "unknown")
            if (
                target_team in ("team_a", "team_b")
                and other_team in ("team_a", "team_b")
                and target_team != other_team
            ):
                continue

            other_start = other_fragment.get("start_frame")
            other_end = other_fragment.get("end_frame")
            if other_start is None or other_end is None:
                continue

            if target_start <= other_end and target_end >= other_start:
                return True

        return False

    for track_id, track_frags in fragments_by_track.items():
        for idx in range(len(track_frags) - 1):
            current = track_frags[idx]
            nxt = track_frags[idx + 1]

            current_id = current.get("fragment_id")
            next_id = nxt.get("fragment_id")
            if not current_id or not next_id:
                continue

            current_start = current.get("start_frame")
            current_end = current.get("end_frame")
            next_start = nxt.get("start_frame")
            if current_start is None or current_end is None or next_start is None:
                continue

            duration = int(current_end - current_start + 1)
            if duration > max_bridge_frames:
                continue

            if not (current.get("identity_jump") or current.get("continuity_locked")):
                continue

            if next_start > current_end + 2:
                continue

            next_source = str(nxt.get("label_source") or "")
            if "kmeans" not in next_source:
                continue

            current_team = current.get("team")
            next_team = nxt.get("team")

            # Guard: for continuity-locked fragments, do not override with next-team bridge
            # if immediate predecessor provides a stronger same-track anchor (especially jersey anchor).
            if idx > 0 and current.get("continuity_locked"):
                prev = track_frags[idx - 1]
                prev_id = prev.get("fragment_id")
                prev_team = prev.get("team")
                prev_end = prev.get("end_frame")
                current_start_for_gap = current.get("start_frame")
                prev_assignment = assignments.get(prev_id, {}) if prev_id else {}
                prev_jersey = prev_assignment.get("jersey_number")
                near_prev = (
                    prev_end is not None
                    and current_start_for_gap is not None
                    and current_start_for_gap - prev_end <= 2
                )
                predecessor_is_anchor = (
                    near_prev
                    and prev_jersey is not None
                    and prev_team in ("team_a", "team_b")
                    and next_team in ("team_a", "team_b")
                    and prev_team != next_team
                )
                if predecessor_is_anchor:
                    print(
                        f"[BRIDGE_REPAIR] SKIP team flip for {current_id}: "
                        f"preserving predecessor anchor {prev_id} ({prev_team}, #{prev_jersey})"
                    )
                    next_team = current_team

            if next_team in ("team_a", "team_b") and current_team != next_team:
                current["team"] = next_team
                current["team_locked"] = next_team
                current["team_confidence"] = "bridge_forward"
                current["label_source"] = f"bridge_from_{next_id}"
                print(
                    f"[BRIDGE_REPAIR] Team {current_id}: {current_team} -> {next_team} "
                    f"(next={next_id}, duration={duration})"
                )

            current_assignment = assignments.get(current_id, {})
            next_assignment = assignments.get(next_id, {})
            current_jersey = current_assignment.get("jersey_number")
            next_jersey = next_assignment.get("jersey_number")

            if current_jersey is None and next_jersey is not None:
                if not _has_jersey_conflict(current_id, next_jersey):
                    assignments[current_id] = {
                        "jersey_number": next_jersey,
                        "confidence": next_assignment.get("confidence", 0.0),
                        "reason": f"bridge_forward_from_{next_id}",
                    }
                    print(
                        f"[BRIDGE_REPAIR] Jersey {current_id}: None -> #{next_jersey} "
                        f"(next={next_id}, duration={duration})"
                    )

    # Temporal sandwich smoothing: prev and next agree, short middle KMeans segment disagrees.
    for track_id, track_frags in fragments_by_track.items():
        for idx in range(1, len(track_frags) - 1):
            prev_frag = track_frags[idx - 1]
            mid_frag = track_frags[idx]
            next_frag = track_frags[idx + 1]

            mid_id = mid_frag.get("fragment_id")
            if not mid_id:
                continue

            mid_start = mid_frag.get("start_frame")
            mid_end = mid_frag.get("end_frame")
            if mid_start is None or mid_end is None:
                continue

            mid_duration = int(mid_end - mid_start + 1)
            if mid_duration > max_bridge_frames:
                continue

            prev_team = prev_frag.get("team")
            mid_team = mid_frag.get("team")
            next_team = next_frag.get("team")
            if prev_team not in ("team_a", "team_b") or next_team not in ("team_a", "team_b"):
                continue
            if prev_team != next_team:
                continue
            if mid_team == prev_team:
                continue

            mid_source = str(mid_frag.get("label_source") or "")
            if "kmeans" not in mid_source:
                continue

            prev_end = prev_frag.get("end_frame")
            next_start = next_frag.get("start_frame")
            if prev_end is None or next_start is None:
                continue
            if mid_start > prev_end + 3 or next_start > mid_end + 3:
                continue

            old_team = mid_team
            mid_frag["team"] = prev_team
            mid_frag["team_locked"] = prev_team
            mid_frag["team_confidence"] = "bridge_sandwich"
            mid_frag["label_source"] = f"sandwich_{prev_frag.get('fragment_id')}_{next_frag.get('fragment_id')}"
            print(
                f"[BRIDGE_REPAIR] Sandwich team {mid_id}: {old_team} -> {prev_team} "
                f"(prev={prev_frag.get('fragment_id')}, next={next_frag.get('fragment_id')}, duration={mid_duration})"
            )


def _suppress_unanchored_continuity_jerseys(
    fragments: list[dict],
    assignments: dict[str, dict[str, Any]],
) -> None:
    """
    Remove jerseys assigned to continuity-locked fragments when no same-team anchor exists.

    This prevents short-lived jersey theft after crossings where a continuity fragment
    gets weak jersey evidence but the track lineage does not carry that jersey.
    """
    fragments_by_track: dict[str, list[dict]] = {}
    for fragment in fragments:
        track_id = fragment.get("original_track_id")
        if track_id is None:
            continue
        fragments_by_track.setdefault(str(track_id), []).append(fragment)

    for track_frags in fragments_by_track.values():
        track_frags.sort(key=lambda f: f.get("start_frame", -1))

    for track_id, track_frags in fragments_by_track.items():
        for idx, fragment in enumerate(track_frags):
            if not fragment.get("continuity_locked"):
                continue

            frag_id = fragment.get("fragment_id")
            if not frag_id:
                continue

            assignment = assignments.get(frag_id, {})
            jersey_num = assignment.get("jersey_number")
            if jersey_num is None:
                continue

            reason = str(assignment.get("reason", ""))
            if "assigned (" not in reason and "inherited_" not in reason:
                continue

            current_team = fragment.get("team")
            has_anchor = False

            for prev in reversed(track_frags[:idx]):
                prev_team = prev.get("team")
                if (
                    current_team in ("team_a", "team_b")
                    and prev_team in ("team_a", "team_b")
                    and current_team != prev_team
                ):
                    continue

                prev_id = prev.get("fragment_id")
                if not prev_id:
                    continue

                prev_assignment = assignments.get(prev_id, {})
                if prev_assignment.get("jersey_number") == jersey_num:
                    has_anchor = True
                    break

                if prev_assignment.get("jersey_number") is not None:
                    break

            if has_anchor:
                continue

            assignments[frag_id] = {
                "jersey_number": None,
                "confidence": 0.0,
                "reason": "continuity_unanchored_jersey_suppressed",
            }
            print(
                f"[JERSEY_SUPPRESS] {frag_id}: removed #{jersey_num} "
                f"(continuity without same-team anchor)"
            )


def _stabilize_short_gap_jerseys(
    fragments: list[dict],
    assignments: dict[str, dict[str, Any]],
    max_gap_fragment_frames: int = 45,
    max_anchor_gap_frames: int = 20,
) -> None:
    """
    Backfill short identity-jump/continuity gaps with same-track anchored jersey.

    If a short gap fragment sits between same-team anchored jersey ownership on the
    same track, keep jersey continuity to avoid brief disappearances.
    """
    fragments_by_track: dict[str, list[dict]] = {}
    fragment_lookup: dict[str, dict] = {}
    for fragment in fragments:
        fragment_id = fragment.get("fragment_id")
        if fragment_id:
            fragment_lookup[fragment_id] = fragment
        track_id = fragment.get("original_track_id")
        if track_id is None:
            continue
        fragments_by_track.setdefault(str(track_id), []).append(fragment)

    for track_frags in fragments_by_track.values():
        track_frags.sort(key=lambda f: f.get("start_frame", -1))

    def _duration(fragment_obj: dict) -> int:
        start = fragment_obj.get("start_frame")
        end = fragment_obj.get("end_frame")
        if start is None or end is None:
            return 0
        return int(end - start + 1)

    def _same_team(team_a: Any, team_b: Any) -> bool:
        if team_a in ("team_a", "team_b") and team_b in ("team_a", "team_b"):
            return team_a == team_b
        return True

    changed = True
    while changed:
        changed = False
        for track_id, track_frags in fragments_by_track.items():
            for idx, fragment in enumerate(track_frags):
                if not (fragment.get("identity_jump") or fragment.get("continuity_locked")):
                    continue

                if _duration(fragment) > max_gap_fragment_frames:
                    continue

                frag_id = fragment.get("fragment_id")
                if not frag_id:
                    continue

                curr_assignment = assignments.get(frag_id, {})
                if curr_assignment.get("jersey_number") is not None:
                    continue

                curr_team = fragment.get("team")

                prev_jersey = None
                prev_id = None
                for prev in reversed(track_frags[:idx]):
                    if not _same_team(curr_team, prev.get("team")):
                        continue
                    prev_id = prev.get("fragment_id")
                    if not prev_id:
                        continue
                    prev_end = prev.get("end_frame")
                    curr_start = fragment.get("start_frame")
                    if prev_end is None or curr_start is None:
                        continue
                    anchor_gap = int(curr_start - prev_end - 1)
                    if anchor_gap > max_anchor_gap_frames:
                        break
                    prev_assignment = assignments.get(prev_id, {})
                    prev_jersey = prev_assignment.get("jersey_number")
                    if prev_jersey is not None:
                        break

                next_jersey = None
                next_id = None
                for nxt in track_frags[idx + 1:]:
                    if not _same_team(curr_team, nxt.get("team")):
                        continue
                    next_id = nxt.get("fragment_id")
                    if not next_id:
                        continue
                    next_start = nxt.get("start_frame")
                    curr_end = fragment.get("end_frame")
                    if next_start is None or curr_end is None:
                        continue
                    anchor_gap = int(next_start - curr_end - 1)
                    if anchor_gap > max_anchor_gap_frames:
                        break
                    next_assignment = assignments.get(next_id, {})
                    next_jersey = next_assignment.get("jersey_number")
                    if next_jersey is not None:
                        break

                chosen_jersey = None
                source_id = None
                if prev_jersey is not None and next_jersey is not None and prev_jersey == next_jersey:
                    chosen_jersey = prev_jersey
                    source_id = prev_id
                elif prev_jersey is not None:
                    chosen_jersey = prev_jersey
                    source_id = prev_id
                elif next_jersey is not None:
                    chosen_jersey = next_jersey
                    source_id = next_id

                if chosen_jersey is None:
                    continue

                temp_assignment = {
                    "jersey_number": chosen_jersey,
                    "confidence": assignments.get(source_id, {}).get("confidence", 0.0),
                    "reason": f"short_gap_stabilized_from_{source_id}",
                }

                # Team-aware conflict check against existing assignments
                curr_start = fragment.get("start_frame")
                curr_end = fragment.get("end_frame")
                if curr_start is None or curr_end is None:
                    continue

                conflict = False
                for other_id, other_assignment in assignments.items():
                    if other_id == frag_id:
                        continue
                    if other_assignment.get("jersey_number") != chosen_jersey:
                        continue
                    other_fragment = fragment_lookup.get(other_id)
                    if not other_fragment:
                        continue
                    if not _same_team(curr_team, other_fragment.get("team")):
                        continue
                    other_start = other_fragment.get("start_frame")
                    other_end = other_fragment.get("end_frame")
                    if other_start is None or other_end is None:
                        continue
                    if curr_start <= other_end and curr_end >= other_start:
                        conflict = True
                        break

                if conflict:
                    continue

                assignments[frag_id] = temp_assignment
                changed = True
                print(
                    f"[JERSEY_STABILIZE] {frag_id}: None -> #{chosen_jersey} "
                    f"(source={source_id})"
                )


def _apply_fragment_color_team_corrections(
    fragments: list[dict],
    min_duration_frames: int = 20,
    min_margin: float = 0.35,
    min_ratio: float = 1.12,
) -> int:
    """
    Correct non-KMeans team labels when fragment mean color strongly supports opposite team.

    Uses team centroids built from KMeans-labeled fragments and applies conservative
    overrides only to non-kmeans labels (continuity/parent/bridge/sandwich).
    """
    team_hist = {"team_a": [], "team_b": []}

    fragments_by_track: dict[str, list[dict]] = {}
    fragment_pos: dict[str, tuple[list[dict], int]] = {}
    for fragment in fragments:
        track_id = fragment.get("original_track_id")
        if track_id is None:
            continue
        fragments_by_track.setdefault(str(track_id), []).append(fragment)

    for track_frags in fragments_by_track.values():
        track_frags.sort(key=lambda f: f.get("start_frame", -1))
        for idx, frag in enumerate(track_frags):
            frag_id = frag.get("fragment_id")
            if frag_id:
                fragment_pos[frag_id] = (track_frags, idx)

    for fragment in fragments:
        team = fragment.get("team")
        source = str(fragment.get("label_source") or "")
        hist = fragment.get("mean_hsv_histogram")
        if team not in ("team_a", "team_b"):
            continue
        if "kmeans" not in source:
            continue
        if not hist or len(hist) != 96:
            continue
        team_hist[team].append(np.array(hist, dtype=np.float32))

    if not team_hist["team_a"] or not team_hist["team_b"]:
        return 0

    centroid_a = np.mean(np.stack(team_hist["team_a"], axis=0), axis=0)
    centroid_b = np.mean(np.stack(team_hist["team_b"], axis=0), axis=0)

    corrected = 0
    for fragment in fragments:
        team = fragment.get("team")
        if team not in ("team_a", "team_b"):
            continue

        source = str(fragment.get("label_source") or "")
        if "kmeans" in source:
            continue

        hist = fragment.get("mean_hsv_histogram")
        if not hist or len(hist) != 96:
            continue

        start_frame = fragment.get("start_frame")
        end_frame = fragment.get("end_frame")
        if start_frame is None or end_frame is None:
            continue
        duration = int(end_frame - start_frame + 1)
        if duration < min_duration_frames:
            continue

        vector = np.array(hist, dtype=np.float32)
        dist_a = float(np.linalg.norm(vector - centroid_a))
        dist_b = float(np.linalg.norm(vector - centroid_b))

        assigned_dist = dist_a if team == "team_a" else dist_b
        opposite_team = "team_b" if team == "team_a" else "team_a"
        opposite_dist = dist_b if team == "team_a" else dist_a

        frag_id = fragment.get("fragment_id")
        pos_info = fragment_pos.get(frag_id)
        if pos_info:
            track_frags, idx = pos_info
            prev_team = track_frags[idx - 1].get("team") if idx > 0 else None
            next_team = track_frags[idx + 1].get("team") if idx + 1 < len(track_frags) else None
            opposite_supported = prev_team == opposite_team or next_team == opposite_team
            current_supported = prev_team == team and next_team == team
            if current_supported:
                continue

            source_lower = source.lower()
            if "continuity_window" in source_lower and not opposite_supported:
                continue

        if assigned_dist <= opposite_dist * min_ratio:
            continue
        if (assigned_dist - opposite_dist) < min_margin:
            continue

        old_team = team
        fragment["team"] = opposite_team
        fragment["team_locked"] = opposite_team
        fragment["team_confidence"] = "color_corrected"
        fragment["label_source"] = f"{source}_color_corrected" if source else "color_corrected"
        corrected += 1
        print(
            f"[COLOR_TEAM_CORRECT] {fragment.get('fragment_id')}: {old_team} -> {opposite_team} "
            f"(dist_a={dist_a:.3f}, dist_b={dist_b:.3f}, duration={duration}, source={source})"
        )

    return corrected


def _stitch_anchor_identity_gaps(
    fragments: list[dict],
    assignments: dict[str, dict[str, Any]],
    max_gap_frames: int = 140,
    max_mid_fragments: int = 4,
) -> None:
    """
    Stitch identity through short/medium gaps bounded by same-track anchor fragments.

    If both sides of a gap on the same track agree on team and jersey number,
    propagate that identity to middle fragments to avoid long wrong-team linger
    and jersey disappearance after crossings.
    """
    fragments_by_track: dict[str, list[dict]] = {}
    fragment_lookup: dict[str, dict] = {}

    for fragment in fragments:
        frag_id = fragment.get("fragment_id")
        if frag_id:
            fragment_lookup[frag_id] = fragment
        track_id = fragment.get("original_track_id")
        if track_id is None:
            continue
        fragments_by_track.setdefault(str(track_id), []).append(fragment)

    for track_frags in fragments_by_track.values():
        track_frags.sort(key=lambda f: f.get("start_frame", -1))

    def _same_known_team(team_a: Any, team_b: Any) -> bool:
        return team_a in ("team_a", "team_b") and team_b in ("team_a", "team_b") and team_a == team_b

    def _has_same_team_conflict(target_fragment: dict, jersey_number: int) -> bool:
        target_id = target_fragment.get("fragment_id")
        target_team = target_fragment.get("team")
        target_start = target_fragment.get("start_frame")
        target_end = target_fragment.get("end_frame")
        if not target_id or target_start is None or target_end is None:
            return True

        for other_id, other_assignment in assignments.items():
            if other_id == target_id:
                continue
            if other_assignment.get("jersey_number") != jersey_number:
                continue

            other_fragment = fragment_lookup.get(other_id)
            if not other_fragment:
                continue

            other_team = other_fragment.get("team")
            if not _same_known_team(target_team, other_team):
                continue

            other_start = other_fragment.get("start_frame")
            other_end = other_fragment.get("end_frame")
            if other_start is None or other_end is None:
                continue

            if target_start <= other_end and target_end >= other_start:
                return True

        return False

    for track_id, track_frags in fragments_by_track.items():
        n = len(track_frags)
        for left_idx in range(n - 2):
            left = track_frags[left_idx]
            left_id = left.get("fragment_id")
            if not left_id:
                continue

            left_assignment = assignments.get(left_id, {})
            left_jersey = left_assignment.get("jersey_number")
            left_team = left.get("team")
            left_end = left.get("end_frame")
            if left_jersey is None or left_end is None or left_team not in ("team_a", "team_b"):
                continue

            for right_idx in range(left_idx + 2, min(n, left_idx + 2 + max_mid_fragments)):
                right = track_frags[right_idx]
                right_id = right.get("fragment_id")
                if not right_id:
                    continue

                right_assignment = assignments.get(right_id, {})
                right_jersey = right_assignment.get("jersey_number")
                right_team = right.get("team")
                right_start = right.get("start_frame")
                if right_jersey is None or right_start is None:
                    continue

                if right_jersey != left_jersey:
                    continue
                if not _same_known_team(left_team, right_team):
                    continue

                gap_frames = int(right_start - left_end - 1)
                if gap_frames < 0 or gap_frames > max_gap_frames:
                    continue

                mids = track_frags[left_idx + 1:right_idx]
                if not mids:
                    continue

                changed_any = False
                for mid in mids:
                    mid_id = mid.get("fragment_id")
                    if not mid_id:
                        continue

                    if mid.get("team") != left_team:
                        old_team = mid.get("team")
                        mid["team"] = left_team
                        mid["team_locked"] = left_team
                        mid["team_confidence"] = "anchor_gap"
                        mid["label_source"] = f"anchor_gap_{left_id}_{right_id}"
                        changed_any = True
                        print(
                            f"[ANCHOR_GAP] Team {mid_id}: {old_team} -> {left_team} "
                            f"(anchors {left_id}/{right_id}, jersey #{left_jersey})"
                        )

                    mid_assignment = assignments.get(mid_id, {})
                    if mid_assignment.get("jersey_number") is None and not _has_same_team_conflict(mid, left_jersey):
                        assignments[mid_id] = {
                            "jersey_number": left_jersey,
                            "confidence": max(
                                float(left_assignment.get("confidence", 0.0)),
                                float(right_assignment.get("confidence", 0.0)),
                            ),
                            "reason": f"anchor_gap_stitch_{left_id}_{right_id}",
                        }
                        changed_any = True
                        print(
                            f"[ANCHOR_GAP] Jersey {mid_id}: None -> #{left_jersey} "
                            f"(anchors {left_id}/{right_id})"
                        )

                if changed_any:
                    break


def process_clip_pass3(pass2_file: Path, output_dir: Path, config: dict, run_dir: Path):
    with open(pass2_file, "r", encoding="utf-8") as f:
        pass2_data = json.load(f)

    clip_name = pass2_data.get("clip_name", pass2_file.stem)
    fragments = pass2_data.get("fragments", [])

    if len(fragments) == 0:
        print(f"  No fragments found")
        return

    # Mark identity-jump fragments and build lookup for parent mapping
    fragment_lookup = _mark_identity_jumps(fragments)

    team_cfg = config.get("team_clustering", {})
    jersey_cfg = config.get("jersey", {})
    n_clusters = team_cfg.get("n_clusters", 2)
    team_cap = int(team_cfg.get("team_cap", 6))
    jersey_lock_threshold = jersey_cfg.get("lock_threshold", 0.7)
    jersey_min_detections = jersey_cfg.get("min_detections", 3)
    jersey_min_coverage = jersey_cfg.get("min_coverage_pct", 0.15)
    jersey_frame_stride = max(1, int(jersey_cfg.get("frame_stride", 1)))

    for fragment in fragments:
        fragment["continuity_locked"] = False

    fragment_histograms = []
    valid_fragment_indices = []
    for i, fragment in enumerate(fragments):
        hist = fragment.get("mean_hsv_histogram")
        if hist and len(hist) == 96:
            fragment_histograms.append(hist)
            valid_fragment_indices.append(i)

    if len(fragment_histograms) < n_clusters:
        print(f"  Not enough valid fragments for K-Means")
        for fragment in fragments:
            fragment["team"] = "unknown"
            fragment["team_confidence"] = "unknown"
            fragment["label_source"] = "insufficient_kmeans_input"
        # Continue with jersey logic even if team is unknown
        n_clusters = 0

    if n_clusters > 0:
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
            fragments[i]["team_confidence"] = "kmeans"
            fragments[i]["label_source"] = "kmeans"

    # Enforce hard team cap before locking team assignments.
    print(f"\n  Enforcing team size cap...")
    enforce_team_size_constraint(fragments, max_per_team=team_cap)

    print(f"  Recovering unknown team assignments...")
    recover_unknown_team_assignments(
        fragments,
        team_cap=team_cap,
        min_margin=float(team_cfg.get("unknown_recovery_min_margin", 0.12)),
        min_ratio=float(team_cfg.get("unknown_recovery_min_ratio", 1.04)),
        max_anchor_gap_frames=int(team_cfg.get("unknown_recovery_anchor_gap_frames", 5)),
    )

    # ========================================================================
    # LOCK TEAM ASSIGNMENTS
    # ========================================================================
    # Explicit lock: from this point, frag.team is READ-ONLY
    # All new logic must respect this invariant
    # ========================================================================
    for fragment in fragments:
        fragment["team_locked"] = fragment.get("team")

    # ========================================================================
    # INTRA-TEAM APPEARANCE SUB-CLUSTERING (metadata-only enrichment)
    # ========================================================================
    # Detect multi-appearance teams and compute appearance modes
    # CRITICAL: This is metadata-only. No feedback to team assignment, size constraints,
    # or divergence logic. Appearance modes are informational only.
    # ========================================================================
    print(f"\n  Computing intra-team appearance variance...")
    team_variances = _calculate_team_variances(fragments)
    print(f"    Team A variance: {team_variances['team_a']:.3f}")
    print(f"    Team B variance: {team_variances['team_b']:.3f}")

    appearance_variance_ratio_threshold = team_cfg.get("appearance_variance_ratio_threshold", 1.5)
    multi_appearance_teams = _detect_multi_appearance_teams(team_variances, appearance_variance_ratio_threshold)

    if multi_appearance_teams:
        print(f"  Sub-clustering multi-appearance teams: {multi_appearance_teams}")
        max_appearance_modes = team_cfg.get("max_appearance_modes", 3)
        for team_id in multi_appearance_teams:
            _subcluster_appearance_modes(fragments, team_id, max_modes=max_appearance_modes)
            # Count assignments
            assigned = sum(1 for f in fragments if f.get("team") == team_id and "appearance_mode_id" in f)
            print(f"    {team_id.upper()}: {assigned} fragments assigned appearance modes")
    else:
        print(f"  [OK] All teams are single-appearance (bibbed/uniform)")

    # ========================================================================
    # VALIDATE TEAM SIZE CONSTRAINT (SAFETY NET)
    # ========================================================================
    # Ensure no more than 6 concurrent players per team at any frame.
    # This catches fragmentation bugs that slipped through appearance-based splitting.
    # ========================================================================
    print(f"\n  Validating team size constraint...")
    validate_team_size_constraint(fragments, max_per_team=team_cap)

    # Get detailed jersey assignments (returns dict with assignment details)
    jersey_assignments_detailed = _infer_jersey_numbers_detailed(
        fragments,
        lock_threshold=jersey_lock_threshold,
        min_detections=jersey_min_detections,
        min_coverage_pct=jersey_min_coverage,
        frame_stride=jersey_frame_stride,
    )

    _retro_backfill_short_unknown_segments(
        fragments,
        jersey_assignments_detailed,
        max_fragment_frames=int(team_cfg.get("retro_backfill_max_fragment_frames", 45)),
        max_anchor_gap_frames=int(team_cfg.get("retro_backfill_max_anchor_gap_frames", 3)),
    )

    _retro_backfill_same_track_jerseys(
        fragments,
        jersey_assignments_detailed,
        max_anchor_gap_frames=int(team_cfg.get("retro_backfill_max_anchor_gap_frames", 3)),
    )

    team_snapshot = {
        fragment.get("fragment_id"): (
            fragment.get("team"),
            fragment.get("team_confidence"),
            fragment.get("label_source"),
        )
        for fragment in fragments
        if fragment.get("fragment_id")
    }
    jersey_snapshot = {
        fragment_id: (
            assignment.get("jersey_number"),
            assignment.get("confidence"),
            assignment.get("reason"),
        )
        for fragment_id, assignment in jersey_assignments_detailed.items()
    }

    # Display assignment summary with statistics
    _print_assignment_summary(fragments, jersey_assignments_detailed)

    for fragment in fragments:
        fragment_id = fragment.get("fragment_id")
        if not fragment_id:
            continue
        current_team_tuple = (
            fragment.get("team"),
            fragment.get("team_confidence"),
            fragment.get("label_source"),
        )
        if team_snapshot.get(fragment_id) != current_team_tuple:
            raise AssertionError(
                f"Pass3 fragment purity violation: team mutated for {fragment_id} "
                f"from {team_snapshot.get(fragment_id)} to {current_team_tuple}"
            )

    current_jersey_snapshot = {
        fragment_id: (
            assignment.get("jersey_number"),
            assignment.get("confidence"),
            assignment.get("reason"),
        )
        for fragment_id, assignment in jersey_assignments_detailed.items()
    }
    if jersey_snapshot != current_jersey_snapshot:
        raise AssertionError("Pass3 fragment purity violation: jersey assignments mutated post-inference")

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
            "identity_jump": fragment.get("identity_jump", False),
            "continuity_locked": fragment.get("continuity_locked", False),
            "team_confidence": fragment.get("team_confidence"),
            "label_source": fragment.get("label_source"),
            "team_constraint_violation": fragment.get("team_constraint_violation", False),
            "violation_reason": fragment.get("violation_reason"),
            "over_capacity_frame_count": fragment.get("over_capacity_frame_count", 0),
            # Appearance mode (metadata-only, for debugging/visualization)
            "appearance_mode_id": fragment.get("appearance_mode_id"),
            "appearance_mode_confidence": fragment.get("appearance_mode_confidence"),
        }
        identities.append(identity)

    output_data = {
        "clip_name": clip_name,
        "identities": identities,
        "team_summary": {
            "team_a_count": len([i for i in identities if i["team"] == "team_a"]),
            "team_b_count": len([i for i in identities if i["team"] == "team_b"]),
            "team_a_jerseys": list(set(i["jersey_number"] for i in identities if i["jersey_number"] is not None)),
        },
        # Appearance mode metadata (for debugging/visualization)
        "appearance_analysis": {
            "team_a_variance": team_variances.get("team_a", 0.0),
            "team_b_variance": team_variances.get("team_b", 0.0),
            "variance_ratio_threshold": appearance_variance_ratio_threshold,
            "multi_appearance_teams": list(multi_appearance_teams),
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

    # Crop export disabled - commented out to prevent disk writes
    # _export_team_crops(
    #     run_dir=run_dir,
    #     clip_name=clip_name,
    #     output_dir=output_dir,
    #     output_stem=output_stem,
    #     fragments=fragments,
    #     config=config,
    # )


def _infer_jersey_numbers_detailed(
    fragments: list[dict[str, Any]],
    lock_threshold: float = 0.7,
    min_detections: int = 3,
    min_coverage_pct: float = 0.15,
    frame_stride: int = 1,
) -> dict[str, dict[str, Any]]:
    """
    Assign jersey numbers to fragments with team-scoped single-owner enforcement.

    Core Invariant: At any frame, a jersey number can belong to AT MOST ONE fragment
    per team. (Same jersey number on opposing teams is allowed.)

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

    # Step 3: Build conflict groups (fragments competing for same jersey within the same team)
    # (team, jersey_num) -> list of candidates
    jersey_candidates = {}

    for frag_meta in fragment_metadata:
        fragment_id = frag_meta["fragment_id"]
        if frag_meta.get("team") not in ("team_a", "team_b"):
            continue
        eligible_jerseys = frag_meta["eligible_jerseys"]

        for jersey_num, jersey_stats in eligible_jerseys.items():
            jersey_key = (frag_meta["team"], jersey_num)
            if jersey_key not in jersey_candidates:
                jersey_candidates[jersey_key] = []

            jersey_candidates[jersey_key].append({
                "fragment_id": fragment_id,
                "evidence_score": jersey_stats["evidence_score"],
                "start_frame": frag_meta["start_frame"],
                "end_frame": frag_meta["end_frame"],
                "team": frag_meta["team"],
                "jersey_stats": jersey_stats,
            })

    # Step 4: Resolve conflicts using greedy algorithm
    # Process jerseys one at a time, assign to non-overlapping fragments with highest evidence
    # Note: Jersey inheritance (Step 5) handles track continuity after initial assignment
    assignments = {}
    assigned_fragments = set()  # Track which fragments have been assigned

    for jersey_key, candidates in jersey_candidates.items():
        _, jersey_num = jersey_key
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
            if team not in ("team_a", "team_b"):
                assignments[fragment_id] = {
                    "jersey_number": None,
                    "confidence": 0.0,
                    "reason": "unknown_team_no_jersey_assignment",
                }
                continue
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


def _retro_backfill_short_unknown_segments(
    fragments: list[dict[str, Any]],
    assignments: dict[str, dict[str, Any]],
    max_fragment_frames: int = 45,
    max_anchor_gap_frames: int = 3,
) -> int:
    """
    Backfill short unknown team/jersey segments from nearby same-track anchors.

    Purpose:
    - Hide brief correction lag in final JSON/video after identity stabilizes.
    - Preserve hard-cap demotions by never backfilling `team_cap_enforced` fragments.
    """
    fragments_by_track: dict[str, list[dict[str, Any]]] = {}
    fragment_lookup: dict[str, dict[str, Any]] = {}

    for fragment in fragments:
        fragment_id = fragment.get("fragment_id")
        if fragment_id:
            fragment_lookup[fragment_id] = fragment
        track_id = fragment.get("original_track_id")
        if track_id is None:
            continue
        fragments_by_track.setdefault(str(track_id), []).append(fragment)

    for track_frags in fragments_by_track.values():
        track_frags.sort(key=lambda f: f.get("start_frame", -1))

    def _duration(fragment_obj: dict[str, Any]) -> int:
        start = fragment_obj.get("start_frame")
        end = fragment_obj.get("end_frame")
        if start is None or end is None:
            return 0
        return int(end - start + 1)

    def _known_team(team_val: Any) -> bool:
        return team_val in ("team_a", "team_b")

    def _has_same_team_jersey_conflict(target_fragment: dict[str, Any], team_val: str, jersey_number: int) -> bool:
        target_id = target_fragment.get("fragment_id")
        target_start = target_fragment.get("start_frame")
        target_end = target_fragment.get("end_frame")
        if not target_id or target_start is None or target_end is None:
            return True

        for other_id, other_assignment in assignments.items():
            if other_id == target_id:
                continue
            if other_assignment.get("jersey_number") != jersey_number:
                continue

            other_fragment = fragment_lookup.get(other_id)
            if not other_fragment:
                continue

            other_team = other_fragment.get("team")
            if other_team != team_val:
                continue

            other_start = other_fragment.get("start_frame")
            other_end = other_fragment.get("end_frame")
            if other_start is None or other_end is None:
                continue

            if int(target_start) <= int(other_end) and int(target_end) >= int(other_start):
                return True

        return False

    updates = 0

    for _, track_frags in fragments_by_track.items():
        for idx, fragment in enumerate(track_frags):
            if _duration(fragment) <= 0 or _duration(fragment) > max_fragment_frames:
                continue

            if fragment.get("label_source") == "team_cap_enforced":
                continue

            frag_id = fragment.get("fragment_id")
            if not frag_id:
                continue

            curr_team = fragment.get("team")
            curr_assignment = assignments.get(frag_id, {})
            curr_jersey = curr_assignment.get("jersey_number")

            if _known_team(curr_team) and curr_jersey is not None:
                continue

            prev_anchor = None
            for prev in reversed(track_frags[:idx]):
                prev_end = prev.get("end_frame")
                curr_start = fragment.get("start_frame")
                if prev_end is None or curr_start is None:
                    continue
                gap = int(curr_start - prev_end - 1)
                if gap > max_anchor_gap_frames:
                    break
                if _known_team(prev.get("team")):
                    prev_anchor = prev
                    break

            next_anchor = None
            for nxt in track_frags[idx + 1:]:
                next_start = nxt.get("start_frame")
                curr_end = fragment.get("end_frame")
                if next_start is None or curr_end is None:
                    continue
                gap = int(next_start - curr_end - 1)
                if gap > max_anchor_gap_frames:
                    break
                if _known_team(nxt.get("team")):
                    next_anchor = nxt
                    break

            team_candidate = None
            if prev_anchor and next_anchor:
                prev_team = prev_anchor.get("team")
                next_team = next_anchor.get("team")
                if prev_team == next_team:
                    team_candidate = prev_team
            elif prev_anchor:
                team_candidate = prev_anchor.get("team")
            elif next_anchor:
                team_candidate = next_anchor.get("team")

            if not _known_team(team_candidate):
                continue

            if curr_team != team_candidate:
                fragment["team"] = team_candidate
                fragment["team_confidence"] = "retro_backfill"
                fragment["label_source"] = "retro_backfill"
                updates += 1

            prev_jersey = None
            next_jersey = None
            if prev_anchor:
                prev_id = prev_anchor.get("fragment_id")
                prev_jersey = assignments.get(prev_id, {}).get("jersey_number") if prev_id else None
            if next_anchor:
                next_id = next_anchor.get("fragment_id")
                next_jersey = assignments.get(next_id, {}).get("jersey_number") if next_id else None

            jersey_candidate = None
            source_id = None
            if prev_jersey is not None and next_jersey is not None and prev_jersey == next_jersey:
                jersey_candidate = prev_jersey
                source_id = prev_anchor.get("fragment_id") if prev_anchor else None
            elif prev_jersey is not None:
                jersey_candidate = prev_jersey
                source_id = prev_anchor.get("fragment_id") if prev_anchor else None
            elif next_jersey is not None:
                jersey_candidate = next_jersey
                source_id = next_anchor.get("fragment_id") if next_anchor else None

            if curr_jersey is None and jersey_candidate is not None:
                if not _has_same_team_jersey_conflict(fragment, team_candidate, int(jersey_candidate)):
                    assignments[frag_id] = {
                        "jersey_number": int(jersey_candidate),
                        "confidence": float(assignments.get(source_id, {}).get("confidence", 0.0)),
                        "reason": f"retro_backfill_from_{source_id}",
                    }
                    updates += 1

    if updates > 0:
        print(f"  [RETRO_BACKFILL] Applied {updates} retro backfill update(s)")

    return updates


def _retro_backfill_same_track_jerseys(
    fragments: list[dict[str, Any]],
    assignments: dict[str, dict[str, Any]],
    max_anchor_gap_frames: int = 3,
) -> int:
    """
    Back-populate jersey numbers backward on the same track when a split fragment
    later reveals a stable jersey.

    Example handled:
    - `track 6: frag_000045` has no visible back yet
    - `track 6: frag_000045_split` later shows `#4`
    - Backfill `#4` to the earlier fragment when safe.

    Safety rules:
    - Same original track only.
    - Same known team only.
    - Only assigns fragments that currently have no jersey.
    - Never violates same-team temporal jersey exclusivity.
    """
    fragments_by_track: dict[str, list[dict[str, Any]]] = {}
    fragment_lookup: dict[str, dict[str, Any]] = {}

    for fragment in fragments:
        fragment_id = fragment.get("fragment_id")
        if fragment_id:
            fragment_lookup[fragment_id] = fragment
        track_id = fragment.get("original_track_id")
        if track_id is None:
            continue
        fragments_by_track.setdefault(str(track_id), []).append(fragment)

    for track_frags in fragments_by_track.values():
        track_frags.sort(key=lambda f: (int(f.get("start_frame", -1)), str(f.get("fragment_id", ""))))

    def _known_team(team_val: Any) -> bool:
        return team_val in ("team_a", "team_b")

    def _has_same_team_temporal_conflict(target_id: str, team_val: str, jersey_number: int) -> bool:
        target_fragment = fragment_lookup.get(target_id)
        if not target_fragment:
            return True

        target_start = target_fragment.get("start_frame")
        target_end = target_fragment.get("end_frame")
        if target_start is None or target_end is None:
            return True

        for other_id, other_assignment in assignments.items():
            if other_id == target_id:
                continue
            if other_assignment.get("jersey_number") != jersey_number:
                continue

            other_fragment = fragment_lookup.get(other_id)
            if not other_fragment:
                continue
            if other_fragment.get("team") != team_val:
                continue

            other_start = other_fragment.get("start_frame")
            other_end = other_fragment.get("end_frame")
            if other_start is None or other_end is None:
                continue

            if int(target_start) <= int(other_end) and int(target_end) >= int(other_start):
                return True

        return False

    updates = 0

    for track_frags in fragments_by_track.values():
        for idx in range(1, len(track_frags)):
            source_fragment = track_frags[idx]
            source_id = source_fragment.get("fragment_id")
            if not source_id:
                continue

            source_team = source_fragment.get("team")
            source_assignment = assignments.get(source_id, {})
            source_jersey = source_assignment.get("jersey_number")

            if not _known_team(source_team) or source_jersey is None:
                continue

            for prev_idx in range(idx - 1, -1, -1):
                target_fragment = track_frags[prev_idx]
                target_id = target_fragment.get("fragment_id")
                if not target_id:
                    continue

                source_start = source_fragment.get("start_frame")
                target_end = target_fragment.get("end_frame")
                if source_start is None or target_end is None:
                    continue

                gap = int(source_start - target_end - 1)
                if gap > max_anchor_gap_frames:
                    break

                target_team = target_fragment.get("team")
                if not _known_team(target_team) or target_team != source_team:
                    continue

                target_assignment = assignments.get(target_id, {})
                if target_assignment.get("jersey_number") is not None:
                    continue

                if _has_same_team_temporal_conflict(target_id, source_team, int(source_jersey)):
                    continue

                assignments[target_id] = {
                    "jersey_number": int(source_jersey),
                    "confidence": float(source_assignment.get("confidence", 0.0)),
                    "reason": f"retro_track_jersey_from_{source_id}",
                }
                updates += 1

    if updates > 0:
        print(f"  [RETRO_TRACK_JERSEY] Applied {updates} jersey backfill update(s)")

    return updates


def _apply_jersey_inheritance(
    fragments: list[dict[str, Any]],
    assignments: dict[str, dict[str, Any]],
    fragment_lookup: dict[str, dict]
) -> dict[str, dict[str, Any]]:
    """
    Apply bidirectional jersey inheritance based on track continuity.

    Logic: A jersey number can't just disappear! Propagate jerseys both forward
    and backward through track fragments to handle:
    - FORWARD: Earlier fragment has jersey → later fragment inherits it
    - BACKWARD: Later fragment has jersey → earlier fragment inherits it

    This handles:
    - Player turns around (jersey appears later): backward inheritance
    - Brief occlusions where tracking is lost and re-acquired: forward inheritance
    - Track gaps where the player temporarily leaves detection range: both directions

    CRITICAL: Jersey inheritance respects temporal exclusivity. A jersey can only
    be inherited if it's NOT already in use by another fragment at overlapping times.

    Args:
        fragments: List of fragment dicts from Pass 2
        assignments: Current jersey assignments from _infer_jersey_numbers_detailed()

    Returns:
        Updated assignments with inherited jerseys (bidirectional)
    """
    # Build fragment metadata lookup (fragment_id -> {start_frame, end_frame, team})
    fragment_metadata = {}
    for fragment in fragments:
        fragment_id = fragment.get("fragment_id")
        fragment_metadata[fragment_id] = {
            "start_frame": fragment.get("start_frame"),
            "end_frame": fragment.get("end_frame"),
            "team": fragment.get("team", "unknown"),
        }

    # Helper function to check for temporal conflicts
    def _has_temporal_conflict(
        jersey_number: int,
        target_fragment_id: str,
        current_assignments: dict[str, dict[str, Any]],
        fragment_metadata: dict[str, dict[str, int]]
    ) -> bool:
        """
        Check if assigning jersey_number to target_fragment_id would create a temporal conflict
        on the same team. Same jersey on opposing teams is allowed.

        Returns True if the jersey is already assigned to another fragment at overlapping times.
        """
        target_meta = fragment_metadata.get(target_fragment_id)
        if not target_meta:
            return False  # No metadata, can't check

        target_start = target_meta["start_frame"]
        target_end = target_meta["end_frame"]
        target_team = target_meta.get("team", "unknown")

        # Check all current assignments
        for frag_id, assignment in current_assignments.items():
            if frag_id == target_fragment_id:
                continue  # Skip self

            assigned_jersey = assignment.get("jersey_number")
            if assigned_jersey != jersey_number:
                continue  # Different jersey, no conflict

            # Same jersey on different fragment - check for temporal overlap
            other_meta = fragment_metadata.get(frag_id)
            if not other_meta:
                continue  # No metadata, can't check

            other_team = other_meta.get("team", "unknown")
            if (
                target_team in ("team_a", "team_b")
                and other_team in ("team_a", "team_b")
                and target_team != other_team
            ):
                continue  # Opposing team ownership is allowed for same jersey number

            other_start = other_meta["start_frame"]
            other_end = other_meta["end_frame"]

            # Check for temporal overlap: [target_start, target_end] overlaps [other_start, other_end]
            if target_start <= other_end and target_end >= other_start:
                return True  # CONFLICT: Same jersey on different fragments at the same time

        return False  # No conflict

    # Group fragments by original_track_id
    tracks = {}  # track_id -> list of fragment metadata
    for fragment in fragments:
        track_id = fragment.get("original_track_id")
        fragment_id = fragment.get("fragment_id")
        start_frame = fragment.get("start_frame")
        end_frame = fragment.get("end_frame")
        team = fragment.get("team", "unknown")
        split_reason = fragment.get("split_reason")

        if track_id not in tracks:
            tracks[track_id] = []
        tracks[track_id].append({
            "fragment_id": fragment_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "team": team,
            "split_reason": split_reason,
        })

    # Sort each track's fragments by start_frame
    for track_id in tracks:
        tracks[track_id].sort(key=lambda f: f["start_frame"])

    # ========================================================================
    # BIDIRECTIONAL JERSEY INHERITANCE
    # ========================================================================
    # A jersey can't just disappear! Propagate jerseys both forward and backward
    # through track fragments to handle cases where:
    # 1. Jersey appears later (player turns around): F000000 (no jersey) ← F000001 (jersey #10)
    # 2. Jersey disappears temporarily (brief occlusion): F000000 (jersey #10) → F000001 (no jersey)
    #
    # Strategy:
    # 1. Forward pass: Inherit jerseys from earlier fragments to later fragments
    # 2. Backward pass: Inherit jerseys from later fragments to earlier fragments
    # ========================================================================
    forward_count = 0
    backward_count = 0
    new_assignments = dict(assignments)  # Copy to avoid modifying original

    # FORWARD PASS: Inherit jerseys from earlier fragments to later fragments
    for track_id, track_fragments in tracks.items():
        for i in range(len(track_fragments) - 1):
            curr_frag = track_fragments[i]
            next_frag = track_fragments[i + 1]
            next_fragment = fragment_lookup.get(next_frag.get("fragment_id"))
            if next_fragment and next_fragment.get("identity_jump"):
                parent_id = next_fragment.get("parent_fragment_id")
                if parent_id != curr_frag.get("fragment_id"):
                    continue

            # Check if current fragment has a jersey
            curr_assignment = new_assignments.get(curr_frag["fragment_id"], {})
            curr_jersey = curr_assignment.get("jersey_number")

            if curr_jersey is None:
                continue  # No jersey to inherit

            # Check if next fragment has no jersey
            next_assignment = new_assignments.get(next_frag["fragment_id"], {})
            next_jersey = next_assignment.get("jersey_number")

            if next_jersey is not None:
                continue  # Next fragment already has a jersey

            curr_team = curr_frag.get("team")
            next_team = next_frag.get("team")
            if (
                curr_team in ("team_a", "team_b")
                and next_team in ("team_a", "team_b")
                and curr_team != next_team
            ):
                print(
                    f"    [FORWARD] SKIPPED jersey #{curr_jersey}: {curr_frag['fragment_id']} "
                    f"(track {track_id}) -> {next_frag['fragment_id']} (team mismatch {curr_team}->{next_team})"
                )
                continue

            # CRITICAL: Check for temporal conflicts before inheriting
            if _has_temporal_conflict(curr_jersey, next_frag["fragment_id"], new_assignments, fragment_metadata):
                print(f"    [FORWARD] SKIPPED jersey #{curr_jersey}: {curr_frag['fragment_id']} (track {track_id}) -> {next_frag['fragment_id']} (temporal conflict)")
                continue  # Skip inheritance - jersey already in use elsewhere

            # Inherit forward: earlier fragment → later fragment
            new_assignments[next_frag["fragment_id"]] = {
                "jersey_number": curr_jersey,
                "confidence": curr_assignment.get("confidence", 0.0),
                "reason": f"inherited_forward_from_{curr_frag['fragment_id']} (track_continuity)",
            }
            forward_count += 1
            print(f"    [FORWARD] Inherited jersey #{curr_jersey}: {curr_frag['fragment_id']} (track {track_id}) -> {next_frag['fragment_id']}")

    # BACKWARD PASS: Inherit jerseys from later fragments to earlier fragments
    for track_id, track_fragments in tracks.items():
        # Iterate in reverse order (from end to beginning)
        for i in range(len(track_fragments) - 1, 0, -1):
            curr_frag = track_fragments[i]
            prev_frag = track_fragments[i - 1]
            prev_fragment = fragment_lookup.get(prev_frag.get("fragment_id"))
            if prev_fragment and prev_fragment.get("identity_jump"):
                parent_id = prev_fragment.get("parent_fragment_id")
                if parent_id != curr_frag.get("fragment_id"):
                    continue

            # Check if current fragment has a jersey
            curr_assignment = new_assignments.get(curr_frag["fragment_id"], {})
            curr_jersey = curr_assignment.get("jersey_number")

            if curr_jersey is None:
                continue  # No jersey to inherit

            # Check if previous fragment has no jersey
            prev_assignment = new_assignments.get(prev_frag["fragment_id"], {})
            prev_jersey = prev_assignment.get("jersey_number")

            if prev_jersey is not None:
                continue  # Previous fragment already has a jersey

            curr_team = curr_frag.get("team")
            prev_team = prev_frag.get("team")
            if (
                curr_team in ("team_a", "team_b")
                and prev_team in ("team_a", "team_b")
                and curr_team != prev_team
            ):
                print(
                    f"    [BACKWARD] SKIPPED jersey #{curr_jersey}: {curr_frag['fragment_id']} "
                    f"(track {track_id}) -> {prev_frag['fragment_id']} (team mismatch {curr_team}->{prev_team})"
                )
                continue

            # CRITICAL: Check for temporal conflicts before inheriting
            if _has_temporal_conflict(curr_jersey, prev_frag["fragment_id"], new_assignments, fragment_metadata):
                print(f"    [BACKWARD] SKIPPED jersey #{curr_jersey}: {curr_frag['fragment_id']} (track {track_id}) -> {prev_frag['fragment_id']} (temporal conflict)")
                continue  # Skip inheritance - jersey already in use elsewhere

            # Inherit backward: later fragment → earlier fragment
            new_assignments[prev_frag["fragment_id"]] = {
                "jersey_number": curr_jersey,
                "confidence": curr_assignment.get("confidence", 0.0),
                "reason": f"inherited_backward_from_{curr_frag['fragment_id']} (track_continuity)",
            }
            backward_count += 1
            print(f"    [BACKWARD] Inherited jersey #{curr_jersey}: {curr_frag['fragment_id']} (track {track_id}) -> {prev_frag['fragment_id']}")

    total_count = forward_count + backward_count
    if total_count == 0:
        print(f"    No jersey inheritance needed")
    else:
        print(f"    Applied {total_count} jersey inheritance(s) ({forward_count} forward, {backward_count} backward)")

    return new_assignments


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
    Validate team-scoped single-owner invariant:
    At any frame, a jersey number belongs to at most one fragment per team.
    (Opposing teams may share the same jersey number.)

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
        team = frag_meta.get("team", "unknown")

        if jersey_num not in jersey_timeline:
            jersey_timeline[jersey_num] = []

        # Check for overlaps with existing assignments
        for existing_start, existing_end, existing_frag, existing_team in jersey_timeline[jersey_num]:
            if (
                team in ("team_a", "team_b")
                and existing_team in ("team_a", "team_b")
                and team != existing_team
            ):
                continue  # Opposing teams may share same jersey number

            # Check temporal overlap
            if not (end_frame < existing_start or existing_end < start_frame):
                raise AssertionError(
                    f"INVARIANT VIOLATION: Jersey {jersey_num} assigned to overlapping fragments "
                    f"{fragment_id} [{start_frame}-{end_frame}] and "
                    f"{existing_frag} [{existing_start}-{existing_end}]"
                )

        jersey_timeline[jersey_num].append((start_frame, end_frame, fragment_id, team))

    # If we reach here, invariant holds
    pass


def _get_bib_stats(crop_bgr: np.ndarray) -> dict[str, float]:
    """Get crop color statistics for debugging."""
    if crop_bgr is None or crop_bgr.size == 0:
        return {"mean_hue": 0, "mean_sat": 0}

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask_valid = (s > 20) & (v > 20)

    if np.count_nonzero(mask_valid) < 10:
        return {"mean_hue": 0, "mean_sat": 0}

    mean_hue = float(np.mean(h[mask_valid]))
    mean_sat = float(np.mean(s[mask_valid]))

    return {
        "mean_hue": mean_hue,
        "mean_sat": mean_sat,
    }


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

    # Diagnostic: Save candidate crops for debugging
    debug_candidates_dir = crops_root / "DEBUG_candidates"
    debug_candidates_dir.mkdir(parents=True, exist_ok=True)

    # Crop export configuration (all from pass3)
    crops_per_fragment = int(pass3_cfg.get("crops_per_fragment", 6))
    min_per_team = int(pass3_cfg.get("min_crops_per_team", 15))
    crop_mode = pass3_cfg.get("crop_mode", "jersey")
    proximity_thresh = float(pass3_cfg.get("proximity_threshold_px", 24.0))

    # Quality thresholds (color-agnostic)
    quality_threshold = float(pass3_cfg.get("quality_threshold", 0.35))
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
        debug_crop_count = 0  # Limit diagnostic crops to ~10 per fragment

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

                # DIAGNOSTIC: Save candidate crops with metadata (limit to 10 per fragment)
                if debug_crop_count < 10:
                    color_stats = _get_bib_stats(jersey_crop)
                    debug_filename = (
                        f"{fragment_id}_f{frame_idx}_"
                        f"q{quality:.2f}_t{quality_threshold:.2f}_"
                        f"h{color_stats['mean_hue']:.0f}_s{color_stats['mean_sat']:.0f}_"
                        f"{team}.jpg"
                    )
                    cv2.imwrite(str(debug_candidates_dir / debug_filename), jersey_crop)
                    debug_crop_count += 1

                # Quality check (same threshold for all players, regardless of color/team)
                if quality < quality_threshold:
                    rejection_reason = f"quality_too_low_q{quality:.2f}_t{quality_threshold:.2f}"
                    rejection_stats[rejection_reason] = rejection_stats.get(rejection_reason, 0) + 1
                    continue

                if crop_bgr is None or crop_bgr.size == 0:
                    rejection_stats["empty_crop"] = rejection_stats.get("empty_crop", 0) + 1
                    continue

                # Accept (standard quality threshold passed)
                accept_reason = f"{team}_accept"
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
