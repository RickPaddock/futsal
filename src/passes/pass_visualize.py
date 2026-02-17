"""
Visualization Pass: Generate annotated video from Pass 3 output.

Loads Pass 1 (bboxes), Pass 2 (divergences), and Pass 3 (team/jersey) to create
frame-by-frame annotated video output.
"""

from pathlib import Path
import json
import numpy as np
import cv2
import supervision as sv
from tqdm import tqdm

from src.utils.video_io import VideoReader, VideoWriter
from src.utils.data_models import BoundingBox

# Import homography for 2D pitch visualization (copied from ARCHIVE to src/)
from src.geometry.homography import create_homography_from_config


# Team colors (BGR format for OpenCV)
TEAM_COLORS = {
    "team_a": (0, 140, 255),    # Orange (bibbed team)
    "team_b": (40, 40, 40),  # Black (diverse team)
    "unknown": (200, 200, 200), # Light gray
}


def _bridge_micro_fragment_gaps(
    fragments: list[dict[str, object]],
    fragment_identity_map: dict[str, dict[str, object]],
    max_gap_frames: int = 5,
) -> dict[tuple[str, int], str]:
    """
    Build visualization-only fallback mapping for tiny fragment gaps.

    This does NOT modify Pass2/Pass3 outputs. It only fills short holes in
    (track_id, frame) → fragment_id lookup when adjacent fragments on the same
    track have consistent team identity.
    """
    if max_gap_frames <= 0:
        return {}

    by_track: dict[str, list[dict[str, object]]] = {}
    for fragment in fragments:
        track_id = fragment.get("original_track_id")
        fragment_id = fragment.get("fragment_id")
        if track_id is None or not fragment_id:
            continue
        by_track.setdefault(str(track_id), []).append(fragment)

    for track_fragments in by_track.values():
        track_fragments.sort(key=lambda f: int(f.get("start_frame", -1)))

    bridged: dict[tuple[str, int], str] = {}

    for track_id, track_fragments in by_track.items():
        for idx in range(len(track_fragments) - 1):
            prev_fragment = track_fragments[idx]
            next_fragment = track_fragments[idx + 1]

            prev_id = str(prev_fragment.get("fragment_id"))
            next_id = str(next_fragment.get("fragment_id"))

            prev_identity = fragment_identity_map.get(prev_id)
            next_identity = fragment_identity_map.get(next_id)
            if not prev_identity or not next_identity:
                continue

            prev_team = prev_identity.get("team")
            next_team = next_identity.get("team")
            if prev_team != next_team or prev_team not in ("team_a", "team_b"):
                continue

            prev_jersey = prev_identity.get("jersey")
            next_jersey = next_identity.get("jersey")
            if prev_jersey is not None and next_jersey is not None and prev_jersey != next_jersey:
                continue

            prev_end = prev_fragment.get("end_frame")
            next_start = next_fragment.get("start_frame")
            if prev_end is None or next_start is None:
                continue

            gap_start = int(prev_end) + 1
            gap_end = int(next_start) - 1
            gap_size = gap_end - gap_start + 1
            if gap_size <= 0 or gap_size > max_gap_frames:
                continue

            source_fragment_id = prev_id if prev_jersey is not None else next_id if next_jersey is not None else prev_id
            for frame_idx in range(gap_start, gap_end + 1):
                bridged[(track_id, frame_idx)] = source_fragment_id

    return bridged


def _build_pre_anchor_identity_backfill(
    fragments: list[dict[str, object]],
    fragment_identity_map: dict[str, dict[str, object]],
    max_anchor_gap_frames: int = 5,
    max_current_fragment_frames: int = 45,
    backfill_tail_frames: int = 30,
    min_next_confidence: float = 12.0,
) -> dict[tuple[str, int], str]:
    """
    Visualization-only backfill from a strong *future* same-track anchor.

    Use-case:
    - Short unstable segment (often unknown / cap-enforced / transient wrong team)
    - Immediately followed by strong known identity on same track
    - Backfill only the tail of the unstable segment for smoother display
    """
    if backfill_tail_frames <= 0:
        return {}

    unstable_sources = {
        "team_cap_enforced",
        "replacement_demoted",
        "unknown_recovered_anchor",
        "unknown_recovered_color",
        "unknown_recovered_replacement_anchor",
        "unknown_recovered_replacement_color",
    }

    by_track: dict[str, list[dict[str, object]]] = {}
    for fragment in fragments:
        track_id = fragment.get("original_track_id")
        fragment_id = fragment.get("fragment_id")
        if track_id is None or not fragment_id:
            continue
        by_track.setdefault(str(track_id), []).append(fragment)

    for track_fragments in by_track.values():
        track_fragments.sort(key=lambda f: int(f.get("start_frame", -1)))

    backfilled: dict[tuple[str, int], str] = {}

    for track_id, track_fragments in by_track.items():
        for idx in range(len(track_fragments) - 1):
            current_fragment = track_fragments[idx]
            next_fragment = track_fragments[idx + 1]

            current_id = str(current_fragment.get("fragment_id"))
            next_id = str(next_fragment.get("fragment_id"))

            current_identity = fragment_identity_map.get(current_id)
            next_identity = fragment_identity_map.get(next_id)
            if not current_identity or not next_identity:
                continue

            next_team = next_identity.get("team")
            next_jersey = next_identity.get("jersey")
            next_confidence = float(next_identity.get("confidence", 0.0) or 0.0)
            if next_team not in ("team_a", "team_b"):
                continue
            if next_jersey is None:
                continue
            if next_confidence < float(min_next_confidence):
                continue

            current_start = current_fragment.get("start_frame")
            current_end = current_fragment.get("end_frame")
            next_start = next_fragment.get("start_frame")
            if current_start is None or current_end is None or next_start is None:
                continue

            current_duration = int(current_end) - int(current_start) + 1
            if current_duration <= 0 or current_duration > int(max_current_fragment_frames):
                continue

            gap = int(next_start) - int(current_end) - 1
            if gap < 0 or gap > int(max_anchor_gap_frames):
                continue

            current_team = current_identity.get("team")
            current_jersey = current_identity.get("jersey")
            current_source = str(current_identity.get("label_source") or "")
            current_unstable = (
                current_team == "unknown"
                or current_source in unstable_sources
                or (current_team in ("team_a", "team_b") and current_team != next_team and current_jersey is None)
            )
            if not current_unstable:
                continue

            tail_start = max(int(current_start), int(current_end) - int(backfill_tail_frames) + 1)
            for frame_idx in range(tail_start, int(current_end) + 1):
                backfilled[(track_id, frame_idx)] = next_id

            gap_start = int(current_end) + 1
            gap_end = int(next_start) - 1
            if gap_start <= gap_end:
                for frame_idx in range(gap_start, gap_end + 1):
                    backfilled[(track_id, frame_idx)] = next_id

    return backfilled


def _build_track_jersey_persistence_map(
    fragments: list[dict[str, object]],
    fragment_identity_map: dict[str, dict[str, object]],
    max_anchor_gap_frames: int = 8,
    block_min_confidence: float = 4.0,
    block_requires_assigned_reason: bool = True,
    block_min_conf_ratio_to_source: float = 0.6,
    track_frame_presence: dict[str, set[int]] | None = None,
) -> dict[tuple[str, int], str]:
    """
    Visualization-only jersey persistence across same-track fragment splits.

    Rule:
    - If a track had jersey J on team T and subsequent same-track fragment(s)
      lose jersey due to splitting, keep showing J frame-by-frame only while no
      other fragment on team T claims J at that frame.
    """
    if max_anchor_gap_frames < 0:
        return {}

    by_track: dict[str, list[dict[str, object]]] = {}
    for fragment in fragments:
        track_id = fragment.get("original_track_id")
        fragment_id = fragment.get("fragment_id")
        if track_id is None or not fragment_id:
            continue
        by_track.setdefault(str(track_id), []).append(fragment)

    for track_fragments in by_track.values():
        track_fragments.sort(key=lambda f: int(f.get("start_frame", -1)))

    # Build frame-level claim index from base fragment identities.
    # key: (team, jersey, frame) -> {track_id: (confidence, assignment_reason)}
    claim_index: dict[tuple[str, int, int], dict[str, tuple[float, str]]] = {}
    for fragment in fragments:
        fragment_id = fragment.get("fragment_id")
        if not fragment_id:
            continue
        identity = fragment_identity_map.get(str(fragment_id), {})
        team = identity.get("team")
        jersey = identity.get("jersey")
        confidence = float(identity.get("confidence", 0.0) or 0.0)
        assignment_reason = str(identity.get("assignment_reason") or "")
        if team not in ("team_a", "team_b") or jersey is None:
            continue

        start_frame = fragment.get("start_frame")
        end_frame = fragment.get("end_frame")
        track_id = fragment.get("original_track_id")
        if start_frame is None or end_frame is None or track_id is None:
            continue

        jersey_i = int(jersey)
        track_key = str(track_id)
        for frame_idx in range(int(start_frame), int(end_frame) + 1):
            if track_frame_presence is not None:
                present_frames = track_frame_presence.get(track_key, set())
                if frame_idx not in present_frames:
                    continue
            key = (str(team), jersey_i, frame_idx)
            claim_index.setdefault(key, {})[track_key] = (confidence, assignment_reason)

    persistence: dict[tuple[str, int], str] = {}

    for track_id, track_fragments in by_track.items():
        last_known_team: str | None = None
        last_known_jersey: int | None = None
        last_known_source_fragment_id: str | None = None
        last_known_end: int | None = None

        for fragment in track_fragments:
            fragment_id = str(fragment.get("fragment_id"))
            identity = fragment_identity_map.get(fragment_id, {})
            team = identity.get("team")
            jersey = identity.get("jersey")
            start_frame = fragment.get("start_frame")
            end_frame = fragment.get("end_frame")
            if start_frame is None or end_frame is None:
                continue

            start_i = int(start_frame)
            end_i = int(end_frame)

            if team in ("team_a", "team_b") and jersey is not None:
                last_known_team = str(team)
                last_known_jersey = int(jersey)
                last_known_source_fragment_id = fragment_id
                last_known_end = end_i
                continue

            if team not in ("team_a", "team_b"):
                continue

            if (
                last_known_team is None
                or last_known_jersey is None
                or last_known_source_fragment_id is None
                or last_known_end is None
            ):
                continue

            if str(team) != last_known_team:
                continue

            gap = start_i - int(last_known_end) - 1
            if gap > int(max_anchor_gap_frames):
                continue

            # Persist only while nobody else on same team claims this jersey.
            for frame_idx in range(start_i, end_i + 1):
                claimers = claim_index.get((last_known_team, int(last_known_jersey), frame_idx), {})
                source_identity = fragment_identity_map.get(last_known_source_fragment_id, {})
                source_confidence = float(source_identity.get("confidence", 0.0) or 0.0)
                required_confidence = max(
                    float(block_min_confidence),
                    float(source_confidence) * float(block_min_conf_ratio_to_source),
                )

                other_strong_claim_exists = False
                for claim_track_id, (claim_confidence, claim_reason) in claimers.items():
                    if claim_track_id == track_id:
                        continue
                    if float(claim_confidence) < required_confidence:
                        continue
                    if block_requires_assigned_reason and not str(claim_reason).startswith("assigned ("):
                        continue
                    other_strong_claim_exists = True
                    break

                if other_strong_claim_exists:
                    continue
                persistence[(track_id, frame_idx)] = last_known_source_fragment_id

            last_known_end = end_i

    return persistence


def _bbox_iou_xyxy(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = area_a + area_b - inter
    if den <= 0:
        return 0.0
    return float(inter / den)


def _annotation_rank(ann: dict) -> float:
    score = float(ann.get("jersey_confidence", 0.0))
    if ann.get("jersey") is not None:
        score += 2.0

    label_source = str(ann.get("label_source") or "")
    team_confidence = str(ann.get("team_confidence") or "")
    if "kmeans" in label_source or "anchor_gap" in label_source:
        score += 1.0
    if team_confidence in ("kmeans", "anchor_gap", "bridge_forward", "bridge_sandwich"):
        score += 0.5

    return score


def _dedupe_overlapping_annotations(
    annotations: dict[str, dict],
    iou_threshold: float = 0.55,
) -> dict[str, dict]:
    """
    Remove duplicate annotations with high spatial overlap.

    CRITICAL: Only suppresses annotations with the SAME track_id.
    Different tracks (e.g., real player vs ghost of different player) are NEVER suppressed
    even if they overlap spatially (ghost may be positioned at occluder's location).
    """
    if not annotations:
        return annotations

    track_items = list(annotations.items())
    suppressed: set[str] = set()

    for i in range(len(track_items)):
        track_id_a, ann_a = track_items[i]
        if track_id_a in suppressed:
            continue
        bbox_a = ann_a.get("bbox")
        if not bbox_a or len(bbox_a) != 4:
            continue

        for j in range(i + 1, len(track_items)):
            track_id_b, ann_b = track_items[j]
            if track_id_b in suppressed:
                continue

            # CRITICAL: Only suppress if SAME track_id (duplicate detection)
            # Different tracks should never suppress each other
            if track_id_a != track_id_b:
                continue

            bbox_b = ann_b.get("bbox")
            if not bbox_b or len(bbox_b) != 4:
                continue

            overlap = _bbox_iou_xyxy(bbox_a, bbox_b)
            if overlap < iou_threshold:
                continue

            score_a = _annotation_rank(ann_a)
            score_b = _annotation_rank(ann_b)

            if score_a >= score_b:
                suppressed.add(track_id_b)
            else:
                suppressed.add(track_id_a)
                break

    if not suppressed:
        return annotations

    return {track_id: ann for track_id, ann in annotations.items() if track_id not in suppressed}


def _enforce_frame_jersey_exclusivity(
    annotations: dict[str, dict],
) -> dict[str, dict]:
    """
    Enforce single jersey owner per frame (global across all tracks).

    When multiple tracks carry the same jersey at a frame, keep only the
    strongest claim and suppress jersey display for the others.
    """
    if not annotations:
        return annotations

    by_jersey: dict[int, list[tuple[str, dict]]] = {}
    for track_id, ann in annotations.items():
        jersey = ann.get("jersey")
        if jersey is None:
            continue
        try:
            jersey_i = int(jersey)
        except (TypeError, ValueError):
            continue
        by_jersey.setdefault(jersey_i, []).append((track_id, ann))

    if not by_jersey:
        return annotations

    updated = {track_id: dict(ann) for track_id, ann in annotations.items()}

    def _claim_rank(item: tuple[str, dict]) -> tuple[float, float, float, int]:
        _, ann = item
        base_rank = _annotation_rank(ann)
        conf = float(ann.get("jersey_confidence", 0.0) or 0.0)
        raw_conf = float(ann.get("raw_jersey_conf", 0.0) or 0.0)
        reason = str(ann.get("assignment_reason") or "")
        assigned_boost = 1 if reason.startswith("assigned (") else 0
        return (base_rank, conf, raw_conf, assigned_boost)

    for _, claimants in by_jersey.items():
        if len(claimants) <= 1:
            continue

        winner_track_id, _ = max(claimants, key=_claim_rank)
        for track_id, _ in claimants:
            if track_id == winner_track_id:
                continue
            updated_ann = updated.get(track_id)
            if not updated_ann:
                continue
            updated_ann["jersey"] = None
            updated_ann["jersey_confidence"] = 0.0
            updated[track_id] = updated_ann

    return updated

class BallAnnotator:
    """Draws ball with a trailing jet-colormap tail showing recent positions."""

    def __init__(
        self,
        radius: int = 12,
        buffer_size: int = 30,
        thickness: int = 2,
        max_age_seconds: float = 2.0,
        fps: float | None = None,
    ):
        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer: list[tuple[tuple[int, int], int | None]] = []
        self.buffer_size = buffer_size
        self.radius = radius
        self.thickness = thickness
        self.max_age_frames = int(max_age_seconds * fps) if fps and max_age_seconds > 0 else None

    def interpolate_radius(self, i: int, max_i: int) -> int:
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def _prune_old(self, frame_idx: int | None) -> None:
        if self.max_age_frames is None or frame_idx is None:
            return
        cutoff = frame_idx - self.max_age_frames
        self.buffer = [(pos, idx) for pos, idx in self.buffer if idx is None or idx >= cutoff]

    def annotate(
        self,
        frame: np.ndarray,
        center: tuple[int, int] | None = None,
        frame_idx: int | None = None,
    ) -> None:
        if center is not None:
            self.buffer.append((center, frame_idx))
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)

        self._prune_old(frame_idx)

        for i, (pos, _) in enumerate(self.buffer):
            color = self.color_palette.by_idx(i).as_bgr()
            radius = self.interpolate_radius(i, len(self.buffer))
            cv2.circle(frame, pos, radius, color, self.thickness)


def visualize_run(
    run_dir: Path,
    clip_filter: str | None,
    input_dir: Path,
    output_scale: float,
    config: dict,
    render_2d_birdseye: bool = False,
    render_2d_voronoi: bool = False,
):
    """
    Visualize all clips (or a specific clip) from a run directory.

    Automatically detects clips from Pass 1 JSON files.

    Args:
        run_dir: Run output directory (output/run_DDMMYY_HHMMSS)
        clip_filter: Optional clip name to filter (None = all clips)
        input_dir: Directory with original video files
        output_scale: Output video scale (0.5 = half size)
        config: Configuration dictionary
        render_2d_birdseye: Whether to render 2D birdseye view
        render_2d_voronoi: Whether to render 2D voronoi overlay
    """
    run_dir = Path(run_dir)
    pass1_dir = run_dir / "pass1_raw"

    # Get all Pass 1 JSON files
    pass1_files = list(pass1_dir.glob("*.json"))
    if not pass1_files:
        print(f"ERROR: No Pass 1 JSON files found in {pass1_dir}")
        return

    # Extract clip names from Pass 1 JSONs
    clips_to_visualize = []
    for pass1_file in pass1_files:
        with open(pass1_file, 'r', encoding='utf-8') as f:
            pass1_data = json.load(f)
        clip_name = pass1_data.get("clip_name")
        if not clip_name:
            print(f"WARNING: No clip_name in {pass1_file.name}, skipping")
            continue

        # Apply filter if specified
        if clip_filter is None or Path(clip_name).stem == Path(clip_filter).stem:
            clips_to_visualize.append(clip_name)

    if not clips_to_visualize:
        if clip_filter:
            print(f"ERROR: No clips matched filter: {clip_filter}")
        else:
            print(f"ERROR: No clips found in {pass1_dir}")
        return

    print(f"\nFound {len(clips_to_visualize)} clip(s) to visualize:")
    for clip_name in clips_to_visualize:
        print(f"  - {clip_name}")

    # Visualize each clip
    for clip_name in clips_to_visualize:
        print(f"\n{'='*60}")
        print(f"Visualizing: {clip_name}")
        print(f"{'='*60}")
        visualize_clip(
            run_dir=run_dir,
            clip_name=clip_name,
            input_dir=input_dir,
            output_scale=output_scale,
            config=config,
            render_2d_birdseye=render_2d_birdseye,
            render_2d_voronoi=render_2d_voronoi,
        )

    print(f"\n{'='*60}")
    print(f"All visualizations complete!")
    print(f"{'='*60}")


def visualize_clip(
    run_dir: Path,
    clip_name: str,
    input_dir: Path,
    output_scale: float,
    config: dict,
    render_2d_birdseye: bool = False,
    render_2d_voronoi: bool = False,
):
    """
    Generate annotated video from Pass 3 output.

    Args:
        run_dir: Run output directory (output/run_DDMMYY_HHMMSS)
        clip_name: Video clip filename (e.g., "clip_001.mp4")
        input_dir: Directory with original video files
        output_scale: Output video scale (0.5 = half size)
        config: Configuration dictionary
        render_2d_birdseye: Whether to render 2D birdseye view
        render_2d_voronoi: Whether to render 2D voronoi overlay
    """
    run_dir = Path(run_dir)
    input_dir = Path(input_dir)

    # Find original video file
    clip_path = input_dir / clip_name
    if not clip_path.exists():
        # Try with different extensions
        stem = Path(clip_name).stem
        candidates = list(input_dir.glob(f"{stem}.*"))
        if candidates:
            clip_path = candidates[0]
        else:
            print(f"ERROR: Video file not found: {clip_path}")
            return

    # Load Pass 1 JSON (bboxes per track)
    pass1_dir = run_dir / "pass1_raw"
    pass1_file = pass1_dir / f"{Path(clip_name).stem}.json"
    if not pass1_file.exists():
        print(f"ERROR: Pass 1 file not found: {pass1_file}")
        return

    with open(pass1_file, 'r', encoding='utf-8') as f:
        pass1_data = json.load(f)

    # Load ball positions from Pass 1 JSON
    ball_data = pass1_data.get("ball", {})
    ball_frames = ball_data.get("frames", [])
    ball_positions = ball_data.get("positions", [])
    ball_by_frame: dict[int, tuple[int, int]] = {}
    for frame_idx, pos in zip(ball_frames, ball_positions):
        if pos and len(pos) == 2:
            ball_by_frame[int(frame_idx)] = (int(pos[0]), int(pos[1]))

    # Load Pass 2 JSON (divergence markers)
    pass2_dir = run_dir / "pass2_identity"
    pass2_file = pass2_dir / f"{Path(clip_name).stem}_fragments.json"
    divergence_markers = {}
    pass2_data = None
    if pass2_file.exists():
        with open(pass2_file, 'r', encoding='utf-8') as f:
            pass2_data = json.load(f)
        print(f"  Pass 2: {len(pass2_data.get('fragments', []))} fragments")
        # Build divergence marker lookup: frame_idx → split_reason
        for fragment in pass2_data.get("fragments", []):
            split_reason = fragment.get("split_reason")
            if split_reason and "frame_idx" in split_reason:
                frame_idx = split_reason["frame_idx"]
                if frame_idx not in divergence_markers:
                    divergence_markers[frame_idx] = []
                divergence_markers[frame_idx].append(split_reason)
    else:
        print(f"  WARNING: Pass 2 file not found: {pass2_file}")

    # Load Pass 3 JSON (team/jersey assignments)
    pass3_dir = run_dir / "pass3_final"
    pass3_file = pass3_dir / f"{Path(clip_name).stem}.json"
    name_map = config.get("jersey", {}).get("name_map", {}) if config else {}
    visualize_cfg = config.get("visualize", {}) if config else {}
    debug_mode = bool(visualize_cfg.get("debug", False))
    fragment_identity_map: dict[str, dict[str, object]] = {}  # fragment_id → {team, jersey, confidence}
    if pass3_file.exists():
        with open(pass3_file, 'r', encoding='utf-8') as f:
            pass3_data = json.load(f)
        print(f"  Pass 3: {len(pass3_data.get('identities', []))} identities")

        # Build identity map: fragment_id → {team, jersey, confidence}
        # Pass 3 outputs ONE identity per fragment (not one per player)
        identities = pass3_data.get("identities", [])
        for identity in identities:
            frag_id = identity.get("fragment_id")  # Single fragment ID
            if frag_id:
                team = identity.get("team", "unknown")
                jersey = identity.get("assigned_jersey_number", identity.get("jersey_number"))
                confidence = identity.get("confidence", 0.0)
                fragment_identity_map[frag_id] = {
                    "team": team,
                    "jersey": jersey,
                    "confidence": confidence,
                    "assignment_reason": identity.get("assignment_reason"),
                    "label_source": identity.get("label_source"),
                    "team_confidence": identity.get("team_confidence"),
                }

        print(f"  Fragment->Identity mappings: {len(fragment_identity_map)}")

        # Count team assignments
        team_breakdown = {"team_a": 0, "team_b": 0, "unknown": 0}
        jersey_breakdown = {}
        for identity in fragment_identity_map.values():
            team = identity.get("team", "unknown")
            jersey = identity.get("jersey")
            team_breakdown[team] = team_breakdown.get(team, 0) + 1
            if jersey is not None:
                jersey_breakdown[jersey] = jersey_breakdown.get(jersey, 0) + 1
        print(f"  Fragments by team: {team_breakdown}")
        if jersey_breakdown:
            print(f"  Fragments by jersey: {jersey_breakdown}")
    else:
        print(f"  WARNING: Pass 3 file not found: {pass3_file}")

    # Build fragment lookup: (original_track_id, frame_idx) → fragment_id
    track_frame_to_fragment = {}
    bridged_track_frame_to_fragment = {}
    preanchor_backfill_map = {}
    jersey_persistence_map = {}
    track_frame_presence = {
        str(track_id): set(int(frame_idx) for frame_idx in track_data.get("frames", []))
        for track_id, track_data in pass1_data.get("tracks", {}).items()
    }
    if pass2_data:
        for fragment in pass2_data.get("fragments", []):
            frag_id = fragment["fragment_id"]
            track_id = str(fragment["original_track_id"])
            start_frame = fragment["start_frame"]
            end_frame = fragment["end_frame"]
            for frame_idx in range(start_frame, end_frame + 1):
                track_frame_to_fragment[(track_id, frame_idx)] = frag_id

        micro_gap_bridge_enabled = bool(visualize_cfg.get("micro_gap_bridge_enabled", True))
        micro_gap_max_frames = int(visualize_cfg.get("micro_gap_max_frames", 5))
        if micro_gap_bridge_enabled:
            bridged_track_frame_to_fragment = _bridge_micro_fragment_gaps(
                pass2_data.get("fragments", []),
                fragment_identity_map,
                max_gap_frames=micro_gap_max_frames,
            )

        preanchor_backfill_enabled = bool(visualize_cfg.get("preanchor_backfill_enabled", True))
        if preanchor_backfill_enabled:
            preanchor_backfill_map = _build_pre_anchor_identity_backfill(
                pass2_data.get("fragments", []),
                fragment_identity_map,
                max_anchor_gap_frames=int(visualize_cfg.get("preanchor_max_gap_frames", 5)),
                max_current_fragment_frames=int(visualize_cfg.get("preanchor_max_current_fragment_frames", 45)),
                backfill_tail_frames=int(visualize_cfg.get("preanchor_backfill_tail_frames", 30)),
                min_next_confidence=float(visualize_cfg.get("preanchor_min_next_confidence", 12.0)),
            )

        jersey_persistence_enabled = bool(visualize_cfg.get("track_jersey_persistence_enabled", True))
        if jersey_persistence_enabled:
            jersey_persistence_map = _build_track_jersey_persistence_map(
                pass2_data.get("fragments", []),
                fragment_identity_map,
                max_anchor_gap_frames=int(visualize_cfg.get("track_jersey_persistence_max_gap_frames", 8)),
                block_min_confidence=float(visualize_cfg.get("track_jersey_persistence_block_min_confidence", 4.0)),
                block_requires_assigned_reason=bool(visualize_cfg.get("track_jersey_persistence_block_requires_assigned_reason", True)),
                block_min_conf_ratio_to_source=float(visualize_cfg.get("track_jersey_persistence_block_min_conf_ratio_to_source", 0.6)),
                track_frame_presence=track_frame_presence,
            )

        print(f"  Track->Fragment mappings: {len(track_frame_to_fragment)} frame entries")
        if bridged_track_frame_to_fragment:
            print(f"  Micro-gap bridge mappings: {len(bridged_track_frame_to_fragment)} frame entries")
        if preanchor_backfill_map:
            print(f"  Pre-anchor backfill mappings: {len(preanchor_backfill_map)} frame entries")
        if jersey_persistence_map:
            print(f"  Track jersey persistence mappings: {len(jersey_persistence_map)} frame entries")

    # Load homography for ghost bbox conversion (court → pixel)
    homography = create_homography_from_config(config)

    # Build frame-by-frame annotation data
    # Structure: {frame_idx: {track_id: {"bbox": [...], "team": "...", "jersey": ...}}}
    frame_annotations = {}

    tracks = pass1_data.get("tracks", {})

    # Debug: Track team/jersey assignment stats
    team_counts = {"team_a": 0, "team_b": 0, "unknown": 0}
    jersey_counts = {}

    for track_id, track_data in tracks.items():
        frames = track_data.get("frames", [])
        bboxes = track_data.get("bboxes", [])
        jersey_decisions = track_data.get("jersey_decisions", [])
        for i, frame_idx in enumerate(frames):
            bbox = bboxes[i]

            # Lookup fragment and identity
            frag_id = track_frame_to_fragment.get((str(track_id), frame_idx))
            override_frag_id = preanchor_backfill_map.get((str(track_id), int(frame_idx)))
            if override_frag_id is not None:
                frag_id = override_frag_id
            persistence_frag_id = jersey_persistence_map.get((str(track_id), int(frame_idx)))
            if persistence_frag_id is not None:
                frag_id = persistence_frag_id
            if frag_id is None:
                frag_id = bridged_track_frame_to_fragment.get((str(track_id), int(frame_idx)))
            team = "unknown"
            jersey = None
            jersey_confidence = 0.0
            if frag_id and frag_id in fragment_identity_map:
                identity = fragment_identity_map[frag_id]
                team = identity.get("team", "unknown")
                jersey = identity.get("jersey")
                jersey_confidence = float(identity.get("confidence", 0.0))
                label_source = identity.get("label_source")
                team_confidence = identity.get("team_confidence")
            else:
                label_source = None
                team_confidence = None

            # Get raw jersey detection from Pass 1 (per-frame)
            raw_jersey_id = None
            raw_jersey_conf = 0.0
            if i < len(jersey_decisions):
                decision = jersey_decisions[i]
                if decision and len(decision) == 2:
                    raw_jersey_id, raw_jersey_conf = decision

            # Track stats
            team_counts[team] = team_counts.get(team, 0) + 1
            if jersey is not None:
                jersey_counts[jersey] = jersey_counts.get(jersey, 0) + 1

            if frame_idx not in frame_annotations:
                frame_annotations[frame_idx] = {}

            frame_annotations[frame_idx][track_id] = {
                "bbox": bbox,
                "team": team,
                "jersey": jersey,
                "jersey_confidence": jersey_confidence,
                "fragment_id": frag_id,
                "assignment_reason": fragment_identity_map.get(frag_id, {}).get("assignment_reason") if frag_id else None,
                "raw_jersey_id": raw_jersey_id,
                "raw_jersey_conf": raw_jersey_conf,
                "label_source": label_source,
                "team_confidence": team_confidence,
            }

    # Process ghost fragments (Pass 2 only - no Pass 1 tracks)
    # Ghosts maintain player count invariant when real players are occluded
    ghost_count = 0
    if pass2_data and homography:
        for fragment in pass2_data.get("fragments", []):
            if not fragment.get("is_ghost", False):
                continue  # Skip real fragments

            ghost_count += 1
            track_id = str(fragment["original_track_id"])  # Ensure string for consistency
            frag_id = fragment["fragment_id"]
            start_frame = fragment["start_frame"]
            end_frame = fragment["end_frame"]
            positions = fragment.get("spatial_footprint", {}).get("court_positions", [])

            # Get ghost identity from fragment_identity_map
            team = "unknown"
            jersey = None
            jersey_confidence = 0.0
            if frag_id in fragment_identity_map:
                identity = fragment_identity_map[frag_id]
                team = identity.get("team", "unknown")
                jersey = identity.get("jersey")
                jersey_confidence = float(identity.get("confidence", 0.0))

            # Use pixel bboxes directly from ghost fragment (Pass 1 resolution)
            pixel_bboxes = fragment.get("pixel_bboxes", [])

            for i in range(len(pixel_bboxes)):
                frame_idx = start_frame + i
                if frame_idx > end_frame:
                    break

                bbox = pixel_bboxes[i]

                if frame_idx not in frame_annotations:
                    frame_annotations[frame_idx] = {}

                # Mark as ghost for visualization (will be rendered as T3/estimated)
                frame_annotations[frame_idx][track_id] = {
                    "bbox": bbox,
                    "team": team,
                    "jersey": jersey,
                    "jersey_confidence": jersey_confidence,
                    "fragment_id": frag_id,
                    "is_ghost": True,  # Mark as ghost for T3 rendering
                    "raw_jersey_id": None,
                    "raw_jersey_conf": 0.0,
                    "label_source": "ghost",
                    "team_confidence": 0.0,
                }

                # Update stats
                team_counts[team] = team_counts.get(team, 0) + 1
                if jersey is not None:
                    jersey_counts[jersey] = jersey_counts.get(jersey, 0) + 1

    if ghost_count > 0:
        print(f"  Ghost fragments processed: {ghost_count}")
        # Debug: Check which frames have ghosts
        ghost_frames = [frame_idx for frame_idx, anns in frame_annotations.items()
                       if any(ann.get("is_ghost", False) for ann in anns.values())]
        if ghost_frames:
            print(f"  Frames with ghosts: {ghost_frames[:10]}")  # Show first 10

    print(f"  Team assignments: {team_counts}")
    print(f"  Jersey assignments: {jersey_counts}")

    # Open video and create output
    reader = VideoReader(str(clip_path))
    fps = reader.fps
    output_width = int(reader.width * output_scale)
    output_height = int(reader.height * output_scale)

    # Create output directory
    pass3_dir = run_dir / "pass3_final"
    pass3_dir.mkdir(parents=True, exist_ok=True)
    output_path = pass3_dir / f"{Path(clip_name).stem}_annotated.mp4"

    writer = VideoWriter(
        str(output_path),
        fps=fps,
        width=output_width,
        height=output_height,
        codec="h264",
    )

    print(f"\nGenerating annotated video: {output_path}")
    print(f"  Input: {clip_path}")
    print(f"  Output size: {output_width}x{output_height} @ {fps} FPS")
    print(f"  Tracks: {len(tracks)}")
    print(f"  Divergence markers: {len(divergence_markers)}")

    # Process frames
    trail_radius = max(2, int(9 * output_scale))
    trail_thickness = max(1, int(2 * output_scale))
    ball_annotator = BallAnnotator(fps=fps, radius=trail_radius, thickness=trail_thickness)
    fragment_debug_shown: set[str] = set()
    for frame_idx, frame in tqdm(reader.frames(), total=reader.total_frames, desc="Rendering"):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize frame
        frame_bgr = cv2.resize(frame_bgr, (output_width, output_height))

        # Draw divergence markers (if any at this frame)
        if frame_idx in divergence_markers:
            for split_reason in divergence_markers[frame_idx]:
                reason_type = split_reason.get("reason", "unknown")
                metric = split_reason.get("metric", 0.0)
                # Draw warning banner at top
                text = f"DIVERGENCE: {reason_type} (metric={metric:.2f})"
                cv2.rectangle(frame_bgr, (0, 0), (output_width, 50), (0, 0, 255), -1)
                cv2.putText(
                    frame_bgr,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        # Draw ball trail (from Pass 1 ball positions)
        if frame_idx in ball_by_frame:
            bx, by = ball_by_frame[frame_idx]
            ball_annotator.annotate(
                frame_bgr,
                center=(int(bx * output_scale), int(by * output_scale)),
                frame_idx=frame_idx,
            )
        else:
            ball_annotator.annotate(frame_bgr, center=None, frame_idx=frame_idx)

        # Draw player annotations
        annotations = frame_annotations.get(frame_idx, {})
        annotations = _dedupe_overlapping_annotations(annotations, iou_threshold=0.55)
        if bool(visualize_cfg.get("jersey_single_owner_enabled", True)):
            annotations = _enforce_frame_jersey_exclusivity(annotations)
        rendered_jerseys: set[int] = set()
        for track_id, data in annotations.items():
            bbox = data["bbox"]
            team = data["team"]
            jersey = data["jersey"]
            jersey_confidence = float(data.get("jersey_confidence", 0.0))
            frag_id = data.get("fragment_id")
            raw_jersey_id = data.get("raw_jersey_id")
            raw_jersey_conf = float(data.get("raw_jersey_conf", 0.0))
            is_ghost = data.get("is_ghost", False)

            # DEFENSIVE: Skip unreasonably large bboxes (false positives from tracker)
            # Player bboxes should be < 50% of frame width/height
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            max_allowed_width = reader.width * 0.5
            max_allowed_height = reader.height * 0.5
            if bbox_width > max_allowed_width or bbox_height > max_allowed_height:
                # Skip rendering this corrupted bbox
                continue

            # Scale bbox to output size
            x1 = int(bbox[0] * output_scale)
            y1 = int(bbox[1] * output_scale)
            x2 = int(bbox[2] * output_scale)
            y2 = int(bbox[3] * output_scale)

            # Get team color (same for real players and ghosts)
            color = TEAM_COLORS.get(team, TEAM_COLORS["unknown"])
            b, g, r = color
            luminance = 0.114 * b + 0.587 * g + 0.299 * r
            text_color = (255, 255, 255) if luminance < 128 else (0, 0, 0)

            # Draw ellipse around player (archive style) or dotted box for ghosts
            det = sv.Detections(
                xyxy=np.array([[x1, y1, x2, y2]]),
                class_id=np.array([0]),
            )
            if is_ghost:
                # Draw dotted rectangle for ghosts (T3/estimated)
                dash_length = 8
                for i in range(x1, x2, dash_length * 2):
                    cv2.line(frame_bgr, (i, y1), (min(i + dash_length, x2), y1), color, 2)
                    cv2.line(frame_bgr, (i, y2), (min(i + dash_length, x2), y2), color, 2)
                for i in range(y1, y2, dash_length * 2):
                    cv2.line(frame_bgr, (x1, i), (x1, min(i + dash_length, y2)), color, 2)
                    cv2.line(frame_bgr, (x2, i), (x2, min(i + dash_length, y2)), color, 2)
            else:
                # Draw ellipse for real players
                ellipse_color = sv.Color(color[2], color[1], color[0])
                frame_bgr = sv.EllipseAnnotator(color=ellipse_color, thickness=2).annotate(frame_bgr, det)

            # Build label (always show track id; jersey overlay is Pass 3 assigned ONLY)
            label_parts = [f"T{track_id}"]
            if is_ghost:
                label_parts.append("(EST)")
            if frag_id:
                frag_suffix = frag_id.replace("frag_", "F")
                label_parts.append(f": {frag_suffix}")
            if jersey is not None:
                jersey_int = int(jersey)
                if jersey_int not in rendered_jerseys:
                    rendered_jerseys.add(jersey_int)
                    jersey_name = name_map.get(jersey_int)
                    if jersey_name is None:
                        jersey_name = name_map.get(str(jersey_int))
                    if debug_mode and frag_id and frag_id not in fragment_debug_shown:
                        label_parts.append(f"(assigned={jersey_int}, conf={jersey_confidence:.2f})")
                        fragment_debug_shown.add(frag_id)
            elif debug_mode:
                label_parts.append("?")

            label = " ".join(label_parts)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.35, 0.5 * output_scale)
            thickness = max(1, int(2 * output_scale))

            # Draw raw jersey detection (per-frame from Pass 1) on right side of bbox
            if raw_jersey_id is not None and raw_jersey_conf > 0.0:
                jersey_detect_label = f"J:{int(raw_jersey_id)} ({raw_jersey_conf:.2f})"
                (jd_width, jd_height), _ = cv2.getTextSize(
                    jersey_detect_label, font, font_scale * 0.8, thickness
                )

                # Position on right side of bbox
                jd_x = x2 + 5
                jd_y = y1 + jd_height + 5

                # Background color: green if high conf (>0.5), yellow if medium (>0.3), red if low
                if raw_jersey_conf >= 0.5:
                    jd_bg_color = (0, 200, 0)  # Green
                elif raw_jersey_conf >= 0.3:
                    jd_bg_color = (0, 200, 200)  # Yellow
                else:
                    jd_bg_color = (0, 0, 200)  # Red

                cv2.rectangle(
                    frame_bgr,
                    (jd_x, jd_y - jd_height - 4),
                    (jd_x + jd_width + 4, jd_y + 2),
                    jd_bg_color,
                    -1,
                )
                cv2.putText(
                    frame_bgr,
                    jersey_detect_label,
                    (jd_x + 2, jd_y),
                    font,
                    font_scale * 0.8,
                    (255, 255, 255),
                    thickness,
                )

            if label:
                # Draw top label background (track + fragment)
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                cv2.rectangle(
                    frame_bgr,
                    (x1, y1 - text_height - 8),
                    (x1 + text_width + 4, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    frame_bgr,
                    label,
                    (x1 + 2, y1 - 5),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                )

            # Draw jersey number/name near feet (bottom of bbox)
            if jersey is not None:
                jersey_int = int(jersey)
                if jersey_int in rendered_jerseys:
                    jersey_name = name_map.get(jersey_int)
                    if jersey_name is None:
                        jersey_name = name_map.get(str(jersey_int))
                    jersey_label = f"#{jersey_int}"
                    if jersey_name is not None:
                        jersey_label = f"{jersey_label} : {jersey_name}"

                    (j_text_w, j_text_h), j_base = cv2.getTextSize(
                        jersey_label, font, font_scale, thickness
                    )
                    y_bottom = min(output_height - 2, y2 + j_text_h + 6)
                    cv2.rectangle(
                        frame_bgr,
                        (x1, y_bottom - j_text_h - 6),
                        (x1 + j_text_w + 4, y_bottom),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame_bgr,
                        jersey_label,
                        (x1 + 2, y_bottom - 4),
                        font,
                        font_scale,
                        text_color,
                        thickness,
                    )

        # Draw frame counter with black background
        frame_text = f"Frame: {frame_idx}"
        frame_font_scale = max(0.5, 0.7 * output_scale)
        frame_thickness = max(1, int(2 * output_scale))
        (frame_text_width, frame_text_height), frame_baseline = cv2.getTextSize(
            frame_text, cv2.FONT_HERSHEY_SIMPLEX, frame_font_scale, frame_thickness
        )

        # Black background
        cv2.rectangle(
            frame_bgr,
            (5, 68),
            (15 + frame_text_width, 90 + frame_text_height),
            (0, 0, 0),
            -1,
        )

        # White text
        cv2.putText(
            frame_bgr,
            frame_text,
            (10, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            frame_font_scale,
            (255, 255, 255),
            frame_thickness,
        )

        # Draw player count ticker (top-right): "Tracked: X | Estimated: Y"
        # Count unique PLAYERS from current frame annotations
        tracked_players = set()
        estimated_players = set()
        for track_id, data in annotations.items():
            if data.get("is_ghost", False):
                estimated_players.add(track_id)
            else:
                tracked_players.add(track_id)

        tracked_count = len(tracked_players)
        estimated_count = len(estimated_players)
        total_count = tracked_count + estimated_count
        ticker_text = f"Tracked: {tracked_count} | Estimated: {estimated_count} | Total: {total_count}"
        ticker_y = 84  # Same Y as frame counter (line 1181)
        ticker_x = output_width - 550  # Top-right (adjusted for longer text)

        # Semi-transparent background
        (text_w, text_h), _ = cv2.getTextSize(ticker_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        overlay = frame_bgr.copy()
        cv2.rectangle(
            overlay,
            (ticker_x - 5, ticker_y - text_h - 5),
            (ticker_x + text_w + 5, ticker_y + 5),
            (0, 0, 0),
            -1,
        )
        frame_bgr = cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0)

        # Draw text
        cv2.putText(
            frame_bgr,
            ticker_text,
            (ticker_x, ticker_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Convert back to RGB for writer
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Write frame
        writer.write_frame(frame_rgb)

    writer.close()
    print(f"\nVisualization saved: {output_path}")

    # Render 2D pitch views (if requested)
    if render_2d_birdseye or render_2d_voronoi:
        print(f"\n{'='*60}")
        print("Rendering 2D Pitch Views")
        print(f"{'='*60}")

        # Get pass3_data if available
        pass3_data_for_render = None
        if pass3_file.exists():
            with open(pass3_file, 'r', encoding='utf-8') as f:
                pass3_data_for_render = json.load(f)

        _render_2d_pitch_views(
            run_dir=run_dir,
            clip_name=clip_name,
            pass1_data=pass1_data,
            pass2_data=pass2_data,
            pass3_data=pass3_data_for_render,
            config=config,
            render_birdseye=render_2d_birdseye,
            render_voronoi=render_2d_voronoi,
        )


def _render_2d_pitch_views(
    run_dir: Path,
    clip_name: str,
    pass1_data: dict,
    pass2_data: dict | None,
    pass3_data: dict | None,
    config: dict,
    render_birdseye: bool,
    render_voronoi: bool,
):
    """
    Render 2D pitch views (birdseye and/or voronoi).

    Args:
        run_dir: Run output directory
        clip_name: Clip name (e.g., "clip_001.mp4")
        pass1_data: Pass 1 JSON data
        pass2_data: Pass 2 JSON data (or None if not available)
        pass3_data: Pass 3 JSON data (or None if not available)
        config: Configuration dictionary
        render_birdseye: Whether to render birdseye view
        render_voronoi: Whether to render voronoi overlay
    """
    from src.viz.pitch_2d_positions import export_mp4, PitchRenderer2D

    # Create homography
    homography = create_homography_from_config(config)
    if homography is None:
        print("ERROR: Homography not configured, skipping 2D pitch rendering")
        return

    if pass2_data is None:
        print("ERROR: Pass 2 data not available, skipping 2D pitch rendering")
        return

    if pass3_data is None:
        print("WARNING: Pass 3 data not available, rendering with unknown teams")
        # Create empty Pass 3 data
        pass3_data = {"identities": []}

    # Get frame range from Pass 1
    fps = pass1_data.get("fps", 30.0)
    total_frames = pass1_data.get("total_frames", 0)

    if total_frames == 0:
        print("ERROR: No frames found in Pass 1 data")
        return

    clip_stem = Path(clip_name).stem
    output_dir = run_dir / "pass3_final"

    # Render birdseye view (no Voronoi)
    if render_birdseye:
        print(f"\n  Rendering birdseye view (0-{total_frames-1} frames @ {fps} FPS)...")
        output_path = output_dir / f"{clip_stem}_pitch_birdseye.mp4"
        export_mp4(
            output_path=output_path,
            start_frame=0,
            end_frame=total_frames - 1,
            pass1_data=pass1_data,
            pass2_data=pass2_data,
            pass3_data=pass3_data,
            homography=homography,
            fps=fps,
            draw_voronoi=False,
        )
        print(f"  ✓ Saved: {output_path}")

    # Render voronoi view
    if render_voronoi:
        print(f"\n  Rendering voronoi view (0-{total_frames-1} frames @ {fps} FPS)...")
        output_path = output_dir / f"{clip_stem}_pitch_voronoi.mp4"
        export_mp4(
            output_path=output_path,
            start_frame=0,
            end_frame=total_frames - 1,
            pass1_data=pass1_data,
            pass2_data=pass2_data,
            pass3_data=pass3_data,
            homography=homography,
            fps=fps,
            draw_voronoi=True,
        )
        print(f"  ✓ Saved: {output_path}")
