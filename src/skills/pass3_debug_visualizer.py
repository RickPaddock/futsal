"""
Pass 3 Debug Video Renderer.

Renders a confirmation-layer video from committed Pass 3 identities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
from tqdm import tqdm

from ..core import constants as const
from ..core.data_models import (
    CommittedIdentity,
    Detection,
    Pass1Output,
    Pass2COutput,
    Pass3COutput,
    ScoredFragment,
)
from ..core.types import TeamID
from ..utils.file_utils import load_json
from ..utils.logging_utils import get_logger
from ..utils.video_io import VideoReader

logger = get_logger("pass3_debug_visualizer")


def _draw_text_with_bg(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    pad_x = 4
    pad_y = 3
    x1 = max(0, x - pad_x)
    y1 = max(0, y - text_h - pad_y)
    x2 = min(img.shape[1] - 1, x + text_w + pad_x)
    y2 = min(img.shape[0] - 1, y + baseline + pad_y)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_dashed_rectangle(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    dash_length: int = 10,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2

    def draw_dashed_line(p1: Tuple[int, int], p2: Tuple[int, int]) -> None:
        dist = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if dist <= 0:
            return
        points: List[Tuple[int, int]] = []
        for i in range(0, dist, dash_length * 2):
            ratio = i / dist
            x = int((p1[0] * (1 - ratio) + p2[0] * ratio) + 0.5)
            y = int((p1[1] * (1 - ratio) + p2[1] * ratio) + 0.5)
            points.append((x, y))
        for i in range(0, len(points) - 1, 2):
            cv2.line(img, points[i], points[min(i + 1, len(points) - 1)], color, thickness)

    draw_dashed_line((x1, y1), (x2, y1))
    draw_dashed_line((x2, y1), (x2, y2))
    draw_dashed_line((x2, y2), (x1, y2))
    draw_dashed_line((x1, y2), (x1, y1))


def _resolve_team_palette(
    pass3_output: Pass3COutput,
) -> Tuple[Dict[TeamID, Tuple[int, int, int]], TeamID, TeamID, str]:
    """
    Resolve team colors for debug overlays.

    Contract for debug rendering:
    - bibbed team -> ORANGE
    - random-shirt team -> BLACK

    Prefer solver hint when available; otherwise infer bibbed side from compactness
    (tighter color cluster -> bibbed), then from fewer visible jersey numbers.
    """
    orange = (0, 165, 255)  # BGR
    black = (0, 0, 0)

    solver_log = dict(getattr(pass3_output, "solver_log", {}) or {})
    evidence = str(solver_log.get("bibbed_team_evidence", "")).strip().lower()

    bibbed_team: Optional[TeamID] = None
    source = "fallback_default"
    if evidence == TeamID.TEAM_A.value:
        bibbed_team = TeamID.TEAM_A
        source = "solver_log:bibbed_team_evidence"
    elif evidence == TeamID.TEAM_B.value:
        bibbed_team = TeamID.TEAM_B
        source = "solver_log:bibbed_team_evidence"

    if bibbed_team is None:
        compactness = solver_log.get("cluster_compactness", {})
        compact_a = compactness.get(TeamID.TEAM_A.value)
        compact_b = compactness.get(TeamID.TEAM_B.value)
        if isinstance(compact_a, (int, float)) and isinstance(compact_b, (int, float)):
            if float(compact_a) < float(compact_b):
                bibbed_team = TeamID.TEAM_A
                source = "cluster_compactness_heuristic"
            elif float(compact_b) < float(compact_a):
                bibbed_team = TeamID.TEAM_B
                source = "cluster_compactness_heuristic"

    if bibbed_team is None:
        team_jersey_counts: Dict[TeamID, int] = {TeamID.TEAM_A: 0, TeamID.TEAM_B: 0}
        for identity in pass3_output.identities:
            if identity.team not in team_jersey_counts:
                continue
            if identity.jersey_number is not None:
                team_jersey_counts[identity.team] += 1

        # Bibbed side often has fewer reliably visible jersey numbers.
        if team_jersey_counts[TeamID.TEAM_A] < team_jersey_counts[TeamID.TEAM_B]:
            bibbed_team = TeamID.TEAM_A
            source = "jersey_count_heuristic"
        elif team_jersey_counts[TeamID.TEAM_B] < team_jersey_counts[TeamID.TEAM_A]:
            bibbed_team = TeamID.TEAM_B
            source = "jersey_count_heuristic"

    if bibbed_team is None:
        # Deterministic fallback to avoid run-to-run palette drift.
        bibbed_team = TeamID.TEAM_B

    random_team = TeamID.TEAM_B if bibbed_team == TeamID.TEAM_A else TeamID.TEAM_A

    logger.info(
        f"Debug palette resolved: bibbed={bibbed_team.value} (orange), "
        f"random={random_team.value} (black), source={source}"
    )

    return ({bibbed_team: orange, random_team: black}, bibbed_team, random_team, source)


def _identity_label(identity: CommittedIdentity) -> str:
    return identity.player_id


def _coerce_int(value: object, fallback: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return fallback
    return fallback


def _interpolate_bbox(
    start_bbox: List[float],
    end_bbox: List[float],
    alpha: float,
) -> List[float]:
    alpha = max(0.0, min(1.0, float(alpha)))
    return [
        float(start_bbox[idx]) + (float(end_bbox[idx]) - float(start_bbox[idx])) * alpha
        for idx in range(4)
    ]


def _first_detection_bbox(fragment, detections_by_id: Dict[str, Detection]) -> Optional[List[float]]:
    for detection_id in getattr(fragment, "detection_ids", []) or []:
        detection = detections_by_id.get(detection_id)
        if detection is not None:
            return list(detection.bbox)
    return None


def _ghost_bbox_for_frame(
    fragment: ScoredFragment,
    frame_idx: int,
    ghost_window: Dict[str, Any],
    fragment_by_id: Dict[str, ScoredFragment],
    detections_by_id: Dict[str, Detection],
) -> Optional[List[float]]:
    start_bbox = cast(
        Optional[List[float]],
        getattr(fragment, "ghost_last_known_bbox", None) or getattr(fragment, "estimated_position", None),
    )
    if start_bbox is None:
        return None

    matched_reappearance_frame = ghost_window.get("matched_reappearance_frame")
    target_fragment_id = ghost_window.get("target_fragment_id")
    if matched_reappearance_frame is None or target_fragment_id is None:
        return list(start_bbox)

    target_fragment = fragment_by_id.get(str(target_fragment_id))
    if target_fragment is None:
        return list(start_bbox)

    target_bbox = _first_detection_bbox(target_fragment, detections_by_id)
    if target_bbox is None:
        return list(start_bbox)

    start_frame = _coerce_int(ghost_window.get("start_frame"), fragment.start_frame)
    matched_frame = _coerce_int(matched_reappearance_frame, start_frame)
    duration = max(1, matched_frame - start_frame)
    alpha = float(frame_idx - start_frame + 1) / float(duration)
    return _interpolate_bbox(list(start_bbox), target_bbox, alpha)


def _draw_pass3_overlay_frame(
    frame: np.ndarray,
    frame_idx: int,
    annotations: List[Dict[str, Any]],
    team_palette: Dict[TeamID, Tuple[int, int, int]],
    palette_line: str,
) -> np.ndarray:
    overlay = frame.copy()

    team_a_count = 0
    team_b_count = 0
    committed_ghost_count = 0
    real_player_count = 0

    for ann in annotations:
        bbox = cast(List[float], ann["bbox"])
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        team = cast(TeamID, ann["team"])
        color = cast(Tuple[int, int, int], ann.get("color") or team_palette.get(team, (0, 0, 0)))
        is_ghost = bool(ann["is_ghost"])
        is_retired_ghost = bool(ann.get("retired_ghost", False))

        if team == TeamID.TEAM_A:
            team_a_count += 1
        elif team == TeamID.TEAM_B:
            team_b_count += 1
        if is_ghost:
            if not is_retired_ghost:
                committed_ghost_count += 1
        else:
            real_player_count += 1

        if is_ghost:
            _draw_dashed_rectangle(overlay, (x1, y1), (x2, y2), color, 2, dash_length=10)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = str(ann["label"])
        jersey_number = ann.get("jersey_number") if isinstance(ann.get("jersey_number"), int) else None
        track_id = ann.get("track_id")
        fragment_id = str(ann.get("fragment_id", ""))
        if is_ghost:
            label = f"GHOST {label}"

        # Top label: player identity (white text on black background for readability).
        _draw_text_with_bg(overlay, str(label), (x1, max(18, y1 - 8)), 0.48, (255, 255, 255), 2)

        # Bottom edge label: track and fragment linkage for debugging split/jump behavior.
        track_fragment_text = f"T{track_id} {fragment_id}"
        _draw_text_with_bg(
            overlay,
            track_fragment_text,
            (x1, max(y1 + 16, y2 - 6)),
            0.42,
            (255, 255, 255),
            1,
        )

        # Bottom label: jersey number and player name (if known).
        if jersey_number is not None:
            name = const.PLAYER_NAME_BY_JERSEY.get(jersey_number)
            jersey_text = f"#{jersey_number} - {name}" if name else f"#{jersey_number}"
        else:
            jersey_text = None
        if jersey_text is not None:
            _draw_text_with_bg(
                overlay,
                jersey_text,
                (x1, min(overlay.shape[0] - 8, y2 + 22)),
                0.62,
                (255, 255, 255),
                2,
            )

    title = "PASS 3 DEBUG: COMMITTED IDENTITY"
    lines = [
        f"{palette_line} | dashed boxes=ghosts",
        "top label=player_id, bottom label=#number - name (if known)",
        "bbox bottom=track+fragment",
        f"frame={frame_idx} team_a={team_a_count} team_b={team_b_count} ghosts={committed_ghost_count}",
    ]

    counter_text = (
        f"Players: {real_player_count}  Ghosts: {committed_ghost_count}  "
        f"Total: {real_player_count + committed_ghost_count}"
    )
    _draw_text_with_bg(overlay, counter_text, (12, 24), 0.62, (255, 255, 255), 2)

    frame_h, frame_w = overlay.shape[:2]
    top = max(12, int(frame_h * 0.055))
    left = 12
    line_gap = 20
    title_y = top + 14
    first_line_y = title_y + line_gap
    footer_y = first_line_y + (len(lines) * line_gap) + 8
    panel_bottom = min(frame_h - 8, footer_y + 18)

    panel = overlay.copy()
    panel_top = max(6, top - 12)
    cv2.rectangle(panel, (6, panel_top), (frame_w - 6, panel_bottom), (0, 0, 0), -1)
    overlay = cv2.addWeighted(panel, 0.55, overlay, 0.45, 0)

    _draw_text_with_bg(overlay, title, (left, title_y), 0.52, (255, 255, 255), 2)
    for idx, line in enumerate(lines):
        y = first_line_y + idx * line_gap
        _draw_text_with_bg(overlay, f"- {line}", (left + 4, y), 0.46, (255, 255, 255), 1)

    return overlay


def render_pass3_debug_video_from_artifact(
    video_path: str,
    pass1_output_path: str,
    pass2c_output_path: str,
    pass3_output_path: str,
    debug_video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> None:
    pass1_output = load_json(pass1_output_path, Pass1Output)
    pass2c_output = load_json(pass2c_output_path, Pass2COutput)
    pass3_output = load_json(pass3_output_path, Pass3COutput)
    team_palette, bibbed_team, random_team, palette_source = _resolve_team_palette(pass3_output)
    palette_line = (
        f"bibbed={bibbed_team.value}(ORANGE), random={random_team.value}(BLACK), source={palette_source}"
    )

    detections_by_id: Dict[str, Detection] = {
        detection.detection_id: detection for detection in pass1_output.detections
    }
    identity_by_fragment: Dict[str, CommittedIdentity] = {
        identity.fragment_id: identity for identity in pass3_output.identities
    }
    fragment_by_id: Dict[str, ScoredFragment] = {
        fragment.fragment_id: fragment for fragment in pass2c_output.fragments
    }
    ghost_activity_windows = pass3_output.solver_log.get("ghost_active_windows") if isinstance(pass3_output.solver_log, dict) else None
    if not isinstance(ghost_activity_windows, dict):
        ghost_activity_windows = {}

    frame_annotations: Dict[int, List[Dict[str, Any]]] = {}

    for fragment in pass2c_output.fragments:
        identity = identity_by_fragment.get(fragment.fragment_id)
        is_ghost = bool(getattr(fragment, "is_ghost", False))

        if identity is None:
            continue

        label = _identity_label(identity)

        if is_ghost:
            ghost_window = ghost_activity_windows.get(fragment.fragment_id)
            if not isinstance(ghost_window, dict):
                continue
            ghost_start_frame = _coerce_int(ghost_window.get("start_frame"), fragment.start_frame)
            ghost_end_frame = _coerce_int(ghost_window.get("end_frame"), fragment.end_frame)
            if ghost_end_frame < ghost_start_frame:
                continue
            for frame_idx in range(ghost_start_frame, ghost_end_frame + 1):
                bbox = _ghost_bbox_for_frame(
                    fragment,
                    frame_idx,
                    ghost_window,
                    fragment_by_id,
                    detections_by_id,
                )
                if bbox is None:
                    continue
                frame_annotations.setdefault(frame_idx, []).append(
                    {
                        "bbox": bbox,
                        "team": identity.team,
                        "label": label,
                        "track_id": fragment.original_track_id,
                        "fragment_id": fragment.fragment_id,
                        "jersey_number": identity.jersey_number,
                        "is_ghost": True,
                    }
                )
            continue

        for detection_id in fragment.detection_ids:
            detection = detections_by_id.get(detection_id)
            if detection is None:
                continue
            frame_annotations.setdefault(detection.frame_idx, []).append(
                {
                    "bbox": detection.bbox,
                    "team": identity.team,
                    "label": label,
                    "track_id": fragment.original_track_id,
                    "fragment_id": fragment.fragment_id,
                    "jersey_number": identity.jersey_number,
                    "is_ghost": False,
                }
            )

    reader = VideoReader(video_path)
    writer = None

    try:
        fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
        if fourcc_fn is None:
            fourcc_fn = cv2.VideoWriter.fourcc

        writer = cv2.VideoWriter(
            debug_video_path,
            fourcc_fn(*"mp4v"),
            float(reader.fps),
            (int(reader.width), int(reader.height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open Pass 3 debug video writer: {debug_video_path}")

        artifact_start = pass1_output.processed_start_frame
        artifact_end_exclusive = pass1_output.processed_end_frame_exclusive
        if artifact_end_exclusive is None:
            artifact_end_exclusive = pass1_output.total_frames

        render_start = max(start_frame, artifact_start)
        requested_end = artifact_end_exclusive if end_frame is None else end_frame
        render_end_exclusive = min(requested_end, artifact_end_exclusive)

        if render_end_exclusive <= render_start:
            raise ValueError(
                f"Invalid render window: start={render_start}, end={render_end_exclusive}. "
                f"Artifact range is [{artifact_start}, {artifact_end_exclusive})."
            )

        total_render_frames = render_end_exclusive - render_start
        with tqdm(total=total_render_frames, desc="Pass 3 debug render", unit="frame") as pbar:
            for frame_idx, frame in reader.iter_frames():
                if frame_idx < render_start:
                    continue
                if frame_idx >= render_end_exclusive:
                    break

                anns = frame_annotations.get(frame_idx, [])
                debug_frame = _draw_pass3_overlay_frame(
                    frame,
                    frame_idx,
                    anns,
                    team_palette,
                    palette_line,
                )
                writer.write(debug_frame)
                pbar.update(1)

        logger.info(f"Pass 3 debug video written: {debug_video_path}")

    finally:
        if writer is not None:
            writer.release()
        reader.close()
