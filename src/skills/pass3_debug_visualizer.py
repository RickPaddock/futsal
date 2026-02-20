"""
Pass 3 Debug Video Renderer.

Renders a confirmation-layer video from committed Pass 3 identities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from ..core.data_models import Detection, Pass1Output, Pass2COutput, Pass3COutput, CommittedIdentity
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


def _resolve_team_palette(pass3_output: Pass3COutput) -> Dict[TeamID, Tuple[int, int, int]]:
    """
    Resolve team colors with bibbed team in ORANGE and the other team in BLACK.

    Uses solver diagnostic `bibbed_team_evidence` when available.
    """
    orange = (0, 165, 255)  # BGR
    black = (0, 0, 0)

    # Prefer identity evidence: team with more resolved jersey numbers is likely bibbed team.
    team_jersey_counts: Dict[TeamID, int] = {TeamID.TEAM_A: 0, TeamID.TEAM_B: 0}
    for identity in pass3_output.identities:
        if identity.jersey_number is None:
            continue
        if identity.team in team_jersey_counts:
            team_jersey_counts[identity.team] += 1

    if team_jersey_counts[TeamID.TEAM_A] != team_jersey_counts[TeamID.TEAM_B]:
        bibbed_team = TeamID.TEAM_A if team_jersey_counts[TeamID.TEAM_A] > team_jersey_counts[TeamID.TEAM_B] else TeamID.TEAM_B
        other_team = TeamID.TEAM_B if bibbed_team == TeamID.TEAM_A else TeamID.TEAM_A
        return {bibbed_team: orange, other_team: black}

    solver_log = pass3_output.solver_log or {}
    bibbed = solver_log.get("bibbed_team_evidence")

    if bibbed == TeamID.TEAM_A.value:
        return {TeamID.TEAM_A: orange, TeamID.TEAM_B: black}
    if bibbed == TeamID.TEAM_B.value:
        return {TeamID.TEAM_A: black, TeamID.TEAM_B: orange}

    # Fallback when evidence is ambiguous: deterministic mapping.
    return {TeamID.TEAM_A: orange, TeamID.TEAM_B: black}


def _identity_label(identity: CommittedIdentity) -> str:
    return identity.player_id


def _draw_pass3_overlay_frame(
    frame: np.ndarray,
    frame_idx: int,
    annotations: List[Dict[str, object]],
    team_palette: Dict[TeamID, Tuple[int, int, int]],
) -> np.ndarray:
    overlay = frame.copy()

    team_a_count = 0
    team_b_count = 0
    committed_ghost_count = 0
    real_player_count = 0

    for ann in annotations:
        bbox = ann["bbox"]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        team = ann["team"]
        color = ann.get("color", team_palette.get(team, (0, 0, 0)))
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

        label = ann["label"]
        jersey_number = ann.get("jersey_number")
        if is_ghost:
            label = f"GHOST {label}"

        # Top label: player identity (white text on black background for readability).
        _draw_text_with_bg(overlay, str(label), (x1, max(18, y1 - 8)), 0.48, (255, 255, 255), 2)

        # Bottom label: large jersey number.
        jersey_text = f"#{jersey_number}" if jersey_number is not None else "#?"
        _draw_text_with_bg(
            overlay,
            jersey_text,
            (x1, min(overlay.shape[0] - 8, y2 + 28)),
            0.82,
            (255, 255, 255),
            2,
        )

    title = "PASS 3 DEBUG: COMMITTED IDENTITY"
    lines = [
        "bibbed team=ORANGE, other team=BLACK | dashed boxes=ghosts",
        "top label=player_id, bottom label=jersey number",
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
    pass1_output = load_json(Path(pass1_output_path), Pass1Output)
    pass2c_output = load_json(Path(pass2c_output_path), Pass2COutput)
    pass3_output = load_json(Path(pass3_output_path), Pass3COutput)
    team_palette = _resolve_team_palette(pass3_output)

    detections_by_id: Dict[str, Detection] = {
        detection.detection_id: detection for detection in pass1_output.detections
    }
    identity_by_fragment: Dict[str, CommittedIdentity] = {
        identity.fragment_id: identity for identity in pass3_output.identities
    }

    frame_annotations: Dict[int, List[Dict[str, object]]] = {}

    for fragment in pass2c_output.fragments:
        identity = identity_by_fragment.get(fragment.fragment_id)
        is_ghost = bool(getattr(fragment, "is_ghost", False))

        if identity is None:
            continue

        label = _identity_label(identity)

        if is_ghost:
            bbox = getattr(fragment, "ghost_last_known_bbox", None)
            if bbox is None:
                continue
            for frame_idx in range(fragment.start_frame, fragment.end_frame + 1):
                frame_annotations.setdefault(frame_idx, []).append(
                    {
                        "bbox": bbox,
                        "team": identity.team,
                        "label": label,
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
                debug_frame = _draw_pass3_overlay_frame(frame, frame_idx, anns, team_palette)
                writer.write(debug_frame)
                pbar.update(1)

        logger.info(f"Pass 3 debug video written: {debug_video_path}")

    finally:
        if writer is not None:
            writer.release()
        reader.close()
