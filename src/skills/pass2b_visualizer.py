"""
Pass 2B: Fragment Quality Scoring Visualization

Renders debug video showing fragment quality scores and metrics.
"""

from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import logging

import cv2
import numpy as np
from tqdm import tqdm

from ..core.data_models import (
    Detection,
    Pass1Output,
    ScoredFragment,
    Pass2BOutput,
)
from ..core.types import FragmentQuality
from ..utils.file_utils import load_json
from ..utils.video_io import VideoReader

logger = logging.getLogger(__name__)


def _quality_color(quality: FragmentQuality) -> Tuple[int, int, int]:
    """Map quality to BGR color."""
    if quality == FragmentQuality.HIGH:
        return (0, 255, 0)  # Green
    elif quality == FragmentQuality.MEDIUM:
        return (0, 255, 255)  # Yellow
    elif quality == FragmentQuality.LOW:
        return (0, 0, 255)  # Red
    elif quality == FragmentQuality.GHOST:
        return (128, 128, 128)  # Gray
    else:
        return (255, 255, 255)  # White (unknown)


def _draw_text_with_bg(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    """Draw text with black background for visibility."""
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


def _draw_pass2b_debug_overlay_frame(
    frame: np.ndarray,
    frame_idx: int,
    frame_annotations: List[Tuple[Detection, ScoredFragment]],
    quality_distribution: Dict[str, int],
) -> np.ndarray:
    """Draw Pass 2B quality scoring overlays for one frame."""
    overlay = frame.copy()

    active_fragments: Dict[FragmentQuality, int] = {}

    for det, fragment in frame_annotations:
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]

        # Color by quality
        quality = fragment.quality
        color = _quality_color(quality)

        # Draw bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Label: fragment_id + quality
        quality_str = quality.value if isinstance(quality, FragmentQuality) else quality
        label = f"{fragment.fragment_id} [{quality_str.upper()}]"
        _draw_text_with_bg(overlay, label, (x1, max(18, y1 - 8)), 0.42, color, 1)

        # Quality score below
        score_text = f"score={fragment.quality_score:.2f}"
        _draw_text_with_bg(overlay, score_text, (x1, min(overlay.shape[0] - 6, y2 + 16)), 0.40, color, 1)

        # Metrics below score (if space)
        metrics_y = min(overlay.shape[0] - 6, y2 + 32)
        metrics_text = (
            f"conf={fragment.avg_confidence:.2f} "
            f"bbox={fragment.avg_bbox_stability:.2f} "
            f"jersey={fragment.jersey_consistency:.2f}"
        )
        _draw_text_with_bg(overlay, metrics_text, (x1, metrics_y), 0.35, color, 1)

        # Count by quality
        active_fragments[quality] = active_fragments.get(quality, 0) + 1

    # Overlay panel
    intent_lines = [
        "Pass 2B: Fragment Quality Scoring (metadata only, no identity decisions)",
        "Quality levels: HIGH (green) / MEDIUM (yellow) / LOW (red) / GHOST (gray)",
        "Metrics: confidence stability, bbox stability, jersey consistency, HSV consistency",
    ]

    # Runtime stats
    high_count = active_fragments.get(FragmentQuality.HIGH, 0)
    medium_count = active_fragments.get(FragmentQuality.MEDIUM, 0)
    low_count = active_fragments.get(FragmentQuality.LOW, 0)
    ghost_count = active_fragments.get(FragmentQuality.GHOST, 0)

    runtime_lines = [
        f"Frame {frame_idx}: HIGH={high_count} MEDIUM={medium_count} LOW={low_count} GHOST={ghost_count}",
        f"Total distribution: HIGH={quality_distribution.get('high', 0)} "
        f"MEDIUM={quality_distribution.get('medium', 0)} "
        f"LOW={quality_distribution.get('low', 0)}",
    ]

    title = "PASS 2B: QUALITY SCORING (ACTIVE)"
    all_lines = intent_lines + runtime_lines

    frame_h, frame_w = overlay.shape[:2]
    top = max(12, int(frame_h * 0.055))
    left = 12
    line_gap = 20
    title_y = top + 14
    first_rule_y = title_y + line_gap
    footer_y = first_rule_y + (len(all_lines) * line_gap) + 8
    panel_bottom = min(frame_h - 8, footer_y + 24)

    # Draw semi-transparent panel
    panel = overlay.copy()
    panel_top = max(6, top - 12)
    cv2.rectangle(panel, (6, panel_top), (frame_w - 6, panel_bottom), (0, 0, 0), -1)
    overlay = cv2.addWeighted(panel, 0.55, overlay, 0.45, 0)

    _draw_text_with_bg(overlay, title, (left, title_y), 0.50, (255, 255, 255), 1)
    for idx, line in enumerate(all_lines):
        y = first_rule_y + (idx * line_gap)
        _draw_text_with_bg(overlay, f"- {line}", (left + 4, y), 0.46, (255, 255, 255), 1)

    _draw_text_with_bg(
        overlay,
        "Box color = quality | Labels show: fragment_id [QUALITY] score metrics",
        (left, footer_y),
        0.45,
        (255, 255, 255),
        1,
    )
    _draw_text_with_bg(overlay, f"frame={frame_idx}", (left, footer_y + 18), 0.52, (255, 255, 255), 1)

    return overlay


def render_pass2b_debug_video_from_artifact(
    video_path: str,
    pass1_output_path: str,
    pass2b_output_path: str,
    debug_video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> None:
    """
    Render Pass 2B debug video showing quality scores and metrics.

    Args:
        video_path: Path to input video
        pass1_output_path: Path to pass1_raw.json
        pass2b_output_path: Path to pass2_fragments.json (with quality scores)
        debug_video_path: Path to output debug video
        start_frame: Start frame (default: 0)
        end_frame: End frame (default: None = entire video)
    """
    pass1_output = load_json(Path(pass1_output_path), Pass1Output)
    pass2b_output = load_json(Path(pass2b_output_path), Pass2BOutput)

    detection_by_id: Dict[str, Detection] = {det.detection_id: det for det in pass1_output.detections}

    # Build frame annotations (only real fragments, not ghosts)
    frame_annotations: Dict[int, List[Tuple[Detection, ScoredFragment]]] = {}
    for fragment in pass2b_output.fragments:
        if fragment.is_ghost:
            continue  # Skip ghosts for quality view (they always have quality=GHOST)

        for detection_id in fragment.detection_ids:
            det = detection_by_id.get(detection_id)
            if det is None:
                continue
            frame_annotations.setdefault(det.frame_idx, []).append((det, fragment))

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
            raise RuntimeError(f"Failed to open Pass 2B debug video writer: {debug_video_path}")

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

        # Get quality distribution from Pass 2B output
        quality_distribution = pass2b_output.quality_distribution if hasattr(pass2b_output, 'quality_distribution') else {}

        with tqdm(total=total_render_frames, desc="Pass 2B debug render", unit="frame") as pbar:
            for frame_idx, frame in reader.iter_frames():
                if frame_idx < render_start:
                    continue
                if frame_idx >= render_end_exclusive:
                    break

                annotations = frame_annotations.get(frame_idx, [])
                debug_frame = _draw_pass2b_debug_overlay_frame(
                    frame,
                    frame_idx,
                    annotations,
                    quality_distribution,
                )
                writer.write(debug_frame)
                pbar.update(1)

        logger.info(f"Pass 2B debug video written: {debug_video_path}")

    finally:
        if writer is not None:
            writer.release()
        reader.close()
