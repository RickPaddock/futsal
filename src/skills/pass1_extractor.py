"""
Pass 1: Raw Evidence Extraction

Entry point for the pipeline. Processes video and extracts raw observations.

NO team assignment, NO identity, NO player_id.
Only raw observations: bbox, track_id, jersey probabilities, HSV histograms.

Multi-layer bbox defense applied at source (R1 enforcement).
"""

from typing import List, Tuple, Optional
import numpy as np
import hashlib
from pathlib import Path
from functools import lru_cache
from tqdm import tqdm
import cv2

from ..core.data_models import Detection, BallDetection, Pass1Output
from ..core.constants import (
    PLAYER_CONF_THRESHOLD,
    BALL_CONF_THRESHOLD,
    PASS1_BALL_BATCH_SIZE,
    JERSEY_CONF_THRESHOLD,
    JERSEY_NUMBER_CLASSIFY_EVERY_N_FRAMES,
    JERSEY_COLOR_SAMPLE_EVERY_N_FRAMES,
    JERSEY_ROI_X_MIN_FRAC,
    JERSEY_ROI_X_MAX_FRAC,
    JERSEY_ROI_Y_MIN_FRAC,
    JERSEY_ROI_Y_MAX_FRAC,
)
from ..detectors import PlayerDetector, BallDetector, JerseyClassifier
from ..utils.video_io import VideoReader
from ..utils.hsv_color import extract_hsv_histogram
from ..utils.geometry import bbox_centroid, clip_bbox_to_frame
from ..utils.logging_utils import get_logger
from ..utils.file_utils import save_json, load_json
from ..validation.validator import Validator
from ..core.schemas import PASS1_OUTPUT_SCHEMA

logger = get_logger("pass1_extractor")


@lru_cache(maxsize=1)
def _load_pass1_intention_lines(max_lines: int = 5) -> List[str]:
    """Load human-readable Pass 1 intent lines from validation/pass1_rules.py docstring."""
    try:
        from ..validation import pass1_rules

        doc = pass1_rules.__doc__ or ""
        lines: List[str] = []

        for raw in doc.splitlines():
            line = raw.strip()
            if line.startswith("- "):
                lines.append(line[2:].strip())

        if not lines:
            return ["Pass 1 validation rules are active."]

        return lines[:max_lines]
    except Exception:
        return ["Pass 1 validation rules are active."]


def _draw_roi_debug_overlay_frame(
    frame: np.ndarray,
    frame_idx: int,
    detections: List[Detection],
    ball_detection: Optional[BallDetection] = None,
) -> np.ndarray:
    """Draw deterministic player/ROI overlays for Pass 1 debug visualization."""
    overlay = frame.copy()

    def _draw_text_with_bg(
        img: np.ndarray,
        text: str,
        org: tuple[int, int],
        font_scale: float,
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
        cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    jersey_cls_sampled_this_frame = (frame_idx % JERSEY_NUMBER_CLASSIFY_EVERY_N_FRAMES == 0)

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 220, 60), 2)

        if det.jersey_roi_valid and det.jersey_roi_bbox:
            rx1, ry1, rx2, ry2 = [int(round(v)) for v in det.jersey_roi_bbox]
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (40, 40, 240), 2)

        quality_pct = (
            int(round(det.jersey_crop_quality * 100))
            if det.jersey_crop_quality is not None
            else None
        )
        if det.jersey_number is not None:
            num_conf_pct = int(round(det.jersey_confidence * 100))
            num_text = f"num={det.jersey_number}:{num_conf_pct}%"
        else:
            num_text = "num=none"

        cropq_text = f"q={quality_pct}%" if quality_pct is not None else "q=n/a"
        top_label = f"trk={det.track_id} {num_text} {cropq_text}"
        _draw_text_with_bg(overlay, top_label, (x1, max(18, y1 - 8)), 0.43, 1)

        sampled_label = (
            f"sample cls:{'Y' if jersey_cls_sampled_this_frame else 'N'} "
            f"hsv:{'Y' if det.jersey_color_sampled else 'N'}"
        )
        sampled_y = min(overlay.shape[0] - 6, y2 + 16)
        _draw_text_with_bg(overlay, sampled_label, (x1, sampled_y), 0.40, 1)

    if ball_detection is not None:
        bx1, by1, bx2, by2 = [int(round(v)) for v in ball_detection.bbox]
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 215, 255), 2)
        cx, cy = [int(round(v)) for v in ball_detection.centroid]
        cv2.circle(overlay, (cx, cy), 4, (0, 215, 255), -1)
        _draw_text_with_bg(
            overlay,
            f"ballConf={ball_detection.confidence:.2f}",
            (bx1, max(18, by1 - 6)),
            0.45,
            1,
        )

    runtime_status_lines = [
        "Player detection + ByteTrack IDs",
        "Jersey number classification (sampled every N frames)",
        "Jersey ROI + HSV extraction (sampled every N frames)",
        "Jersey crop quality score (heuristic, 0-100%)",
        "Ball detection",
        f"Jersey number sampling: every {JERSEY_NUMBER_CLASSIFY_EVERY_N_FRAMES} frames",
        f"Jersey HSV sampling: every {JERSEY_COLOR_SAMPLE_EVERY_N_FRAMES} frames",
    ]
    sampled_count = sum(1 for det in detections if det.jersey_color_sampled)
    quality_values = [det.jersey_crop_quality for det in detections if det.jersey_crop_quality is not None]
    avg_quality_pct = int(round((sum(quality_values) / len(quality_values)) * 100)) if quality_values else 0
    frame_summary_line = (
        f"frame stats: players={len(detections)} ball={'Y' if ball_detection else 'N'} "
        f"hsvSampledCount={sampled_count} avgCropQ={avg_quality_pct}% jerseyClsSampled={'Y' if jersey_cls_sampled_this_frame else 'N'}"
    )
    all_lines = runtime_status_lines + [frame_summary_line]
    title = "PASS 1 RESPONSIBILITIES (ACTIVE)"

    frame_h, frame_w = overlay.shape[:2]
    top = max(12, int(frame_h * 0.055))
    left = 12
    line_gap = 20
    title_y = top + 14
    first_rule_y = title_y + line_gap
    footer_y = first_rule_y + (len(all_lines) * line_gap) + 8
    panel_bottom = min(frame_h - 8, footer_y + 24)

    panel = overlay.copy()
    panel_top = max(6, top - 12)
    cv2.rectangle(panel, (6, panel_top), (frame_w - 6, panel_bottom), (0, 0, 0), -1)
    overlay = cv2.addWeighted(panel, 0.55, overlay, 0.45, 0)

    _draw_text_with_bg(overlay, title, (left, title_y), 0.50, 1)

    for idx, rule_line in enumerate(all_lines):
        y = first_rule_y + (idx * line_gap)
        _draw_text_with_bg(overlay, f"- {rule_line}", (left + 4, y), 0.48, 1)

    _draw_text_with_bg(
        overlay,
        "GREEN=playerBBox  RED=jerseyROI  YELLOW=ballBBox",
        (left, footer_y),
        0.56,
        1,
    )
    _draw_text_with_bg(
        overlay,
        "Label: num=<topJersey:prob%> q=<cropQuality%> | sample cls=jerseyClassifier hsv=HSVExtraction",
        (left, footer_y + 18),
        0.44,
        1,
    )
    _draw_text_with_bg(
        overlay,
        f"frame={frame_idx}",
        (left, footer_y + 36),
        0.56,
        1,
    )

    return overlay


def _compute_jersey_crop_quality(
    frame: np.ndarray,
    player_bbox: list,
    jersey_roi_bbox: Optional[list],
) -> Optional[float]:
    """
    Compute heuristic jersey crop quality in [0, 1].

    This is not model-based; it combines ROI size, sharpness, saturation, and hue homogeneity.
    """
    if jersey_roi_bbox is None:
        return None

    px1, py1, px2, py2 = [int(round(v)) for v in player_bbox]
    rx1, ry1, rx2, ry2 = [int(round(v)) for v in jersey_roi_bbox]

    player_w = max(0, px2 - px1)
    player_h = max(0, py2 - py1)
    roi_w = max(0, rx2 - rx1)
    roi_h = max(0, ry2 - ry1)
    if player_w == 0 or player_h == 0 or roi_w < 2 or roi_h < 2:
        return None

    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None

    roi_area = roi_w * roi_h
    player_area = player_w * player_h
    area_ratio = roi_area / max(1, player_area)
    expected_area_ratio = (JERSEY_ROI_X_MAX_FRAC - JERSEY_ROI_X_MIN_FRAC) * (JERSEY_ROI_Y_MAX_FRAC - JERSEY_ROI_Y_MIN_FRAC)
    area_score = max(0.0, 1.0 - min(1.0, abs(area_ratio - expected_area_ratio) / max(1e-6, expected_area_ratio)))

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = min(1.0, lap_var / 150.0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    sat_score = float(np.mean(s) / 255.0)

    mask = s > 40.0
    if np.any(mask):
        hue_std = float(np.std(h[mask]))
        homogeneity_score = max(0.0, 1.0 - min(1.0, hue_std / 45.0))
    else:
        homogeneity_score = 0.0

    quality = (
        0.25 * area_score
        + 0.25 * sharpness_score
        + 0.20 * sat_score
        + 0.30 * homogeneity_score
    )

    return float(max(0.0, min(1.0, quality)))


def render_pass1_debug_video_from_artifact(
    video_path: str,
    pass1_output_path: str,
    debug_video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> None:
    """
    Render Pass 1 debug video from existing pass1_raw.json artifact.

    This allows deterministic visualization without re-running detectors.
    """
    pass1_output = load_json(Path(pass1_output_path), Pass1Output)

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
            raise RuntimeError(f"Failed to open Pass 1 debug video writer: {debug_video_path}")

        detections_by_frame = {}
        for det in pass1_output.detections:
            detections_by_frame.setdefault(det.frame_idx, []).append(det)

        ball_by_frame = {
            ball_det.frame_idx: ball_det
            for ball_det in pass1_output.ball_detections
        }

        artifact_start = pass1_output.processed_start_frame
        artifact_end_exclusive = pass1_output.processed_end_frame_exclusive

        render_start = max(start_frame, artifact_start)
        requested_end = artifact_end_exclusive if end_frame is None else end_frame
        render_end_exclusive = min(requested_end, artifact_end_exclusive)

        if render_end_exclusive <= render_start:
            raise ValueError(
                f"Invalid render window: start={render_start}, end={render_end_exclusive}. "
                f"Artifact range is [{artifact_start}, {artifact_end_exclusive})."
            )

        total_render_frames = render_end_exclusive - render_start

        with tqdm(
            total=total_render_frames,
            desc="Pass 1 debug render",
            unit="frame",
        ) as pbar:
            for frame_idx, frame in reader.iter_frames():
                if frame_idx < render_start:
                    continue
                if frame_idx >= render_end_exclusive:
                    break

                frame_detections = detections_by_frame.get(frame_idx, [])
                frame_ball = ball_by_frame.get(frame_idx)
                debug_frame = _draw_roi_debug_overlay_frame(frame, frame_idx, frame_detections, frame_ball)
                writer.write(debug_frame)
                pbar.update(1)
    finally:
        if writer is not None:
            writer.release()
        reader.close()


def render_pass1_roi_debug_video_from_artifact(
    video_path: str,
    pass1_output_path: str,
    debug_roi_video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> None:
    """Backward-compatible alias for older call sites."""
    render_pass1_debug_video_from_artifact(
        video_path=video_path,
        pass1_output_path=pass1_output_path,
        debug_video_path=debug_roi_video_path,
        start_frame=start_frame,
        end_frame=end_frame,
    )


class Pass1Extractor:
    """
    Pass 1: Raw Evidence Extraction

    Extracts raw observations from video without any interpretation.
    """

    def __init__(
        self,
        player_model_path: str = "models/PLAYER_MODEL_best_v1.pt",
        ball_model_path: str = "models/BALL_MODEL_best_v2.pt",
        jersey_model_path: str = "models/JERSEY_MODEL_best_v1.pt",
    ):
        """
        Initialize Pass 1 extractor.

        Args:
            player_model_path: Path to player detection model
            ball_model_path: Path to ball detection model
            jersey_model_path: Path to jersey classification model
        """
        self.logger = logger

        self.logger.info("Initializing detectors...")
        self.player_detector = PlayerDetector(player_model_path)
        self.ball_detector = BallDetector(ball_model_path)
        self.jersey_classifier = JerseyClassifier(jersey_model_path)

        self.logger.info("Pass 1 extractor initialized")

    def extract(
        self,
        video_path: str,
        output_path: str,
        validation_output_path: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        debug_roi_video_path: Optional[str] = None,
    ) -> Pass1Output:
        """
        Extract raw evidence from video.

        Args:
            video_path: Path to input video
            output_path: Path to save pass1_raw.json
            validation_output_path: Path to save pass1_validation.json
            start_frame: Start frame index (default 0)
            end_frame: End frame index (default None = process all)
            debug_roi_video_path: Optional output path for Pass 1 debug overlay video

        Returns:
            Pass1Output with detections and ball detections
        """
        self.logger.info(f"Starting Pass 1 extraction: {video_path}")
        self.logger.info(f"  Output: {output_path}")
        self.logger.info(f"  Validation: {validation_output_path}")

        reader = VideoReader(video_path)
        total_frames = reader.total_frames if end_frame is None else min(end_frame, reader.total_frames)
        frames_to_process = total_frames - start_frame

        self.logger.info(f"Video info: {reader.width}x{reader.height} @ {reader.fps}fps, {reader.total_frames} frames")
        self.logger.info(f"Processing frames {start_frame} to {total_frames}")

        all_detections: List[Detection] = []
        all_ball_detections: List[BallDetection] = []

        debug_writer = None
        if debug_roi_video_path:
            fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
            if fourcc_fn is None:
                fourcc_fn = cv2.VideoWriter.fourcc
            debug_writer = cv2.VideoWriter(
                debug_roi_video_path,
                fourcc_fn(*"mp4v"),
                float(reader.fps),
                (int(reader.width), int(reader.height)),
            )
            if not debug_writer.isOpened():
                raise RuntimeError(f"Failed to open Pass 1 debug video writer: {debug_roi_video_path}")
            self.logger.info(f"Pass 1 debug video enabled: {debug_roi_video_path}")

        try:
            with tqdm(total=frames_to_process, desc="Pass 1: Extracting raw evidence") as pbar:
                frame_batch: List[Tuple[int, np.ndarray]] = []

                for frame_idx, frame in reader.iter_frames():
                    if frame_idx < start_frame:
                        continue
                    if end_frame is not None and frame_idx >= end_frame:
                        break

                    frame_batch.append((frame_idx, frame))

                    if len(frame_batch) >= PASS1_BALL_BATCH_SIZE:
                        batch_detections, batch_ball_detections = self._process_frame_batch(
                            frame_batch=frame_batch,
                            frame_width=reader.width,
                            frame_height=reader.height,
                            debug_writer=debug_writer,
                        )
                        all_detections.extend(batch_detections)
                        all_ball_detections.extend(batch_ball_detections)
                        pbar.update(len(frame_batch))
                        frame_batch = []

                if frame_batch:
                    batch_detections, batch_ball_detections = self._process_frame_batch(
                        frame_batch=frame_batch,
                        frame_width=reader.width,
                        frame_height=reader.height,
                        debug_writer=debug_writer,
                    )
                    all_detections.extend(batch_detections)
                    all_ball_detections.extend(batch_ball_detections)
                    pbar.update(len(frame_batch))
        finally:
            if debug_writer is not None:
                debug_writer.release()

        video_name = Path(video_path).stem
        video_fps = reader.fps
        video_width = reader.width
        video_height = reader.height
        video_total_frames = reader.total_frames

        reader.close()

        self.logger.info(f"Extraction complete: {len(all_detections)} player detections, {len(all_ball_detections)} ball detections")

        result = Pass1Output(
            video_name=video_name,
            fps=video_fps,
            width=video_width,
            height=video_height,
            total_frames=video_total_frames,
            processed_start_frame=start_frame,
            processed_end_frame_exclusive=total_frames,
            detections=all_detections,
            ball_detections=all_ball_detections,
        )

        self.logger.info("Validating Pass 1 output...")
        validator = Validator()
        validation_result = validator.validate_pass1(result)

        validation_data = validation_result.model_dump()
        save_json(validation_data, validation_output_path, None)

        if not validation_result.passed:
            error_msg = f"Pass 1 validation failed with {len(validation_result.violations)} violations:\n"
            error_msg += "\n".join(v.message for v in validation_result.violations[:10])
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Pass 1 validation passed ✓")

        output_data = result.model_dump(exclude_none=True, exclude_defaults=True)
        save_json(output_data, output_path, PASS1_OUTPUT_SCHEMA, indent=0)

        self.logger.info(f"Pass 1 complete: {output_path}")

        return result

    def _process_frame_batch(
        self,
        frame_batch: List[Tuple[int, np.ndarray]],
        frame_width: int,
        frame_height: int,
        debug_writer=None,
    ) -> Tuple[List[Detection], List[BallDetection]]:
        """Process a frame chunk with batched ball detection and per-frame player tracking."""
        frames = [frame for _, frame in frame_batch]
        ball_results = self.ball_detector.detect_batch(frames, BALL_CONF_THRESHOLD)

        all_detections: List[Detection] = []
        all_ball_detections: List[BallDetection] = []

        for (frame_idx, frame), ball_result in zip(frame_batch, ball_results):
            detections, ball_detection = self._process_frame(
                frame=frame,
                frame_idx=frame_idx,
                frame_width=frame_width,
                frame_height=frame_height,
                precomputed_ball_result=ball_result,
            )

            all_detections.extend(detections)
            if ball_detection is not None:
                all_ball_detections.append(ball_detection)

            if debug_writer is not None:
                debug_frame = self._draw_roi_debug_overlay(frame, frame_idx, detections, ball_detection)
                debug_writer.write(debug_frame)

        return all_detections, all_ball_detections

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        frame_width: int,
        frame_height: int,
        precomputed_ball_result: Optional[Tuple[List[float], float]] = None,
    ) -> Tuple[List[Detection], Optional[BallDetection]]:
        """
        Process a single frame.

        Args:
            frame: RGB frame (H, W, 3)
            frame_idx: Frame index
            frame_width: Frame width
            frame_height: Frame height
            precomputed_ball_result: Optional precomputed ball detection for this frame

        Returns:
            (detections, ball_detection) tuple
        """
        tracked_dets = self.player_detector.detect_and_track(frame, PLAYER_CONF_THRESHOLD)

        jersey_results = [None] * len(tracked_dets)
        if tracked_dets and (frame_idx % JERSEY_NUMBER_CLASSIFY_EVERY_N_FRAMES == 0):
            jersey_bboxes = [bbox for bbox, _, _ in tracked_dets]
            jersey_results = self.jersey_classifier.classify_batch(
                frame,
                jersey_bboxes,
                JERSEY_CONF_THRESHOLD,
            )

        detections = []
        for det_idx, (bbox, conf, track_id) in enumerate(tracked_dets):
            jersey_result = jersey_results[det_idx] if det_idx < len(jersey_results) else None
            if jersey_result is not None:
                jersey_number, jersey_conf = jersey_result
            else:
                jersey_number = None
                jersey_conf = 0.0

            jersey_roi_bbox = self._get_jersey_roi_bbox(bbox, reader_width=frame_width, reader_height=frame_height)
            jersey_roi_valid = jersey_roi_bbox is not None
            jersey_color_sampled = jersey_roi_valid and (frame_idx % JERSEY_COLOR_SAMPLE_EVERY_N_FRAMES == 0)
            hsv_histogram_jersey = (
                extract_hsv_histogram(frame, jersey_roi_bbox)
                if (jersey_color_sampled and jersey_roi_bbox is not None)
                else None
            )

            if jersey_color_sampled and hsv_histogram_jersey is None:
                jersey_roi_valid = False
                jersey_roi_bbox = None
                jersey_color_sampled = False

            detection_id = self._create_detection_id(frame_idx, track_id, bbox)
            centroid = bbox_centroid(bbox)

            detection = Detection(
                detection_id=detection_id,
                frame_idx=frame_idx,
                track_id=track_id,
                bbox=bbox,
                centroid=centroid,
                confidence=conf,
                jersey_number=jersey_number,
                jersey_confidence=jersey_conf,
                hsv_histogram_jersey=hsv_histogram_jersey,
                jersey_color_sampled=jersey_color_sampled,
                jersey_roi_valid=jersey_roi_valid,
                jersey_roi_bbox=jersey_roi_bbox,
                jersey_crop_quality=_compute_jersey_crop_quality(frame, bbox, jersey_roi_bbox),
            )

            detections.append(detection)

        ball_detection = None
        ball_result = precomputed_ball_result
        if ball_result is None:
            ball_result = self.ball_detector.detect(frame, BALL_CONF_THRESHOLD)

        if ball_result:
            ball_bbox, ball_conf = ball_result
            ball_centroid = bbox_centroid(ball_bbox)
            ball_detection = BallDetection(
                frame_idx=frame_idx,
                bbox=ball_bbox,
                centroid=ball_centroid,
                confidence=ball_conf,
            )

        return detections, ball_detection

    def _get_jersey_roi_bbox(self, bbox: list, reader_width: int, reader_height: int) -> Optional[list]:
        """
        Compute jersey ROI inside player bbox.

        Default ROI ratios (relative to player bbox):
        - Height: 20% to 55% (upper torso band)
        - Width: 25% to 75% (center torso band)

        Returns:
            Cropped jersey ROI bbox [x1, y1, x2, y2], or None if invalid
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return None

        roi_x1 = x1 + (JERSEY_ROI_X_MIN_FRAC * width)
        roi_x2 = x1 + (JERSEY_ROI_X_MAX_FRAC * width)
        roi_y1 = y1 + (JERSEY_ROI_Y_MIN_FRAC * height)
        roi_y2 = y1 + (JERSEY_ROI_Y_MAX_FRAC * height)

        roi_bbox = [roi_x1, roi_y1, roi_x2, roi_y2]
        roi_bbox = clip_bbox_to_frame(roi_bbox, reader_width, reader_height)

        rx1, ry1, rx2, ry2 = roi_bbox
        if rx2 <= rx1 or ry2 <= ry1:
            return None

        if (rx2 - rx1) < 2 or (ry2 - ry1) < 2:
            return None

        return roi_bbox

    def _create_detection_id(self, frame_idx: int, track_id: int, bbox: list) -> str:
        """
        Create unique detection_id.

        Format: {frame_idx}_{track_id}_{bbox_hash}
        """
        bbox_str = f"{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}"
        bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()[:8]
        return f"{frame_idx}_{track_id}_{bbox_hash}"

    def _draw_roi_debug_overlay(
        self,
        frame: np.ndarray,
        frame_idx: int,
        detections: List[Detection],
        ball_detection: Optional[BallDetection] = None,
    ) -> np.ndarray:
        """Draw deterministic player/ROI overlays for Pass 1 debug visualization."""
        return _draw_roi_debug_overlay_frame(frame, frame_idx, detections, ball_detection)


def run_pass1(
    video_path: str,
    output_path: str,
    validation_output_path: str,
    player_model_path: str = "models/PLAYER_MODEL_best_v1.pt",
    ball_model_path: str = "models/BALL_MODEL_best_v2.pt",
    jersey_model_path: str = "models/JERSEY_MODEL_best_v1.pt",
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    debug_roi_video_path: Optional[str] = None,
) -> Pass1Output:
    """
    Run Pass 1: Raw Evidence Extraction.
    """
    extractor = Pass1Extractor(
        player_model_path=player_model_path,
        ball_model_path=ball_model_path,
        jersey_model_path=jersey_model_path,
    )

    return extractor.extract(
        video_path=video_path,
        output_path=output_path,
        validation_output_path=validation_output_path,
        start_frame=start_frame,
        end_frame=end_frame,
        debug_roi_video_path=debug_roi_video_path,
    )
