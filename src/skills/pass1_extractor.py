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
from tqdm import tqdm
import cv2

from ..core.data_models import Detection, BallDetection, Pass1Output
from ..core.constants import (
    PLAYER_CONF_THRESHOLD,
    BALL_CONF_THRESHOLD,
    JERSEY_CONF_THRESHOLD,
    JERSEY_CLASSIFY_EVERY_N_FRAMES,
    JERSEY_ROI_X_MIN_FRAC,
    JERSEY_ROI_X_MAX_FRAC,
    JERSEY_ROI_Y_MIN_FRAC,
    JERSEY_ROI_Y_MAX_FRAC,
    MAX_BBOX_HEIGHT_PX,
    MAX_BBOX_WIDTH_PX,
    MAX_BBOX_AREA_FRACTION,
)
from ..detectors import PlayerDetector, BallDetector, JerseyClassifier
from ..utils.video_io import VideoReader
from ..utils.hsv_color import extract_hsv_histogram
from ..utils.geometry import bbox_centroid, clip_bbox_to_frame
from ..utils.logging_utils import get_logger
from ..utils.file_utils import save_json
from ..validation.validator import Validator
from ..core.schemas import PASS1_OUTPUT_SCHEMA

logger = get_logger("pass1_extractor")


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

        # Initialize detectors
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
        end_frame: int = None,
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
            debug_roi_video_path: Optional output path for Pass 1 ROI overlay video

        Returns:
            Pass1Output with detections and ball detections
        """
        self.logger.info(f"Starting Pass 1 extraction: {video_path}")
        self.logger.info(f"  Output: {output_path}")
        self.logger.info(f"  Validation: {validation_output_path}")

        # Open video
        reader = VideoReader(video_path)
        total_frames = reader.total_frames if end_frame is None else min(end_frame, reader.total_frames)
        frames_to_process = total_frames - start_frame

        self.logger.info(f"Video info: {reader.width}x{reader.height} @ {reader.fps}fps, {reader.total_frames} frames")
        self.logger.info(f"Processing frames {start_frame} to {total_frames}")

        # Process frames
        all_detections = []
        all_ball_detections = []

        debug_writer = None
        if debug_roi_video_path:
            debug_writer = cv2.VideoWriter(
                debug_roi_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(reader.fps),
                (int(reader.width), int(reader.height)),
            )
            if not debug_writer.isOpened():
                raise RuntimeError(f"Failed to open debug ROI video writer: {debug_roi_video_path}")
            self.logger.info(f"Pass 1 ROI debug video enabled: {debug_roi_video_path}")

        try:
            with tqdm(total=frames_to_process, desc="Pass 1: Extracting raw evidence") as pbar:
                for frame_idx, frame in reader.iter_frames():
                    if frame_idx < start_frame:
                        continue
                    if end_frame is not None and frame_idx >= end_frame:
                        break

                    # Process frame
                    detections, ball_detection = self._process_frame(
                        frame=frame,
                        frame_idx=frame_idx,
                        frame_width=reader.width,
                        frame_height=reader.height,
                    )

                    all_detections.extend(detections)
                    if ball_detection:
                        all_ball_detections.append(ball_detection)

                    if debug_writer is not None:
                        debug_frame = self._draw_roi_debug_overlay(frame, frame_idx, detections)
                        debug_writer.write(debug_frame)

                    pbar.update(1)
        finally:
            if debug_writer is not None:
                debug_writer.release()

        # Get video metadata before closing
        video_name = Path(video_path).stem
        video_fps = reader.fps
        video_width = reader.width
        video_height = reader.height
        video_total_frames = reader.total_frames

        reader.close()

        self.logger.info(f"Extraction complete: {len(all_detections)} player detections, {len(all_ball_detections)} ball detections")

        # Create output
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

        # Validate BEFORE saving (CRITICAL)
        self.logger.info("Validating Pass 1 output...")
        validator = Validator()
        validation_result = validator.validate_pass1(result)

        # Save validation result
        validation_data = validation_result.model_dump()
        save_json(validation_data, validation_output_path, None)  # No schema for validation output

        # FAIL-FAST if validation failed
        if not validation_result.passed:
            error_msg = f"Pass 1 validation failed with {len(validation_result.violations)} violations:\n"
            error_msg += "\n".join(validation_result.violations[:10])
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Pass 1 validation passed ✓")

        # Save output (only after validation passes)
        output_data = result.model_dump(exclude_none=True, exclude_defaults=True)
        save_json(output_data, output_path, PASS1_OUTPUT_SCHEMA, indent=None)

        self.logger.info(f"Pass 1 complete: {output_path}")

        return result

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[List[Detection], Optional[BallDetection]]:
        """
        Process a single frame.

        Args:
            frame: RGB frame (H, W, 3)
            frame_idx: Frame index
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            (detections, ball_detection) tuple
        """
        # 1. Detect AND track players (with huge bbox filter built-in)
        # Uses Ultralytics built-in tracking (BoT-SORT) - no separate tracker needed
        tracked_dets = self.player_detector.detect_and_track(frame, PLAYER_CONF_THRESHOLD)

        # 3. Process each tracked detection
        detections = []
        for bbox, conf, track_id in tracked_dets:
            # 4. Classify jersey number (optimization: only every N frames)
            # This reduces YOLO calls from 12/frame to ~2.4/frame (5x speedup on jersey classification)
            if frame_idx % JERSEY_CLASSIFY_EVERY_N_FRAMES == 0:
                jersey_result = self.jersey_classifier.classify(frame, bbox, JERSEY_CONF_THRESHOLD)
                if jersey_result:
                    jersey_number, jersey_conf = jersey_result
                else:
                    jersey_number = None
                    jersey_conf = 0.0
            else:
                # Skip classification on non-sampled frames
                jersey_number = None
                jersey_conf = 0.0

            # 5. Extract HSV evidence
            # Primary (team identity): jersey ROI only
            jersey_roi_bbox = self._get_jersey_roi_bbox(bbox, reader_width=frame_width, reader_height=frame_height)
            jersey_roi_valid = jersey_roi_bbox is not None
            hsv_histogram_jersey = extract_hsv_histogram(frame, jersey_roi_bbox) if jersey_roi_valid else None

            # If extraction failed, mark ROI invalid and keep jersey histogram unset
            if jersey_roi_valid and hsv_histogram_jersey is None:
                jersey_roi_valid = False
                jersey_roi_bbox = None

            # 6. Create detection_id (unique identifier)
            detection_id = self._create_detection_id(frame_idx, track_id, bbox)

            # 7. Calculate centroid
            centroid = bbox_centroid(bbox)

            # 8. Create Detection object (NO team, NO player_id)
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
                jersey_roi_valid=jersey_roi_valid,
                jersey_roi_bbox=jersey_roi_bbox,
            )

            detections.append(detection)

        # 9. Detect ball
        ball_detection = None
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

        Args:
            frame_idx: Frame index
            track_id: Track ID
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Unique detection_id string
        """
        # Hash bbox to create unique identifier
        bbox_str = f"{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}"
        bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()[:8]

        return f"{frame_idx}_{track_id}_{bbox_hash}"

    def _draw_roi_debug_overlay(
        self,
        frame: np.ndarray,
        frame_idx: int,
        detections: List[Detection],
    ) -> np.ndarray:
        """Draw deterministic player/ROI overlays for Pass 1 debug visualization."""
        overlay = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 220, 60), 2)

            if det.jersey_roi_valid and det.jersey_roi_bbox:
                rx1, ry1, rx2, ry2 = [int(round(v)) for v in det.jersey_roi_bbox]
                cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (40, 40, 240), 2)

            label = f"trk:{det.track_id}"
            cv2.putText(
                overlay,
                label,
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            overlay,
            "GREEN=Player bbox  RED=Jersey ROI(HSV)",
            (14, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"frame={frame_idx}",
            (14, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return overlay


def run_pass1(
    video_path: str,
    output_path: str,
    validation_output_path: str,
    player_model_path: str = "models/PLAYER_MODEL_best_v1.pt",
    ball_model_path: str = "models/BALL_MODEL_best_v2.pt",
    jersey_model_path: str = "models/JERSEY_MODEL_best_v1.pt",
    start_frame: int = 0,
    end_frame: int = None,
    debug_roi_video_path: Optional[str] = None,
) -> Pass1Output:
    """
    Run Pass 1: Raw Evidence Extraction.

    Entry point for the pipeline.

    Args:
        video_path: Path to input video
        output_path: Path to save pass1_raw.json
        validation_output_path: Path to save pass1_validation.json
        player_model_path: Path to player detection model
        ball_model_path: Path to ball detection model
        jersey_model_path: Path to jersey classification model
        start_frame: Start frame index (default 0)
        end_frame: End frame index (default None = process all)
        debug_roi_video_path: Optional output path for Pass 1 ROI overlay video

    Returns:
        Pass1Output with detections
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
