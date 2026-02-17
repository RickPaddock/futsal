"""
Player Detector - YOLO wrapper for detecting players in frames.

Loads PLAYER_MODEL_best_v1.pt and applies PLAYER_CONF_THRESHOLD.
"""

from typing import List, Tuple
import numpy as np
from pathlib import Path

from ..core.constants import (
    PLAYER_CONF_THRESHOLD,
    MAX_BBOX_HEIGHT_PX,
    MAX_BBOX_WIDTH_PX,
    MAX_BBOX_AREA_FRACTION,
    PLAYER_TRACKER_CONFIG,
)
from ..utils.logging_utils import get_logger
from ..utils.geometry import bbox_is_valid, clip_bbox_to_frame, bbox_width, bbox_height, bbox_area

logger = get_logger("player_detector")


class PlayerDetector:
    """
    YOLO-based player detector with multi-layer bbox defense.

    Applies pathological detection filters (huge bbox defense) at source.
    """

    def __init__(self, model_path: str = "models/PLAYER_MODEL_best_v1.pt"):
        """
        Initialize player detector.

        Args:
            model_path: Path to YOLO model weights
        """
        self.model_path = Path(model_path)
        self.model = None
        self.logger = logger

        if not self.model_path.exists():
            raise FileNotFoundError(f"Player model not found: {self.model_path}")

        self._load_model()

    def _load_model(self):
        """Load YOLO model. FAIL HARD if not available."""
        from ultralytics import YOLO  # Hard import - no try/except

        self.model = YOLO(str(self.model_path))
        self.logger.info(f"Loaded player detection model: {self.model_path}")

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = PLAYER_CONF_THRESHOLD,
    ) -> List[Tuple[List[float], float]]:
        """
        Detect players in frame.

        Multi-layer defense against pathological YOLO detections:
        - Layer 1: Absolute size limits (800px height, 600px width)
        - Layer 2: Relative size limit (25% frame area)

        Args:
            frame: RGB frame (H, W, 3)
            conf_threshold: Confidence threshold (default from constants)

        Returns:
            List of (bbox, confidence) tuples
            bbox format: [x1, y1, x2, y2]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        frame_height, frame_width = frame.shape[:2]

        # Run YOLO detection
        results = self.model(frame, conf=conf_threshold, verbose=False)

        detections = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confidences = result.boxes.conf.cpu().numpy()

            for bbox, conf in zip(boxes, confidences):
                bbox = bbox.tolist()

                # Validate bbox is within frame
                if not bbox_is_valid(bbox, frame_width, frame_height):
                    self.logger.warning(f"Invalid bbox (out of frame): {bbox}")
                    continue

                # Clip to frame bounds (defensive)
                bbox = clip_bbox_to_frame(bbox, frame_width, frame_height)

                # LAYER 1: Absolute size limits
                width = bbox_width(bbox)
                height = bbox_height(bbox)

                if height > MAX_BBOX_HEIGHT_PX:
                    self.logger.warning(
                        f"Filtered huge bbox (height={height:.0f}px > {MAX_BBOX_HEIGHT_PX}px): {bbox}"
                    )
                    continue

                if width > MAX_BBOX_WIDTH_PX:
                    self.logger.warning(
                        f"Filtered huge bbox (width={width:.0f}px > {MAX_BBOX_WIDTH_PX}px): {bbox}"
                    )
                    continue

                # LAYER 2: Relative size limit (area fraction)
                area = bbox_area(bbox)
                frame_area = frame_width * frame_height
                area_fraction = area / frame_area

                if area_fraction > MAX_BBOX_AREA_FRACTION:
                    self.logger.warning(
                        f"Filtered huge bbox (area={area_fraction:.2%} > {MAX_BBOX_AREA_FRACTION:.2%}): {bbox}"
                    )
                    continue

                # Bbox passed all filters
                detections.append((bbox, float(conf)))

        self.logger.debug(f"Detected {len(detections)} players in frame")

        return detections

    def detect_and_track(
        self,
        frame: np.ndarray,
        conf_threshold: float = PLAYER_CONF_THRESHOLD,
    ) -> List[Tuple[List[float], float, int]]:
        """
        Detect AND track players in frame using Ultralytics built-in tracking.

        Uses BoT-SORT tracker (better than ByteTrack) built into ultralytics v8+.

        Args:
            frame: RGB frame (H, W, 3)
            conf_threshold: Confidence threshold (default from constants)

        Returns:
            List of (bbox, confidence, track_id) tuples
            bbox format: [x1, y1, x2, y2]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        frame_height, frame_width = frame.shape[:2]

        # Run YOLO detection + tracking (persist=True maintains track state)
        results = self.model.track(
            frame,
            conf=conf_threshold,
            persist=True,
            verbose=False,
            tracker=PLAYER_TRACKER_CONFIG,
        )

        tracked_detections = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confidences = result.boxes.conf.cpu().numpy()

            # Get track IDs (if available)
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                # No tracking info - assign sequential IDs
                track_ids = list(range(1, len(boxes) + 1))

            for bbox, conf, track_id in zip(boxes, confidences, track_ids):
                bbox = bbox.tolist()

                # Validate bbox is within frame
                if not bbox_is_valid(bbox, frame_width, frame_height):
                    self.logger.warning(f"Invalid bbox (out of frame): {bbox}")
                    continue

                # Clip to frame bounds (defensive)
                bbox = clip_bbox_to_frame(bbox, frame_width, frame_height)

                # LAYER 1: Absolute size limits
                width = bbox_width(bbox)
                height = bbox_height(bbox)

                if height > MAX_BBOX_HEIGHT_PX:
                    self.logger.warning(
                        f"Filtered huge bbox (height={height:.0f}px > {MAX_BBOX_HEIGHT_PX}px): {bbox}"
                    )
                    continue

                if width > MAX_BBOX_WIDTH_PX:
                    self.logger.warning(
                        f"Filtered huge bbox (width={width:.0f}px > {MAX_BBOX_WIDTH_PX}px): {bbox}"
                    )
                    continue

                # LAYER 2: Relative size limit (area fraction)
                area = bbox_area(bbox)
                frame_area = frame_width * frame_height
                area_fraction = area / frame_area

                if area_fraction > MAX_BBOX_AREA_FRACTION:
                    self.logger.warning(
                        f"Filtered huge bbox (area={area_fraction:.2%} > {MAX_BBOX_AREA_FRACTION:.2%}): {bbox}"
                    )
                    continue

                # Bbox passed all filters
                tracked_detections.append((bbox, float(conf), int(track_id)))

        self.logger.debug(f"Detected and tracked {len(tracked_detections)} players in frame")

        return tracked_detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        conf_threshold: float = PLAYER_CONF_THRESHOLD,
    ) -> List[List[Tuple[List[float], float]]]:
        """
        Detect players in multiple frames (batch processing).

        Args:
            frames: List of RGB frames
            conf_threshold: Confidence threshold

        Returns:
            List of detections per frame
        """
        return [self.detect(frame, conf_threshold) for frame in frames]
