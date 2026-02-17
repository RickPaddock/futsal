"""
Ball Detector - YOLO wrapper for detecting the ball in frames.

Loads BALL_MODEL_best_v2.pt and applies BALL_CONF_THRESHOLD.
"""

from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path

from ..core.constants import BALL_CONF_THRESHOLD
from ..utils.logging_utils import get_logger
from ..utils.geometry import bbox_is_valid, clip_bbox_to_frame

logger = get_logger("ball_detector")


class BallDetector:
    """
    YOLO-based ball detector.

    Returns at most one ball detection per frame (highest confidence).
    """

    def __init__(self, model_path: str = "models/BALL_MODEL_best_v2.pt"):
        """
        Initialize ball detector.

        Args:
            model_path: Path to YOLO model weights
        """
        self.model_path = Path(model_path)
        self.model = None
        self.logger = logger

        if not self.model_path.exists():
            raise FileNotFoundError(f"Ball model not found: {self.model_path}")

        self._load_model()

    def _load_model(self):
        """Load YOLO model. FAIL HARD if not available."""
        from ultralytics import YOLO  # Hard import - no try/except

        self.model = YOLO(str(self.model_path))
        self.logger.info(f"Loaded ball detection model: {self.model_path}")

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = BALL_CONF_THRESHOLD,
    ) -> Optional[Tuple[List[float], float]]:
        """
        Detect ball in frame.

        Returns at most one detection (highest confidence).

        Args:
            frame: RGB frame (H, W, 3)
            conf_threshold: Confidence threshold (default from constants)

        Returns:
            (bbox, confidence) tuple if ball detected, else None
            bbox format: [x1, y1, x2, y2]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        frame_height, frame_width = frame.shape[:2]

        # Run YOLO detection
        results = self.model(frame, conf=conf_threshold, verbose=False)

        best_detection = None
        best_conf = 0.0

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confidences = result.boxes.conf.cpu().numpy()

            for bbox, conf in zip(boxes, confidences):
                bbox = bbox.tolist()

                # Validate bbox is within frame
                if not bbox_is_valid(bbox, frame_width, frame_height):
                    self.logger.warning(f"Invalid ball bbox (out of frame): {bbox}")
                    continue

                # Clip to frame bounds (defensive)
                bbox = clip_bbox_to_frame(bbox, frame_width, frame_height)

                # Keep highest confidence detection
                if conf > best_conf:
                    best_detection = (bbox, float(conf))
                    best_conf = float(conf)

        if best_detection:
            self.logger.debug(f"Detected ball with confidence {best_conf:.3f}")
        else:
            self.logger.debug("No ball detected")

        return best_detection

    def detect_batch(
        self,
        frames: List[np.ndarray],
        conf_threshold: float = BALL_CONF_THRESHOLD,
    ) -> List[Optional[Tuple[List[float], float]]]:
        """
        Detect ball in multiple frames (batch processing).

        Args:
            frames: List of RGB frames
            conf_threshold: Confidence threshold

        Returns:
            List of detections per frame (None if no ball detected)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if not frames:
            return []

        try:
            results = self.model(frames, conf=conf_threshold, verbose=False)
        except Exception as exc:
            self.logger.warning(f"Batch ball detection failed; falling back to per-frame inference: {exc}")
            return [self.detect(frame, conf_threshold) for frame in frames]

        detections_per_frame: List[Optional[Tuple[List[float], float]]] = []

        for frame, result in zip(frames, results):
            frame_height, frame_width = frame.shape[:2]
            best_detection = None
            best_conf = 0.0

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for bbox, conf in zip(boxes, confidences):
                    bbox = bbox.tolist()

                    if not bbox_is_valid(bbox, frame_width, frame_height):
                        self.logger.warning(f"Invalid ball bbox (out of frame): {bbox}")
                        continue

                    bbox = clip_bbox_to_frame(bbox, frame_width, frame_height)

                    if conf > best_conf:
                        best_detection = (bbox, float(conf))
                        best_conf = float(conf)

            detections_per_frame.append(best_detection)

        if len(detections_per_frame) != len(frames):
            self.logger.warning("Batch output size mismatch; falling back to per-frame ball inference")
            return [self.detect(frame, conf_threshold) for frame in frames]

        return detections_per_frame
