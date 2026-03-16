"""
Ball Detector - YOLO wrapper for detecting the ball in frames.

Loads BALL_MODEL_best_v2.pt and applies BALL_CONF_THRESHOLD.
Supports InferenceSlicer for small-object detection in high-resolution footage.
"""

from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path

from ..core.constants import (
    BALL_CONF_THRESHOLD,
    USE_INFERENCE_SLICER,
    SLICER_OVERLAP_PX,
    SLICER_IOU_THRESHOLD,
)
from ..utils.logging_utils import get_logger
from ..utils.geometry import bbox_is_valid, clip_bbox_to_frame

logger = get_logger("ball_detector")


class BallDetector:
    """
    YOLO-based ball detector with optional InferenceSlicer for small-object detection.

    Returns at most one ball detection per frame (highest confidence).
    InferenceSlicer uses 2x2 tiling with overlap for improved recall on small balls.
    """

    def __init__(
        self,
        model_path: str = "models/BALL_MODEL_best_v2.pt",
        use_inference_slicer: bool = USE_INFERENCE_SLICER,
    ):
        """
        Initialize ball detector.

        Args:
            model_path: Path to YOLO model weights
            use_inference_slicer: Enable tiling for small-object detection
        """
        self.model_path = Path(model_path)
        self.model = None
        self.logger = logger
        self.use_inference_slicer = use_inference_slicer

        # Slicer will be lazily initialized on first frame (needs frame dimensions)
        self.slicer = None
        self._slicer_initialized = False
        self._slicer_conf_threshold = BALL_CONF_THRESHOLD

        if not self.model_path.exists():
            raise FileNotFoundError(f"Ball model not found: {self.model_path}")

        self._load_model()

    def _load_model(self):
        """Load YOLO model. FAIL HARD if not available."""
        from ultralytics import YOLO  # Hard import - no try/except

        self.model = YOLO(str(self.model_path))
        self.logger.info(f"Loaded ball detection model: {self.model_path}")
        if self.use_inference_slicer:
            self.logger.info("InferenceSlicer enabled (will init on first frame)")

    def _ensure_slicer(
        self,
        frame_width: int,
        frame_height: int,
        conf_threshold: float = BALL_CONF_THRESHOLD,
    ):
        """
        Lazily initialize InferenceSlicer with frame dimensions.

        Creates 2x2 grid with overlap for improved small-object detection.
        For 4K (3840x2160): tiles are ~2120x1280 pixels with 200px overlap.

        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        if (
            self._slicer_initialized
            and self.slicer is not None
            and abs(self._slicer_conf_threshold - conf_threshold) < 1e-9
        ):
            return

        try:
            import supervision as sv
        except ImportError:
            self.logger.warning("supervision library not available, disabling InferenceSlicer")
            self.use_inference_slicer = False
            self._slicer_initialized = True
            return

        def callback(frame_slice: np.ndarray) -> "sv.Detections":
            """Inference callback for each tile."""
            results = self.model(
                frame_slice,
                conf=conf_threshold,
                verbose=False,
            )
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = np.zeros(len(boxes), dtype=int)
                    return sv.Detections(
                        xyxy=boxes,
                        confidence=confidences,
                        class_id=class_ids,
                    )
            return sv.Detections.empty()

        # Calculate tile dimensions: (width//2 + overlap, height//2 + overlap)
        # Matches utils/tile_dataset.py tiling strategy
        tile_w = frame_width // 2 + SLICER_OVERLAP_PX
        tile_h = frame_height // 2 + SLICER_OVERLAP_PX

        self.slicer = sv.InferenceSlicer(
            callback=callback,
            slice_wh=(tile_w, tile_h),
            overlap_wh=(SLICER_OVERLAP_PX, SLICER_OVERLAP_PX),
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            iou_threshold=SLICER_IOU_THRESHOLD,
        )
        self._slicer_initialized = True
        self._slicer_conf_threshold = conf_threshold
        self.logger.info(
            f"InferenceSlicer initialized: {tile_w}x{tile_h} tiles "
            f"(2x2 grid + {SLICER_OVERLAP_PX}px overlap, IOU={SLICER_IOU_THRESHOLD})"
        )

    def _extract_valid_candidates(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> List[Tuple[List[float], float]]:
        candidates: List[Tuple[List[float], float]] = []

        for bbox, conf in zip(boxes, confidences):
            bbox_list = bbox.tolist()

            if not bbox_is_valid(bbox_list, frame_width, frame_height):
                self.logger.warning(f"Invalid ball bbox (out of frame): {bbox_list}")
                continue

            clipped_bbox = clip_bbox_to_frame(bbox_list, frame_width, frame_height)
            candidates.append((clipped_bbox, float(conf)))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates

    def detect_all(
        self,
        frame: np.ndarray,
        conf_threshold: float = BALL_CONF_THRESHOLD,
    ) -> List[Tuple[List[float], float]]:
        """
        Detect all ball candidates in a frame.

        Uses the same model path as detect(), but returns every valid candidate
        sorted by confidence descending. This is intended for diagnostics and
        audit workflows where low-confidence alternatives matter.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        frame_height, frame_width = frame.shape[:2]

        if self.use_inference_slicer and (
            not self._slicer_initialized
            or abs(self._slicer_conf_threshold - conf_threshold) >= 1e-9
        ):
            self._ensure_slicer(frame_width, frame_height, conf_threshold)

        if self.use_inference_slicer and self.slicer is not None:
            try:
                detections_sv = self.slicer(frame)
                if len(detections_sv) > 0:
                    boxes = detections_sv.xyxy
                    confidences = (
                        detections_sv.confidence
                        if detections_sv.confidence is not None
                        else np.ones(len(boxes))
                    )
                    return self._extract_valid_candidates(boxes, confidences, frame_width, frame_height)
            except Exception as e:
                self.logger.error(f"InferenceSlicer failed, falling back to full-frame: {e}")
                self.use_inference_slicer = False

        results = self.model(frame, conf=conf_threshold, verbose=False)

        all_candidates: List[Tuple[List[float], float]] = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            all_candidates.extend(
                self._extract_valid_candidates(boxes, confidences, frame_width, frame_height)
            )

        all_candidates.sort(key=lambda item: item[1], reverse=True)
        return all_candidates

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = BALL_CONF_THRESHOLD,
    ) -> Optional[Tuple[List[float], float]]:
        """
        Detect ball in frame.

        Returns at most one detection (highest confidence).
        Uses InferenceSlicer if enabled for improved small-object detection.

        Args:
            frame: RGB frame (H, W, 3)
            conf_threshold: Confidence threshold (default from constants)

        Returns:
            (bbox, confidence) tuple if ball detected, else None
            bbox format: [x1, y1, x2, y2]
        """
        all_candidates = self.detect_all(frame, conf_threshold=conf_threshold)
        best_detection = all_candidates[0] if all_candidates else None

        if best_detection:
            self.logger.debug(f"Detected ball with confidence {best_detection[1]:.3f}")
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

        Note: InferenceSlicer doesn't support batch mode, so we fall back to
        per-frame detection when slicer is enabled.

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

        # InferenceSlicer doesn't support batch mode - use per-frame
        if self.use_inference_slicer:
            return [self.detect(frame, conf_threshold) for frame in frames]

        # Standard batch inference
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
