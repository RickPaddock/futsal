"""
Ball detection module for futsal tracking.

Detects balls in video frames using a YOLOv11 model trained on tiled GoPro footage.
Uses supervision.InferenceSlicer for small object detection (per Roboflow blog).
"""

from pathlib import Path
from typing import Optional
from collections import deque
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv

from src.utils.data_models import BoundingBox, Detection


class BallTracker:
    """
    Simple ball tracker to filter false positives.
    
    Per Roboflow blog: keeps buffer of recent detections, calculates centroid,
    and selects the ball closest to centroid (assumes single ball, physically realistic motion).
    """

    def __init__(self, buffer_size: int = 10, max_distance: float = 500.0):
        """
        Initialize ball tracker.

        Args:
            buffer_size: Number of previous positions to keep in buffer
            max_distance: Max pixels ball can move from centroid (filters anomalies)
        """
        self.buffer = deque(maxlen=buffer_size)
        self.max_distance = max_distance

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Filter detections by selecting one closest to centroid of recent positions.

        Args:
            detections: Supervision Detections object

        Returns:
            Filtered detections (0 or 1 ball)
        """
        if len(detections) == 0:
            return detections

        # Get center coordinates of each detection
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        # Calculate centroid of all recent positions
        if len(self.buffer) > 0:
            all_positions = np.concatenate(list(self.buffer), axis=0)
            centroid = np.mean(all_positions, axis=0)

            # Find detection closest to centroid
            distances = np.linalg.norm(xy - centroid, axis=1)
            closest_idx = np.argmin(distances)
            closest_distance = distances[closest_idx]

            # Only return if within max distance (filters anomalies)
            if closest_distance <= self.max_distance:
                return detections[[closest_idx]]

        return sv.Detections.empty()


class BallDetector:
    """Ball detector using YOLO model with InferenceSlicer for small objects."""

    def __init__(
        self,
        model_path: str = "models/futsal_ball_detector.pt",
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.1,
        device: str = "cuda",
        max_detections: int = 100,  # Allow all detections to see raw model performance
        input_scale: float = 1.0,
        use_inference_slicer: bool = False,
    ):
        """
        Initialize the ball detector.

        Args:
            model_path: Path to YOLOv11 ball detector weights
            confidence_threshold: Minimum confidence for detections (0.3 per Roboflow blog)
            iou_threshold: IoU threshold for NMS (0.1 per Roboflow blog)
            device: Device to run inference on ('cuda' or 'cpu')
            max_detections: Maximum detections per frame (allow multiples, tracker filters to 1)
            input_scale: Input image scale (1.0 = full resolution for InferenceSlicer)
            use_inference_slicer: Use overlapping tiles at inference for small object detection
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.max_detections = max_detections  # Allow all detections, no restriction
        self.input_scale = input_scale
        self.use_inference_slicer = use_inference_slicer

        # Load YOLO model
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Ball detector model not found at {model_path}")

        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Slicer will be lazily initialized on first frame (needs frame dimensions)
        self.slicer = None
        
        # Initialize ball tracker to filter anomalies
        self.tracker = BallTracker(buffer_size=10, max_distance=500.0)

        print(f"[Ball Detector] Loaded model from {model_path} on device {device}")
        if self.use_inference_slicer:
            print(f"[Ball Detector] InferenceSlicer enabled (will init on first frame)")

    def _ensure_slicer(self, frame_width: int, frame_height: int):
        """
        Lazily initialize InferenceSlicer with frame dimensions.

        Uses Roboflow blog formula: creates 2x2 grid with overlap.
        For 4K (3840x2160): tiles are (2020, 1180) pixels.
        """
        if self.slicer is not None:
            return

        def callback(frame_slice: np.ndarray) -> sv.Detections:
            """Inference callback for each tile."""
            results = self.model(
                frame_slice,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False,
            )
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    dets = self._get_detection_array(result)
                    # Debug: show tile detection with coords (commented to reduce noise)
                    # print(f"    [Tile] Found {len(dets)} det(s)")
                    return dets
            return sv.Detections.empty()

        # Creates 2x2 grid with 200px overlap (~10% of tile, optimal for small objects)
        # For 3840x2160: (1920+200, 1080+200) = (2120, 1280)
        tile_w = frame_width // 2 + 200
        tile_h = frame_height // 2 + 200

        self.slicer = sv.InferenceSlicer(
            callback=callback,
            slice_wh=(tile_w, tile_h),
            overlap_wh=(200, 200),
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            iou_threshold=self.iou_threshold,  # Use config value (0.05)
        )
        print(f"[Ball Detector] InferenceSlicer: {tile_w}x{tile_h} tiles (2x2 grid + 100px overlap)")

    def detect_batch(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
    ) -> list[list[Detection]]:
        """
        Detect balls in a batch of frames.

        Args:
            frames: List of BGR frames (numpy arrays)
            frame_indices: List of frame indices (for tracking)

        Returns:
            List of lists of Detection objects (one list per frame)
        """
        batch_results = []

        for frame_idx, frame in zip(frame_indices, frames):
            detections = self.detect_frame(frame, frame_idx)
            batch_results.append(detections)

        return batch_results

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> list[Detection]:
        """
        Detect balls in a single frame.

        Args:
            frame: BGR frame (numpy array)
            frame_idx: Frame index

        Returns:
            List of Detection objects
        """
        h, w = frame.shape[:2]

        # Scale frame if requested
        if self.input_scale != 1.0:
            scaled_w = int(w * self.input_scale)
            scaled_h = int(h * self.input_scale)
            scaled_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        else:
            scaled_frame = frame
            scaled_w, scaled_h = w, h

        detections = []

        # Initialize slicer on first frame if enabled (needs frame dimensions)
        if self.use_inference_slicer:
            self._ensure_slicer(scaled_w, scaled_h)

        # Run inference with or without slicer
        if self.use_inference_slicer and self.slicer is not None:
            # InferenceSlicer handles the callback and tiling automatically
            detections_sv = self.slicer(scaled_frame)
            if frame_idx % 30 == 0:  # Print every 30 frames (~1 sec)
                if len(detections_sv) > 0:
                    coords = detections_sv.xyxy[0]
                    print(f"[Ball] Frame {frame_idx}: {len(detections_sv)} det(s) after NMS, first at ({coords[0]:.0f},{coords[1]:.0f})-({coords[2]:.0f},{coords[3]:.0f})")
                else:
                    print(f"[Ball] Frame {frame_idx}: 0 detections after NMS")
        else:
            # Standard full-frame inference
            results = self.model(
                scaled_frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False,
            )
            if results and len(results) > 0:
                result = results[0]
                detections_sv = self._get_detection_array(result)
            else:
                detections_sv = sv.Detections.empty()

        # Apply BallTracker to filter anomalies (per Roboflow blog)
        # DISABLED: We need raw model performance first before applying filters
        # detections_sv = self.tracker.update(detections_sv)

        # Convert supervision Detections to our Detection objects
        if len(detections_sv) > 0:
            boxes = detections_sv.xyxy
            confidences = detections_sv.confidence if detections_sv.confidence is not None else np.ones(len(boxes))

            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box

                # Scale back to original frame size if needed
                if self.input_scale != 1.0:
                    scale_x = w / scaled_w
                    scale_y = h / scaled_h
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y

                bbox = BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(conf),
                )

                det = Detection(
                    frame_idx=frame_idx,
                    bbox=bbox,
                    class_id=0,
                    class_name="ball",
                )
                detections.append(det)

        return detections

    def _get_detection_array(self, result) -> sv.Detections:
        """Convert YOLO result to supervision Detections."""
        if result.boxes is None or len(result.boxes) == 0:
            return sv.Detections.empty()

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = np.zeros(len(boxes), dtype=int)

        return sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids,
        )

    def get_detection_array(self, detections: list[Detection]) -> np.ndarray:
        """
        Convert Detection objects to supervision detection array format.

        Args:
            detections: List of Detection objects

        Returns:
            Supervision Detections object
        """
        if not detections:
            return sv.Detections.empty()

        boxes = np.array([
            [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]
            for d in detections
        ])

        confidences = np.array([d.bbox.confidence for d in detections])
        class_ids = np.array([d.class_id for d in detections])

        return sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids,
        )
