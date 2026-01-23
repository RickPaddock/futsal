"""
Player detection module supporting both local YOLO and Roboflow models.

Detects players (persons) in video frames using supervision for detection handling.
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import supervision as sv

from src.utils.data_models import BoundingBox, Detection, PlayerDetection

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass



class PlayerDetector:
    """Player detector supporting local YOLO or Roboflow models."""

    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        classes: list[int] = None,
        use_roboflow: bool = False,
        roboflow_model_id: str = None,
        max_detections: int = 12,
        input_scale: float = 1.0,
    ):
        """
        Initialize the player detector.

        Args:
            model_path: Path to YOLO weights or model name (for local mode)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS (lower = more aggressive duplicate suppression)
            device: Device to run inference on ('cuda' or 'cpu')
            classes: List of class IDs to detect (default [0] for person in COCO)
            use_roboflow: If True, use Roboflow inference API
            roboflow_model_id: Roboflow model ID (e.g., "football-players-detection-3zvbc/20")
            max_detections: Maximum number of detections per frame
            input_scale: Scale factor for input frames (0.5 = half resolution, faster)
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes if classes is not None else [0]
        self.use_roboflow = use_roboflow
        self.max_detections = max_detections
        self.input_scale = input_scale
        self.model = None
        self.class_names = {}

        if use_roboflow and roboflow_model_id:
            self._init_roboflow(roboflow_model_id)
        else:
            self._init_ultralytics(model_path)

    def _init_ultralytics(self, model_path: str):
        """Initialize local Ultralytics YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Ultralytics is required. Install with: pip install ultralytics")

        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.class_names = self.model.names
        self.use_roboflow = False

    def _init_roboflow(self, model_id: str):
        """Initialize Roboflow inference model."""
        try:
            from inference import get_model
        except ImportError:
            raise ImportError("Roboflow inference is required. Install with: pip install inference")

        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY not found in environment. Add it to .env file.")

        self.model = get_model(model_id=model_id, api_key=api_key)
        self.roboflow_model_id = model_id
        self.use_roboflow = True

        # Roboflow football models typically have these classes
        # Will be populated from first inference
        self.class_names = {}

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[PlayerDetection]:
        """
        Detect players in a single frame.

        Args:
            frame: RGB numpy array (H, W, 3)
            frame_idx: Frame number for tracking

        Returns:
            List of PlayerDetection objects
        """
        if self.use_roboflow:
            return self._detect_roboflow(frame, frame_idx)
        else:
            return self._detect_ultralytics(frame, frame_idx)

    def _detect_ultralytics(self, frame: np.ndarray, frame_idx: int) -> list[PlayerDetection]:
        """Detect using local Ultralytics model."""
        # Scale down input for faster inference
        if self.input_scale != 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.input_scale)
            new_h = int(h * self.input_scale)
            scaled_frame = cv2.resize(frame, (new_w, new_h))
            scale_factor = 1.0 / self.input_scale
        else:
            scaled_frame = frame
            scale_factor = 1.0

        # For COCO models, filter to person class (0)
        results = self.model(
            scaled_frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            max_det=self.max_detections,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            # Scale coordinates back to original frame size
            bbox = BoundingBox(
                x1=float(box.xyxy[0][0]) * scale_factor,
                y1=float(box.xyxy[0][1]) * scale_factor,
                x2=float(box.xyxy[0][2]) * scale_factor,
                y2=float(box.xyxy[0][3]) * scale_factor,
                confidence=float(box.conf[0]),
            )

            # All detections are players (class filtering done in model call)
            detection = PlayerDetection(
                frame_idx=frame_idx,
                bbox=bbox,
                class_id=2,
                class_name="player",
            )
            detections.append(detection)

        return detections

    def _detect_roboflow(self, frame: np.ndarray, frame_idx: int) -> list[PlayerDetection]:
        """Detect using Roboflow inference API with supervision."""
        import supervision as sv

        result = self.model.infer(
            frame,
            confidence=self.confidence_threshold,
            iou_threshold=self.iou_threshold
        )[0]

        # Use supervision for cleaner detection handling
        sv_detections = sv.Detections.from_inference(result)

        # Convert all detections to PlayerDetection objects
        detections = []
        for i in range(len(sv_detections)):
            x1, y1, x2, y2 = sv_detections.xyxy[i]
            confidence = sv_detections.confidence[i]

            bbox = BoundingBox(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(confidence),
            )

            detection = PlayerDetection(
                frame_idx=frame_idx,
                bbox=bbox,
                class_id=2,  # Normalize all to player
                class_name="player",
            )
            detections.append(detection)

        return detections

    def detect_batch(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
    ) -> list[list[PlayerDetection]]:
        """
        Detect players in a batch of frames.

        Args:
            frames: List of RGB numpy arrays
            frame_indices: Corresponding frame numbers

        Returns:
            List of detection lists, one per frame
        """
        if self.use_roboflow:
            # Roboflow doesn't support true batch inference in the same way
            # Process frames one by one
            return [self.detect(frame, idx) for frame, idx in zip(frames, frame_indices)]

        # Scale down input for faster inference
        if self.input_scale != 1.0:
            h, w = frames[0].shape[:2]
            new_w = int(w * self.input_scale)
            new_h = int(h * self.input_scale)
            scaled_frames = [cv2.resize(f, (new_w, new_h)) for f in frames]
            scale_factor = 1.0 / self.input_scale
        else:
            scaled_frames = frames
            scale_factor = 1.0

        # Ultralytics batch inference
        results = self.model(
            scaled_frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            max_det=self.max_detections,
            verbose=False,
        )

        all_detections = []
        for result, frame_idx in zip(results, frame_indices):
            frame_detections = []
            for box in result.boxes:
                # Scale coordinates back to original frame size
                bbox = BoundingBox(
                    x1=float(box.xyxy[0][0]) * scale_factor,
                    y1=float(box.xyxy[0][1]) * scale_factor,
                    x2=float(box.xyxy[0][2]) * scale_factor,
                    y2=float(box.xyxy[0][3]) * scale_factor,
                    confidence=float(box.conf[0]),
                )

                # All detections are players (class filtering done in model call)
                detection = PlayerDetection(
                    frame_idx=frame_idx,
                    bbox=bbox,
                    class_id=2,
                    class_name="player",
                )
                frame_detections.append(detection)

            all_detections.append(frame_detections)

        return all_detections

    def get_detection_array(
        self,
        detections: list[PlayerDetection],
    ) -> np.ndarray:
        """
        Convert detections to numpy array for tracker input.

        Args:
            detections: List of PlayerDetection objects

        Returns:
            Array of shape (N, 5) with [x1, y1, x2, y2, confidence]
        """
        if not detections:
            return np.empty((0, 5))

        return np.array([
            [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.bbox.confidence]
            for d in detections
        ])


def filter_detections_by_size(
    detections: list[PlayerDetection],
    min_height: int = 30,
    max_height: int = 800,
    min_aspect_ratio: float = 0.2,
    max_aspect_ratio: float = 2.0,
) -> list[PlayerDetection]:
    """
    Filter detections by size constraints.

    Removes detections that are too small, too large, or have unusual aspect ratios.

    Args:
        detections: List of detections to filter
        min_height: Minimum bbox height in pixels
        max_height: Maximum bbox height in pixels
        min_aspect_ratio: Minimum width/height ratio
        max_aspect_ratio: Maximum width/height ratio

    Returns:
        Filtered list of detections
    """
    filtered = []
    for det in detections:
        height = det.bbox.height
        aspect_ratio = det.bbox.width / height if height > 0 else 0

        if (
            min_height <= height <= max_height
            and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio
        ):
            filtered.append(det)

    return filtered


def filter_detections_by_region(
    detections: list[PlayerDetection],
    roi_x1: float = 0,
    roi_y1: float = 0,
    roi_x2: float = float('inf'),
    roi_y2: float = float('inf'),
) -> list[PlayerDetection]:
    """
    Filter detections to those within a region of interest.

    Args:
        detections: List of detections to filter
        roi_x1, roi_y1: Top-left corner of ROI
        roi_x2, roi_y2: Bottom-right corner of ROI

    Returns:
        Detections with centers inside the ROI
    """
    filtered = []
    for det in detections:
        cx, cy = det.bbox.center
        if roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2:
            filtered.append(det)
    return filtered

def extract_color_histogram(
    frame: np.ndarray,
    bbox: BoundingBox,
    bins: int = 32,
) -> np.ndarray:
    """
    Extract HSV color histogram from a bounding box region.

    Args:
        frame: RGB frame (H, W, 3)
        bbox: Bounding box to extract from
        bins: Number of bins per HSV channel (default 32)

    Returns:
        Flattened histogram of shape (3*bins,) - HSV channels concatenated
    """
    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
    
    # Ensure bbox is within frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        # Invalid bbox, return zero histogram
        return np.zeros(3 * bins, dtype=np.float32)
    
    # Extract region
    roi = frame[y1:y2, x1:x2]
    
    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    
    # Compute histograms for each channel
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    
    # Normalize and concatenate
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    
    return np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)