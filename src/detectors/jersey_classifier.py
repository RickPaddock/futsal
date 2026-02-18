"""
Jersey Classifier - YOLO wrapper for classifying jersey numbers.

Loads JERSEY_MODEL_best_v1.pt and applies JERSEY_CONF_THRESHOLD.
Uses lower confidence threshold (0.3) per memory learnings.

Based on reference: https://github.com/RickPaddock/futsal/blob/rick_claude_2pass_mvp1/src/detection/jersey_classifier.py
"""

from typing import Dict, Optional, Tuple, List
import numpy as np
import cv2
from pathlib import Path

from ..core.constants import JERSEY_CONF_THRESHOLD
from ..utils.logging_utils import get_logger

logger = get_logger("jersey_classifier")


class JerseyClassifier:
    """
    YOLO-based jersey number classifier.

    Returns jersey number with confidence, or None if below threshold.
    Uses conf >= 0.3 for detecting jersey visibility (not 0.5).

    Supports both classification and detection YOLO models.
    """

    def __init__(self, model_path: str = "models/JERSEY_MODEL_best_v1.pt", top_crop_ratio: float = 0.5):
        """
        Initialize jersey classifier.

        Args:
            model_path: Path to YOLO model weights
            top_crop_ratio: Fraction of player bbox to use (0.5 = top half)
        """
        self.model_path = Path(model_path)
        self.model = None
        self.logger = logger
        self.top_crop_ratio = np.clip(top_crop_ratio, 0.1, 1.0)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Jersey model not found: {self.model_path}")

        self._load_model()

    def _load_model(self):
        """Load YOLO model and detect task type."""
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))

        # Auto-detect model task type (classify vs detect)
        self._task = getattr(self.model.model, "task", getattr(self.model, "task", "detect"))
        self._is_classify = (self._task == "classify")

        # Build class mapping (class_idx -> jersey_number)
        self._class_mapping = self._build_class_mapping()

        self.logger.info(f"Loaded jersey model: {self.model_path} (task={self._task}, mode={'classify' if self._is_classify else 'detect'})")

    def _build_class_mapping(self) -> Dict[int, int]:
        """
        Build mapping from model class indices to jersey numbers.

        Returns:
            Dict mapping class_idx -> jersey_number
        """
        mapping = {}
        names = getattr(self.model, "names", None)

        if isinstance(names, dict):
            iterable = names.items()
        elif isinstance(names, (list, tuple)):
            iterable = enumerate(names)
        else:
            iterable = []

        for idx, name in iterable:
            number = self._parse_digit_name(name)
            if number is not None:
                mapping[int(idx)] = number

        self.logger.debug(f"Class mapping: {mapping}")
        return mapping

    @staticmethod
    def _parse_digit_name(name: Optional[str]) -> Optional[int]:
        """
        Parse jersey number from class name.

        Handles aliases like "seven" -> 7, "jersey_10" -> 10, etc.

        Args:
            name: Class name from model

        Returns:
            Jersey number or None
        """
        if not name:
            return None

        normalized = name.strip().lower()

        # Alias map for common names
        alias_map = {
            "0": 0, "zero": 0,
            "1": 1, "one": 1,
            "2": 2, "two": 2,
            "3": 3, "three": 3,
            "4": 4, "four": 4,
            "5": 5, "five": 5,
            "6": 6, "six": 6,
            "7": 7, "07": 7, "seven": 7,
            "8": 8, "eight": 8,
            "9": 9, "nine": 9,
            "10": 10, "ten": 10,
            "11": 11, "eleven": 11,
            "12": 12, "twelve": 12,
        }

        if normalized in alias_map:
            return alias_map[normalized]

        # Extract digits from formatted strings like "jersey_7" or "number_10"
        digits = "".join(ch for ch in normalized if ch.isdigit())
        if digits:
            try:
                return int(digits)
            except ValueError:
                pass

        return None

    def _crop_player_region(
        self,
        frame: np.ndarray,
        bbox: list,
    ) -> Optional[np.ndarray]:
        """
        Extract jersey crop from player bbox (top portion only).

        Args:
            frame: BGR frame
            bbox: Player bbox [x1, y1, x2, y2]

        Returns:
            Cropped image (BGR format for YOLO) or None if invalid
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Clip to frame bounds
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        # Crop top portion (where jersey typically is)
        bbox_height = y2 - y1
        crop_y2 = max(y1 + int(bbox_height * self.top_crop_ratio), y1 + 1)
        crop_y2 = min(y2, crop_y2 + int(bbox_height * 0.05))  # Small buffer

        crop = frame[y1:crop_y2, x1:x2]

        if crop.size == 0:
            return None

        # Frame input is already BGR from VideoReader.
        return crop

    def classify(
        self,
        frame: np.ndarray,
        bbox: list,
        conf_threshold: float = JERSEY_CONF_THRESHOLD,
    ) -> Optional[Tuple[int, float]]:
        """
        Classify jersey number within a player bbox.

        Args:
            frame: BGR frame (H, W, 3)
            bbox: Player bbox [x1, y1, x2, y2]
            conf_threshold: Confidence threshold (default 0.3 per memory learnings)

        Returns:
            (jersey_number, confidence) tuple if detected, else None
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        player_crop = self._crop_player_region(frame, bbox)
        if player_crop is None:
            return None

        # Run YOLO
        results = self.model(player_crop, conf=conf_threshold, verbose=False)
        result = results[0]

        # Handle classification mode
        if self._is_classify:
            if result.probs is None:
                return None

            class_idx = int(result.probs.top1)
            confidence = float(result.probs.top1conf)

            if confidence < conf_threshold:
                return None

            jersey_number = self._class_mapping.get(class_idx, class_idx)
            return (jersey_number, confidence)

        # Handle detection mode
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.cls is None or boxes.conf is None:
            return None

        confidences = boxes.conf.tolist()
        classes = boxes.cls.tolist()

        if not confidences or not classes:
            return None

        # Get best detection
        best_idx = int(np.argmax(confidences))
        cls_id = int(classes[best_idx])
        confidence = float(confidences[best_idx])

        if confidence < conf_threshold:
            return None

        jersey_number = self._class_mapping.get(cls_id, cls_id)
        return (jersey_number, confidence)

    def classify_batch(
        self,
        frame: np.ndarray,
        bboxes: list,
        conf_threshold: float = JERSEY_CONF_THRESHOLD,
    ) -> list:
        """
        Classify jersey numbers for multiple player bboxes.

        Args:
            frame: BGR frame (H, W, 3)
            bboxes: List of player bboxes
            conf_threshold: Confidence threshold

        Returns:
            List of (jersey_number, confidence) tuples or None per bbox.
            Output order matches input bboxes exactly.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if not bboxes:
            return []

        # Prepare crops
        crops: List[np.ndarray] = []
        crop_to_bbox_idx: List[int] = []
        results: List[Optional[Tuple[int, float]]] = [None] * len(bboxes)

        for idx, bbox in enumerate(bboxes):
            player_crop = self._crop_player_region(frame, bbox)
            if player_crop is None:
                continue

            crops.append(player_crop)
            crop_to_bbox_idx.append(idx)

        if not crops:
            return results

        # Run batch inference
        try:
            batch_outputs = self.model(crops, conf=conf_threshold, verbose=False)
        except Exception as exc:
            self.logger.warning(f"Batch inference failed, falling back to sequential: {exc}")
            return [self.classify(frame, bbox, conf_threshold) for bbox in bboxes]

        # Process results based on model type
        for crop_result_idx, result in enumerate(batch_outputs):
            bbox_idx = crop_to_bbox_idx[crop_result_idx]

            if self._is_classify:
                # Classification mode
                if result.probs is None:
                    continue

                class_idx = int(result.probs.top1)
                confidence = float(result.probs.top1conf)

                if confidence >= conf_threshold:
                    jersey_number = self._class_mapping.get(class_idx, class_idx)
                    results[bbox_idx] = (jersey_number, confidence)

            else:
                # Detection mode
                boxes = getattr(result, "boxes", None)
                if boxes is None or boxes.cls is None or boxes.conf is None:
                    continue

                confidences = boxes.conf.tolist()
                classes = boxes.cls.tolist()

                if not confidences or not classes:
                    continue

                best_idx = int(np.argmax(confidences))
                cls_id = int(classes[best_idx])
                confidence = float(confidences[best_idx])

                if confidence >= conf_threshold:
                    jersey_number = self._class_mapping.get(cls_id, cls_id)
                    results[bbox_idx] = (jersey_number, confidence)

        return results

    def get_probabilities(
        self,
        frame: np.ndarray,
        bbox: list,
    ) -> Dict[int, float]:
        """
        Get probability distribution over all jersey numbers.

        Used for Pass 3A candidate generation.

        Args:
            frame: BGR frame (H, W, 3)
            bbox: Player bbox [x1, y1, x2, y2]

        Returns:
            Dict mapping jersey_number -> probability
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        player_crop = self._crop_player_region(frame, bbox)
        if player_crop is None:
            return {}

        # Run YOLO with permissive settings for candidate probability extraction.
        # This path is used for explainability (top-k display / Pass 3 evidence),
        # not for the primary top-1 assignment decision.
        results = self.model(
            player_crop,
            conf=0.01,
            max_det=50,
            iou=0.7,
            verbose=False,
        )
        result = results[0]

        probs_dict = {}

        # Handle classification mode
        if self._is_classify:
            if result.probs is None:
                return {}

            # Get all class probabilities
            all_probs = result.probs.data.cpu().numpy()

            for class_idx, prob in enumerate(all_probs):
                jersey_number = self._class_mapping.get(class_idx)
                if jersey_number is not None and prob > 0.01:  # Filter very low probs
                    probs_dict[jersey_number] = float(prob)

        else:
            # Detection mode - use box confidences as probabilities
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.cls is None or boxes.conf is None:
                return {}

            confidences = boxes.conf.tolist()
            classes = boxes.cls.tolist()

            for cls_id, confidence in zip(classes, confidences):
                jersey_number = self._class_mapping.get(int(cls_id))
                if jersey_number is not None and confidence > 0.01:
                    # Aggregate if same number detected multiple times
                    probs_dict[jersey_number] = max(probs_dict.get(jersey_number, 0.0), float(confidence))

            # Keep only top-k entries for stable/debug-friendly output.
            if len(probs_dict) > 5:
                sorted_items = sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)[:5]
                probs_dict = {k: v for k, v in sorted_items}

        return probs_dict
