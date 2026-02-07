"""Lightweight jersey digit identification using YOLO models (adapted for Pass 1).

For Pass 1, we need to return full probability vectors per frame (NO voting).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - explicit error for missing dependency
    raise ImportError("Ultralytics is required for jersey classification. Install with: pip install ultralytics") from exc

from src.utils.data_models import BoundingBox


class JerseyClassifier:
    """Wrapper around a YOLO classification model for jersey digits."""

    def __init__(
        self,
        model_path: str | Path,
        imgsz: int = 224,
        top_crop_ratio: float = 0.5,
    ) -> None:
        self.model_path = str(model_path)
        self.imgsz = imgsz
        self.top_crop_ratio = np.clip(top_crop_ratio, 0.1, 1.0)
        self._model = YOLO(self.model_path)
        self._model.to('cuda')
        print(f"[Jersey Classifier] Loaded model from {self.model_path} on device cuda")
        self._task: str = getattr(self._model.model, "task", getattr(self._model, "task", "detect"))
        self._is_classify = self._task == "classify"
        self._class_mapping = self._build_class_mapping()

    def classify(self, frame: np.ndarray, bbox: BoundingBox) -> tuple[Optional[int], float]:
        """
        Classify jersey digits using the top region of the player crop.

        Returns only the top-1 prediction (for backward compatibility).
        For Pass 1, use get_probabilities() instead to get full probability vector.
        """
        x1, y1, x2, y2 = self._bbox_to_int(frame, bbox)
        if x2 <= x1 or y2 <= y1:
            return None, 0.0

        height = y2 - y1
        crop_y2 = max(y1 + int(height * self.top_crop_ratio), y1 + 1)
        crop_y2 = min(y2, crop_y2 + int(height * 0.05))
        crop = frame[y1:crop_y2, x1:x2]
        if crop.size == 0:
            return None, 0.0

        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        result = self._model(crop_bgr, imgsz=self.imgsz, verbose=False)[0]
        if self._is_classify:
            if result.probs is None:
                return None, 0.0

            class_idx = int(result.probs.top1)
            number = self._class_mapping.get(class_idx, class_idx)
            confidence = float(result.probs.top1conf)
            return number, confidence

        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.cls is None or boxes.conf is None:
            return None, 0.0

        confidences = boxes.conf.tolist()
        classes = boxes.cls.tolist()
        if not confidences or not classes:
            return None, 0.0

        best_idx = int(np.argmax(confidences))
        cls_id = int(classes[best_idx])
        confidence = float(confidences[best_idx])
        number = self._class_mapping.get(cls_id)

        if number is None:
            # Fallback attempt: parse class name directly
            name = None
            names = getattr(self._model, "names", None)
            if isinstance(names, dict):
                name = names.get(cls_id)
            elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                name = names[cls_id]
            number = self._parse_digit_name(name)

        if number is None:
            return None, 0.0

        return number, confidence

    def get_probabilities(self, frame: np.ndarray, bbox: BoundingBox) -> dict[int, float]:
        """
        Get full probability distribution for all jersey numbers (Pass 1 version).

        Args:
            frame: RGB frame
            bbox: Player bounding box

        Returns:
            Dictionary mapping jersey number -> probability
            Empty dict if classification fails
        """
        x1, y1, x2, y2 = self._bbox_to_int(frame, bbox)
        if x2 <= x1 or y2 <= y1:
            return {}

        height = y2 - y1
        crop_y2 = max(y1 + int(height * self.top_crop_ratio), y1 + 1)
        crop_y2 = min(y2, crop_y2 + int(height * 0.05))
        crop = frame[y1:crop_y2, x1:x2]
        if crop.size == 0:
            return {}

        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        result = self._model(crop_bgr, imgsz=self.imgsz, verbose=False)[0]

        probs_dict = {}

        if self._is_classify:
            if result.probs is None:
                return {}

            # Get all class probabilities
            all_probs = result.probs.data.cpu().numpy()  # Shape: (num_classes,)
            for class_idx, prob in enumerate(all_probs):
                number = self._class_mapping.get(class_idx)
                if number is not None and prob > 0.01:  # Filter very low probs
                    probs_dict[number] = float(prob)

        else:
            # Detection mode - get probabilities from boxes
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.cls is None or boxes.conf is None:
                return {}

            confidences = boxes.conf.tolist()
            classes = boxes.cls.tolist()

            for cls_id, confidence in zip(classes, confidences):
                number = self._class_mapping.get(int(cls_id))
                if number is not None and confidence > 0.01:
                    # Aggregate probabilities if same number detected multiple times
                    probs_dict[number] = max(probs_dict.get(number, 0.0), float(confidence))

        return probs_dict

    @staticmethod
    def _bbox_to_int(frame: np.ndarray, bbox: BoundingBox) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, int(bbox.x1)))
        y1 = max(0, min(height - 1, int(bbox.y1)))
        x2 = max(0, min(width, int(np.ceil(bbox.x2))))
        y2 = max(0, min(height, int(np.ceil(bbox.y2))))
        return x1, y1, x2, y2

    def _build_class_mapping(self) -> dict[int, int]:
        mapping: dict[int, int] = {}
        names = getattr(self._model, "names", None)
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
        return mapping

    @staticmethod
    def _parse_digit_name(name: Optional[str]) -> Optional[int]:
        if not name:
            return None
        normalized = name.strip().lower()
        alias_map = {
            "0": 0,
            "zero": 0,
            "4": 4,
            "four": 4,
            "07": 7,
            "7": 7,
            "seven": 7,
            "10": 10,
            "ten": 10,
        }
        if normalized in alias_map:
            return alias_map[normalized]

        # Extract standalone integers if present (e.g., "jersey_7")
        digits = "".join(ch for ch in normalized if ch.isdigit())
        if digits:
            try:
                return int(digits)
            except ValueError:
                return None
        return None
