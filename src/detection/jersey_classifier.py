"""Lightweight jersey digit classification using YOLO11n-cls."""

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
        device: str = "cuda",
        imgsz: int = 224,
        top_crop_ratio: float = 0.5,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.imgsz = imgsz
        self.top_crop_ratio = np.clip(top_crop_ratio, 0.1, 1.0)
        self._model = YOLO(self.model_path)
        self._model.to(self.device)

    def classify(self, frame: np.ndarray, bbox: BoundingBox) -> tuple[Optional[int], float]:
        """Classify jersey digits using the top region of the player crop."""
        x1, y1, x2, y2 = self._bbox_to_int(frame, bbox)
        if x2 <= x1 or y2 <= y1:
            return None, 0.0

        height = y2 - y1
        crop_y2 = max(y1 + int(height * self.top_crop_ratio), y1 + 1)
        crop = frame[y1:crop_y2, x1:x2]
        if crop.size == 0:
            return None, 0.0

        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        result = self._model(crop_bgr, imgsz=self.imgsz, verbose=False)[0]
        if result.probs is None:
            return None, 0.0

        number = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        return number, confidence

    @staticmethod
    def _bbox_to_int(frame: np.ndarray, bbox: BoundingBox) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, int(bbox.x1)))
        y1 = max(0, min(height - 1, int(bbox.y1)))
        x2 = max(0, min(width, int(np.ceil(bbox.x2))))
        y2 = max(0, min(height, int(np.ceil(bbox.y2))))
        return x1, y1, x2, y2
