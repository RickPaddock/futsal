"""
Jersey Classifier - YOLO wrapper for classifying jersey numbers.

Loads JERSEY_MODEL_best_v1.pt and applies JERSEY_CONF_THRESHOLD.
Uses lower confidence threshold (0.3) per memory learnings.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from pathlib import Path

from ..core.constants import JERSEY_CONF_THRESHOLD
from ..utils.logging_utils import get_logger

logger = get_logger("jersey_classifier")


class JerseyClassifier:
    """
    YOLO-based jersey number classifier.

    Returns jersey number with confidence, or None if below threshold.
    Uses conf >= 0.3 for detecting jersey visibility (not 0.5).
    """

    def __init__(self, model_path: str = "models/JERSEY_MODEL_best_v1.pt"):
        """
        Initialize jersey classifier.

        Args:
            model_path: Path to YOLO model weights
        """
        self.model_path = Path(model_path)
        self.model = None
        self.logger = logger

        if not self.model_path.exists():
            raise FileNotFoundError(f"Jersey model not found: {self.model_path}")

        self._load_model()

    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self.logger.info(f"Loaded jersey classification model: {self.model_path}")
        except ImportError:
            raise ImportError(
                "ultralytics package not found. Install with: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load jersey model: {e}")

    def classify(
        self,
        frame: np.ndarray,
        bbox: list,
        conf_threshold: float = JERSEY_CONF_THRESHOLD,
    ) -> Optional[Tuple[int, float]]:
        """
        Classify jersey number within a player bbox.

        Args:
            frame: RGB frame (H, W, 3)
            bbox: Player bbox [x1, y1, x2, y2]
            conf_threshold: Confidence threshold (default 0.3 per memory learnings)

        Returns:
            (jersey_number, confidence) tuple if detected, else None
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Crop to player bbox
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            self.logger.warning(f"Invalid bbox for jersey classification: {bbox}")
            return None

        player_crop = frame[y1:y2, x1:x2]

        # Run YOLO classification
        results = self.model(player_crop, conf=conf_threshold, verbose=False)

        best_jersey = None
        best_conf = 0.0

        for result in results:
            if result.probs is None:
                continue

            # Get top prediction
            top_class_idx = result.probs.top1
            top_conf = float(result.probs.top1conf)

            if top_conf >= conf_threshold and top_conf > best_conf:
                # Map class index to jersey number (depends on model training)
                # Assuming classes are jersey numbers 1-12
                jersey_number = int(top_class_idx) + 1  # Adjust based on actual model

                best_jersey = (jersey_number, top_conf)
                best_conf = top_conf

        if best_jersey:
            self.logger.debug(f"Classified jersey #{best_jersey[0]} with confidence {best_jersey[1]:.3f}")
        else:
            self.logger.debug("No jersey number detected")

        return best_jersey

    def classify_batch(
        self,
        frame: np.ndarray,
        bboxes: list,
        conf_threshold: float = JERSEY_CONF_THRESHOLD,
    ) -> list:
        """
        Classify jersey numbers for multiple player bboxes.

        Args:
            frame: RGB frame (H, W, 3)
            bboxes: List of player bboxes
            conf_threshold: Confidence threshold

        Returns:
            List of (jersey_number, confidence) tuples or None per bbox
        """
        return [self.classify(frame, bbox, conf_threshold) for bbox in bboxes]

    def get_probabilities(
        self,
        frame: np.ndarray,
        bbox: list,
    ) -> Dict[int, float]:
        """
        Get probability distribution over all jersey numbers.

        Used for Pass 3A candidate generation.

        Args:
            frame: RGB frame (H, W, 3)
            bbox: Player bbox [x1, y1, x2, y2]

        Returns:
            Dict mapping jersey_number -> probability
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Crop to player bbox
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return {}

        player_crop = frame[y1:y2, x1:x2]

        # Run YOLO classification
        results = self.model(player_crop, verbose=False)

        probabilities = {}

        for result in results:
            if result.probs is None:
                continue

            # Get all class probabilities
            probs = result.probs.data.cpu().numpy()

            for class_idx, prob in enumerate(probs):
                jersey_number = int(class_idx) + 1  # Adjust based on actual model
                probabilities[jersey_number] = float(prob)

        return probabilities
