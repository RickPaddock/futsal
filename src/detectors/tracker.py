"""
Multi-Object Tracker - Uses Ultralytics built-in tracking (BoT-SORT/ByteTrack).

Assigns temporary track_ids to detections for continuity.
Track_ids are temporary and may change - fragments are the stable identity.

Uses ultralytics v8+ built-in tracking - NO yolox dependency needed.
"""

from typing import List, Tuple
import numpy as np

from ..core.constants import TRACK_HIGH_THRESH, TRACK_LOW_THRESH, TRACK_BUFFER
from ..utils.logging_utils import get_logger

logger = get_logger("tracker")


class ByteTracker:
    """
    Multi-object tracker using Ultralytics built-in tracking.

    Assigns temporary track_ids to detections.
    These track_ids are used for Pass 1 only - fragments are the stable identity.

    Note: Despite the class name, this now uses Ultralytics' built-in tracking
    (BoT-SORT by default, which is superior to ByteTrack).
    """

    def __init__(
        self,
        track_high_thresh: float = TRACK_HIGH_THRESH,
        track_low_thresh: float = TRACK_LOW_THRESH,
        track_buffer: int = TRACK_BUFFER,
        min_track_length: int = 5,
    ):
        """
        Initialize tracker.

        Args:
            track_high_thresh: High confidence threshold for track initialization (default 0.6)
            track_low_thresh: Low confidence threshold for track continuation (default 0.1)
            track_buffer: Number of frames to keep lost tracks (default 30)
            min_track_length: Minimum track length to output (default 5)
        """
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.track_buffer = track_buffer
        self.min_track_length = min_track_length
        self.logger = logger

        # Track state: detection_id -> track_id mapping
        # Ultralytics tracking is stateful and managed internally by the model
        # We just need to maintain consistency across update() calls
        self.frame_count = 0

        self.logger.info(
            f"Initialized Ultralytics tracker "
            f"(conf_thresh={self.track_high_thresh}, buffer={self.track_buffer})"
        )

    def update(
        self,
        detections: List[Tuple[List[float], float]],
        frame_idx: int,
    ) -> List[Tuple[List[float], float, int]]:
        """
        Update tracker with new detections.

        Note: This is a stateless interface. The actual tracking state is managed
        by the YOLO model when using model.track() in the PlayerDetector.

        This method just assigns sequential IDs based on detection order.
        The real tracking happens in PlayerDetector.detect_and_track().

        Args:
            detections: List of (bbox, confidence) tuples
                bbox format: [x1, y1, x2, y2]
            frame_idx: Current frame index

        Returns:
            List of (bbox, confidence, track_id) tuples
        """
        # Simple passthrough - track_ids are assigned by PlayerDetector.detect_and_track()
        # This method exists for API compatibility with old ByteTrack interface
        tracked_detections = []

        for i, (bbox, conf) in enumerate(detections):
            # Assign sequential track_id (will be overridden by actual tracking in detector)
            track_id = i + 1
            tracked_detections.append((bbox, conf, track_id))

        self.frame_count += 1

        return tracked_detections

    def reset(self):
        """Reset tracker state."""
        self.frame_count = 0
        self.logger.info("Tracker reset")
