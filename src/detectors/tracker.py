"""
ByteTrack Tracker - Wrapper for ByteTrack multi-object tracking.

Assigns temporary track_ids to detections for continuity.
Track_ids are temporary and may change - fragments are the stable identity.
"""

from typing import List, Tuple, Optional
import numpy as np

from ..core.constants import TRACK_HIGH_THRESH, TRACK_LOW_THRESH, TRACK_BUFFER, MIN_TRACK_LENGTH
from ..utils.logging_utils import get_logger

logger = get_logger("bytetrack")


class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Assigns temporary track_ids to detections.
    These track_ids are used for Pass 1 only - fragments are the stable identity.
    """

    def __init__(
        self,
        track_high_thresh: float = TRACK_HIGH_THRESH,
        track_low_thresh: float = TRACK_LOW_THRESH,
        track_buffer: int = TRACK_BUFFER,
        min_track_length: int = MIN_TRACK_LENGTH,
    ):
        """
        Initialize ByteTrack tracker.

        Args:
            track_high_thresh: High confidence threshold for first association (default 0.6)
            track_low_thresh: Low confidence threshold for second association (default 0.1)
            track_buffer: Number of frames to keep lost tracks (default 30)
            min_track_length: Minimum track length to output (default 5)
        """
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.track_buffer = track_buffer
        self.min_track_length = min_track_length
        self.tracker = None
        self.logger = logger

        self._initialize_tracker()

    def _initialize_tracker(self):
        """Initialize ByteTrack tracker."""
        try:
            # Try importing from multiple possible locations
            try:
                from yolox.tracker.byte_tracker import BYTETracker
                from yolox.tracker.basetrack import TrackState
            except ImportError:
                # Alternative import path
                from byte_tracker import BYTETracker
                from basetrack import TrackState

            # Create tracker arguments
            class TrackerArgs:
                track_thresh = self.track_high_thresh
                track_buffer = self.track_buffer
                match_thresh = 0.8  # IOU threshold for matching
                mot20 = False  # Use standard MOT metrics

            self.tracker = BYTETracker(TrackerArgs())
            self.TrackState = TrackState
            self.logger.info(
                f"Initialized ByteTracker (high_thresh={self.track_high_thresh}, "
                f"low_thresh={self.track_low_thresh}, buffer={self.track_buffer})"
            )

        except ImportError as e:
            self.logger.warning(
                f"ByteTrack not found ({e}). Using fallback tracker (simple IOU matching)."
            )
            self.tracker = None
            self._initialize_fallback_tracker()

    def _initialize_fallback_tracker(self):
        """Initialize simple fallback tracker if ByteTrack not available."""
        self.fallback_tracks = {}
        self.next_track_id = 1

    def update(
        self,
        detections: List[Tuple[List[float], float]],
        frame_idx: int,
    ) -> List[Tuple[List[float], float, int]]:
        """
        Update tracker with new detections.

        Args:
            detections: List of (bbox, confidence) tuples
                bbox format: [x1, y1, x2, y2]
            frame_idx: Current frame index

        Returns:
            List of (bbox, confidence, track_id) tuples
        """
        if self.tracker is not None:
            return self._update_bytetrack(detections, frame_idx)
        else:
            return self._update_fallback(detections, frame_idx)

    def _update_bytetrack(
        self,
        detections: List[Tuple[List[float], float]],
        frame_idx: int,
    ) -> List[Tuple[List[float], float, int]]:
        """Update using ByteTrack."""
        if not detections:
            return []

        # Convert to ByteTrack format: [x1, y1, x2, y2, score]
        dets = np.array([[*bbox, conf] for bbox, conf in detections], dtype=np.float32)

        # Update tracker
        online_targets = self.tracker.update(dets, [frame_idx] * len(dets), [frame_idx] * len(dets))

        # Convert back to our format
        tracked_detections = []
        for track in online_targets:
            bbox = track.tlbr.tolist()  # [x1, y1, x2, y2]
            track_id = track.track_id
            conf = track.score

            tracked_detections.append((bbox, float(conf), int(track_id)))

        return tracked_detections

    def _update_fallback(
        self,
        detections: List[Tuple[List[float], float]],
        frame_idx: int,
    ) -> List[Tuple[List[float], float, int]]:
        """Update using simple IOU-based fallback tracker."""
        from ..utils.geometry import iou

        tracked_detections = []

        # Match detections to existing tracks using IOU
        matched_tracks = set()

        for bbox, conf in detections:
            best_track_id = None
            best_iou = 0.3  # Minimum IOU threshold

            # Try to match to existing tracks
            for track_id, track_bbox in self.fallback_tracks.items():
                iou_score = iou(bbox, track_bbox)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_track_id = track_id

            if best_track_id is not None:
                # Matched to existing track
                track_id = best_track_id
                self.fallback_tracks[track_id] = bbox
                matched_tracks.add(track_id)
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                self.fallback_tracks[track_id] = bbox
                matched_tracks.add(track_id)

            tracked_detections.append((bbox, conf, track_id))

        # Remove unmatched tracks (simple - no buffer)
        self.fallback_tracks = {
            tid: bbox for tid, bbox in self.fallback_tracks.items()
            if tid in matched_tracks
        }

        return tracked_detections

    def reset(self):
        """Reset tracker state."""
        if self.tracker is not None:
            self._initialize_tracker()
        else:
            self._initialize_fallback_tracker()

        self.logger.info("Tracker reset")
