"""
Multi-object tracking module using ByteTrack.

Implements ByteTrack algorithm for persistent player tracking across frames.
Uses IoU + velocity consistency for association, with distance and team gating.
Based on: https://github.com/ifzhang/ByteTrack
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass, field

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    raise ImportError("SciPy is required. Install with: pip install scipy")

from src.utils.data_models import BoundingBox, TeamID


@dataclass
class STrack:
    """Single track representation for ByteTrack."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    state: str = "tracked"  # tracked, lost, removed
    frame_id: int = 0
    start_frame: int = 0
    tracklet_len: int = 0
    is_activated: bool = False

    # Kalman filter state
    mean: np.ndarray = field(default_factory=lambda: np.zeros(8))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(8))

    def __post_init__(self):
        if self.mean.sum() == 0:
            self._init_kalman()

    def _init_kalman(self):
        """Initialize Kalman filter state from bbox."""
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.mean = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.covariance = np.eye(8, dtype=np.float32) * 10

    def predict(self):
        """Predict next state using Kalman filter."""
        # Simple constant velocity model
        F = np.eye(8, dtype=np.float32)
        F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = 1  # Position += Velocity

        Q = np.eye(8, dtype=np.float32) * 0.1  # Process noise

        self.mean = F @ self.mean
        self.covariance = F @ self.covariance @ F.T + Q

    def update(self, bbox: np.ndarray, score: float, frame_id: int):
        """Update track with new detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.score = score

        # Measurement
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([cx, cy, w, h], dtype=np.float32)

        # Kalman update
        H = np.eye(4, 8, dtype=np.float32)  # Observation matrix
        # Scale measurement noise by bbox size: larger boxes tolerate more pixel noise
        noise_scale = max(w, h) / 100.0  # normalized to ~100px reference
        R = np.diag(np.array([w, h, w, h], dtype=np.float32)) * noise_scale * 0.5

        y = z - H @ self.mean
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.mean = self.mean + K @ y
        self.covariance = (np.eye(8) - K @ H) @ self.covariance

        # Update bbox from state
        self.bbox = self._state_to_bbox()

    def _state_to_bbox(self) -> np.ndarray:
        """Convert Kalman state to bbox."""
        cx, cy, w, h = self.mean[:4]
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    @property
    def tlbr(self) -> np.ndarray:
        """Get bbox as [x1, y1, x2, y2]."""
        return self._state_to_bbox()


class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Implements the ByteTrack algorithm which associates both high and low
    confidence detections to maintain tracks through occlusions.
    Uses IoU + velocity consistency + distance/team gating for matching.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.7,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        max_center_distance: float = 100.0,
        velocity_weight: float = 0.3,
    ):
        """
        Initialize ByteTracker.

        Args:
            track_high_thresh: Detection threshold for first association
            track_low_thresh: Detection threshold for second association
            new_track_thresh: Threshold for initializing new tracks
            track_buffer: Frames to keep lost tracks
            match_thresh: IoU threshold for matching
            max_center_distance: Maximum pixel distance between track and detection centers for matching.
                                 Prevents ID swaps when players are close together.
            velocity_weight: Weight for velocity consistency in cost [0-1]. Penalizes matches
                           that deviate from predicted Kalman motion direction.
        """
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.max_center_distance = max_center_distance
        self.velocity_weight = velocity_weight

        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

        self.frame_id = 0
        self._next_id = 1

    def reset(self) -> None:
        """Clear all tracker state and restart IDs."""
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        self._next_id = 1

    def _get_next_id(self) -> int:
        """Get next unique track ID."""
        track_id = self._next_id
        self._next_id += 1
        return track_id

    def update(
        self,
        detections: np.ndarray,
        frame_id: int,
        team_preds: Optional[list[Optional[TeamID]]] = None,
        track_team_map: Optional[dict[int, TeamID]] = None,
    ) -> list[STrack]:
        """
        Update tracker with new detections.

        Args:
            detections: Array of shape (N, 5) with [x1, y1, x2, y2, score]
            frame_id: Current frame number
            team_preds: Optional list of team predictions per detection (aligned to detections)
            track_team_map: Optional map of track_id -> locked TeamID for gating associations

        Returns:
            List of active tracks
        """
        self.frame_id = frame_id
        # Clear per-frame removal log
        self.removed_stracks = []

        # Predict existing tracks
        for track in self.tracked_stracks + self.lost_stracks:
            track.predict()

        # Split detections by confidence
        if len(detections) > 0:
            scores = detections[:, 4]
            high_mask = scores >= self.track_high_thresh
            low_mask = (scores >= self.track_low_thresh) & ~high_mask

            detections_high = detections[high_mask]
            detections_low = detections[low_mask]

            team_preds_high = None
            team_preds_low = None
            if team_preds is not None:
                team_preds_high = [p for i, p in enumerate(team_preds) if high_mask[i]]
                team_preds_low = [p for i, p in enumerate(team_preds) if low_mask[i]]
        else:
            detections_high = np.empty((0, 5))
            detections_low = np.empty((0, 5))
            team_preds_high = None
            team_preds_low = None

        # First association: high confidence detections with tracked tracks
        unmatched_tracks = []
        unmatched_detections_high = []

        if len(detections_high) > 0 and len(self.tracked_stracks) > 0:
            cost_matrix = self._compute_cost_matrix(
                self.tracked_stracks,
                detections_high[:, :4],
                det_team_preds=team_preds_high,
                track_team_map=track_team_map,
            )
            matched_indices, unmatched_track_idx, unmatched_det_idx = self._linear_assignment(
                cost_matrix, self.match_thresh
            )

            for track_idx, det_idx in matched_indices:
                self.tracked_stracks[track_idx].update(
                    detections_high[det_idx, :4],
                    detections_high[det_idx, 4],
                    frame_id,
                )

            unmatched_tracks = [self.tracked_stracks[i] for i in unmatched_track_idx]
            unmatched_detections_high = detections_high[unmatched_det_idx]
            if team_preds_high is not None:
                team_preds_high = [team_preds_high[i] for i in unmatched_det_idx]
        else:
            unmatched_tracks = self.tracked_stracks.copy()
            unmatched_detections_high = detections_high

        # Second association: low confidence detections with remaining tracks
        if len(detections_low) > 0 and len(unmatched_tracks) > 0:
            cost_matrix = self._compute_cost_matrix(
                unmatched_tracks,
                detections_low[:, :4],
                det_team_preds=team_preds_low,
                track_team_map=track_team_map,
            )
            matched_indices, unmatched_track_idx, _ = self._linear_assignment(
                cost_matrix, self.match_thresh
            )

            for track_idx, det_idx in matched_indices:
                unmatched_tracks[track_idx].update(
                    detections_low[det_idx, :4],
                    detections_low[det_idx, 4],
                    frame_id,
                )

            # Update unmatched tracks list
            unmatched_tracks = [unmatched_tracks[i] for i in unmatched_track_idx]
            # Update team preds for remaining tracks/dets not needed here (tracks only)

        # Third association: remaining tracks with lost tracks
        # Use stricter distance threshold for lost track reactivation to prevent "teleporting"
        if len(unmatched_detections_high) > 0 and len(self.lost_stracks) > 0:
            cost_matrix = self._compute_cost_matrix(
                self.lost_stracks,
                unmatched_detections_high[:, :4],
                det_team_preds=team_preds_high,
                track_team_map=track_team_map,
            )
            matched_indices, _, unmatched_det_idx = self._linear_assignment(
                cost_matrix, self.match_thresh
            )

            # Filter matches by maximum allowed distance (prevent teleporting)
            # Lost tracks should only reactivate near where they were last seen
            valid_matches = []
            for track_idx, det_idx in matched_indices:
                track = self.lost_stracks[track_idx]
                det_bbox = unmatched_detections_high[det_idx, :4]

                # Calculate distance from track's last known position to detection
                track_cx, track_cy = track.mean[0], track.mean[1]
                det_cx = (det_bbox[0] + det_bbox[2]) / 2
                det_cy = (det_bbox[1] + det_bbox[3]) / 2
                distance = np.sqrt((track_cx - det_cx)**2 + (track_cy - det_cy)**2)

                # Allow more movement for tracks lost longer (they may have moved)
                # But cap it to prevent cross-pitch teleporting
                frames_lost = frame_id - track.frame_id
                max_allowed_distance = min(
                    self.max_center_distance + (frames_lost * 20),  # 20px per frame lost
                    300  # Absolute max - no teleporting across pitch
                )

                if distance <= max_allowed_distance:
                    valid_matches.append((track_idx, det_idx))

            for track_idx, det_idx in valid_matches:
                track = self.lost_stracks[track_idx]
                track.update(
                    unmatched_detections_high[det_idx, :4],
                    unmatched_detections_high[det_idx, 4],
                    frame_id,
                )
                track.state = "tracked"
                self.tracked_stracks.append(track)

            # Remove re-activated tracks from lost
            reactivated_ids = {self.lost_stracks[i].track_id for i, _ in valid_matches}
            self.lost_stracks = [t for t in self.lost_stracks if t.track_id not in reactivated_ids]

            # Update unmatched detections (exclude those matched to valid reactivations)
            valid_det_indices = {det_idx for _, det_idx in valid_matches}
            unmatched_det_idx = [i for i in range(len(unmatched_detections_high))
                                 if i not in valid_det_indices and i in set(unmatched_det_idx)]
            unmatched_detections_high = unmatched_detections_high[unmatched_det_idx] if unmatched_det_idx else np.empty((0, 5))
            if team_preds_high is not None:
                team_preds_high = [team_preds_high[i] for i in unmatched_det_idx] if unmatched_det_idx else []

        # Mark unmatched tracks as lost
        for track in unmatched_tracks:
            track.state = "lost"
            self.lost_stracks.append(track)

        # Remove from tracked
        tracked_ids = {t.track_id for t in unmatched_tracks}
        self.tracked_stracks = [t for t in self.tracked_stracks if t.track_id not in tracked_ids]

        # Initialize new tracks from unmatched high-confidence detections
        # But first check if there's a nearby lost track - prefer reactivating over creating new IDs
        for i, det in enumerate(unmatched_detections_high):
            if det[4] >= self.new_track_thresh:
                det_cx = (det[0] + det[2]) / 2
                det_cy = (det[1] + det[3]) / 2

                # Check if any lost track is close enough to claim this detection
                # This prevents creating new track IDs when a player briefly disappears
                best_lost_track = None
                best_distance = float('inf')

                for lost_track in self.lost_stracks:
                    track_cx, track_cy = lost_track.mean[0], lost_track.mean[1]
                    distance = np.sqrt((track_cx - det_cx)**2 + (track_cy - det_cy)**2)
                    frames_lost = frame_id - lost_track.frame_id

                    # More lenient matching for recently lost tracks
                    max_dist = min(150 + frames_lost * 15, 250)

                    if distance < max_dist and distance < best_distance:
                        best_distance = distance
                        best_lost_track = lost_track

                if best_lost_track is not None:
                    # Reactivate the lost track instead of creating new
                    best_lost_track.update(det[:4], det[4], frame_id)
                    best_lost_track.state = "tracked"
                    self.tracked_stracks.append(best_lost_track)
                    self.lost_stracks.remove(best_lost_track)
                else:
                    # No nearby lost track - create new track
                    new_track = STrack(
                        track_id=self._get_next_id(),
                        bbox=det[:4],
                        score=det[4],
                        frame_id=frame_id,
                        start_frame=frame_id,
                        is_activated=True,
                    )
                    self.tracked_stracks.append(new_track)

        # Remove old lost tracks
        expired_lost = [
            t for t in self.lost_stracks
            if frame_id - t.frame_id > self.track_buffer
        ]
        for track in expired_lost:
            track.state = "removed"
        self.removed_stracks.extend(expired_lost)
        self.lost_stracks = [t for t in self.lost_stracks if t not in expired_lost]

        return [t for t in self.tracked_stracks if t.is_activated]

    def _compute_center_distance_matrix(
        self,
        tracks: list[STrack],
        detections: np.ndarray,
    ) -> np.ndarray:
        """
        Compute center-to-center distance matrix between tracks and detections.

        Args:
            tracks: List of STrack objects
            detections: Detection array (N, 4) with [x1, y1, x2, y2]

        Returns:
            Distance matrix (M, N) with Euclidean distances between centers
        """
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((len(tracks), len(detections)))

        # Get track centers from Kalman state
        track_centers = np.array([[t.mean[0], t.mean[1]] for t in tracks])  # (M, 2)

        # Get detection centers
        det_centers = np.stack([
            (detections[:, 0] + detections[:, 2]) / 2,  # cx
            (detections[:, 1] + detections[:, 3]) / 2,  # cy
        ], axis=1)  # (N, 2)

        # Compute pairwise Euclidean distances
        diff = track_centers[:, None, :] - det_centers[None, :, :]  # (M, N, 2)
        distances = np.sqrt(np.sum(diff ** 2, axis=2))  # (M, N)

        return distances

    def _compute_cost_matrix(
        self,
        tracks: list[STrack],
        detections: np.ndarray,
        det_team_preds: Optional[list[Optional[TeamID]]] = None,
        track_team_map: Optional[dict[int, TeamID]] = None,
    ) -> np.ndarray:
        """
        Compute cost matrix: IoU + velocity consistency, with distance and team gating.

        Args:
            tracks: List of STrack objects
            detections: Detection array (N, 4) with [x1, y1, x2, y2]
            det_team_preds: Optional team predictions per detection
            track_team_map: Optional map of track_id -> locked TeamID

        Returns:
            Cost matrix (M, N) blending IoU and velocity consistency.
            Entries gated by distance or team mismatch are set to 0.
        """
        iou_matrix = self._compute_iou_matrix(
            [t.tlbr for t in tracks],
            detections,
        )

        # Apply perspective-scaled distance gating to prevent ID swaps
        # Scale max_center_distance per track by bbox height relative to a reference
        _REF_HEIGHT = 100.0  # typical mid-pitch player height in pixels
        if self.max_center_distance > 0 and len(tracks) > 0 and len(detections) > 0:
            distance_matrix = self._compute_center_distance_matrix(tracks, detections)
            for i, track in enumerate(tracks):
                bbox_h = max(track.mean[3], 1.0)  # Kalman state index 3 = height
                scale = bbox_h / _REF_HEIGHT
                scaled_gate = self.max_center_distance * scale
                too_far = distance_matrix[i, :] > scaled_gate
                iou_matrix[i, too_far] = 0.0

        # Team gating: zero out matches between different locked teams
        if det_team_preds is not None and track_team_map is not None:
            for i, track in enumerate(tracks):
                locked_team = track_team_map.get(track.track_id)
                if locked_team not in (TeamID.TEAM_A, TeamID.TEAM_B):
                    continue
                for j, team_pred in enumerate(det_team_preds):
                    if team_pred in (TeamID.TEAM_A, TeamID.TEAM_B) and team_pred != locked_team:
                        iou_matrix[i, j] = 0.0

        # Velocity consistency: penalize matches that contradict Kalman-predicted motion
        if self.velocity_weight > 0 and len(tracks) > 0 and len(detections) > 0:
            vel_matrix = self._compute_velocity_consistency(tracks, detections)
            blended = (1.0 - self.velocity_weight) * iou_matrix + self.velocity_weight * vel_matrix
            return blended

        return iou_matrix

    def _compute_velocity_consistency(
        self,
        tracks: list[STrack],
        detections: np.ndarray,
    ) -> np.ndarray:
        """
        Compute velocity consistency score between tracks and detections.

        For each (track, detection) pair, compare the displacement vector
        (track predicted center -> detection center) against the Kalman velocity.
        Score is in [0, 1] where 1 = perfect consistency, 0 = contradicts motion.

        Tracks with near-zero velocity (stationary) get a neutral score of 0.5
        so they don't dominate or penalize any direction.
        """
        n_tracks = len(tracks)
        n_dets = len(detections)
        vel_scores = np.full((n_tracks, n_dets), 0.5, dtype=np.float32)

        # Detection centers
        det_cx = (detections[:, 0] + detections[:, 2]) / 2  # (N,)
        det_cy = (detections[:, 1] + detections[:, 3]) / 2  # (N,)

        for i, track in enumerate(tracks):
            vx, vy = track.mean[4], track.mean[5]
            speed = np.sqrt(vx * vx + vy * vy)

            # Skip stationary tracks -- no directional preference
            if speed < 2.0:
                continue

            # Predicted center (before Kalman update, i.e. current mean after predict())
            pred_cx, pred_cy = track.mean[0], track.mean[1]

            # Displacement from predicted position to each detection
            dx = det_cx - pred_cx  # (N,)
            dy = det_cy - pred_cy  # (N,)
            disp_speed = np.sqrt(dx * dx + dy * dy)

            for j in range(n_dets):
                if disp_speed[j] < 1.0:
                    # Detection is right on top of prediction -- perfect match
                    vel_scores[i, j] = 1.0
                    continue

                # Cosine similarity between velocity vector and displacement vector
                cos_sim = (vx * dx[j] + vy * dy[j]) / (speed * disp_speed[j] + 1e-8)
                # Map from [-1, 1] to [0, 1]: same direction = 1, opposite = 0
                vel_scores[i, j] = (cos_sim + 1.0) / 2.0

        return vel_scores

    def _compute_iou_matrix(
        self,
        bboxes_a: list[np.ndarray],
        bboxes_b: np.ndarray,
    ) -> np.ndarray:
        """Compute IoU matrix between two sets of bboxes."""
        if len(bboxes_a) == 0 or len(bboxes_b) == 0:
            return np.empty((len(bboxes_a), len(bboxes_b)))

        bboxes_a = np.array(bboxes_a)
        bboxes_b = np.array(bboxes_b)

        # Compute intersection
        x1 = np.maximum(bboxes_a[:, None, 0], bboxes_b[None, :, 0])
        y1 = np.maximum(bboxes_a[:, None, 1], bboxes_b[None, :, 1])
        x2 = np.minimum(bboxes_a[:, None, 2], bboxes_b[None, :, 2])
        y2 = np.minimum(bboxes_a[:, None, 3], bboxes_b[None, :, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Compute union
        area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
        area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter

        return inter / (union + 1e-6)
    
    def _linear_assignment(
        self,
        cost_matrix: np.ndarray,
        thresh: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Solve linear assignment problem.

        Returns:
            matched_indices: List of (track_idx, det_idx) pairs
            unmatched_track_idx: List of unmatched track indices
            unmatched_det_idx: List of unmatched detection indices
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        # Convert IoU to cost (1 - IoU)
        cost = 1 - cost_matrix

        # Solve assignment
        row_indices, col_indices = linear_sum_assignment(cost)

        # Filter by threshold
        matched_indices = []
        unmatched_track_idx = set(range(cost_matrix.shape[0]))
        unmatched_det_idx = set(range(cost_matrix.shape[1]))

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] >= thresh:
                matched_indices.append((r, c))
                unmatched_track_idx.discard(r)
                unmatched_det_idx.discard(c)

        return matched_indices, list(unmatched_track_idx), list(unmatched_det_idx)

    def get_active_tracks(self) -> list[STrack]:
        """Get all currently active tracks."""
        return [t for t in self.tracked_stracks if t.is_activated]

    def get_removed_tracks(self) -> list[STrack]:
        """Get tracks that were just removed this frame."""
        return self.removed_stracks


def convert_stracks_to_player_tracks(
    stracks: list[STrack],
    frame_idx: int,
) -> list[tuple[int, BoundingBox]]:
    """
    Convert STrack objects to simplified format for downstream use.

    Args:
        stracks: List of active STracks
        frame_idx: Current frame number

    Returns:
        List of (track_id, bbox) tuples
    """
    results = []
    for track in stracks:
        bbox = BoundingBox(
            x1=float(track.tlbr[0]),
            y1=float(track.tlbr[1]),
            x2=float(track.tlbr[2]),
            y2=float(track.tlbr[3]),
            confidence=track.score,
        )
        results.append((track.track_id, bbox))
    return results
