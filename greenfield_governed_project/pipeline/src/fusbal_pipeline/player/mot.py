# PROV: FUSBAL.PIPELINE.PLAYER.MOT.01
# REQ: FUSBAL-V1-PLAYER-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Provide a conservative swap-avoidant MOT that prefers explicit breaks over identity swaps.

from __future__ import annotations

from dataclasses import dataclass, field

from ..contract import TrackRecordV1
from ..diagnostics_keys import (
    ASSOCIATION_SCORE,
    REASON,
    BREAK_AMBIGUOUS,
    BREAK_DETECTOR_MISSING,
    BREAK_OCCLUSION,
    BREAK_OUT_OF_VIEW,
    UNKNOWN_REASON,
    UNKNOWN_REASON_AMBIGUOUS,
)
from .track_types import BreakReason, MotConfig


def _bbox_center_xy(bbox: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _association_score(distance_px: float, *, max_distance_px: float) -> float:
    if max_distance_px <= 0:
        return 0.0
    score = 1.0 - (distance_px / max_distance_px)
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return float(score)


@dataclass
class _ActiveTrack:
    track_id: str
    segment_id: str
    last_center_xy: tuple[float, float]
    last_t_ms: int
    confidence: float
    frames_missing: int = 0


@dataclass
class SwapAvoidantMOT:
    source: str
    cfg: MotConfig = field(default_factory=MotConfig)

    _next_track_seq: int = 1
    _next_segment_seq: int = 1
    _active: dict[str, _ActiveTrack] = field(default_factory=dict)

    def _new_track_id(self) -> str:
        tid = f"trk_{self._next_track_seq:04d}"
        self._next_track_seq += 1
        return tid

    def _new_segment_id(self) -> str:
        sid = f"seg_{self._next_segment_seq:04d}"
        self._next_segment_seq += 1
        return sid

    def update(
        self,
        *,
        t_ms: int,
        detections: list[TrackRecordV1],
    ) -> list[TrackRecordV1]:
        """Update tracker from detection-style player records (frame=image_px).

        Ambiguous associations trigger breaks; detections start new tracks.
        """
        
        # Enforce bounded state - evict oldest tracks if over limit
        if len(self._active) > self.cfg.max_active_tracks:
            # Sort by last update time and evict oldest
            sorted_tracks = sorted(self._active.items(), key=lambda x: x[1].last_t_ms)
            tracks_to_evict = len(self._active) - self.cfg.max_active_tracks
            for track_id, _ in sorted_tracks[:tracks_to_evict]:
                # Create a temporary end_track function for bounded state cleanup
                trk = self._active.get(track_id)
                if trk:
                    out.append({
                        "schema_version": 1,
                        "t_ms": int(t_ms),
                        "entity_type": "player", 
                        "entity_id": track_id,
                        "track_id": track_id,
                        "segment_id": trk.segment_id,
                        "source": str(self.source),
                        "frame": "image_px",
                        "pos_state": "missing",
                        "confidence": float(trk.confidence),
                        "break_reason": "out_of_view",
                        "diagnostics": {ASSOCIATION_SCORE: 0.0, REASON: "bounded_state_eviction"},
                    })
                    del self._active[track_id]

        dets = [
            d
            for d in detections
            if d.get("entity_type") == "player"
            and d.get("frame") == "image_px"
            and d.get("pos_state") == "present"
            and isinstance(d.get("bbox_xyxy_px"), list)
        ]
        det_centers: list[tuple[int, tuple[float, float]]] = []
        for i, d in enumerate(dets):
            bbox = d.get("bbox_xyxy_px")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            det_centers.append((i, _bbox_center_xy([int(x) for x in bbox])))

        # Build all candidate associations.
        candidates: list[tuple[str, int, float, float]] = []
        for track_id, trk in self._active.items():
            for det_idx, cxy in det_centers:
                dist_px = _dist(trk.last_center_xy, cxy)
                if dist_px > self.cfg.max_center_distance_px:
                    continue
                score = _association_score(dist_px, max_distance_px=self.cfg.max_center_distance_px)
                candidates.append((track_id, det_idx, score, dist_px))

        # Greedy matching by score (deterministic).
        candidates.sort(key=lambda x: (-x[2], x[3], x[0], x[1]))

        assigned_tracks: dict[str, int] = {}
        assigned_dets: dict[int, str] = {}
        ambiguous_tracks: set[str] = set()
        ambiguous_dets: set[int] = set()

        for track_id, det_idx, score, _dist_px in candidates:
            if score < self.cfg.min_association_score:
                continue
            # If the track or detection is already assigned, mark ambiguity instead of swapping.
            if track_id in assigned_tracks and assigned_tracks[track_id] != det_idx:
                ambiguous_tracks.add(track_id)
                ambiguous_dets.add(det_idx)
                ambiguous_dets.add(assigned_tracks[track_id])
                continue
            if det_idx in assigned_dets and assigned_dets[det_idx] != track_id:
                ambiguous_dets.add(det_idx)
                ambiguous_tracks.add(track_id)
                ambiguous_tracks.add(assigned_dets[det_idx])
                continue
            assigned_tracks[track_id] = det_idx
            assigned_dets[det_idx] = track_id

        out: list[TrackRecordV1] = []

        def end_track(track_id: str, *, break_reason: BreakReason) -> None:
            trk = self._active.get(track_id)
            if not trk:
                return
            out.append(
                {
                    "schema_version": 1,
                    "t_ms": int(t_ms),
                    "entity_type": "player",
                    "entity_id": track_id,
                    "track_id": track_id,
                    "segment_id": trk.segment_id,
                    "source": str(self.source),
                    "frame": "image_px",
                    "pos_state": "missing",
                    "confidence": float(trk.confidence),
                    "break_reason": break_reason,
                    "diagnostics": {ASSOCIATION_SCORE: 0.0, REASON: str(break_reason)},
                }
            )
            del self._active[track_id]

        # Explicitly break ambiguous tracks.
        for track_id in sorted(ambiguous_tracks):
            end_track(track_id, break_reason="ambiguous_association")

        # End tracks that were not matched (with out_of_view heuristic).
        if not det_centers:
            for track_id in sorted(self._active.keys()):
                end_track(track_id, break_reason="detector_missing")
        else:
            for track_id in sorted(list(self._active.keys())):
                if track_id in assigned_tracks:
                    # Reset missing counter for matched tracks
                    self._active[track_id].frames_missing = 0
                    continue
                
                # Increment missing counter
                self._active[track_id].frames_missing += 1
                
                # Determine break reason using heuristic
                track = self._active[track_id]
                min_dist_to_detection = float('inf')
                for _, det_center in det_centers:
                    dist = _dist(track.last_center_xy, det_center)
                    min_dist_to_detection = min(min_dist_to_detection, dist)
                
                # Out of view if far from all detections or timed out
                if (min_dist_to_detection > self.cfg.out_of_view_distance_px or 
                    track.frames_missing >= self.cfg.track_timeout_frames):
                    end_track(track_id, break_reason="out_of_view")
                else:
                    end_track(track_id, break_reason="occlusion")

        # Emit matched updates.
        for track_id in sorted(assigned_tracks.keys()):
            if track_id not in self._active:
                continue
            det = dets[assigned_tracks[track_id]]
            bbox = det.get("bbox_xyxy_px")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            center = _bbox_center_xy([int(x) for x in bbox])
            dist_px = _dist(self._active[track_id].last_center_xy, center)
            score = _association_score(dist_px, max_distance_px=self.cfg.max_center_distance_px)
            conf = float(det.get("confidence") or 0.0) * float(score)
            self._active[track_id] = _ActiveTrack(
                track_id=track_id,
                segment_id=self._active[track_id].segment_id,
                last_center_xy=center,
                last_t_ms=int(t_ms),
                confidence=conf,
                frames_missing=0,
            )
            out.append(
                {
                    "schema_version": 1,
                    "t_ms": int(t_ms),
                    "entity_type": "player",
                    "entity_id": track_id,
                    "track_id": track_id,
                    "segment_id": self._active[track_id].segment_id,
                    "source": str(self.source),
                    "frame": "image_px",
                    "pos_state": "present",
                    "bbox_xyxy_px": [int(x) for x in bbox],
                    "confidence": conf,
                    "diagnostics": {ASSOCIATION_SCORE: float(score)},
                }
            )

        # Start new tracks for unmatched, non-ambiguous detections.
        used_det_indices = set(assigned_dets.keys()) | set(ambiguous_dets)
        new_det_indices = [i for i in range(len(dets)) if i not in used_det_indices]
        
        # Emit explicit unknown records for ambiguous detections
        for det_idx in sorted(ambiguous_dets):
            if det_idx >= len(dets):
                continue
            det = dets[det_idx]
            bbox = det.get("bbox_xyxy_px")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            # Create a unique entity_id for the ambiguous detection
            entity_id = f"amb_{t_ms:08d}_{det_idx:02d}"
            conf = float(det.get("confidence") or 0.0)
            out.append(
                {
                    "schema_version": 1,
                    "t_ms": int(t_ms),
                    "entity_type": "player",
                    "entity_id": entity_id,
                    "track_id": entity_id,
                    "source": str(self.source),
                    "frame": "image_px",
                    "pos_state": "unknown",
                    "bbox_xyxy_px": [int(x) for x in bbox],
                    "confidence": conf,
                    "diagnostics": {UNKNOWN_REASON: UNKNOWN_REASON_AMBIGUOUS, ASSOCIATION_SCORE: 0.0},
                }
            )
        for det_idx in new_det_indices:
            det = dets[det_idx]
            bbox = det.get("bbox_xyxy_px")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            track_id = self._new_track_id()
            segment_id = self._new_segment_id()
            center = _bbox_center_xy([int(x) for x in bbox])
            conf = float(det.get("confidence") or 0.0)
            self._active[track_id] = _ActiveTrack(
                track_id=track_id,
                segment_id=segment_id,
                last_center_xy=center,
                last_t_ms=int(t_ms),
                confidence=conf,
                frames_missing=0,
            )
            out.append(
                {
                    "schema_version": 1,
                    "t_ms": int(t_ms),
                    "entity_type": "player",
                    "entity_id": track_id,
                    "track_id": track_id,
                    "segment_id": segment_id,
                    "source": str(self.source),
                    "frame": "image_px",
                    "pos_state": "present",
                    "bbox_xyxy_px": [int(x) for x in bbox],
                    "confidence": conf,
                    "diagnostics": {ASSOCIATION_SCORE: 1.0, REASON: "new_track"},
                }
            )

        out.sort(key=lambda r: (str(r.get("track_id")), str(r.get("segment_id", "")), int(r.get("t_ms", 0)), str(r.get("pos_state"))))
        return out
