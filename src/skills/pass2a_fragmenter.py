"""
Pass 2A: Mechanical Fragmentation  —  full rewrite per IMPLEMENTATION_PLAN.md

Input : pass1_raw.json
Output: pass2_fragments.json, pass2_validation.json

Split triggers (T1–T5 only — no other triggers allowed):
  T1 — Track Collision      : same track_id, >1 detection per frame
  T2 — Jersey Change        : number X → Y (both non-None, X≠Y), persists ≥15 frames
  T3 — Jersey Temporal Conflict: same jersey on two tracks simultaneously
  T4 — Team Assignment Discontinuity: windowed HSV + global K-means cluster change
    T5 — Impossible Motion Spike: centroid jump > PASS2_MAX_PLAYER_SPEED px in 1 frame

FORBIDDEN triggers:
  - jersey disappearance  (X → None)
  - jersey first-appearance (None → X)
  - standalone HSV drift without cluster change
  - occlusion / confidence drops / tracker jitter
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mode
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from ..core.data_models import Detection, Fragment, Pass1Output, Pass2AOutput
from ..core.types import DetectionID, FragmentID, FragmentQuality, TrackID
from ..core import constants as const
from ..utils.file_utils import load_json, save_json
from ..utils.geometry import centroid_distance
from ..utils.hsv_color import compare_hsv_histograms
from ..validation.validator import Validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# K-means helpers (pure numpy, no external deps needed beyond sklearn)
# ---------------------------------------------------------------------------

def _kmeans_fit(X: np.ndarray, k: int, n_init: int = 5, max_iter: int = 100,
                random_state: int = 42) -> np.ndarray:
    """
    Fit K-means on rows of X (shape N×D).
    Returns cluster centroids (shape k×D).
    Uses sklearn if available, falls back to numpy.
    """
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter,
                    random_state=random_state)
        km.fit(X)
        return km.cluster_centers_
    except ImportError:
        pass

    # Fallback: pure numpy K-means
    rng = np.random.default_rng(random_state)
    best_centroids = None
    best_inertia = float("inf")
    for _ in range(n_init):
        idx = rng.choice(len(X), k, replace=False)
        centroids = X[idx].copy()
        for _ in range(max_iter):
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=-1)  # N×k
            labels = dists.argmin(axis=1)
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                new_centroids[j] = X[mask].mean(axis=0) if mask.any() else centroids[j]
            if np.allclose(new_centroids, centroids, atol=1e-6):
                break
            centroids = new_centroids
        inertia = dists.min(axis=1).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids.copy()
    return best_centroids


def _nearest_cluster(hist: np.ndarray, centroids: np.ndarray) -> int:
    """Return index of the centroid nearest to hist (L2 distance)."""
    dists = np.linalg.norm(centroids - hist, axis=1)
    return int(dists.argmin())


def _hist_mean(histograms: List[np.ndarray]) -> np.ndarray:
    """Element-wise mean of a list of histogram arrays."""
    return np.stack(histograms).mean(axis=0)


# ---------------------------------------------------------------------------
# Fragment-id counter
# ---------------------------------------------------------------------------

class _Counter:
    def __init__(self):
        self._n = 0

    def next_id(self) -> FragmentID:
        fid = f"F{self._n:06d}"
        self._n += 1
        return fid


# ---------------------------------------------------------------------------
# Main fragmenter class
# ---------------------------------------------------------------------------

class Pass2AFragmenter:
    """
    Mechanical track fragmenter.

    Does NOT assign teams, player identities, or cluster labels.
    Only fires splits on mechanical evidence (T1–T5).
    """

    def __init__(self):
        self._counter = _Counter()
        self.split_log: List[Dict] = []
        self._cluster_centroids: Optional[np.ndarray] = None  # shape (k, 512)
        self._cluster_to_team: Dict[int, int] = {}
        self._team_to_clusters: Dict[int, List[int]] = {}
        self._all_dets_by_frame: Dict[int, List[Detection]] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, pass1_path: Path, output_path: Path) -> Pass2AOutput:
        logger.info(f"Pass 2A: loading {pass1_path}")
        pass1: Pass1Output = load_json(pass1_path, Pass1Output)
        logger.info(f"  {len(pass1.detections)} detections across clip")

        # Build cross-frame lookup (for T4 proximity gate if needed)
        for det in pass1.detections:
            self._all_dets_by_frame.setdefault(det.frame_idx, []).append(det)

        # Step 0 — global K-means (used by T4)
        self._cluster_centroids = self._build_color_clusters(pass1.detections)
        if self._cluster_centroids is not None:
            self._cluster_to_team = self._build_team_cluster_map(self._cluster_centroids)
            self._team_to_clusters = defaultdict(list)
            for cluster_idx, team_id in self._cluster_to_team.items():
                self._team_to_clusters[int(team_id)].append(int(cluster_idx))
            logger.info(f"  K-means: {len(self._cluster_centroids)} colour clusters")
            logger.info(f"  Team cluster map: {self._cluster_to_team}")
            logger.info(f"  Team subclusters: {dict(self._team_to_clusters)}")
        else:
            logger.warning("  K-means skipped (insufficient valid HSV data) — T4 disabled")

        # Group detections by track
        tracks = self._group_by_track(pass1.detections)
        logger.info(f"  Processing {len(tracks)} tracks")

        # Per-track splitting (T1, T2, T4, T5)
        all_fragments: List[Fragment] = []
        for track_id, dets in tqdm(tracks.items(), desc="Fragmenting tracks"):
            all_fragments.extend(self._fragment_track(track_id, dets))

        logger.info(f"  After per-track splits: {len(all_fragments)} fragments")

        # T3 — jersey temporal conflicts (cross-track)
        all_fragments = self._resolve_t3(all_fragments, pass1.detections)
        logger.info(f"  After T3 jersey conflicts: {len(all_fragments)} fragments")

        output = Pass2AOutput(fragments=all_fragments, split_log=self.split_log)

        # Validate (fail-fast)
        validator = Validator()
        result = validator.validate_pass2a(output, pass1)
        val_path = output_path.parent / const.PASS2_VALIDATION_JSON
        save_json(result.dict(), str(val_path))

        if not result.passed:
            msg = f"Pass 2A validation FAILED: {len(result.violations)} violations"
            logger.error(msg)
            for v in result.violations[:5]:
                logger.error(f"  {v.rule}: {v.message}")
            raise ValueError(msg)

        save_json(output.dict(), str(output_path))
        logger.info(f"Pass 2A: {len(all_fragments)} fragments written → {output_path}")
        return output

    # ------------------------------------------------------------------
    # K-means colour clustering
    # ------------------------------------------------------------------

    def _build_color_clusters(
        self, detections: List[Detection]
    ) -> Optional[np.ndarray]:
        """
        Fit K-means on all valid jersey HSV histograms.
        Returns centroids (k × 512) or None if not enough data.
        """
        histos = [
            np.array(d.hsv_histogram_jersey, dtype=np.float32)
            for d in detections
            if d.jersey_roi_valid and d.hsv_histogram_jersey
        ]
        if len(histos) < const.N_COLOR_CLUSTERS * 10:
            return None

        X = np.stack(histos)
        # Subsample for speed
        if len(X) > const.KMEANS_SUBSAMPLE_SIZE:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), const.KMEANS_SUBSAMPLE_SIZE, replace=False)
            X = X[idx]

        return _kmeans_fit(X, k=const.N_COLOR_CLUSTERS)

    def _build_team_cluster_map(self, colour_centroids: np.ndarray) -> Dict[int, int]:
        """
        Collapse colour clusters into team-level groups (k=2).

        This prevents T4 from splitting on colour changes that remain within
        the same team palette (e.g., black <-> white).
        """
        if len(colour_centroids) < const.TEAM_SWITCH_TEAM_CLUSTERS:
            return {i: i for i in range(len(colour_centroids))}

        team_centroids = _kmeans_fit(
            colour_centroids,
            k=const.TEAM_SWITCH_TEAM_CLUSTERS,
            n_init=10,
            max_iter=100,
            random_state=42,
        )

        mapping: Dict[int, int] = {}
        for idx, centroid in enumerate(colour_centroids):
            team_id = _nearest_cluster(centroid, team_centroids)
            mapping[idx] = int(team_id)
        return mapping

    # ------------------------------------------------------------------
    # Track grouping
    # ------------------------------------------------------------------

    def _group_by_track(
        self, detections: List[Detection]
    ) -> Dict[TrackID, List[Detection]]:
        tracks: Dict[TrackID, List[Detection]] = defaultdict(list)
        for det in detections:
            tracks[det.track_id].append(det)
        for tid in tracks:
            tracks[tid].sort(key=lambda d: d.frame_idx)
        return dict(tracks)

    # ------------------------------------------------------------------
    # Per-track fragmentation
    # ------------------------------------------------------------------

    def _fragment_track(
        self, track_id: TrackID, dets: List[Detection]
    ) -> List[Fragment]:
        """Apply T1, T2, T4, T5 and create Fragment objects."""
        split_points: List[Tuple[int, str]] = []

        split_points.extend(self._detect_t1(track_id, dets))
        split_points.extend(self._detect_t2(track_id, dets))
        split_points.extend(self._detect_t4(track_id, dets))
        split_points.extend(self._detect_t5(track_id, dets))

        # Deduplicate: same frame can only have one split trigger
        seen: Set[int] = set()
        unique: List[Tuple[int, str]] = []
        for frame, reason in sorted(split_points, key=lambda x: x[0]):
            if frame not in seen:
                seen.add(frame)
                unique.append((frame, reason))

        return self._build_fragments(track_id, dets, unique)

    # ------------------------------------------------------------------
    # T1 — Track Collision
    # ------------------------------------------------------------------

    def _detect_t1(
        self, track_id: TrackID, dets: List[Detection]
    ) -> List[Tuple[int, str]]:
        """Same track_id, >1 detection in the same frame → immediate split."""
        frame_counts: Dict[int, int] = Counter(d.frame_idx for d in dets)
        splits = []
        for frame, count in frame_counts.items():
            if count > 1:
                splits.append((frame, "track_collision"))
                self.split_log.append({
                    "track_id": track_id, "frame_idx": frame,
                    "reason": "track_collision",
                    "details": f"{count} detections at frame {frame}",
                })
        return splits

    # ------------------------------------------------------------------
    # T2 — Jersey Change (X → Y, both non-None, persists ≥ N frames)
    # ------------------------------------------------------------------

    def _detect_t2(
        self, track_id: TrackID, dets: List[Detection]
    ) -> List[Tuple[int, str]]:
        """
        Jersey changes from number X to number Y (both non-None, X ≠ Y).
        The new jersey must be confirmed by enough consecutive sampled observations
        so that it has persisted for at least JERSEY_CHANGE_PERSISTENCE_FRAMES frames.
        None → X and X → None are NOT splits.
        """
        step = max(1, const.JERSEY_NUMBER_CLASSIFY_EVERY_N_FRAMES)
        # Observations where jersey classifier actually ran
        sampled = [d for d in dets if d.frame_idx % step == 0]
        if len(sampled) < 2:
            return []

        # Minimum consecutive confirmations ≥ persistence / sample_interval
        confirm_needed = max(
            2,
            int(math.ceil(const.JERSEY_CHANGE_PERSISTENCE_FRAMES / step)) + 1,
        )

        splits = []
        i = 0
        while i < len(sampled) - 1:
            prev = sampled[i]
            curr = sampled[i + 1]

            # Both must have a number, they must differ, both must be confident
            if (prev.jersey_number is not None
                    and curr.jersey_number is not None
                    and prev.jersey_number != curr.jersey_number
                    and prev.jersey_confidence >= const.JERSEY_CHANGE_MIN_CONF
                    and curr.jersey_confidence >= const.JERSEY_CHANGE_MIN_CONF):

                # Old jersey must also be stable before the boundary.
                old_confirmed = 1  # prev already qualifies
                for k in range(i - 1, max(-1, i - confirm_needed), -1):
                    look = sampled[k]
                    if (look.jersey_number == prev.jersey_number
                            and look.jersey_confidence >= const.JERSEY_CHANGE_MIN_CONF):
                        old_confirmed += 1
                    else:
                        break
                if old_confirmed < confirm_needed:
                    i += 1
                    continue

                # Confirm with subsequent samples
                confirmed = 1  # curr already qualifies
                for k in range(i + 2, min(i + 1 + confirm_needed, len(sampled))):
                    look = sampled[k]
                    if (look.jersey_number == curr.jersey_number
                            and look.jersey_confidence >= const.JERSEY_CHANGE_MIN_CONF):
                        confirmed += 1
                    else:
                        break

                if confirmed >= confirm_needed:
                    splits.append((curr.frame_idx, "jersey_change"))
                    self.split_log.append({
                        "track_id": track_id,
                        "frame_idx": curr.frame_idx,
                        "reason": "jersey_change",
                        "details": (
                            f"#{prev.jersey_number}→#{curr.jersey_number} "
                            f"confirmed old/new by {old_confirmed}/{confirmed} samples "
                            f"(frames {prev.frame_idx}→{curr.frame_idx})"
                        ),
                    })
                    i += confirmed  # Skip past confirmed region
                    continue

            i += 1
        return splits

    # ------------------------------------------------------------------
    # T4 — Team Assignment Discontinuity (windowed HSV + K-means)
    # ------------------------------------------------------------------

    def _detect_t4(
        self, track_id: TrackID, dets: List[Detection]
    ) -> List[Tuple[int, str]]:
        """
        Compare mean HSV of window [t-W, t-1] vs [t+1, t+W].
        Fires only when:
          - Both windows have ≥ TEAM_SWITCH_MIN_SAMPLES valid HSV samples.
          - The two window means are assigned to DIFFERENT K-means clusters.
          - The 1-correlation distance between means exceeds TEAM_SWITCH_HSV_THRESHOLD.
        Cooldown: no second split within TEAM_SWITCH_WINDOW frames of the last.
        """
        if self._cluster_centroids is None:
            return []

        # Build HSV sample list:
        # (frame_idx, histogram, nearest_cluster, crop_quality, nearest_team, team_margin)
        hsv_samples: List[Tuple[int, np.ndarray, int, float, int, float]] = sorted(
            [
                self._build_t4_sample(d)
                for d in dets
                if d.jersey_roi_valid and d.hsv_histogram_jersey
            ],
            key=lambda x: x[0],
        )
        if len(hsv_samples) < 2 * const.TEAM_SWITCH_MIN_SAMPLES:
            return []

        # Index for fast range queries
        frames_arr = np.array([f for f, _, _, _, _, _ in hsv_samples])
        hsvs_arr = np.stack([h for _, h, _, _, _, _ in hsv_samples])
        cluster_arr = np.array([c for _, _, c, _, _, _ in hsv_samples], dtype=np.int32)
        quality_arr = np.array([q for _, _, _, q, _, _ in hsv_samples], dtype=np.float32)
        team_arr = np.array([tm for _, _, _, _, tm, _ in hsv_samples], dtype=np.int32)
        margin_arr = np.array([m for _, _, _, _, _, m in hsv_samples], dtype=np.float32)

        has_two_team_evidence = self._track_has_two_team_evidence(
            team_arr,
            margin_arr,
            quality_arr,
        )

        candidate_scan_frames = self._local_track_team_transition_frames(
            frames_arr=frames_arr,
            hsvs_arr=hsvs_arr,
        )

        det_frames = sorted(d.frame_idx for d in dets)
        if not det_frames:
            return []

        W = const.TEAM_SWITCH_WINDOW
        step = const.TEAM_SWITCH_SCAN_STEP
        splits = []
        last_split_frame = -W * 2
        det_by_frame: Dict[int, Detection] = {d.frame_idx: d for d in dets}

        track_start = det_frames[0]
        track_end = det_frames[-1]
        if track_end <= track_start:
            return []

        # Always include edge candidates: if less than W frames exist on either side,
        # evaluate with truncated windows rather than skipping the boundary region.
        full_scan_frames = list(range(track_start + 1, track_end, step))
        candidate_scan_frames = sorted(set(candidate_scan_frames + full_scan_frames))

        for t in candidate_scan_frames:
            if t <= track_start or t >= track_end:
                continue
            if t - last_split_frame < W:
                continue  # Cooldown

            # Indices in window
            left = max(track_start, t - W)
            right = min(track_end, t + W)
            before_mask = (frames_arr >= left) & (frames_arr < t)
            after_mask = (frames_arr > t) & (frames_arr <= right)

            n_before = before_mask.sum()
            n_after = after_mask.sum()
            if n_before < const.TEAM_SWITCH_MIN_SAMPLES:
                continue
            if n_after < const.TEAM_SWITCH_MIN_SAMPLES:
                continue

            before_clusters = cluster_arr[before_mask]
            after_clusters = cluster_arr[after_mask]
            before_teams = team_arr[before_mask]
            after_teams = team_arr[after_mask]

            c_before, c_before_count = Counter(before_clusters.tolist()).most_common(1)[0]
            c_after, c_after_count = Counter(after_clusters.tolist()).most_common(1)[0]
            conf_before = c_before_count / float(n_before)
            conf_after = c_after_count / float(n_after)

            # Team vote is weighted by margin and crop quality so weak assignments
            # do not dominate window decisions.
            team_before, team_before_conf = self._dominant_team_vote(
                before_teams, margin_arr[before_mask], quality_arr[before_mask]
            )
            team_after, team_after_conf = self._dominant_team_vote(
                after_teams, margin_arr[after_mask], quality_arr[after_mask]
            )

            if conf_before < const.TEAM_SWITCH_CONFIDENCE:
                continue
            if conf_after < const.TEAM_SWITCH_CONFIDENCE:
                continue
            if team_before_conf < const.TEAM_SWITCH_CONFIDENCE:
                continue
            if team_after_conf < const.TEAM_SWITCH_CONFIDENCE:
                continue

            # Keep strict two-team gate for full-window checks, but allow edge-truncated
            # windows to proceed even when global track-level evidence is sparse.
            full_before = (t - W) >= track_start
            full_after = (t + W) <= track_end
            if not has_two_team_evidence and full_before and full_after:
                continue

            if team_before == team_after:
                continue  # Team-level cluster unchanged → no team switch

            # Crop quality gate: reject noisy windows with weak jersey visibility.
            q_before = float(quality_arr[before_mask].mean())
            q_after = float(quality_arr[after_mask].mean())
            if q_before < const.TEAM_SWITCH_MIN_CROP_QUALITY:
                continue
            if q_after < const.TEAM_SWITCH_MIN_CROP_QUALITY:
                continue

            before_mean = hsvs_arr[before_mask].mean(axis=0)
            after_mean = hsvs_arr[after_mask].mean(axis=0)

            # Confirm with histogram distance
            dist = 1.0 - compare_hsv_histograms(
                before_mean.tolist(), after_mean.tolist()
            )
            if dist < const.TEAM_SWITCH_HSV_THRESHOLD:
                continue  # Distance below threshold → noise

            # Find nearest actual detection frame to t
            split_frame = self._nearest_det_frame(t, det_frames)
            curr_det = det_by_frame.get(split_frame)
            if curr_det is None:
                continue
            if not self._has_crossing_near_split(
                track_id=track_id,
                split_frame=split_frame,
                det_by_frame=det_by_frame,
            ):
                continue

            splits.append((split_frame, "team_switch"))
            self.split_log.append({
                "track_id": track_id,
                "frame_idx": split_frame,
                "reason": "team_switch",
                "details": (
                    f"cluster {c_before}→{c_after}, "
                    f"team {team_before}→{team_after}, "
                    f"cconf={conf_before:.2f}/{conf_after:.2f}, "
                    f"tconf={team_before_conf:.2f}/{team_after_conf:.2f}, "
                    f"q={q_before:.2f}/{q_after:.2f}, "
                    f"dist={dist:.3f} (>{const.TEAM_SWITCH_HSV_THRESHOLD}), "
                    f"before_n={int(n_before)} after_n={int(n_after)}, "
                    f"edge_trunc={int(not (full_before and full_after))}, "
                    f"scan_t={t}"
                ),
            })
            last_split_frame = t

        return splits

    @staticmethod
    def _nearest_det_frame(t: int, det_frames: List[int]) -> int:
        """Return the detection frame index closest to t."""
        if not det_frames:
            return t
        return min(det_frames, key=lambda f: abs(f - t))

    def _build_t4_sample(self, det: Detection) -> Tuple[int, np.ndarray, int, float, int, float]:
        hist = np.array(det.hsv_histogram_jersey, dtype=np.float32)
        cluster_id = _nearest_cluster(hist, self._cluster_centroids)
        team_id, margin = self._nearest_team_with_margin(hist)
        return (
            int(det.frame_idx),
            hist,
            int(cluster_id),
            float(det.jersey_crop_quality or 0.0),
            int(team_id),
            float(margin),
        )

    def _nearest_team_with_margin(self, hist: np.ndarray) -> Tuple[int, float]:
        # Uses nearest subcluster for each team, preserving non-bib subcluster detail.
        if not self._team_to_clusters:
            nearest_cluster = _nearest_cluster(hist, self._cluster_centroids)
            team_id = int(self._cluster_to_team.get(int(nearest_cluster), int(nearest_cluster)))
            return team_id, 1.0

        team_dists: Dict[int, float] = {}
        for team_id, cluster_ids in self._team_to_clusters.items():
            if not cluster_ids:
                continue
            min_dist = min(float(np.linalg.norm(hist - self._cluster_centroids[cid])) for cid in cluster_ids)
            team_dists[int(team_id)] = float(min_dist)

        if not team_dists:
            nearest_cluster = _nearest_cluster(hist, self._cluster_centroids)
            team_id = int(self._cluster_to_team.get(int(nearest_cluster), int(nearest_cluster)))
            return team_id, 1.0

        ordered = sorted(team_dists.items(), key=lambda kv: kv[1])
        best_team, best_dist = ordered[0]
        if len(ordered) == 1:
            return int(best_team), 1.0
        second_dist = ordered[1][1]
        margin = max(0.0, (second_dist - best_dist) / max(second_dist, 1e-6))
        return int(best_team), float(margin)

    def _track_has_two_team_evidence(
        self,
        team_arr: np.ndarray,
        margin_arr: np.ndarray,
        quality_arr: np.ndarray,
    ) -> bool:
        if team_arr.size == 0:
            return False

        weights = np.maximum(margin_arr, const.TEAM_SWITCH_MIN_TEAM_MARGIN) * np.maximum(quality_arr, 1e-3)
        weighted_by_team: Dict[int, float] = defaultdict(float)
        count_by_team: Dict[int, int] = defaultdict(int)
        for team_id, w in zip(team_arr.tolist(), weights.tolist()):
            weighted_by_team[int(team_id)] += float(w)
            if w >= const.TEAM_SWITCH_MIN_TEAM_MARGIN:
                count_by_team[int(team_id)] += 1

        if len(weighted_by_team) < 2:
            return False

        ordered = sorted(weighted_by_team.items(), key=lambda kv: kv[1], reverse=True)
        minority_team, minority_weight = ordered[-1]
        total_weight = sum(weighted_by_team.values())
        if total_weight <= 1e-6:
            return False

        minority_ratio = minority_weight / total_weight
        minority_samples = int(count_by_team.get(int(minority_team), 0))
        return (
            minority_ratio >= const.TEAM_SWITCH_TRACK_MINORITY_RATIO
            and minority_samples >= const.TEAM_SWITCH_TRACK_MINORITY_SAMPLES
        )

    def _dominant_team_vote(
        self,
        teams: np.ndarray,
        margins: np.ndarray,
        qualities: np.ndarray,
    ) -> Tuple[int, float]:
        weighted_by_team: Dict[int, float] = defaultdict(float)
        weights = np.maximum(margins, const.TEAM_SWITCH_MIN_TEAM_MARGIN) * np.maximum(qualities, 1e-3)
        for team_id, w in zip(teams.tolist(), weights.tolist()):
            weighted_by_team[int(team_id)] += float(w)

        if not weighted_by_team:
            return 0, 0.0

        dominant_team, dominant_weight = max(weighted_by_team.items(), key=lambda kv: kv[1])
        total = sum(weighted_by_team.values())
        conf = dominant_weight / max(total, 1e-6)
        return int(dominant_team), float(conf)

    def _has_nearby_crossing(
        self,
        track_id: TrackID,
        frame_idx: int,
        det: Detection,
        max_bbox_widths: Optional[float] = None,
    ) -> bool:
        """
        Require nearby-player evidence before firing T4.

        Genuine team-switch steals usually happen at crossings; isolated tracks are
        more likely lighting/pose shifts.
        """
        bbox_w = max(1.0, float(det.bbox[2] - det.bbox[0]))
        if max_bbox_widths is None:
            max_bbox_widths = const.TEAM_SWITCH_PROXIMITY_BBOX_WIDTHS
        max_dist = bbox_w * max_bbox_widths
        for other in self._all_dets_by_frame.get(frame_idx, []):
            if other.track_id == track_id:
                continue
            if centroid_distance(det.centroid, other.centroid) <= max_dist:
                return True
        return False

    def _has_crossing_near_split(
        self,
        track_id: TrackID,
        split_frame: int,
        det_by_frame: Dict[int, Detection],
    ) -> bool:
        tolerance = int(const.TEAM_SWITCH_CROSSING_FRAME_TOLERANCE)
        for frame_idx in range(split_frame - tolerance, split_frame + tolerance + 1):
            det = det_by_frame.get(frame_idx)
            if det is None:
                continue
            if self._has_nearby_crossing(track_id, frame_idx, det):
                return True
        return False

    def _local_track_team_transition_frames(
        self,
        frames_arr: np.ndarray,
        hsvs_arr: np.ndarray,
    ) -> List[int]:
        """
        Build candidate split scan frames using local track colour clustering.

        A track contributes candidates only when its two local colour modes map to
        different global teams (via nearest team subcluster).
        """
        n = int(frames_arr.size)
        if n < 2 * const.TEAM_SWITCH_MIN_SAMPLES:
            return []

        local_centroids = _kmeans_fit(
            hsvs_arr,
            k=2,
            n_init=8,
            max_iter=100,
            random_state=42,
        )
        local_labels = np.array(
            [_nearest_cluster(h, local_centroids) for h in hsvs_arr],
            dtype=np.int32,
        )

        counts = Counter(local_labels.tolist())
        if len(counts) < 2:
            return []
        if min(counts.values()) < const.TEAM_SWITCH_MIN_SAMPLES:
            return []

        local_to_team: Dict[int, int] = {}
        for local_idx in range(2):
            mapped_team, _ = self._nearest_team_with_margin(local_centroids[local_idx])
            local_to_team[int(local_idx)] = int(mapped_team)

        if local_to_team[0] == local_to_team[1]:
            return []

        p = int(const.TEAM_SWITCH_MIN_SAMPLES)
        step = max(1, int(const.TEAM_SWITCH_SCAN_STEP // 2))
        candidates: List[int] = []
        last_candidate = -10_000

        for idx in range(p, n - p, step):
            before = local_labels[idx - p:idx]
            after = local_labels[idx:idx + p]

            b_label, b_count = Counter(before.tolist()).most_common(1)[0]
            a_label, a_count = Counter(after.tolist()).most_common(1)[0]
            b_conf = b_count / float(len(before))
            a_conf = a_count / float(len(after))

            if b_conf < const.TEAM_SWITCH_CONFIDENCE:
                continue
            if a_conf < const.TEAM_SWITCH_CONFIDENCE:
                continue
            if local_to_team[int(b_label)] == local_to_team[int(a_label)]:
                continue

            t = int(frames_arr[idx])
            if t - last_candidate < const.TEAM_SWITCH_WINDOW:
                continue
            candidates.append(t)
            last_candidate = t

        return candidates

    def _detect_t4_edge_rescue(
        self,
        track_id: TrackID,
        dets: List[Detection],
        existing_split_frames: Set[int],
        last_split_scan_t: int,
    ) -> List[Tuple[int, str]]:
        """
        Edge rescue for short tracks.

        Core T4 requires full +/-W windows and intentionally skips boundaries.
        This rescue path only targets short tracks where a true handoff happens
        near track end/start and uses stricter gates to avoid false positives.
        """
        if not dets:
            return []

        track_start = min(d.frame_idx for d in dets)
        track_end = max(d.frame_idx for d in dets)
        if (track_end - track_start + 1) > const.TEAM_SWITCH_EDGE_RESCUE_MAX_TRACK_FRAMES:
            return []

        if self._cluster_centroids is None:
            return []

        hsv_samples: List[Tuple[int, np.ndarray, int, float, int, float]] = sorted(
            [
                self._build_t4_sample(d)
                for d in dets
                if d.jersey_roi_valid and d.hsv_histogram_jersey
            ],
            key=lambda x: x[0],
        )

        if len(hsv_samples) < 2 * const.TEAM_SWITCH_MIN_SAMPLES:
            return []

        frames_arr = np.array([f for f, _, _, _, _, _ in hsv_samples])
        hsvs_arr = np.stack([h for _, h, _, _, _, _ in hsv_samples])
        cluster_arr = np.array([c for _, _, c, _, _, _ in hsv_samples], dtype=np.int32)
        quality_arr = np.array([q for _, _, _, q, _, _ in hsv_samples], dtype=np.float32)
        team_arr = np.array([tm for _, _, _, _, tm, _ in hsv_samples], dtype=np.int32)
        margin_arr = np.array([m for _, _, _, _, _, m in hsv_samples], dtype=np.float32)

        if not self._track_has_two_team_evidence(team_arr, margin_arr, quality_arr):
            return []

        candidate_scan_frames = self._local_track_team_transition_frames(
            frames_arr=frames_arr,
            hsvs_arr=hsvs_arr,
        )
        det_frames = sorted(d.frame_idx for d in dets)
        det_by_frame: Dict[int, Detection] = {d.frame_idx: d for d in dets}

        W = const.TEAM_SWITCH_WINDOW
        step = const.TEAM_SWITCH_SCAN_STEP
        splits: List[Tuple[int, str]] = []
        last_scan_t = last_split_scan_t

        full_scan_frames = list(range(track_start + 1, track_end, step))
        candidate_scan_frames = sorted(set(candidate_scan_frames + full_scan_frames))

        for t in candidate_scan_frames:
            if t < track_start + 1 or t >= track_end:
                continue
            if t - last_scan_t < W:
                continue

            left = max(track_start, t - W)
            right = min(track_end, t + W)

            # Edge-only: require at least one truncated side vs full +/-W windows.
            if (t - W >= track_start) and (t + W <= track_end):
                continue

            before_mask = (frames_arr >= left) & (frames_arr < t)
            after_mask = (frames_arr > t) & (frames_arr <= right)
            n_before = int(before_mask.sum())
            n_after = int(after_mask.sum())
            if n_before < const.TEAM_SWITCH_MIN_SAMPLES:
                continue
            if n_after < const.TEAM_SWITCH_MIN_SAMPLES:
                continue

            c_before, c_before_count = Counter(cluster_arr[before_mask].tolist()).most_common(1)[0]
            c_after, c_after_count = Counter(cluster_arr[after_mask].tolist()).most_common(1)[0]
            conf_before = c_before_count / float(n_before)
            conf_after = c_after_count / float(n_after)
            if conf_before < const.TEAM_SWITCH_EDGE_RESCUE_MIN_MAIN_CONF:
                continue
            if conf_after < const.TEAM_SWITCH_CONFIDENCE:
                continue

            team_before, team_before_conf = self._dominant_team_vote(
                team_arr[before_mask], margin_arr[before_mask], quality_arr[before_mask]
            )
            team_after, team_after_conf = self._dominant_team_vote(
                team_arr[after_mask], margin_arr[after_mask], quality_arr[after_mask]
            )
            if team_before_conf < const.TEAM_SWITCH_EDGE_RESCUE_MIN_MAIN_CONF:
                continue
            if team_after_conf < const.TEAM_SWITCH_CONFIDENCE:
                continue
            if team_before == team_after:
                continue

            q_before = float(quality_arr[before_mask].mean())
            q_after = float(quality_arr[after_mask].mean())
            if q_before < const.TEAM_SWITCH_MIN_CROP_QUALITY:
                continue
            if q_after < const.TEAM_SWITCH_MIN_CROP_QUALITY:
                continue

            before_mean = hsvs_arr[before_mask].mean(axis=0)
            after_mean = hsvs_arr[after_mask].mean(axis=0)
            dist = 1.0 - compare_hsv_histograms(before_mean.tolist(), after_mean.tolist())
            if dist < const.TEAM_SWITCH_HSV_THRESHOLD:
                continue

            split_frame = self._nearest_det_frame(t, det_frames)
            if split_frame in existing_split_frames:
                continue
            curr_det = det_by_frame.get(split_frame)
            if curr_det is None:
                continue
            if not self._has_crossing_near_split(
                track_id=track_id,
                split_frame=split_frame,
                det_by_frame=det_by_frame,
            ):
                continue

            splits.append((split_frame, "team_switch"))
            self.split_log.append({
                "track_id": track_id,
                "frame_idx": split_frame,
                "reason": "team_switch",
                "details": (
                    f"edge_rescue=1, cluster {c_before}→{c_after}, "
                    f"team {team_before}→{team_after}, "
                    f"cconf={conf_before:.2f}/{conf_after:.2f}, "
                    f"tconf={team_before_conf:.2f}/{team_after_conf:.2f}, "
                    f"q={q_before:.2f}/{q_after:.2f}, "
                    f"dist={dist:.3f} (>{const.TEAM_SWITCH_HSV_THRESHOLD}), "
                    f"before_n={n_before} after_n={n_after}, scan_t={t}"
                ),
            })
            existing_split_frames.add(split_frame)
            last_scan_t = t

        return splits

    # ------------------------------------------------------------------
    # T5 — Impossible Motion Spike
    # ------------------------------------------------------------------

    def _detect_t5(
        self, track_id: TrackID, dets: List[Detection]
    ) -> List[Tuple[int, str]]:
        """
        Centroid displacement between adjacent frames > PASS2_MAX_PLAYER_SPEED → split.
        Only fires for gap == 1 (teleports, not occlusion gaps).
        """
        splits = []
        for i in range(1, len(dets)):
            prev, curr = dets[i - 1], dets[i]
            if curr.frame_idx - prev.frame_idx != 1:
                continue
            dist = centroid_distance(prev.centroid, curr.centroid)
            if dist > const.PASS2_MAX_PLAYER_SPEED:
                splits.append((curr.frame_idx, "motion_spike"))
                self.split_log.append({
                    "track_id": track_id,
                    "frame_idx": curr.frame_idx,
                    "reason": "motion_spike",
                    "details": f"jump {dist:.1f}px > {const.PASS2_MAX_PLAYER_SPEED}px/frame",
                })
        return splits

    # ------------------------------------------------------------------
    # Build Fragment objects from split points
    # ------------------------------------------------------------------

    def _build_fragments(
        self,
        track_id: TrackID,
        dets: List[Detection],
        split_points: List[Tuple[int, str]],
    ) -> List[Fragment]:
        """
        Slice detections at split_points and create Fragment objects.
        Each detection goes into exactly one fragment.
        """
        if not dets:
            return []

        # Build a quick lookup: frame_idx → reason
        split_map: Dict[int, str] = {frame: reason for frame, reason in split_points}

        segments: List[Tuple[List[Detection], Optional[str], Optional[int]]] = []
        current: List[Detection] = []
        current_reason: Optional[str] = None
        current_trigger: Optional[int] = None

        for det in dets:
            if det.frame_idx in split_map and current:
                segments.append((current, current_reason, current_trigger))
                current = [det]
                current_reason = split_map[det.frame_idx]
                current_trigger = det.frame_idx
            else:
                current.append(det)

        if current:
            segments.append((current, current_reason, current_trigger))

        fragments = []
        for seg_dets, reason, trigger in segments:
            if not seg_dets:
                continue
            fragments.append(
                self._make_fragment(track_id, seg_dets, reason, trigger)
            )

        return fragments

    def _make_fragment(
        self,
        track_id: TrackID,
        dets: List[Detection],
        split_reason: Optional[str],
        split_trigger_frame: Optional[int],
    ) -> Fragment:
        fid = self._counter.next_id()
        start = min(d.frame_idx for d in dets)
        end = max(d.frame_idx for d in dets)
        fragment_length = end - start + 1

        # Dominant jersey number (mode of non-None confident observations)
        jersey_obs = [
            d.jersey_number
            for d in dets
            if d.jersey_number is not None
            and d.jersey_confidence >= const.JERSEY_MIN_CONFIDENCE
        ]
        dominant_jersey: Optional[int] = None
        if jersey_obs:
            try:
                dominant_jersey = mode(jersey_obs)
            except Exception:
                dominant_jersey = Counter(jersey_obs).most_common(1)[0][0]

        jersey_visible_ratio = self._compute_jersey_visible_ratio(dets)
        occlusion_ratio = self._compute_occlusion_ratio(dets)
        mean_velocity = self._compute_mean_velocity(dets)
        appearance_stability_score = self._compute_appearance_stability_score(dets)
        quality = self._assign_fragment_quality(
            fragment_length=fragment_length,
            occlusion_ratio=occlusion_ratio,
            appearance_stability_score=appearance_stability_score,
        )

        # Normalise split_trigger_frame to be within [start, end]
        eff_trigger = split_trigger_frame
        if eff_trigger is not None and not (start <= eff_trigger <= end):
            eff_trigger = start

        rule_map = {
            "track_collision": "TRACK_COLLISION",
            "jersey_change": "JERSEY_CHANGE",
            "jersey_temporal_conflict": "JERSEY_TEMPORAL_CONFLICT",
            "team_switch": "TEAM_SWITCH",
            "motion_spike": "MOTION_SPIKE",
        }

        return Fragment(
            fragment_id=fid,
            track_id=track_id,
            start_frame=start,
            end_frame=end,
            detection_ids=[d.detection_id for d in dets],
            split_reason=split_reason,
            split_trigger_frame=eff_trigger,
            split_rule_id=rule_map.get(split_reason) if split_reason else None,
            parent_fragment_id=None,
            dominant_jersey_number=dominant_jersey,
            jersey_visible_ratio=jersey_visible_ratio,
            occlusion_ratio=occlusion_ratio,
            mean_velocity=mean_velocity,
            appearance_stability_score=appearance_stability_score,
            quality=quality,
            is_ghost=False,
        )

    def _compute_jersey_visible_ratio(self, dets: List[Detection]) -> float:
        """Fraction of detections where a jersey number is visible."""
        if not dets:
            return 0.0
        visible = sum(1 for d in dets if d.jersey_number is not None)
        return float(visible / len(dets))

    def _compute_occlusion_ratio(self, dets: List[Detection]) -> float:
        """Fraction of detections where jersey ROI extraction failed."""
        if not dets:
            return 0.0
        occluded = sum(1 for d in dets if not d.jersey_roi_valid)
        return float(occluded / len(dets))

    def _compute_mean_velocity(self, dets: List[Detection]) -> float:
        """Mean centroid displacement per frame in pixels/frame."""
        if len(dets) < 2:
            return 0.0

        ordered = sorted(dets, key=lambda d: d.frame_idx)
        velocities: List[float] = []
        for i in range(len(ordered) - 1):
            current = ordered[i]
            nxt = ordered[i + 1]
            frame_delta = max(1, nxt.frame_idx - current.frame_idx)
            displacement = centroid_distance(current.centroid, nxt.centroid)
            velocities.append(displacement / frame_delta)

        return float(np.mean(velocities)) if velocities else 0.0

    def _compute_appearance_stability_score(self, dets: List[Detection]) -> float:
        """
        Mean pairwise HSV similarity in [0, 1].

        Uses a capped subset to keep runtime bounded on long fragments.
        """
        histograms = [
            d.hsv_histogram_jersey
            for d in dets
            if d.hsv_histogram_jersey is not None
        ]

        if not histograms:
            return 0.0
        if len(histograms) == 1:
            return 1.0

        max_samples = 40
        if len(histograms) > max_samples:
            idx = np.linspace(0, len(histograms) - 1, max_samples, dtype=int)
            histograms = [histograms[i] for i in idx]

        similarities: List[float] = []
        for i in range(len(histograms) - 1):
            for j in range(i + 1, len(histograms)):
                corr = compare_hsv_histograms(histograms[i], histograms[j])
                similarities.append(float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0)))

        return float(np.mean(similarities)) if similarities else 0.0

    def _assign_fragment_quality(
        self,
        fragment_length: int,
        occlusion_ratio: float,
        appearance_stability_score: float,
    ) -> FragmentQuality:
        """Assign Pass 2A quality tier; short fragments are always low quality."""
        if fragment_length < const.MIN_FRAGMENT_LENGTH:
            return FragmentQuality.LOW

        if (
            appearance_stability_score >= const.QUALITY_HIGH_THRESHOLD
            and occlusion_ratio <= 0.2
        ):
            return FragmentQuality.HIGH

        if appearance_stability_score >= const.QUALITY_MEDIUM_THRESHOLD:
            return FragmentQuality.MEDIUM

        return FragmentQuality.LOW

    # ------------------------------------------------------------------
    # T3 — Jersey Temporal Conflict (cross-track, post per-track splits)
    # ------------------------------------------------------------------

    def _resolve_t3(
        self,
        fragments: List[Fragment],
        all_detections: List[Detection],
    ) -> List[Fragment]:
        """
        Detect fragments that concurrently "own" the same jersey number and split
        the later-appearing fragment at the first frame it claims that jersey.

        Ownership: fragment must have ≥ JERSEY_MIN_OBSERVATIONS confident detections
        with the same jersey, AND ≥ JERSEY_MAJORITY_THRESHOLD fraction of its
        jersey-observed frames showing that number.

        Skip: if the conflicting jersey appears at the very start of the fragment
        (within the first MIN_FRAGMENT_LENGTH frames of track start).
        """
        det_by_id: Dict[str, Detection] = {d.detection_id: d for d in all_detections}

        # Determine jersey ownership for each fragment
        jersey_owner: Dict[str, Tuple[int, int, int]] = {}
        # fragment_id → (jersey_num, start_frame, first_jersey_frame)

        for frag in fragments:
            obs: List[Tuple[int, int]] = []  # (frame_idx, jersey_number)
            for did in frag.detection_ids:
                det = det_by_id.get(did)
                if det and det.jersey_number is not None:
                    if det.jersey_confidence >= const.JERSEY_TEMPORAL_MIN_CONFIDENCE:
                        obs.append((det.frame_idx, det.jersey_number))

            if len(obs) < const.JERSEY_MIN_OBSERVATIONS:
                continue

            counts = Counter(j for _, j in obs)
            best_jersey, best_count = counts.most_common(1)[0]
            ratio = best_count / len(obs)
            if ratio < const.JERSEY_MAJORITY_THRESHOLD:
                continue
            density = len(obs) / max(1, len(frag.detection_ids))
            if density < const.JERSEY_TEMPORAL_MIN_DENSITY:
                continue

            first_jersey_frame = min(f for f, j in obs if j == best_jersey)
            jersey_owner[frag.fragment_id] = (best_jersey, frag.start_frame, first_jersey_frame)

        # Group by jersey number and find temporal overlaps
        # jersey_num → [(frag_id, frag_start, frag_end, first_jersey_frame)]
        jersey_groups: Dict[int, List[Tuple[str, int, int, int]]] = defaultdict(list)
        frag_by_id = {f.fragment_id: f for f in fragments}

        for fid, (jnum, fstart, fjframe) in jersey_owner.items():
            frag = frag_by_id[fid]
            jersey_groups[jnum].append((fid, frag.start_frame, frag.end_frame, fjframe))

        to_split: Dict[str, int] = {}  # fragment_id → split_frame
        seen_pairs: Set[Tuple[str, int]] = set()

        for jnum, group in jersey_groups.items():
            if len(group) < 2:
                continue
            group_sorted = sorted(group, key=lambda x: x[3])  # by first_jersey_frame

            for i in range(len(group_sorted)):
                for j in range(i + 1, len(group_sorted)):
                    fid_a, sa, ea, fja = group_sorted[i]
                    fid_b, sb, eb, fjb = group_sorted[j]

                    # Compute frame overlap
                    overlap_start = max(sa, sb)
                    overlap_end = min(ea, eb)
                    overlap = overlap_end - overlap_start + 1
                    # Minimum overlap to avoid false positives from classifier noise
                    # at fragment boundaries. ~5 seconds at 30fps is deliberate.
                    if overlap < const.JERSEY_TEMPORAL_MIN_OVERLAP_FRAMES:
                        continue

                    # Fragment B appeared later (by first_jersey_frame)
                    # Split B at fjb (first frame it claims this jersey)
                    frag_b = frag_by_id[fid_b]

                    # Skip if jersey appears at very start of fragment B's track
                    if fjb <= frag_b.start_frame + const.MIN_FRAGMENT_LENGTH:
                        continue

                    pair_key = (fid_b, jnum)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    if fid_b not in to_split:
                        to_split[fid_b] = fjb
                        self.split_log.append({
                            "track_id": frag_b.track_id,
                            "frame_idx": fjb,
                            "reason": "jersey_temporal_conflict",
                            "details": (
                                f"jersey #{jnum} already owned by {fid_a} "
                                f"(track {frag_by_id[fid_a].track_id})"
                            ),
                        })

        # Apply splits
        result: List[Fragment] = []
        for frag in fragments:
            if frag.fragment_id not in to_split:
                result.append(frag)
                continue

            split_frame = to_split[frag.fragment_id]
            before_dets = [
                det_by_id[did] for did in frag.detection_ids
                if det_by_id.get(did) and det_by_id[did].frame_idx < split_frame
            ]
            after_dets = [
                det_by_id[did] for did in frag.detection_ids
                if det_by_id.get(did) and det_by_id[did].frame_idx >= split_frame
            ]

            if before_dets:
                result.append(self._make_fragment(
                    frag.track_id, before_dets,
                    frag.split_reason, frag.split_trigger_frame,
                ))
            if after_dets:
                result.append(self._make_fragment(
                    frag.track_id, after_dets,
                    "jersey_temporal_conflict", split_frame,
                ))
            if not before_dets or not after_dets:
                # Fallback: keep original if split would produce empty fragment
                result.append(frag)

        return result


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------

def run_pass2a(
    input_dir: Path,
    output_dir: Optional[Path] = None,
) -> Pass2AOutput:
    """Execute Pass 2A mechanical fragmentation."""
    if output_dir is None:
        output_dir = input_dir

    pass1_path = input_dir / const.PASS1_RAW_JSON
    output_path = output_dir / const.PASS2_FRAGMENTS_JSON

    if not pass1_path.exists():
        raise FileNotFoundError(f"pass1_raw.json not found: {pass1_path}")

    fragmenter = Pass2AFragmenter()
    return fragmenter.run(pass1_path, output_path)


def render_pass2a_debug_video_from_artifact(
    video_path: str,
    pass1_output_path: str,
    pass2a_output_path: str,
    debug_video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    pass2c_ghosts_path: Optional[str] = None,
) -> None:
    """Render a debug video showing fragment bboxes. Minimal implementation."""
    import cv2
    from ..core.data_models import Pass2COutput, ScoredFragment
    from ..utils.video_io import VideoReader

    pass1_output = load_json(Path(pass1_output_path), Pass1Output)
    pass2a_output = load_json(Path(pass2a_output_path), Pass2AOutput)

    all_fragments = list(pass2a_output.fragments)
    if pass2c_ghosts_path and Path(pass2c_ghosts_path).exists():
        pass2c_output = load_json(Path(pass2c_ghosts_path), Pass2COutput)
        all_fragments = pass2c_output.fragments

    det_by_id = {d.detection_id: d for d in pass1_output.detections}

    # frame → [(det, fragment)]
    frame_annot: Dict[int, list] = defaultdict(list)
    for frag in all_fragments:
        if not frag.is_ghost:
            for did in frag.detection_ids:
                det = det_by_id.get(did)
                if det:
                    frame_annot[det.frame_idx].append((det, frag))
        else:
            if hasattr(frag, "ghost_last_known_bbox") and frag.ghost_last_known_bbox:
                for fidx in range(frag.start_frame, frag.end_frame + 1):
                    frame_annot[fidx].append((None, frag))

    reader = VideoReader(video_path)
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", cv2.VideoWriter.fourcc)
    writer = cv2.VideoWriter(
        debug_video_path,
        fourcc_fn(*"mp4v"),
        float(reader.fps),
        (int(reader.width), int(reader.height)),
    )

    try:
        art_start = pass1_output.processed_start_frame
        art_end = pass1_output.processed_end_frame_exclusive or pass1_output.total_frames
        render_start = max(start_frame, art_start)
        render_end = art_end if end_frame is None else min(end_frame, art_end)

        import hashlib
        def _frag_color(fid: str):
            d = hashlib.md5(fid.encode()).digest()
            return (60 + d[0] % 170, 60 + d[1] % 170, 60 + d[2] % 170)

        def _draw_dotted_rect(img, x1, y1, x2, y2, color, spacing=8, radius=1):
            # Draw a dotted rectangle by placing small circles along each edge.
            for x in range(x1, x2 + 1, spacing):
                cv2.circle(img, (x, y1), radius, color, -1)
                cv2.circle(img, (x, y2), radius, color, -1)
            for y in range(y1, y2 + 1, spacing):
                cv2.circle(img, (x1, y), radius, color, -1)
                cv2.circle(img, (x2, y), radius, color, -1)

        def _draw_label_with_banner(img, text, x, y_top, text_color):
            """Draw larger label text on a local black banner for readability."""
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.62
            thickness = 2
            pad_x = 5
            pad_y = 4

            (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
            box_w = text_w + 2 * pad_x
            box_h = text_h + baseline + 2 * pad_y

            h, w = img.shape[:2]
            x = max(0, min(int(x), max(0, w - box_w - 1)))
            y_top = max(0, min(int(y_top), max(0, h - box_h - 1)))

            cv2.rectangle(img, (x, y_top), (x + box_w, y_top + box_h), (0, 0, 0), -1)
            text_org = (x + pad_x, y_top + pad_y + text_h)
            cv2.putText(img, text, text_org, font, scale, text_color, thickness, cv2.LINE_AA)

        for frame_idx, frame in reader.iter_frames():
            if frame_idx < render_start:
                continue
            if frame_idx >= render_end:
                break
            for det, frag in frame_annot.get(frame_idx, []):
                if det is None:
                    ghost_bbox = getattr(frag, "ghost_last_known_bbox", None)
                    if not ghost_bbox:
                        continue
                    x1, y1, x2, y2 = [int(round(v)) for v in ghost_bbox]
                    color = (160, 160, 160)
                    _draw_dotted_rect(frame, x1, y1, x2, y2, color)
                    label = f"GHOST {frag.fragment_id} T{frag.track_id}"
                    _draw_label_with_banner(frame, label, x1, y1 - 30, color)
                    continue

                x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
                color = _frag_color(frag.fragment_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{frag.fragment_id} T{frag.track_id}"
                _draw_label_with_banner(frame, label, x1, y1 - 30, color)
            writer.write(frame)
    finally:
        writer.release()
        reader.close()
