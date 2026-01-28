"""
Team color clustering utility.

Supports sampling jersey crops, fitting k-means (k=2) on HSV histograms,
producing a vivid palette, predicting team IDs, and optionally saving
sample crops split by cluster for inspection.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2
from sklearn.cluster import KMeans

from src.utils.data_models import BoundingBox, TeamID


class TeamClustering:
    """Per-run team color clustering helper."""

    def __init__(
        self,
        bins: int = 32,
        n_clusters: int = 2,
        min_samples: int = 10,
        max_samples: int = 60,
        force_palette: Optional[dict] = None,
        save_crops: bool = False,
        crops_dir: Optional[Path] = None,
        vividify: bool = True,
    ) -> None:
        self.bins = bins
        self.n_clusters = max(2, n_clusters)
        self.min_samples = max(1, min_samples)
        self.max_samples = max_samples
        self.force_palette = force_palette or {}
        self.save_crops_enabled = save_crops
        self.crops_dir = Path(crops_dir) if crops_dir else None
        self.vividify = vividify

        self.samples: list[dict] = []  # {'hist': np.ndarray, 'mean_bgr': np.ndarray, 'crop': np.ndarray, 'frame': int, 'track_id': int}
        self.kmeans: Optional[KMeans] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_colors: list[np.ndarray] = []
        self.cluster_distance_threshold: Optional[float] = None
        self.label_to_team: dict[int, TeamID] = {0: TeamID.TEAM_A, 1: TeamID.TEAM_B}
        self._label_to_team_locked: bool = False  # Cache mapping after first fit to prevent flipping
        self.team_palette: dict[TeamID, tuple[int, int, int]] = {
            TeamID.TEAM_A: (255, 100, 100),
            TeamID.TEAM_B: (100, 100, 255),
            TeamID.UNKNOWN: (200, 200, 200),
            TeamID.REFEREE: (0, 255, 255),
        }

    def _is_valid_bbox(self, frame_shape: Tuple[int, int, int], bbox: BoundingBox) -> bool:
        """Filter obviously bad samples (e.g., fences/skinny artifacts).

        Reject if the box is too small, too skinny/wide, or implausibly large relative to the frame.
        """
        w = bbox.x2 - bbox.x1
        h = bbox.y2 - bbox.y1
        if w <= 0 or h <= 0:
            return False

        ar = w / h
        min_w, min_h = 20, 35  # pixels
        max_ar = 3.5            # too wide/flat (fence spans)
        min_ar = 0.3            # too skinny
        max_h = frame_shape[0] * 0.8  # avoid nearly full-height artifacts

        if w < min_w or h < min_h:
            return False
        if ar > max_ar or ar < min_ar:
            return False
        if h > max_h:
            return False
        return True

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _jersey_mask(roi_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Build a mask that keeps jersey pixels by suppressing known background.

        Strategy: start with ALL pixels, then subtract:
        1. Court floor         (measured H≈28, S≈47, V≈143-187; use H 25-90)
        2. Skin                (YCrCb range, but spare vivid fabric S>110)
        3. Overexposed white   (S < 20, V > 230)

        This approach works for dark/black jerseys (low S, low V — not court
        coloured) as well as vivid orange bibs (H≈10-20, below court range).
        """
        if roi_rgb is None or roi_rgb.size == 0:
            return None

        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)

        # Start: keep everything
        jersey_mask = np.ones(roi_rgb.shape[:2], dtype=np.uint8) * 255

        # 1) Suppress court floor (yellow-green through green, H 25-90)
        #    Measured court: H≈28, S≈47, V≈143-187.
        #    Orange bibs are H≈10-20 so H≥25 cleanly avoids them.
        court_mask = cv2.inRange(hsv, (25, 15, 50), (90, 255, 255))
        jersey_mask = cv2.bitwise_and(jersey_mask, cv2.bitwise_not(court_mask))

        # 2) Suppress skin (YCrCb), but spare:
        #    - vivid fabric (S > 110)
        #    - orange-hue pixels (H 5-22) which are bib, not skin, in
        #      the torso crop.  Shadow can drop bib S to 50-80 which
        #      overlaps the skin range, so hue is the safer discriminator.
        ycrcb = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2YCrCb)
        skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        vivid = hsv[:, :, 1] > 110
        orange_hue = (hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 22)
        skin_mask[vivid | orange_hue] = 0
        jersey_mask = cv2.bitwise_and(jersey_mask, cv2.bitwise_not(skin_mask))

        # 3) Suppress overexposed / washed-out white (S < 20, V > 230)
        white_mask = cv2.inRange(hsv, (0, 0, 230), (180, 20, 255))
        jersey_mask = cv2.bitwise_and(jersey_mask, cv2.bitwise_not(white_mask))

        # Clean small speckles
        if np.any(jersey_mask):
            kernel = np.ones((3, 3), np.uint8)
            jersey_mask = cv2.morphologyEx(jersey_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            jersey_mask = cv2.morphologyEx(jersey_mask, cv2.MORPH_DILATE, kernel, iterations=1)
            return jersey_mask
        return None
    @staticmethod
    def _crop_jersey_region(
        bbox: BoundingBox,
        frame_shape: Tuple[int, int, int],
        top_skip: float = 0.2,
        bottom_cut: float = 0.55,
        width_shrink: float = 0.8,
    ) -> tuple[int, int, int, int]:
        """Crop the shirt/jersey region, skipping head and legs.

        Default: vertical 20%-55% of bbox (upper torso), 80% width.
        """
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        width = x2 - x1
        height = y2 - y1
        cx = (x1 + x2) / 2

        new_w = width * width_shrink
        ny1 = int(max(0, y1 + height * top_skip))
        ny2 = int(min(frame_shape[0], y1 + height * bottom_cut))
        nx1 = int(max(0, cx - new_w / 2))
        nx2 = int(min(frame_shape[1], cx + new_w / 2))
        return nx1, ny1, nx2, ny2

    def extract_features(self, frame: np.ndarray, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], float]:
        """Extract HSV histogram, mean colour, crop image, and mask quality.

        Returns:
            hist:     normalised HSV histogram (3*bins,)
            mean_bgr: mean jersey colour in BGR
            crop_bgr: raw crop image (BGR) for saving
            quality:  fraction of crop pixels kept by the jersey mask (0.0–1.0).
                      Low values indicate the crop is mostly background.
        """
        x1, y1, x2, y2 = self._crop_jersey_region(bbox, frame.shape)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(3 * self.bins, dtype=np.float32), np.array([200.0, 200.0, 200.0], dtype=np.float32), None, 0.0

        roi_rgb = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)

        # Use the jersey mask to exclude background / skin pixels.
        # Fall back to unmasked if the mask rejects everything (e.g.
        # very bright / white kits where S is low).
        mask = self._jersey_mask(roi_rgb)

        total_pixels = roi_rgb.shape[0] * roi_rgb.shape[1]
        if mask is not None and np.any(mask):
            quality = float(np.count_nonzero(mask)) / total_pixels
        else:
            quality = 0.0

        hist_h = cv2.calcHist([hsv], [0], mask, [self.bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], mask, [self.bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], mask, [self.bins], [0, 256])
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)

        # Mean colour from masked pixels only (fall back to whole ROI)
        if mask is not None and np.any(mask):
            masked_hsv = hsv[mask > 0]
            mean_hsv = np.mean(masked_hsv, axis=0).reshape(1, 1, 3).astype(np.uint8)
        else:
            mean_hsv = np.mean(hsv, axis=(0, 1)).reshape(1, 1, 3).astype(np.uint8)
        mean_bgr = cv2.cvtColor(mean_hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)

        crop_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
        return hist, mean_bgr, crop_bgr, quality

    # ------------------------------------------------------------------
    # Sampling and fitting
    # ------------------------------------------------------------------
    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def ready(self) -> bool:
        return self.sample_count >= self.min_samples

    @property
    def fitted(self) -> bool:
        return self.kmeans is not None

    # Minimum fraction of jersey pixels required for a crop to be usable.
    MIN_QUALITY = 0.05

    def add_sample(self, frame: np.ndarray, bbox: BoundingBox, frame_idx: Optional[int] = None, track_id: Optional[int] = None) -> np.ndarray:
        if self.sample_count >= self.max_samples:
            hist, _, _, _ = self.extract_features(frame, bbox)
            return hist

        if not self._is_valid_bbox(frame.shape, bbox):
            # Return empty feature to signal skip
            return np.zeros(3 * self.bins, dtype=np.float32)

        hist, mean_bgr, crop_bgr, quality = self.extract_features(frame, bbox)
        if quality < self.MIN_QUALITY:
            return hist  # too much background — don't pollute the training set

        self.samples.append({
            "hist": hist,
            "mean_bgr": mean_bgr,
            "crop": crop_bgr,
            "frame": frame_idx,
            "track_id": track_id,
        })
        return hist

    def _vividify(self, bgr: np.ndarray) -> tuple[int, int, int]:
        if not self.vividify:
            return tuple(int(x) for x in bgr)
        bgr_uint8 = np.clip(bgr.reshape(1, 1, 3), 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = hsv[0, 0]
        if s < 120:
            s = 180
        if v < 160:
            v = 190
        vivid = cv2.cvtColor(np.array([[[h, min(s, 255), min(v, 255)]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        return int(vivid[0]), int(vivid[1]), int(vivid[2])

    def _build_palette(self) -> None:
        if not self.cluster_colors:
            return

        # If we've already locked the cluster-to-team mapping, use it
        if self._label_to_team_locked:
            # Keep the locked palette from the first fit - don't recalculate
            # The palette was already set in the first call to _build_palette()
            return

        if self.force_palette.get("enabled", False):
            target_a = np.array(self.force_palette.get("team_a", [0, 140, 255]), dtype=float)
            target_b = np.array(self.force_palette.get("team_b", [0, 0, 0]), dtype=float)
            print(f"[TEAM] Force palette enabled: target_A (team_a BGR)={target_a}, target_B (team_b BGR)={target_b}")
            # Optimal assignment: try both pairings and pick the one
            # with lowest total distance (avoids greedy mis-match when
            # one target is close to both clusters, e.g. black [0,0,0]).
            if len(self.cluster_colors) > 1:
                c0, c1 = self.cluster_colors[0], self.cluster_colors[1]
                print(f"[TEAM] Cluster 0 color: {c0}, Cluster 1 color: {c1}")
                cost_0a = np.linalg.norm(c0 - target_a) + np.linalg.norm(c1 - target_b)
                cost_0b = np.linalg.norm(c0 - target_b) + np.linalg.norm(c1 - target_a)
                print(f"[TEAM] Cost if cluster 0→A, 1→B: {cost_0a:.2f}, Cost if cluster 0→B, 1→A: {cost_0b:.2f}")
                a_label = 0 if cost_0a <= cost_0b else 1
            else:
                a_label = 0
            b_label = 1 - a_label if len(self.cluster_colors) > 1 else a_label
            self.label_to_team = {a_label: TeamID.TEAM_A, b_label: TeamID.TEAM_B}
            self._label_to_team_locked = True  # Lock it!
            self.team_palette = {
                TeamID.TEAM_A: tuple(int(x) for x in target_a),
                TeamID.TEAM_B: tuple(int(x) for x in target_b),
                TeamID.UNKNOWN: (200, 200, 200),
                TeamID.REFEREE: (0, 255, 255),
            }
            print(f"[TEAM] Selected mapping: cluster {a_label} → TEAM_A, cluster {b_label} → TEAM_B")
            return

        self.label_to_team = {0: TeamID.TEAM_A, 1: TeamID.TEAM_B}
        self._label_to_team_locked = True  # Lock default mapping too
        palette: dict[TeamID, tuple[int, int, int]] = {}
        for idx, color in enumerate(self.cluster_colors[:2]):
            team = self.label_to_team[idx]
            palette[team] = self._vividify(color)
        palette.setdefault(TeamID.TEAM_A, (255, 100, 100))
        palette.setdefault(TeamID.TEAM_B, (100, 100, 255))
        palette[TeamID.UNKNOWN] = (200, 200, 200)
        palette[TeamID.REFEREE] = (0, 255, 255)
        self.team_palette = palette

    def fit(self) -> None:
        if not self.ready:
            return
        data = np.stack([s["hist"] for s in self.samples])
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=0)
        self.cluster_labels = self.kmeans.fit_predict(data)

        # Compute outlier rejection threshold from training distances.
        # Samples beyond mean + 2*std from their cluster center are outliers.
        distances = self.kmeans.transform(data)  # (N, n_clusters)
        per_sample_dist = distances[np.arange(len(data)), self.cluster_labels]
        self.cluster_distance_threshold = float(
            np.mean(per_sample_dist) + 2.0 * np.std(per_sample_dist)
        )

        # CRITICAL: Enforce team_cap (6 players max per team) by rebalancing clusters
        # Count unique track IDs per cluster
        cluster_track_counts = {0: set(), 1: set()}
        for idx, label in enumerate(self.cluster_labels):
            track_id = self.samples[idx].get("track_id")
            if track_id is not None:
                cluster_track_counts[int(label)].add(track_id)
        
        unique_0 = len(cluster_track_counts[0])
        unique_1 = len(cluster_track_counts[1])
        
        # If one cluster has >6 unique tracks, rebalance by moving weakest matches to other cluster
        max_per_team = 6
        if unique_0 > max_per_team or unique_1 > max_per_team:
            print(f"[TEAM REBALANCE] K-means found {unique_0} tracks in cluster 0, {unique_1} in cluster 1")
            
            # For each sample in the oversized cluster, compute distance to both clusters
            # Move the weakest matches (highest distance) to the other cluster
            for oversize_label in [0, 1]:
                if len(cluster_track_counts[oversize_label]) > max_per_team:
                    # Get all samples from this cluster
                    oversize_indices = np.where(self.cluster_labels == oversize_label)[0]
                    
                    # Compute distance to assigned cluster for each sample
                    sample_distances = []
                    for idx in oversize_indices:
                        track_id = self.samples[idx].get("track_id")
                        dist_to_assigned = distances[idx, oversize_label]
                        dist_to_other = distances[idx, 1 - oversize_label]
                        sample_distances.append({
                            'idx': idx,
                            'track_id': track_id,
                            'dist_to_assigned': dist_to_assigned,
                            'dist_to_other': dist_to_other,
                            'dist_diff': dist_to_assigned - dist_to_other,  # positive = closer to other
                        })
                    
                    # Sort by distance difference (move samples that are closer to other cluster)
                    sample_distances.sort(key=lambda x: x['dist_diff'], reverse=True)
                    
                    # Move tracks until we're at max_per_team
                    moved_tracks = set()
                    for sample_info in sample_distances:
                        if len(cluster_track_counts[oversize_label]) <= max_per_team:
                            break
                        
                        track_id = sample_info['track_id']
                        # Move all samples from this track to the other cluster
                        if track_id is not None and track_id not in moved_tracks:
                            # Reassign all samples from this track
                            track_sample_count = 0
                            for idx in oversize_indices:
                                if self.samples[idx].get("track_id") == track_id:
                                    self.cluster_labels[idx] = 1 - oversize_label
                                    track_sample_count += 1
                            
                            cluster_track_counts[oversize_label].discard(track_id)
                            cluster_track_counts[1 - oversize_label].add(track_id)
                            moved_tracks.add(track_id)
                            
                            print(f"  Moved track {track_id} ({track_sample_count} samples) from cluster {oversize_label} to {1-oversize_label}")
            
            final_0 = len(cluster_track_counts[0])
            final_1 = len(cluster_track_counts[1])
            print(f"[TEAM REBALANCE] After rebalance: {final_0} tracks in cluster 0, {final_1} in cluster 1")

        self.cluster_colors = []
        for label in range(self.n_clusters):
            idxs = np.where(self.cluster_labels == label)[0].tolist()
            if idxs:
                mean_color = np.mean([self.samples[i]["mean_bgr"] for i in idxs], axis=0)
            else:
                mean_color = np.array([180.0, 180.0, 180.0], dtype=np.float32)
            self.cluster_colors.append(mean_color)
            print(f"[TEAM] Cluster {label} mean color (BGR): ({mean_color[0]:.0f}, {mean_color[1]:.0f}, {mean_color[2]:.0f})")

        self._build_palette()

    def predict(self, hist: np.ndarray) -> Optional[TeamID]:
        if self.kmeans is None or hist is None or hist.size == 0:
            return None
        label = int(self.kmeans.predict(hist.reshape(1, -1))[0])
        return self.label_to_team.get(label, TeamID.UNKNOWN)

    def predict_with_confidence(self, hist: np.ndarray) -> Tuple[Optional[TeamID], float]:
        """Return (team, distance_to_cluster). Lower distance = higher confidence."""
        if self.kmeans is None or hist is None or hist.size == 0:
            return None, float("inf")
        data = hist.reshape(1, -1)
        label = int(self.kmeans.predict(data)[0])
        dist = float(self.kmeans.transform(data)[0, label])
        return self.label_to_team.get(label, TeamID.UNKNOWN), dist

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def sample_full_video(self, reader, detector, step: int, max_samples: int, expected_players: Optional[int] = None, frame_limit: Optional[int] = None, start_frame: int = 0, proximity_thresh: float = 40.0) -> int:
        """Lightweight pre-pass to gather samples over the video.

        Only samples frames that meet the expected player count when provided.
        Stops after frame_limit frames if specified.
        Skips detections that are close to another player (within proximity_thresh px center-to-center),
        or if another detection's bbox overlaps the jersey crop region, or if another detection's center is inside the crop.
        Args:
            start_frame: First frame to start sampling from (0-indexed)
            proximity_thresh: Minimum center-to-center distance (in px) to consider a detection isolated
        """
        sampled = 0
        frames_processed = 0
        end_frame = start_frame + frame_limit if frame_limit is not None else None
        for frame_idx, frame in reader.frames(start_frame=start_frame, end_frame=end_frame, step=step):
            frames_processed += 1
            detections = detector.detect(frame, frame_idx)
            if expected_players is not None and len(detections) != expected_players:
                continue

            n = len(detections)
            skip_mask = [False] * n
            # Precompute centers and bbox arrays
            centers = [det.bbox.center for det in detections]
            bboxes = [det.bbox for det in detections]
            # For each detection, get the jersey crop region (x1, y1, x2, y2)
            crop_regions = [self._crop_jersey_region(det.bbox, frame.shape) for det in detections]

            def iou(boxA, boxB):
                # boxA, boxB: (x1, y1, x2, y2)
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                interW = max(0, xB - xA)
                interH = max(0, yB - yA)
                interArea = interW * interH
                boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
                boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
                unionArea = boxAArea + boxBArea - interArea
                if unionArea == 0:
                    return 0.0
                return interArea / unionArea

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # 1. Center-to-center proximity
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < proximity_thresh:
                        skip_mask[i] = True
                        if frame_idx % 100 == 0:
                            print(f"[SAMPLE] Skipping crop {i} at frame {frame_idx}: center too close to {j} (dist={dist:.1f})")
                        break
                    # 2. Bbox overlap with crop region (IoU > 0.01)
                    cropA = crop_regions[i]
                    bboxB = (bboxes[j].x1, bboxes[j].y1, bboxes[j].x2, bboxes[j].y2)
                    iou_val = iou(cropA, bboxB)
                    if iou_val > 0.01:
                        skip_mask[i] = True
                        if frame_idx % 100 == 0:
                            print(f"[SAMPLE] Skipping crop {i} at frame {frame_idx}: bbox {j} overlaps crop (IoU={iou_val:.3f})")
                        break
                    # 3. Center of other detection inside crop region
                    cx, cy = centers[j]
                    if cropA[0] <= cx <= cropA[2] and cropA[1] <= cy <= cropA[3]:
                        skip_mask[i] = True
                        if frame_idx % 100 == 0:
                            print(f"[SAMPLE] Skipping crop {i} at frame {frame_idx}: center of {j} inside crop region")
                        break

            for idx, det in enumerate(detections):
                if skip_mask[idx]:
                    continue  # Skip detections too close/overlapping/contaminated
                self.add_sample(frame, det.bbox, frame_idx=frame_idx, track_id=None)
                sampled += 1
                if sampled >= max_samples:
                    break
            if sampled >= max_samples:
                break
        return sampled

    # ------------------------------------------------------------------
    # Export crops
    # ------------------------------------------------------------------
    def save_crops(self) -> None:
        if not self.save_crops_enabled or self.crops_dir is None or not self.fitted or self.cluster_labels is None:
            return
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        cluster_dirs = {}
        for label, team in self.label_to_team.items():
            name = team.value if isinstance(team, TeamID) else f"cluster_{label}"
            cluster_dir = self.crops_dir / name
            cluster_dir.mkdir(parents=True, exist_ok=True)
            cluster_dirs[label] = cluster_dir

        for idx, sample in enumerate(self.samples):
            label = int(self.cluster_labels[idx]) if self.cluster_labels is not None else 0
            cluster_dir = cluster_dirs.get(label, self.crops_dir)
            frame_idx = sample.get("frame")
            track_id = sample.get("track_id")
            filename = f"f{frame_idx}_tid{track_id}_{idx}.jpg" if frame_idx is not None else f"sample_{idx}.jpg"
            if sample.get("crop") is not None:
                cv2.imwrite(str(cluster_dir / filename), sample["crop"])

    def describe_palette(self) -> str:
        return f"palette A={self.team_palette.get(TeamID.TEAM_A)} B={self.team_palette.get(TeamID.TEAM_B)}"
