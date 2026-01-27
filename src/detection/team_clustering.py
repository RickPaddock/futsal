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
        self.label_to_team: dict[int, TeamID] = {0: TeamID.TEAM_A, 1: TeamID.TEAM_B}
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
        """Build a mask that keeps likely jersey pixels and suppresses shorts/grass/skin.

        Combines:
        - HSV S/V threshold to drop very dark/low-saturation pixels
        - YCrCb skin suppression (typical ranges for human skin)
        Returns a binary mask or None if no pixels pass filters.
        """
        if roi_rgb is None or roi_rgb.size == 0:
            return None

        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
        base_mask = cv2.inRange(hsv, (0, 50, 60), (180, 255, 255))

        ycrcb = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2YCrCb)
        skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

        # Remove skin from base mask
        jersey_mask = cv2.bitwise_and(base_mask, cv2.bitwise_not(skin_mask))

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

    def extract_features(self, frame: np.ndarray, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        x1, y1, x2, y2 = self._crop_jersey_region(bbox, frame.shape)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(3 * self.bins, dtype=np.float32), np.array([200.0, 200.0, 200.0], dtype=np.float32), None

        roi_rgb = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)

        hist_h = cv2.calcHist([hsv], [0], None, [self.bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [self.bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [self.bins], [0, 256])
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)

        mean_bgr = cv2.cvtColor(np.mean(hsv, axis=(0, 1)).reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
        crop_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
        return hist, mean_bgr, crop_bgr

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

    def add_sample(self, frame: np.ndarray, bbox: BoundingBox, frame_idx: Optional[int] = None, track_id: Optional[int] = None) -> np.ndarray:
        if self.sample_count >= self.max_samples:
            hist, _, _ = self.extract_features(frame, bbox)
            return hist

        if not self._is_valid_bbox(frame.shape, bbox):
            # Return empty feature to signal skip
            return np.zeros(3 * self.bins, dtype=np.float32)

        hist, mean_bgr, crop_bgr = self.extract_features(frame, bbox)
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

        if self.force_palette.get("enabled", False):
            target_a = np.array(self.force_palette.get("team_a", [0, 140, 255]), dtype=float)
            target_b = np.array(self.force_palette.get("team_b", [0, 0, 0]), dtype=float)
            dists_a = [np.linalg.norm(c - target_a) for c in self.cluster_colors]
            a_label = int(np.argmin(dists_a)) if dists_a else 0
            b_label = 1 - a_label if len(self.cluster_colors) > 1 else a_label
            self.label_to_team = {a_label: TeamID.TEAM_A, b_label: TeamID.TEAM_B}
            self.team_palette = {
                TeamID.TEAM_A: tuple(int(x) for x in target_a),
                TeamID.TEAM_B: tuple(int(x) for x in target_b),
                TeamID.UNKNOWN: (200, 200, 200),
                TeamID.REFEREE: (0, 255, 255),
            }
            return

        self.label_to_team = {0: TeamID.TEAM_A, 1: TeamID.TEAM_B}
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

        self.cluster_colors = []
        for label in range(self.n_clusters):
            idxs = np.where(self.cluster_labels == label)[0].tolist()
            if idxs:
                mean_color = np.mean([self.samples[i]["mean_bgr"] for i in idxs], axis=0)
            else:
                mean_color = np.array([180.0, 180.0, 180.0], dtype=np.float32)
            self.cluster_colors.append(mean_color)

        self._build_palette()

    def predict(self, hist: np.ndarray) -> Optional[TeamID]:
        if self.kmeans is None or hist is None or hist.size == 0:
            return None
        label = int(self.kmeans.predict(hist.reshape(1, -1))[0])
        return self.label_to_team.get(label, TeamID.UNKNOWN)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def sample_full_video(self, reader, detector, step: int, max_samples: int, expected_players: Optional[int] = None) -> int:
        """Lightweight pre-pass to gather samples over the full video.

        Only samples frames that meet the expected player count when provided.
        """
        sampled = 0
        for frame_idx, frame in reader.frames(step=step):
            detections = detector.detect(frame, frame_idx)
            if expected_players is not None and len(detections) != expected_players:
                continue

            for det in detections:
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
