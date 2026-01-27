"""
Main processing pipeline for futsal player tracking.

Orchestrates the detection, tracking, and output generation stages.
"""

from pathlib import Path
from typing import Optional
from collections import defaultdict, deque
import sys
import numpy as np
from tqdm import tqdm
import cv2

from src.utils.video_io import VideoReader, VideoWriter
from src.utils.data_models import (
    PlayerDetection,
    PlayerTrack,
    MatchData,
    BoundingBox,
    TeamID,
)
from src.detection.player_detector import PlayerDetector, filter_detections_by_size, extract_color_histogram
from src.detection.ball_detector import BallDetector
from src.detection.tracking import ByteTracker, convert_stracks_to_player_tracks
from src.detection.segmentation_sam2 import SamSegmenter2
from src.geometry.homography import CourtHomography, create_homography_from_config
from src.detection.team_clustering import TeamClustering
from src.utils.visualization import TEAM_COLORS


class Pipeline:
    """Main processing pipeline for futsal match analysis."""

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        debug: bool = False,
    ):
        """
        Initialize the pipeline.

        Args:
            config: Configuration dictionary
            output_dir: Directory for output files
            debug: Enable debug visualization
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.debug = debug

        # Processing settings
        self.batch_size = config["processing"]["batch_size"]
        self.device = config["processing"]["device"]

        # Initialize components (lazy loading)
        self._player_detector: Optional[PlayerDetector] = None
        self._ball_detector: Optional[BallDetector] = None
        self._tracker: Optional[ByteTracker] = None
        self._sam_segmenter: Optional[SamSegmenter2] = None
        self._homography: Optional[CourtHomography] = None

        # Match data storage
        self.match_data: Optional[MatchData] = None
        # Runtime team palette (can be overridden by clustering or config)
        self.team_colors = TEAM_COLORS.copy()

    @property
    def player_detector(self) -> PlayerDetector:
        """Lazy-load player detector."""
        if self._player_detector is None:
            cfg = self.config["player_detection"]
            self._player_detector = PlayerDetector(
                model_path=cfg["model"],
                confidence_threshold=cfg["confidence_threshold"],
                iou_threshold=cfg["iou_threshold"],
                device=self.device,
                classes=cfg.get("classes", [0]),
                use_roboflow=cfg.get("use_roboflow", False),
                roboflow_model_id=cfg.get("roboflow_model_id"),
                max_detections=cfg.get("max_detections", 12),
                input_scale=cfg.get("input_scale", 1.0),
            )
        return self._player_detector

    @property
    def ball_detector(self) -> BallDetector:
        """Lazy-load ball detector."""
        if self._ball_detector is None:
            cfg = self.config.get("ball_detection")
            if cfg is None:
                raise ValueError("ball_detection config not found in config file")
            self._ball_detector = BallDetector(
                model_path=cfg["model"],
                confidence_threshold=cfg["confidence_threshold"],
                iou_threshold=cfg["iou_threshold"],
                device=self.device,
                max_detections=cfg.get("max_detections", 1),
                input_scale=cfg.get("input_scale", 1.0),
                use_inference_slicer=cfg.get("use_inference_slicer", False),
            )
        return self._ball_detector

    @property
    def tracker(self) -> ByteTracker:
        """Lazy-load tracker."""
        if self._tracker is None:
            self._tracker = self._build_tracker()
        return self._tracker

    @property
    def sam_segmenter(self) -> Optional[SamSegmenter2]:
        """Lazy-load SAM2 segmenter if enabled and available."""
        if self._sam_segmenter is None:
            seg_cfg = self.config.get("segmentation", {"enabled": False})
            if seg_cfg.get("enabled", False):
                sam2_cfg = seg_cfg.get("sam2", {})
                self._sam_segmenter = SamSegmenter2(
                    model_type=sam2_cfg.get("model_type", "hiera_small"),
                    checkpoint_path=sam2_cfg.get("checkpoint_path"),
                    device=self.device,
                )
            else:
                self._sam_segmenter = None
        return self._sam_segmenter

    @property
    def homography(self) -> Optional[CourtHomography]:
        """Lazy-load homography from config if calibration points available."""
        if self._homography is None:
            self._homography = create_homography_from_config(self.config)
        return self._homography

    def _build_tracker(self) -> ByteTracker:
        """Construct a fresh tracker instance from config."""
        cfg = self.config["tracking"]
        return ByteTracker(
            track_high_thresh=cfg["track_high_thresh"],
            track_low_thresh=cfg["track_low_thresh"],
            new_track_thresh=cfg["new_track_thresh"],
            track_buffer=cfg["track_buffer"],
            match_thresh=cfg["match_thresh"],
            max_center_distance=cfg.get("max_center_distance", 100.0),
            velocity_weight=cfg.get("velocity_weight", 0.3),
        )

    def run(
        self,
        video_path: Path,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> MatchData:
        """
        Run the full processing pipeline.

        Args:
            video_path: Path to input video
            start_frame: First frame to process
            end_frame: Last frame to process (exclusive)

        Returns:
            MatchData with all tracking results
        """
        # Reset tracker state per video to keep IDs consistent between runs
        self._tracker = self._build_tracker()

        reader = VideoReader(video_path)
        print(f"Video info:")
        print(f"  Resolution: {reader.width}x{reader.height}")
        print(f"  FPS: {reader.fps:.2f}")
        print(f"  Total frames: {reader.total_frames}")

        # Initialize match data
        self.match_data = MatchData(
            video_path=str(video_path),
            fps=reader.fps,
            total_frames=reader.total_frames,
            width=reader.width,
            height=reader.height,
        )

        # Team classification (per-run, optional)
        team_cfg = self.config.get("team_classification", {})
        team_enabled = team_cfg.get("enable", False)
        expected_players = self.config["tracking"].get("min_players_enforce", 12)
        team_bins = team_cfg.get("histogram_bins", 32)
        team_min_samples = team_cfg.get("min_samples", 10)
        team_max_samples = team_cfg.get("max_samples", 60)
        team_vote_count = team_cfg.get("vote_count", 3)
        team_recheck_interval = team_cfg.get("recheck_interval", 15)
        team_recheck_votes = team_cfg.get("recheck_votes", 5)
        team_recheck_near_px = team_cfg.get("recheck_near_px", 140)
        team_recheck_far_px = team_cfg.get("recheck_far_px", 260)
        team_recheck_min_interval = team_cfg.get("recheck_min_interval", 10)
        team_recheck_max_interval = team_cfg.get("recheck_max_interval", 30)
        team_cap = team_cfg.get("team_cap")
        team_force_palette = team_cfg.get("force_team_colors", {})
        team_sampling_stride = team_cfg.get("sampling_stride")
        team_sample_full_video = team_cfg.get("sample_full_video", False)
        sample_min_players = max(8, expected_players - 2)

        team_classifier = TeamClustering(
            bins=team_bins,
            n_clusters=team_cfg.get("n_clusters", 2),
            min_samples=team_min_samples,
            max_samples=team_max_samples,
            force_palette=team_force_palette,
            save_crops=team_cfg.get("save_crops", False),
            crops_dir=(self.output_dir / team_cfg.get("crops_output_dir", "team_crops")) if team_cfg.get("save_crops", False) else None,
            vividify=team_cfg.get("vividify", True),
        ) if team_enabled else None
        team_votes: dict[int, dict[TeamID, int]] = defaultdict(lambda: defaultdict(int)) if team_enabled else {}
        team_histories: dict[int, deque] = defaultdict(lambda: deque(maxlen=30)) if team_enabled else {}  # 30-frame window for robust rechecks
        team_last_fit_count = 0
        team_last_sample_log = -1
        team_last_recheck_frame = start_frame - 1
        self.team_colors = TEAM_COLORS.copy()
        team_init_logged = False
        t3_tracks_active: set[int] = set()
        
        # Pre-warm SAM2 if enabled (initialize it now to avoid first-frame lag)
        sam_available = False
        if self.config.get("segmentation", {}).get("enabled", False):
            print(f"DEBUG: Segmentation enabled. sam_segmenter type: {type(self.sam_segmenter)}")
            if self.sam_segmenter is None:
                print("Warning: SAM2 is enabled in config but segmenter is None")
            elif not self.sam_segmenter.available:
                print(f"Warning: SAM2 is enabled but not available. available={self.sam_segmenter.available}, _predictor={self.sam_segmenter._predictor}")
            else:
                sam_available = True
                print("[OK] SAM2 segmenter initialized and ready (temporal tracking mode).")

        # Track storage
        track_histories: dict[int, PlayerTrack] = {}  # Long-term track history for re-ID
        
        # Frame-to-frame detection state (T1/T2/T3 tier tracking)
        previous_frame_state: dict[int, dict] = {}  # {track_id: {tier, bbox, frame_idx}}
        current_frame_state: dict[int, dict] = {}

        # Low-confidence detections for debug visualization (show what was filtered)
        low_conf_detections: dict[int, list[PlayerDetection]] = {}

        # Ball position history (one per frame where detected)
        ball_positions: list = []  # Will store Detection objects with frame_idx

        # =============================================================
        # FULL SAM MODE STATE
        # When YOLO drops below 12, switch to SAM for ALL players
        # =============================================================
        full_sam_mode_active: bool = False
        last_full_t1_frame_idx: Optional[int] = None  # Frame where T1 found all 12
        last_full_t1_state: dict[int, dict] = {}  # Reference state: {track_id: {bbox, detection}}

        # Expected player count is fixed per run
        team_cap = team_cap if team_cap is not None else max(1, expected_players // 2)

        # =============================================================
        # DETECTION MODE STATE: All-or-nothing tier switching
        # =============================================================
        # T1_MODE: YOLO found 12 players, use YOLO for all
        # T2_MODE: YOLO found < 12 players, use SAM for ALL players
        # Once in T2_MODE, stay there until YOLO finds 12 again
        # =============================================================
        T1_MODE = "T1"  # YOLO mode
        T2_MODE = "T2"  # SAM mode (all players)
        current_detection_mode = T1_MODE  # Start in YOLO mode

        # Focused debug logging for problematic frames (optional)
        debug_frames = set()  # Empty set - disable debug logging, or add specific frame ranges

        # SAM T2 Recovery: Track which players we're SAM-tracking (fill gaps from YOLO)
        sam_tracking_players: set[int] = set()  # track_ids being SAM-tracked this frame

        def format_state(state: dict[int, dict]) -> str:
            if not state:
                return "-"
            parts = []
            for tid in sorted(state.keys()):
                info = state[tid]
                bbox = info.get('bbox')
                tier = info.get('tier', '?')
                if bbox is not None:
                    parts.append(f"{tid}:{tier}@{bbox.x1:.0f},{bbox.y1:.0f}-{bbox.x2:.0f},{bbox.y2:.0f}")
                else:
                    parts.append(f"{tid}:{tier}")
            return "; ".join(parts)

        def log_frame_state(frame_idx: int, label: str, state: dict[int, dict]):
            if frame_idx in debug_frames:
                print(f"Frame {frame_idx}: {label} -> {format_state(state)}")

        def log_counts(frame_idx: int, detections: list, active_track_ids: set[int], current_state: dict[int, dict], sam_mode: bool, sam_first: bool, prev_count: Optional[int]):
            if frame_idx in debug_frames:
                print(
                    f"Frame {frame_idx}: counts det={len(detections)} tracks={len(active_track_ids)} state={len(current_state)} "
                    f"sam_mode={sam_mode} sam_first={sam_first} prev_t1={prev_count}"
                )

        def min_pair_distance_px(state: dict[int, dict]) -> float:
            """Compute minimum pairwise distance between bbox centers in the current frame."""
            centers = [s['bbox'].center for s in state.values() if 'bbox' in s]
            if len(centers) < 2:
                return float('inf')
            min_d = float('inf')
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    d = (dx * dx + dy * dy) ** 0.5
                    if d < min_d:
                        min_d = d
            return min_d

        # Process frames
        if end_frame is None:
            end_frame = reader.total_frames

        run_frame_count = max(1, end_frame - start_frame)
        if team_enabled and (team_sampling_stride is None or team_sampling_stride <= 0):
            team_sampling_stride = max(1, run_frame_count // max(team_max_samples, 1))

        total_frames = end_frame - start_frame
        print(f"\nProcessing frames {start_frame} to {end_frame}...")

        # Optional full-video sampling for team colors (before main pass)
        if team_enabled and team_sample_full_video and team_classifier is not None:
            stride = max(1, reader.total_frames // max(team_max_samples, 1))
            print(f"[TEAM] Pre-pass sampling full video with stride={stride} (max {team_max_samples})")
            sampled = team_classifier.sample_full_video(
                reader,
                self.player_detector,
                step=stride,
                max_samples=team_max_samples,
                expected_players=None,  # allow sampling even if fewer players are visible
            )
            print(f"[TEAM] Pre-pass collected {sampled} samples")
            if team_classifier.ready:
                team_classifier.fit()
                self.team_colors = team_classifier.team_palette.copy()
                team_last_fit_count = team_classifier.sample_count
                print(f"[TEAM] pre-fit with {team_last_fit_count} samples | {team_classifier.describe_palette()}")
                team_classifier.save_crops()
            else:
                print(f"[TEAM] pre-pass collected {sampled} samples but not enough to fit (min={team_classifier.min_samples})")

        if team_enabled and not team_init_logged:
            print(
                f"[TEAM] Enabled | stride={team_sampling_stride} min={team_min_samples} max={team_max_samples} "
                f"vote={team_vote_count} force={team_force_palette.get('enabled', False)} run_frames={run_frame_count} "
                f"save_crops={team_cfg.get('save_crops', False)}"
            )
            team_init_logged = True

        with tqdm(total=total_frames, desc="Processing", unit="frame",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {percentage:3.0f}%') as pbar:
            for batch in reader.batch_frames(
                self.batch_size,
                start_frame=start_frame,
                end_frame=end_frame,
            ):
                frame_indices = [idx for idx, _ in batch]
                frames = [frame for _, frame in batch]

                # Player detection
                all_detections = self.player_detector.detect_batch(frames, frame_indices)
                
                # Ball detection
                all_ball_detections = self.ball_detector.detect_batch(frames, frame_indices)

                for (frame_idx, frame), detections, ball_detections in zip(batch, all_detections, all_ball_detections):
                    log_frame_state(frame_idx, "prev_state (memory)", previous_frame_state)

                    # Capture low-confidence detections for debug visualization
                    debug_low_thresh = self.config["player_detection"].get("debug_low_threshold", 0.1)
                    low_conf_detections[frame_idx] = [
                        d for d in detections
                        if d.bbox.confidence < self.config["player_detection"]["confidence_threshold"]
                        and d.bbox.confidence >= debug_low_thresh
                    ]
                    
                    # Store ball detections
                    if ball_detections:
                        # Update frame_idx on ball detections
                        for ball in ball_detections:
                            ball.frame_idx = frame_idx
                        self.match_data.ball_positions.extend(ball_detections)
                    
                    # Filter detections
                    detections = filter_detections_by_size(detections)
                    
                    # Set current frame for SAM2 temporal tracking (if enabled)
                    if sam_available:
                        self.sam_segmenter.set_image(frame, frame_idx)

                    # Extract color histograms for team classification
                    # Use team_classifier.extract_features (jersey-crop) to match training data,
                    # fall back to full-bbox histogram if classifier not available
                    det_team_preds = []
                    for det in detections:
                        if team_enabled and team_classifier is not None:
                            hist, _, _ = team_classifier.extract_features(frame, det.bbox)
                        else:
                            hist = extract_color_histogram(frame, det.bbox)
                        det.color_histogram = hist
                        if team_enabled and team_classifier is not None and team_classifier.fitted:
                            det_team_preds.append(team_classifier.predict(hist))
                        else:
                            det_team_preds.append(None)

                    # Update tracker (IoU + velocity + gating, no appearance)
                    det_array = self.player_detector.get_detection_array(detections)

                    # Build track-to-team map for ByteTrack gating (prevent cross-team swaps)
                    track_team_map = None
                    if team_enabled and team_classifier is not None and team_classifier.fitted:
                        track_team_map = {}
                        for tid, track in track_histories.items():
                            if track.team in (TeamID.TEAM_A, TeamID.TEAM_B):
                                track_team_map[tid] = track.team

                    active_tracks = self.tracker.update(
                        det_array,
                        frame_idx,
                        team_preds=det_team_preds if det_team_preds else None,
                        track_team_map=track_team_map,
                    )

                    # Convert active tracks to preliminary format (before tier decision)
                    yolo_track_results: dict[int, BoundingBox] = {}
                    for track_id, bbox in convert_stracks_to_player_tracks(
                        active_tracks, frame_idx
                    ):
                        yolo_track_results[track_id] = bbox

                    # =============================================================
                    # HYBRID TIER DETECTION LOGIC
                    # =============================================================
                    # T1: YOLO-detected players (always use when available)
                    # T2: SAM recovery for MISSING players only (not detected by YOLO)
                    # T3: Occlusion estimation for players not found by T1 or T2
                    # =============================================================
                    yolo_count = len(yolo_track_results)

                    # Initialize current frame state
                    current_frame_state = {}
                    active_track_ids = set()

                    # =============================================================
                    # STEP 1: T1 - Build YOLO detections (don't add to track_histories yet)
                    # We'll add to track_histories after deciding T1 vs Full SAM Mode
                    # =============================================================
                    for track_id, bbox in yolo_track_results.items():
                        active_track_ids.add(track_id)

                        # Ensure track exists
                        if track_id not in track_histories:
                            track_histories[track_id] = PlayerTrack(
                                track_id=track_id,
                                detections=[],
                            )

                        # Find matching original detection for class info
                        class_name = "player"
                        class_id = 2
                        mask = None
                        best_match_det = None
                        best_iou = 0.0

                        for orig_det in detections:
                            iou = self._compute_iou(orig_det.bbox, bbox)
                            if iou > best_iou:
                                best_iou = iou
                                best_match_det = orig_det

                        if best_match_det and best_iou > 0.5:
                            class_name = best_match_det.class_name
                            class_id = best_match_det.class_id
                            if hasattr(best_match_det, 'mask') and best_match_det.mask is not None:
                                mask = best_match_det.mask

                        # Create T1 detection (stored in current_frame_state, NOT track_histories yet)
                        det = PlayerDetection(
                            frame_idx=frame_idx,
                            bbox=bbox,
                            class_id=class_id,
                            class_name=class_name,
                            mask=mask,
                            is_sam_recovered=False,
                        )
                        # NOTE: Don't add to track_histories here - wait until after Full SAM Mode decision

                        current_frame_state[track_id] = {
                            'tier': 'T1',
                            'bbox': bbox,
                            'frame_idx': frame_idx,
                            'detection': det
                        }

                    log_frame_state(frame_idx, "T1 state (YOLO)", current_frame_state)

                    # =============================================================
                    # FULL SAM MODE: Check if we should store reference or switch modes
                    # =============================================================
                    # Debug: log state every 100 frames
                    if frame_idx % 100 == 0:
                        print(f"Frame {frame_idx}: DEBUG - T1 count={len(current_frame_state)}, ref_exists={bool(last_full_t1_state)}, ref_count={len(last_full_t1_state)}, full_sam_active={full_sam_mode_active}")

                    if len(current_frame_state) == expected_players:
                        # T1 found all 12 players - store as reference frame
                        last_full_t1_frame_idx = frame_idx
                        last_full_t1_state = {
                            tid: {
                                'bbox': info['bbox'],
                                'detection': info['detection'],
                                'class_id': info['detection'].class_id,
                                'class_name': info['detection'].class_name,
                            }
                            for tid, info in current_frame_state.items()
                        }

                        # Exit full SAM mode if we were in it
                        if full_sam_mode_active:
                            print(f"Frame {frame_idx}: EXITING Full SAM Mode - YOLO recovered all {expected_players} players")
                            full_sam_mode_active = False

                    elif len(current_frame_state) < expected_players and sam_available and last_full_t1_state:
                        # T1 dropped below 12, SAM is available, and we have a reference - enter full SAM mode
                        if not full_sam_mode_active:
                            print(f"Frame {frame_idx}: ENTERING Full SAM Mode - YOLO found {len(current_frame_state)}, ref has {len(last_full_t1_state)} players from frame {last_full_t1_frame_idx}")
                            full_sam_mode_active = True
                    elif len(current_frame_state) < expected_players and sam_available and not last_full_t1_state:
                        # No reference yet - will use fallback T2
                        if frame_idx % 100 == 0:
                            print(f"Frame {frame_idx}: No reference yet (YOLO never found {expected_players}), using fallback T2")

                    # =============================================================
                    # STEP 2: T2 - SAM processing
                    # =============================================================
                    # If in full SAM mode: run SAM on ALL players (not just missing)
                    # Otherwise: fall back to SAM recovery for missing players only

                    if full_sam_mode_active and sam_available and self.sam_segmenter is not None:
                        # FULL SAM MODE: Clear T1 results and use SAM for ALL players
                        # This ensures SAM can properly distinguish overlapping players
                        print(f"Frame {frame_idx}: Executing Full SAM Mode on {len(last_full_t1_state)} players (clearing {len(current_frame_state)} T1 results)")

                        # Snapshot YOLO detections before clearing — used for T2 proximity validation
                        yolo_frame_detections = dict(current_frame_state)

                        current_frame_state = {}
                        active_track_ids = set()

                        sam_success_count = 0
                        sam_fail_count = 0

                        # Track successfully segmented players this frame for negative prompts & proximity checks
                        segmented_this_frame = {}  # track_id -> sam_bbox

                        for track_id, ref_state in last_full_t1_state.items():
                            ref_bbox = ref_state['bbox']

                            # Build negative point prompts from already-segmented overlapping players
                            # This tells SAM "not this person" when two players overlap
                            negative_points = []
                            for seg_tid, seg_bbox in segmented_this_frame.items():
                                iou = self._compute_iou(ref_bbox, seg_bbox)
                                if iou > 0.1:  # any meaningful overlap
                                    cx, cy = seg_bbox.center
                                    negative_points.append((cx, cy))

                            # SAM segment using reference bbox + negative points
                            mask = self.sam_segmenter.segment_by_box(
                                ref_bbox,
                                negative_points=negative_points or None,
                            )

                            if mask is not None and np.sum(mask) >= 300:
                                # Derive bbox from mask for accuracy
                                ys, xs = np.where(mask > 0)
                                if len(xs) > 0 and len(ys) > 0:
                                    sam_x1, sam_y1 = int(xs.min()), int(ys.min())
                                    sam_x2, sam_y2 = int(xs.max()), int(ys.max())
                                    sam_bbox = BoundingBox(
                                        x1=sam_x1, y1=sam_y1, x2=sam_x2, y2=sam_y2,
                                        confidence=0.7
                                    )

                                    # Validate movement isn't too extreme
                                    sam_cx, sam_cy = sam_bbox.center
                                    ref_cx, ref_cy = ref_bbox.center
                                    center_dist = ((sam_cx - ref_cx)**2 + (sam_cy - ref_cy)**2)**0.5
                                    max_movement = 200  # Slightly more lenient for full SAM mode

                                    if center_dist <= max_movement:
                                        # ===== T2 PROXIMITY VALIDATION =====
                                        # A T2 detection is only valid if it's near a T1 or another T2.
                                        # An isolated T2 with no nearby player means SAM latched onto
                                        # background (fence, pitch lines) — reject it.
                                        has_nearby_player = False

                                        # Check against YOLO T1 detections from this frame
                                        for yolo_state in yolo_frame_detections.values():
                                            yolo_cx, yolo_cy = yolo_state['bbox'].center
                                            dist = ((sam_cx - yolo_cx)**2 + (sam_cy - yolo_cy)**2)**0.5
                                            if dist < 150:
                                                has_nearby_player = True
                                                break

                                        # Check against already-segmented T2 players this frame
                                        if not has_nearby_player:
                                            for seg_tid, seg_bbox in segmented_this_frame.items():
                                                seg_cx, seg_cy = seg_bbox.center
                                                dist = ((sam_cx - seg_cx)**2 + (sam_cy - seg_cy)**2)**0.5
                                                if dist < 150:
                                                    has_nearby_player = True
                                                    break

                                        if not has_nearby_player:
                                            # Isolated T2 — no player nearby to cause occlusion
                                            # Do NOT update reference bbox (keep last good one for next frame)
                                            if frame_idx % 50 == 0:
                                                print(f"  SAM reject #{track_id}: isolated T2 at ({sam_cx:.0f},{sam_cy:.0f}) - no nearby T1/T2")
                                            sam_fail_count += 1
                                            continue

                                        # Create T2 detection
                                        det = PlayerDetection(
                                            frame_idx=frame_idx,
                                            bbox=sam_bbox,
                                            class_id=ref_state['class_id'],
                                            class_name=ref_state['class_name'],
                                            mask=mask,
                                            is_sam_recovered=True,
                                        )

                                        # Add to tracking
                                        active_track_ids.add(track_id)
                                        if track_id not in track_histories:
                                            track_histories[track_id] = PlayerTrack(track_id=track_id, detections=[])
                                        track_histories[track_id].detections.append(det)

                                        current_frame_state[track_id] = {
                                            'tier': 'T2',
                                            'bbox': sam_bbox,
                                            'frame_idx': frame_idx,
                                            'detection': det
                                        }
                                        sam_tracking_players.add(track_id)

                                        # Track successful segmentation for negative prompts on subsequent players
                                        segmented_this_frame[track_id] = sam_bbox

                                        # Update reference bbox for next frame (so prompts stay accurate)
                                        last_full_t1_state[track_id]['bbox'] = sam_bbox
                                        sam_success_count += 1
                                    else:
                                        if frame_idx % 50 == 0:
                                            print(f"  SAM fail #{track_id}: moved too far ({center_dist:.0f}px > {max_movement}px)")
                                        sam_fail_count += 1
                                else:
                                    if frame_idx % 50 == 0:
                                        print(f"  SAM fail #{track_id}: empty mask coords")
                                    sam_fail_count += 1
                            else:
                                if frame_idx % 50 == 0:
                                    mask_size = np.sum(mask) if mask is not None else 0
                                    print(f"  SAM fail #{track_id}: mask too small ({mask_size} pixels)")
                                sam_fail_count += 1

                        print(f"Frame {frame_idx}: Full SAM Mode result - {sam_success_count} success, {sam_fail_count} fail"
                              f" (neg_prompts used, {len(yolo_frame_detections)} YOLO refs for proximity)")

                    elif sam_available and self.sam_segmenter is not None and len(current_frame_state) < expected_players:
                        # FALLBACK T2: No reference frame yet - use legacy per-player SAM recovery
                        # This only runs before we first see all 12 players
                        # Find players missing from YOLO detection
                        all_prev_track_ids = set(previous_frame_state.keys())
                        missing_track_ids = all_prev_track_ids - set(current_frame_state.keys())

                        if missing_track_ids and frame_idx % 100 == 0:
                            print(f"Frame {frame_idx}: T2 SAM - YOLO has {yolo_count}, SAMing {len(missing_track_ids)} missing players")

                        for track_id in sorted(missing_track_ids):
                            # Stop if we've reached expected count
                            if len(current_frame_state) >= expected_players:
                                break

                            # Get previous position for this player
                            if track_id not in previous_frame_state:
                                continue
                            prev_state = previous_frame_state[track_id]
                            prev_bbox = prev_state['bbox']
                            frames_since_seen = frame_idx - prev_state['frame_idx']

                            # Skip stale tracks (not seen in 6+ frames)
                            if frames_since_seen > 6:
                                continue

                            # Get class info from track history
                            if track_id in track_histories and len(track_histories[track_id].detections) > 0:
                                last_track_det = track_histories[track_id].detections[-1]
                                class_id = last_track_det.class_id
                                class_name = last_track_det.class_name
                            else:
                                class_id = 2
                                class_name = "player"

                            # SAM the player using previous bbox as prompt
                            mask = self.sam_segmenter.segment_by_box(prev_bbox)

                            if mask is not None and np.sum(mask) >= 300:
                                # SAM succeeded - derive bbox from mask for accuracy
                                ys, xs = np.where(mask > 0)
                                if len(xs) > 0 and len(ys) > 0:
                                    sam_x1, sam_y1 = int(xs.min()), int(ys.min())
                                    sam_x2, sam_y2 = int(xs.max()), int(ys.max())
                                    sam_bbox = BoundingBox(
                                        x1=sam_x1, y1=sam_y1, x2=sam_x2, y2=sam_y2,
                                        confidence=0.6
                                    )

                                    # VALIDATION 1: Check SAM bbox doesn't overlap too much with existing T1 detections
                                    # This prevents SAM from "stealing" an already-tracked player
                                    overlaps_existing = False
                                    for existing_tid, existing_state in current_frame_state.items():
                                        if existing_state['tier'] == 'T1':
                                            existing_bbox = existing_state['bbox']
                                            iou = self._compute_iou(sam_bbox, existing_bbox)
                                            if iou > 0.3:  # If >30% overlap, SAM found wrong player
                                                overlaps_existing = True
                                                if frame_idx % 100 == 0:
                                                    print(f"  T2 reject #{track_id}: SAM overlaps T1 #{existing_tid} (IoU={iou:.2f})")
                                                break

                                    if overlaps_existing:
                                        continue  # Skip this SAM recovery, it found the wrong player

                                    # VALIDATION 2: Check SAM bbox center is within reasonable distance of expected position
                                    # Prevents SAM from jumping to a completely different player across the pitch
                                    sam_cx, sam_cy = sam_bbox.center
                                    prev_cx, prev_cy = prev_bbox.center
                                    center_dist = ((sam_cx - prev_cx)**2 + (sam_cy - prev_cy)**2)**0.5
                                    max_movement = 150  # pixels - max reasonable movement between frames
                                    if center_dist > max_movement:
                                        if frame_idx % 100 == 0:
                                            print(f"  T2 reject #{track_id}: SAM moved too far ({center_dist:.0f}px > {max_movement}px)")
                                        continue  # SAM found something too far away

                                    # Create T2 detection
                                    det = PlayerDetection(
                                        frame_idx=frame_idx,
                                        bbox=sam_bbox,
                                        class_id=class_id,
                                        class_name=class_name,
                                        mask=mask,
                                        is_sam_recovered=True,
                                    )

                                    # Add to tracking
                                    active_track_ids.add(track_id)
                                    if track_id not in track_histories:
                                        track_histories[track_id] = PlayerTrack(track_id=track_id, detections=[])
                                    track_histories[track_id].detections.append(det)

                                    current_frame_state[track_id] = {
                                        'tier': 'T2',
                                        'bbox': sam_bbox,
                                        'frame_idx': frame_idx,
                                        'detection': det
                                    }
                                    sam_tracking_players.add(track_id)

                    # =============================================================
                    # COMMIT T1 detections to track_histories (if not in Full SAM Mode)
                    # Full SAM Mode already committed T2 detections; T1 detections were skipped
                    # Fallback T2 needs T1 detections committed now
                    # =============================================================
                    if not full_sam_mode_active:
                        for track_id, state in current_frame_state.items():
                            if state['tier'] == 'T1':
                                # Commit T1 detection to track_histories
                                track_histories[track_id].detections.append(state['detection'])

                    log_frame_state(frame_idx, "T1+T2 state", current_frame_state)
                    log_counts(frame_idx, detections, active_track_ids, current_frame_state,
                              len(current_frame_state) < expected_players, False, yolo_count)

                    # T3 OCCLUSION ESTIMATION: If still <12 after SAM, estimate positions
                    # Use positional inference: players don't teleport, assume behind closest active player

                    current_player_count = len(active_track_ids)

                    # Define edge zones (players near edges might have walked off camera)
                    edge_margin = 50  # pixels from frame edge
                    frame_h, frame_w = frame.shape[:2]

                    def is_near_edge(bbox: BoundingBox) -> bool:
                        """Check if bbox is near frame edge (player might have walked off)."""
                        return (bbox.x1 < edge_margin or
                                bbox.x2 > frame_w - edge_margin or
                                bbox.y1 < edge_margin or
                                bbox.y2 > frame_h - edge_margin)

                    def bbox_distance(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
                        """Calculate center-to-center distance between two bboxes."""
                        cx1, cy1 = bbox1.center
                        cx2, cy2 = bbox2.center
                        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

                    # Process lost tracks - try to recover them with T3 (estimated position)
                    if current_player_count < expected_players:
                        boxes_added = 0
                        max_to_add = expected_players - current_player_count

                        print(f"Frame {frame_idx}: T3 check - {current_player_count} active, need {expected_players}, missing={expected_players-current_player_count}")

                        # T3: Recover tracks that were in previous frame but missing from current frame
                        # Find which tracks disappeared
                        missing_track_ids = [tid for tid in previous_frame_state.keys() if tid not in current_frame_state]
                        print(f"    T3: {len(missing_track_ids)} tracks missing from current frame: {missing_track_ids}")

                        # Sort by track_id for consistency (prefer lower IDs first)
                        missing_track_ids.sort()

                        for track_id in missing_track_ids:
                            if boxes_added >= max_to_add:
                                break

                            # Get the previous state for this track
                            prev_state = previous_frame_state[track_id]
                            last_bbox = prev_state['bbox']
                            frames_since_seen = frame_idx - prev_state['frame_idx']

                            # Skip ONLY if track is very stale (not seen in 6+ frames)
                            # We MUST recover recently-seen players to maintain player count
                            if frames_since_seen > 6:
                                continue

                            # Find the closest active player to where this player disappeared
                            closest_active = None
                            closest_active_id = None
                            closest_distance = float('inf')

                            head_to_foot = getattr(self.homography, "head_to_foot_offset", 150)
                            prev_center_x, _ = last_bbox.center
                            prev_foot_y = last_bbox.y1 + head_to_foot

                            # Helper to reuse previous T3 anchor (keeps consistent offset)
                            anchor = prev_state.get('t3_anchor') if isinstance(prev_state, dict) else None
                            if anchor:
                                anchor_occluder_id = anchor.get('occluder_id')
                                if anchor_occluder_id in current_frame_state:
                                    closest_active_id = anchor_occluder_id
                                    closest_active = current_frame_state[anchor_occluder_id]['bbox']

                            if closest_active is None:
                                for active_track_id in active_track_ids:
                                    if active_track_id not in current_frame_state:
                                        continue
                                    active_state = current_frame_state[active_track_id]
                                    if active_state['tier'] not in ('T1', 'T2'):
                                        continue  # only anchor to visible players
                                    active_bbox = active_state['bbox']

                                    # Prefer occluders that are closer to camera (larger foot_y)
                                    active_foot_y = active_bbox.y1 + head_to_foot
                                    if active_foot_y < prev_foot_y - 30:
                                        continue  # occluder must be in front of (or reasonably near) missing player

                                    dist = bbox_distance(last_bbox, active_bbox)
                                    if dist < closest_distance:
                                        closest_distance = dist
                                        closest_active = active_bbox
                                        closest_active_id = active_track_id

                            # If there's a player within reasonable distance, assume occlusion.
                            # Allow larger gap to avoid losing the anchor; if we have an anchor match, trust it.
                            occlusion_distance_threshold = 280  # pixels
                            if hasattr(self, '_t2_recovery_history') and track_id in self._t2_recovery_history:
                                if frame_idx - self._t2_recovery_history[track_id] <= 2:
                                    occlusion_distance_threshold = 240

                            anchored_match = anchor is not None and closest_active_id == anchor.get('occluder_id')

                            if closest_active and (anchored_match or closest_distance < occlusion_distance_threshold):
                                # Lost player is behind the closest active player
                                full_height = closest_active.y2 - closest_active.y1
                                occluded_height = full_height * 0.5  # keep them visibly shorter than a true box

                                occ_center_x, _ = closest_active.center
                                occ_foot_y = closest_active.y1 + head_to_foot

                                # Compute / reuse anchor offset relative to occluder (in pixels)
                                if not anchor or anchor.get('occluder_id') != closest_active_id:
                                    dx_anchor = prev_center_x - occ_center_x
                                    dy_anchor = prev_foot_y - occ_foot_y
                                    anchor = {
                                        'occluder_id': closest_active_id,
                                        'dx': dx_anchor,
                                        'dy': dy_anchor,
                                        'created_frame': frame_idx,
                                        'last_occ_x': occ_center_x,
                                        'last_occ_y': occ_foot_y,
                                    }
                                else:
                                    dx_anchor = anchor.get('dx', 0)
                                    dy_anchor = anchor.get('dy', -50)

                                desired_foot_x = occ_center_x + dx_anchor
                                desired_foot_y = occ_foot_y + dy_anchor

                                # Prevent teleports: only limit movement if occluder itself moved too far
                                # (someone else became the occluder). Track occluder's last position.
                                last_occ_x = anchor.get('last_occ_x', occ_center_x)
                                last_occ_y = anchor.get('last_occ_y', occ_foot_y)
                                occ_movement = ((occ_center_x - last_occ_x)**2 + (occ_foot_y - last_occ_y)**2)**0.5
                                
                                # Only cap if occluder teleported (different player took over anchor)
                                max_occluder_movement = 150  # reasonable single-frame movement for occluder
                                if occ_movement > max_occluder_movement:
                                    # Occluder teleported - likely switched to different player, smooth the transition
                                    prev_foot_x = prev_center_x
                                    max_step_px = 80
                                    step_x = max(-max_step_px, min(max_step_px, desired_foot_x - prev_foot_x))
                                    step_y = max(-max_step_px, min(max_step_px, desired_foot_y - prev_foot_y))
                                    foot_x = prev_foot_x + step_x
                                    foot_y = prev_foot_y + step_y
                                else:
                                    # Follow occluder directly - no smoothing
                                    foot_x = desired_foot_x
                                    foot_y = desired_foot_y
                                
                                # Update occluder's last position for next frame
                                anchor['last_occ_x'] = occ_center_x
                                anchor['last_occ_y'] = occ_foot_y

                                occluded_width = closest_active.x2 - closest_active.x1
                                half_w = occluded_width / 2

                                y1_est = foot_y - head_to_foot
                                y2_est = y1_est + occluded_height

                                occluded_bbox = BoundingBox(
                                    x1=foot_x - half_w,
                                    y1=y1_est,
                                    x2=foot_x + half_w,
                                    y2=y2_est,
                                    confidence=0.3,  # Low confidence = interpolated
                                )

                            else:
                                # Fallback: no valid occluder found.
                                # MUST still create a box to maintain player count - use last known position
                                # Reduce height to indicate uncertainty
                                occluded_width = last_bbox.x2 - last_bbox.x1
                                occluded_height = (last_bbox.y2 - last_bbox.y1) * 0.4  # very short to indicate uncertainty
                                occluded_bbox = BoundingBox(
                                    x1=last_bbox.x1,
                                    y1=last_bbox.y2 - occluded_height,
                                    x2=last_bbox.x2,
                                    y2=last_bbox.y2,
                                    confidence=0.2,  # very low confidence
                                )
                                anchor = None  # no anchor since no occluder

                            det = PlayerDetection(
                                frame_idx=frame_idx,
                                bbox=occluded_bbox,
                                class_id=prev_state['detection'].class_id,
                                class_name=prev_state['detection'].class_name,
                                is_interpolated=True,
                            )

                            # Ensure track exists in history
                            if track_id not in track_histories:
                                track_histories[track_id] = PlayerTrack(track_id=track_id, detections=[])
                            track_histories[track_id].detections.append(det)

                            # Add to current_frame_state and active set
                            current_frame_state[track_id] = {
                                'tier': 'T3',
                                'bbox': occluded_bbox,
                                'frame_idx': frame_idx,
                                'detection': det,
                                't3_anchor': anchor,
                            }
                            # Preserve team from track history if present
                            if track_id in track_histories:
                                det.team = track_histories[track_id].team
                            t3_tracks_active.add(track_id)
                            active_track_ids.add(track_id)

                            boxes_added += 1
                            print(f"    T3 - Player #{track_id} occluded behind active player (dist={closest_distance:.0f}px)")

                        if boxes_added > 0:
                            print(f"Frame {frame_idx}: T3 recovered {boxes_added} players")

                    # Log any T3 additions and the state that carries to the next frame
                    t3_state: dict[int, dict] = {}
                    for tid, history in track_histories.items():
                        if not history.detections:
                            continue
                        last_det = history.detections[-1]
                        if last_det.frame_idx == frame_idx and getattr(last_det, "is_interpolated", False):
                            t3_state[tid] = {
                                'tier': 'T3',
                                'bbox': last_det.bbox,
                                'frame_idx': frame_idx,
                                'detection': last_det,
                            }

                    log_frame_state(frame_idx, "T3 additions", t3_state)
                    log_frame_state(frame_idx, "end_state (carry to next)", current_frame_state)

                    # =============================================================
                    # TEAM CLASSIFICATION (optional, per-run)
                    # =============================================================
                    if team_enabled and team_classifier is not None:
                        # Sample jersey colors across the video at a fixed stride (only when full roster is present)
                        if (
                            team_classifier.sample_count < team_max_samples
                            and ((frame_idx - start_frame) % team_sampling_stride == 0)
                            and (len(current_frame_state) >= sample_min_players or not team_classifier.ready)
                        ):
                            for track_id, state in current_frame_state.items():
                                hist = team_classifier.add_sample(frame, state['bbox'], frame_idx=frame_idx, track_id=track_id)
                                state['team_hist'] = hist
                                team_histories[track_id].append(hist)

                            if frame_idx != team_last_sample_log:
                                print(f"[TEAM] frame={frame_idx} samples={team_classifier.sample_count}/{team_max_samples}")
                                team_last_sample_log = frame_idx

                        # Fit/re-fit when new samples arrive
                        if team_classifier.ready and team_classifier.sample_count > team_last_fit_count:
                            team_classifier.fit()
                            self.team_colors = team_classifier.team_palette.copy()
                            team_last_fit_count = team_classifier.sample_count
                            print(f"[TEAM] fit with {team_last_fit_count} samples | {team_classifier.describe_palette()}")
                            team_classifier.save_crops()

                        # Constant recheck interval to prevent ID swaps when players are close
                        if team_classifier.fitted and team_recheck_max_interval > 0:
                            # Fixed 5-frame interval for robust team tracking
                            effective_interval = 5

                            if (frame_idx - team_last_recheck_frame) >= effective_interval:
                                team_last_recheck_frame = frame_idx
                                
                                # Collect correction candidates: players whose histogram votes disagree with current team
                                swap_candidates: dict[TeamID, list[tuple[int, TeamID, int, float, bool]]] = {
                                    TeamID.TEAM_A: [],
                                    TeamID.TEAM_B: [],
                                }
                                
                                for track_id, hist_deque in team_histories.items():
                                    if not hist_deque or track_id not in track_histories:
                                        continue
                                    
                                    # Minimum sample gate: need at least 12 samples for reliable recheck
                                    if len(hist_deque) < 12:
                                        continue
                                    
                                    current_team = track_histories[track_id].team
                                    if current_team not in (TeamID.TEAM_A, TeamID.TEAM_B):
                                        continue
                                    
                                    # Weighted voting: 2x weight for recent 10 frames, 1x for older
                                    vote_counts: dict[TeamID, float] = defaultdict(float)
                                    hist_list = list(hist_deque)
                                    recent_cutoff = max(0, len(hist_list) - 10)
                                    
                                    for i, h in enumerate(hist_list):
                                        team_pred = team_classifier.predict(h)
                                        if team_pred is not None:
                                            weight = 2.0 if i >= recent_cutoff else 1.0
                                            vote_counts[team_pred] += weight

                                    if not vote_counts:
                                        continue

                                    majority_team = max(vote_counts.items(), key=lambda x: x[1])[0]
                                    total_votes = sum(vote_counts.values())
                                    confidence = vote_counts[majority_team] / total_votes if total_votes > 0 else 0
                                    
                                    # Strong evidence for different team than current assignment
                                    # 4+ weighted votes OR 75%+ confidence
                                    strong_evidence = (vote_counts[majority_team] >= 4 or confidence >= 0.75) and majority_team != current_team
                                    
                                    # Very strong evidence for hard reset: 85%+ confidence
                                    very_strong = confidence >= 0.85 and majority_team != current_team
                                    
                                    if strong_evidence:
                                        # Add to swap candidates: (track_id, should_be_team, vote_strength, confidence, very_strong)
                                        swap_candidates[majority_team].append((track_id, majority_team, int(vote_counts[majority_team]), confidence, very_strong))
                                
                                # Execute swaps in pairs to maintain balance
                                # Players on TEAM_A who should be TEAM_B <-> Players on TEAM_B who should be TEAM_A
                                a_to_b = swap_candidates[TeamID.TEAM_B]  # Currently A, should be B
                                b_to_a = swap_candidates[TeamID.TEAM_A]  # Currently B, should be A
                                
                                # Sort by confidence first, then vote strength
                                a_to_b.sort(key=lambda x: (x[3], x[2]), reverse=True)
                                b_to_a.sort(key=lambda x: (x[3], x[2]), reverse=True)
                                
                                # Pair up and swap
                                swaps_made = 0
                                hard_resets = 0
                                for i in range(min(len(a_to_b), len(b_to_a))):
                                    tid_a, new_team_a, _, _, very_strong_a = a_to_b[i]
                                    tid_b, new_team_b, _, _, very_strong_b = b_to_a[i]
                                    
                                    # Swap teams
                                    if tid_a in current_frame_state:
                                        current_frame_state[tid_a]['detection'].team = new_team_a
                                    if tid_a in track_histories:
                                        track_histories[tid_a].team = new_team_a
                                    team_votes[tid_a].clear()
                                    team_votes[tid_a][new_team_a] = team_vote_count
                                    if very_strong_a:
                                        team_histories[tid_a].clear()  # Hard reset: clear history
                                        hard_resets += 1
                                    
                                    if tid_b in current_frame_state:
                                        current_frame_state[tid_b]['detection'].team = new_team_b
                                    if tid_b in track_histories:
                                        track_histories[tid_b].team = new_team_b
                                    team_votes[tid_b].clear()
                                    team_votes[tid_b][new_team_b] = team_vote_count
                                    if very_strong_b:
                                        team_histories[tid_b].clear()  # Hard reset: clear history
                                        hard_resets += 1
                                    
                                    swaps_made += 1
                                
                                # Solo corrections: if high confidence (≥75%) and no swap partner, force correction
                                # This handles off-screen or T3 cases
                                unpaired_a_to_b = a_to_b[swaps_made:]
                                unpaired_b_to_a = b_to_a[swaps_made:]
                                solo_corrections = 0
                                
                                for tid, new_team, votes, conf, very_strong in unpaired_a_to_b:
                                    if conf >= 0.75:  # Lowered threshold
                                        if tid in current_frame_state:
                                            current_frame_state[tid]['detection'].team = new_team
                                        if tid in track_histories:
                                            track_histories[tid].team = new_team
                                        team_votes[tid].clear()
                                        team_votes[tid][new_team] = team_vote_count
                                        if very_strong:
                                            team_histories[tid].clear()
                                            hard_resets += 1
                                        solo_corrections += 1
                                
                                for tid, new_team, votes, conf, very_strong in unpaired_b_to_a:
                                    if conf >= 0.75:  # Lowered threshold
                                        if tid in current_frame_state:
                                            current_frame_state[tid]['detection'].team = new_team
                                        if tid in track_histories:
                                            track_histories[tid].team = new_team
                                        team_votes[tid].clear()
                                        team_votes[tid][new_team] = team_vote_count
                                        if very_strong:
                                            team_histories[tid].clear()
                                            hard_resets += 1
                                        solo_corrections += 1
                                
                                if swaps_made > 0 or solo_corrections > 0:
                                    msg = f"[TEAM] Frame {frame_idx}: Swapped {swaps_made} pairs, {solo_corrections} solo"
                                    if hard_resets > 0:
                                        msg += f" ({hard_resets} hard resets)"
                                    print(msg)

                        # Assign teams per frame with votes and cap; natural voting only (no forced initial assignment)
                        if team_classifier.fitted:
                            frame_team_counts: dict[TeamID, int] = defaultdict(int)

                            # Per-frame vote
                            for track_id, state in current_frame_state.items():
                                hist = state.get('team_hist') if isinstance(state, dict) else None
                                if hist is None or (hasattr(hist, "size") and hist.size == 0):
                                    hist = team_classifier.extract_features(frame, state['bbox'])[0]

                                team = team_classifier.predict(hist) or TeamID.UNKNOWN

                                votes = team_votes[track_id]
                                votes[team] += 1

                                # Use vote if it meets threshold; else keep per-frame prediction
                                if votes[team] >= team_vote_count:
                                    assigned_team = team
                                else:
                                    assigned_team = team

                                # Enforce per-team cap
                                cap_blocks = assigned_team in (TeamID.TEAM_A, TeamID.TEAM_B) and frame_team_counts[assigned_team] >= team_cap
                                if cap_blocks:
                                    assigned_team = TeamID.UNKNOWN

                                state['detection'].team = assigned_team
                                if assigned_team in (TeamID.TEAM_A, TeamID.TEAM_B):
                                    frame_team_counts[assigned_team] += 1

                                if track_id in track_histories:
                                    track_histories[track_id].team = assigned_team

                                team_histories[track_id].append(hist)

                            # Hard cap cleanup: demote excess if any team exceeds cap
                            for team_id, count in list(frame_team_counts.items()):
                                if count <= team_cap:
                                    continue
                                overflow = count - team_cap
                                for track_id, state in current_frame_state.items():
                                    if overflow <= 0:
                                        break
                                    assigned = state['detection'].team if 'detection' in state else TeamID.UNKNOWN
                                    if assigned == team_id:
                                        state['detection'].team = TeamID.UNKNOWN
                                        overflow -= 1
                                        frame_team_counts[team_id] -= 1
                    
                    # Store ball detection if found (single ball per frame)
                    if ball_detections:
                        ball_positions.extend(ball_detections)
                    
                    # Update previous_frame_state for next iteration
                    previous_frame_state = current_frame_state.copy()
                    # Cache current frame for SAM baseline (in case YOLO drops on next frame)
                    previous_frame_image = frame.copy()

                pbar.update(len(batch))

        # Store results
        self.match_data.player_tracks = list(track_histories.values())
        self.match_data.ball_positions = ball_positions
        self.match_data.low_confidence_detections = low_conf_detections

        print(f"\nResults:")
        print(f"  Player tracks: {len(self.match_data.player_tracks)}")
        print(f"  Ball detections: {len(self.match_data.ball_positions)}")

        # Generate debug output if enabled
        if self.debug:
            self._generate_debug_video(video_path, start_frame, end_frame)

        # Save results
        self._save_results()

        print(f"\nOutput files:")
        print(f"  Directory: {self.output_dir}")
        if self.debug:
            print(f"  Debug video: {self.output_dir / 'debug_tracking.mp4'}")
        print(f"  Results JSON: {self.output_dir / 'tracking_results.json'}")

        return self.match_data

    def _generate_debug_video(
        self,
        video_path: Path,
        start_frame: int,
        end_frame: int,
    ):
        """Generate debug visualization video."""
        from src.utils.visualization import draw_frame_annotations, BallAnnotator

        print("\nGenerating debug video...")
        reader = VideoReader(video_path)
        ball_annotator = BallAnnotator(
            radius=6,
            buffer_size=60,
            thickness=2,
            max_age_seconds=2.0,
            fps=reader.fps,
        )

        # Scale down for faster rendering (0.5 = half resolution)
        scale = self.config.get("output", {}).get("debug_scale", 0.5)
        out_width = int(reader.width * scale)
        out_height = int(reader.height * scale)
        # Ensure even dimensions
        out_width = out_width - (out_width % 2)
        out_height = out_height - (out_height % 2)

        output_path = self.output_dir / "debug_tracking.mp4"
        with VideoWriter(
            output_path,
            out_width,
            out_height,
            reader.fps,
        ) as writer:
            for frame_idx, frame in tqdm(
                reader.frames(start_frame=start_frame, end_frame=end_frame),
                total=end_frame - start_frame,
                desc="Rendering",
                unit="frame",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {percentage:3.0f}%'
            ):
                # Draw annotations at full resolution for accuracy
                pitch_top_y = self.config.get("segmentation", {}).get("pitch_top_y")
                pitch_bottom_y = self.config.get("segmentation", {}).get("pitch_bottom_y")
                annotated = draw_frame_annotations(
                    frame,
                    frame_idx,
                    self.match_data,
                    pitch_top_y=pitch_top_y,
                    pitch_bottom_y=pitch_bottom_y,
                    homography=self.homography,
                    ball_annotator=ball_annotator,
                    team_colors=self.team_colors,
                )
                # Scale down for output
                if scale != 1.0:
                    annotated = cv2.resize(annotated, (out_width, out_height))
                writer.write_frame(annotated)

    def _compute_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = bbox1.area + bbox2.area - intersection
        return intersection / union if union > 0 else 0.0

    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        if mask1 is None or mask2 is None:
            return 0.0
        
        # Ensure same shape
        if mask1.shape != mask2.shape:
            return 0.0
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return float(intersection / union) if union > 0 else 0.0

    def _interpolate_track_gaps(
        self,
        tracks: list[PlayerTrack],
        max_gap: int = 30,
    ):
        """
        Fill gaps in player tracks by interpolating missing frames.
        Creates synthetic detections for frames where track is lost but prediction is available.

        Args:
            tracks: List of player tracks to interpolate
            max_gap: Maximum gap size to interpolate (frames)
        """
        for track in tracks:
            if len(track.detections) < 2:
                continue

            # Sort detections by frame index
            detections = sorted(track.detections, key=lambda d: d.frame_idx)
            track.detections = detections

            # Find gaps and interpolate
            new_detections = []
            for i, det in enumerate(detections):
                new_detections.append(det)

                # Check gap to next detection
                if i < len(detections) - 1:
                    next_det = detections[i + 1]
                    gap = next_det.frame_idx - det.frame_idx

                    # Interpolate if gap is small enough
                    if 1 < gap <= max_gap:
                        # Linear interpolation between two real detections
                        for frame_offset in range(1, gap):
                            frame_idx = det.frame_idx + frame_offset
                            t = frame_offset / gap  # 0 to 1

                            # Interpolate bbox
                            x1 = det.bbox.x1 + t * (next_det.bbox.x1 - det.bbox.x1)
                            y1 = det.bbox.y1 + t * (next_det.bbox.y1 - det.bbox.y1)
                            x2 = det.bbox.x2 + t * (next_det.bbox.x2 - det.bbox.x2)
                            y2 = det.bbox.y2 + t * (next_det.bbox.y2 - det.bbox.y2)
                            conf = det.bbox.confidence + t * (next_det.bbox.confidence - det.bbox.confidence)

                            bbox = BoundingBox(
                                x1=x1, y1=y1, x2=x2, y2=y2,
                                confidence=conf,
                            )

                            interp_det = PlayerDetection(
                                frame_idx=frame_idx,
                                bbox=bbox,
                                class_id=det.class_id,
                                class_name=det.class_name,
                                team=track.team,
                                jersey_number=track.jersey_number,
                                is_interpolated=True,
                            )
                            new_detections.append(interp_det)

            # Update track with interpolated detections
            track.detections = sorted(new_detections, key=lambda d: d.frame_idx)

    def _save_results(self):
        """Save tracking results to files."""
        import json

        # Save as JSON
        output_path = self.output_dir / "tracking_results.json"

        # Convert to serializable format
        results = {
            "video_path": self.match_data.video_path,
            "fps": self.match_data.fps,
            "total_frames": self.match_data.total_frames,
            "width": self.match_data.width,
            "height": self.match_data.height,
            "num_tracks": len(self.match_data.player_tracks),
            "tracks": [],
            "ball_detections": [],
        }

        for track in self.match_data.player_tracks:
            track_data = {
                "track_id": track.track_id,
                "start_frame": track.start_frame,
                "end_frame": track.end_frame,
                "team": track.team.value if track.team else None,
                "jersey_number": track.jersey_number,
                "positions": [
                    {
                        "frame": d.frame_idx,
                        "x": d.bbox.center[0],
                        "y": d.bbox.center[1],
                    }
                    for d in track.detections
                ],
            }
            results["tracks"].append(track_data)

        # Add ball detections
        for detection in self.match_data.ball_positions:
            ball_data = {
                "frame": detection.frame_idx if hasattr(detection, "frame_idx") else None,
                "x": detection.bbox.center[0],
                "y": detection.bbox.center[1],
                "confidence": detection.bbox.confidence,
            }
            results["ball_detections"].append(ball_data)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
