"""
Main processing pipeline for futsal player tracking.

Orchestrates the detection, tracking, and output generation stages.
"""

from pathlib import Path
from typing import Optional
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
from src.detection.tracking import ByteTracker, convert_stracks_to_player_tracks
from src.detection.segmentation_sam2 import SamSegmenter2
from src.geometry.homography import CourtHomography, create_homography_from_config


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
        self._tracker: Optional[ByteTracker] = None
        self._sam_segmenter: Optional[SamSegmenter2] = None
        self._homography: Optional[CourtHomography] = None

        # Match data storage
        self.match_data: Optional[MatchData] = None

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
            appearance_weight=cfg.get("appearance_weight", 0.5),
            max_center_distance=cfg.get("max_center_distance", 100.0),
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

        # Expected player count is fixed per run
        expected_players = self.config["tracking"].get("min_players_enforce", 12)

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

        # Process frames
        if end_frame is None:
            end_frame = reader.total_frames

        total_frames = end_frame - start_frame
        print(f"\nProcessing frames {start_frame} to {end_frame}...")

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

                for (frame_idx, frame), detections in zip(batch, all_detections):
                    log_frame_state(frame_idx, "prev_state (memory)", previous_frame_state)

                    # Capture low-confidence detections for debug visualization
                    debug_low_thresh = self.config["player_detection"].get("debug_low_threshold", 0.1)
                    low_conf_detections[frame_idx] = [
                        d for d in detections
                        if d.bbox.confidence < self.config["player_detection"]["confidence_threshold"]
                        and d.bbox.confidence >= debug_low_thresh
                    ]
                    
                    # Filter detections
                    detections = filter_detections_by_size(detections)
                    
                    # Set current frame for SAM2 temporal tracking (if enabled)
                    if sam_available:
                        self.sam_segmenter.set_image(frame, frame_idx)

                    # Extract color histograms for appearance matching
                    histograms = []
                    for det in detections:
                        hist = extract_color_histogram(frame, det.bbox)
                        det.color_histogram = hist
                        histograms.append(hist)

                    # Update tracker with appearance cues
                    det_array = self.player_detector.get_detection_array(detections)
                    active_tracks = self.tracker.update(det_array, frame_idx, histograms=histograms if histograms else None)

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
                    # STEP 1: T1 - Process all YOLO-detected players first
                    # =============================================================
                    for track_id, bbox in yolo_track_results.items():
                        active_track_ids.add(track_id)

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

                        # Create T1 detection
                        det = PlayerDetection(
                            frame_idx=frame_idx,
                            bbox=bbox,
                            class_id=class_id,
                            class_name=class_name,
                            mask=mask,
                            is_sam_recovered=False,
                        )
                        track_histories[track_id].detections.append(det)

                        current_frame_state[track_id] = {
                            'tier': 'T1',
                            'bbox': bbox,
                            'frame_idx': frame_idx,
                            'detection': det
                        }

                    log_frame_state(frame_idx, "T1 state (YOLO)", current_frame_state)

                    # =============================================================
                    # STEP 2: T2 - SAM recovery for MISSING players only
                    # =============================================================
                    # Only use SAM for players that YOLO missed but were in previous frame
                    if sam_available and self.sam_segmenter is not None and len(current_frame_state) < expected_players:
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

                            # Skip stale tracks (not seen in 3+ frames)
                            if frames_since_seen > 3:
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

                            # Check if player was near edge - they might have walked off
                            if is_near_edge(last_bbox):
                                # Don't hallucinate players who walked off camera
                                continue

                            # Find the closest active player to where this player disappeared
                            closest_active = None
                            closest_distance = float('inf')

                            for active_track_id in active_track_ids:
                                if active_track_id not in current_frame_state:
                                    continue
                                active_state = current_frame_state[active_track_id]
                                active_bbox = active_state['bbox']

                                dist = bbox_distance(last_bbox, active_bbox)
                                if dist < closest_distance:
                                    closest_distance = dist
                                    closest_active = active_bbox

                            # If there's a player within reasonable distance, assume occlusion
                            # Typical player bbox is ~100-200 pixels wide, so 150px is "very close"
                            # For recently T2-recovered players, use a slightly relaxed threshold (200px)
                            occlusion_distance_threshold = 150  # pixels
                            if hasattr(self, '_t2_recovery_history') and track_id in self._t2_recovery_history:
                                if frame_idx - self._t2_recovery_history[track_id] <= 2:
                                    occlusion_distance_threshold = 200  # More lenient for recently recovered

                            if closest_active and closest_distance < occlusion_distance_threshold:
                                # Lost player is behind the closest active player
                                # Place them at the occluder's position but OFFSET UPWARD to show depth (behind)
                                # Offset upward (negative y) to indicate the player is behind/occluded
                                # Use larger offset so it's clearly "behind" in the camera perspective
                                y_offset = -50  # pixels offset UP to show player is further behind
                                occluded_bbox = BoundingBox(
                                    x1=closest_active.x1,
                                    y1=closest_active.y1 + y_offset,
                                    x2=closest_active.x2,
                                    y2=closest_active.y2 + y_offset,
                                    confidence=0.3,  # Low confidence = interpolated
                                )

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
                                    'detection': det
                                }
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
                    
                    # Update previous_frame_state for next iteration
                    previous_frame_state = current_frame_state.copy()
                    # Cache current frame for SAM baseline (in case YOLO drops on next frame)
                    previous_frame_image = frame.copy()

                pbar.update(len(batch))

        # Store results
        self.match_data.player_tracks = list(track_histories.values())
        self.match_data.low_confidence_detections = low_conf_detections

        print(f"\nResults:")
        print(f"  Player tracks: {len(self.match_data.player_tracks)}")

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
        from src.utils.visualization import draw_frame_annotations

        print("\nGenerating debug video...")
        reader = VideoReader(video_path)

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

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
