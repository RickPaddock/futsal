"""
Pass 2C: Ghost Generation

Per CLAUDE.md Section 5 (Pass 2C):
- Input: pass2_fragments.json (from Pass 2B with quality scores)
- Output: pass2_ghosts.json (fragments + ghosts)
- Maintains player count continuity (R4: Players Never Disappear)

Per CLAUDE.md Rule R4 (Player Continuity):
- Dynamic level (high water mark): increases as more players enter, never decreases
- Track players by original_track_id (NOT fragment_id)
- Create ghosts when tracked_count < level
- Ghost position: HOLD last known position (no interpolation)
- Ghost duration: until reappearance or MAX_GAP (60 frames)
- Mark ghosts: is_ghost=True, quality="ghost"
- Exclude ghosts from K-means clustering (Pass 3C responsibility)

Per CLAUDE.md Principle P1 (Pass Immutability):
- Reads Pass 2B output (read-only)
- Never modifies Pass 2B artifacts
- Emits new artifacts only
"""

from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
import logging
from collections import defaultdict
import bisect

from ..core.data_models import (
    Fragment,
    ScoredFragment,
    Pass2AOutput,
    Pass2BOutput,
)
from ..core.types import FragmentID, TrackID, FragmentQuality
from ..utils.file_utils import load_json, save_json
from ..validation.validator import Validator
from ..core import constants as const

logger = logging.getLogger(__name__)


class Pass2CGhostGenerator:
    """
    Pass 2C: Ghost fragment generation.

    Responsibilities:
    - Maintain player count continuity (R4)
    - Initialize level from first N frames
    - Implement dynamic level (high water mark)
    - Track players by original_track_id (NOT fragment_id)
    - Create ghosts when tracked_count < level
    - Ghost position: HOLD last known position
    - Mark ghosts with is_ghost=True, quality="ghost"

    Does NOT:
    - Assign teams or player identities (Pass 3 responsibility)
    - Enforce team balance (Pass 3C responsibility)
    - Interpolate positions (position hold only)
    """

    def __init__(self):
        self.ghost_counter = 0
        self.ghost_log: List[Dict] = []

    def run(
        self,
        pass2b_path: Path,
        output_path: Path,
        pass1_path: Optional[Path] = None,
    ) -> Pass2BOutput:
        """
        Execute Pass 2C ghost generation.

        Args:
            pass2b_path: Path to pass2_fragments.json (with quality scores from Pass 2B)
            output_path: Path to write pass2_ghosts.json
            pass1_path: Optional path to pass1_raw.json (for frame range metadata)

        Returns:
            Pass2BOutput with fragments + ghosts

        Raises:
            ValidationError if output fails validation
        """
        logger.info(f"Pass 2C: Loading Pass 2B data from {pass2b_path}")
        pass2b_output = load_json(pass2b_path, Pass2BOutput)

        logger.info(f"Pass 2C: Generating ghosts for {len(pass2b_output.fragments)} fragments")

        # Load Pass 1 for bbox/centroid data (required for ghost positioning)
        pass1_output = None
        if pass1_path and pass1_path.exists():
            from ..core.data_models import Pass1Output
            pass1_output = load_json(pass1_path, Pass1Output)
            start_frame = pass1_output.processed_start_frame
            end_frame = (
                pass1_output.processed_end_frame_exclusive - 1
                if pass1_output.processed_end_frame_exclusive
                else pass1_output.total_frames - 1
            )
        else:
            # Infer from fragments
            start_frame = min(f.start_frame for f in pass2b_output.fragments)
            end_frame = max(f.end_frame for f in pass2b_output.fragments)
            logger.warning("Pass 2C: Pass 1 data not available - ghost positions will be None")

        logger.info(f"Pass 2C: Frame range [{start_frame}, {end_frame}]")

        # Generate ghosts
        all_fragments = self._generate_ghosts(
            pass2b_output.fragments,
            start_frame,
            end_frame,
            pass1_output,
        )

        logger.info(
            f"Pass 2C: Generated {sum(1 for f in all_fragments if f.is_ghost)} ghosts, "
            f"total {len(all_fragments)} fragments"
        )

        # Create output (reuse Pass2BOutput structure)
        output = Pass2BOutput(
            fragments=all_fragments,
            split_log=pass2b_output.split_log,  # Preserve from Pass 2A
        )

        # Validate BEFORE writing (CRITICAL - fail-fast)
        logger.info("Pass 2C: Validating output")
        validator = Validator()
        validation_result = validator.validate_pass2(output, pass1_output)

        if not validation_result.passed:
            # Write validation JSON ONLY (no ghosts JSON)
            validation_path = output_path.parent / const.PASS2_VALIDATION_JSON
            save_json(validation_result.dict(), str(validation_path))

            # Raise error (fail-fast)
            error_msg = f"Pass 2C validation failed: {len(validation_result.violations)} violations"
            logger.error(error_msg)
            for v in validation_result.violations[:5]:  # Show first 5
                logger.error(f"  - {v.rule}: {v.message}")
            raise ValueError(error_msg)

        # Validation passed - write output JSON
        logger.info(f"Pass 2C: Writing output to {output_path}")
        save_json(output.dict(), str(output_path))

        # Write validation JSON (passed)
        validation_path = output_path.parent / const.PASS2_VALIDATION_JSON
        save_json(validation_result.dict(), str(validation_path))

        logger.info(
            f"Pass 2C: Complete - {len(all_fragments)} total fragments "
            f"({sum(1 for f in all_fragments if f.is_ghost)} ghosts)"
        )

        return output

    def _generate_ghosts(
        self,
        fragments: List[Fragment],
        start_frame: int,
        end_frame: int,
        pass1_output: Optional,
    ) -> List[Fragment]:
        """
        Generate ghost fragments to maintain player count continuity.

        Per CLAUDE.md R4 and MEMORY.md learnings:
        - Initialize level from first INITIAL_LEVEL_FRAMES frames
        - Dynamic level (high water mark): increases as more players enter, never decreases
        - Track players by original_track_id (NOT fragment_id)
        - Do NOT resolve identity across different track_id values (Pass 3 responsibility)
        - Create ghosts for every disappearance (same-track continuity)
        - Ghost position: HOLD last known position
        - Ghost duration: until reappearance or MAX_GAP frames

        Args:
            fragments: Real fragments from Pass 2B
            start_frame: First frame to process
            end_frame: Last frame to process
            pass1_output: Pass 1 output (for bbox/centroid data), or None

        Returns:
            All fragments (real + ghosts)
        """
        # Filter out existing ghosts (shouldn't be any, but defensive)
        real_fragments = [f for f in fragments if not f.is_ghost]

        # Build detection lookup (detection_id -> Detection)
        detection_by_id = {}
        track_detection_frames: Dict[TrackID, List[int]] = defaultdict(list)
        if pass1_output:
            detection_by_id = {det.detection_id: det for det in pass1_output.detections}
            for det in pass1_output.detections:
                track_detection_frames[det.track_id].append(det.frame_idx)
            for track_id in track_detection_frames:
                track_detection_frames[track_id].sort()

        # Build frame-by-frame inventory: which fragments exist at each frame
        # CRITICAL: Only count frames with actual detections (handles internal gaps)
        frame_inventory = self._build_frame_inventory(real_fragments, start_frame, end_frame, detection_by_id)

        # Initialize level from first N frames
        level = self._initialize_level(frame_inventory, start_frame)
        logger.info(f"Pass 2C: Initial level = {level}")

        # Track player states across frames
        all_fragments = list(real_fragments)
        player_states: Dict[TrackID, Dict] = {}  # track_id -> {last_frame, last_bbox, last_centroid}

        for frame_idx in range(start_frame, end_frame + 1):
            # Get fragments active at this frame
            active_fragments = frame_inventory.get(frame_idx, [])
            active_track_ids = {f.original_track_id for f in active_fragments}

            # CRITICAL: Zero-gap ghost chaining (CLAUDE.md R4).
            # If a ghost expired last frame and track has not reappeared, create
            # a successor ghost immediately at current frame.
            if frame_idx > start_frame:
                expiring_ghosts = [
                    f for f in all_fragments
                    if f.is_ghost and f.end_frame == frame_idx - 1
                ]

                for expired in expiring_ghosts:
                    track_id = expired.original_track_id

                    # If same-track real detection exists now, chain terminates.
                    if track_id in active_track_ids:
                        continue

                    # Already chained by another ghost at this frame.
                    if self._find_active_ghost(all_fragments, track_id, frame_idx):
                        continue

                    # Need player state to position held ghost.
                    if track_id not in player_states:
                        continue

                    chained_ghost = self._create_ghost_fragment(
                        track_id,
                        player_states[track_id],
                        frame_idx,
                        end_frame,
                        real_fragments,
                        track_detection_frames,
                    )
                    if chained_ghost:
                        all_fragments.append(chained_ghost)

            # Include currently active ghosts in presence count.
            # R4 presence is tracked + ghosts, not tracked-only.
            active_ghost_track_ids = {
                f.original_track_id
                for f in all_fragments
                if f.is_ghost and f.start_frame <= frame_idx <= f.end_frame
            }

            active_presence_track_ids = active_track_ids | active_ghost_track_ids

            # Update level (high water mark) from REAL detections only.
            # Presence (real + ghosts) is used for deficit checking, not level estimation.
            real_count = len(active_track_ids)
            if real_count > level:
                logger.info(f"Pass 2C: Level increased from {level} to {real_count} at frame {frame_idx}")
                level = min(real_count, const.DYNAMIC_LEVEL_MAX)

            # Update player states
            for frag in active_fragments:
                # Get bbox and centroid from Pass 1 detections for this frame
                last_bbox, last_centroid = self._get_last_detection_for_frame(
                    frag,
                    frame_idx,
                    detection_by_id,
                )

                player_states[frag.original_track_id] = {
                    "last_frame": frame_idx,
                    "last_bbox": last_bbox,
                    "last_centroid": last_centroid,
                    "fragment_id": frag.fragment_id,
                }

            # CRITICAL: Update last presence timestamp for active ghosts too.
            # Ghost chaining must be based on presence continuity, not real-only detections.
            # Without this, a track falls out of missing-player eligibility immediately
            # after one MAX_FRAGMENT_GAP segment and chaining breaks.
            for track_id in active_ghost_track_ids:
                if track_id in active_track_ids:
                    continue

                active_ghost = self._find_active_ghost(all_fragments, track_id, frame_idx)
                if not active_ghost:
                    continue

                prior_state = player_states.get(track_id, {})
                player_states[track_id] = {
                    "last_frame": frame_idx,
                    "last_bbox": prior_state.get("last_bbox") or active_ghost.ghost_last_known_bbox,
                    "last_centroid": prior_state.get("last_centroid") or active_ghost.ghost_last_known_centroid,
                    "fragment_id": prior_state.get("fragment_id") or active_ghost.parent_fragment_id,
                }

            # Create ghosts for every missing track (identity-local continuity).
            # CRITICAL: this is independent of global count/physical cap.
            missing_players = self._find_missing_players(
                player_states,
                active_presence_track_ids,
                frame_idx,
            )

            for track_id in missing_players:
                # Check if we already have a ghost for this player
                existing_ghost = self._find_active_ghost(
                    all_fragments,
                    track_id,
                    frame_idx,
                )

                if existing_ghost:
                    # Ghost already exists - extend it
                    if frame_idx > existing_ghost.end_frame:
                        existing_ghost.end_frame = frame_idx
                else:
                    # Create new ghost
                    ghost = self._create_ghost_fragment(
                        track_id,
                        player_states[track_id],
                        frame_idx,
                        end_frame,
                        real_fragments,
                        track_detection_frames,
                    )
                    if ghost:
                        all_fragments.append(ghost)

        return all_fragments

    def _build_frame_inventory(
        self,
        fragments: List[Fragment],
        start_frame: int,
        end_frame: int,
        detection_by_id: Optional[Dict] = None,
    ) -> Dict[int, List[Fragment]]:
        """
        Build frame-by-frame inventory of which fragments are active.

        CRITICAL FIX: Only count frames where fragment has ACTUAL detections,
        not the full start_frame-end_frame range (which may include internal gaps).

        Args:
            fragments: All fragments
            start_frame: First frame
            end_frame: Last frame
            detection_by_id: Pass 1 detections (detection_id -> Detection)

        Returns:
            Dict mapping frame_idx -> list of active fragments
        """
        inventory: Dict[int, List[Fragment]] = defaultdict(list)

        for frag in fragments:
            if detection_by_id and frag.detection_ids:
                # Only count frames with actual detections (handles internal gaps)
                for det_id in frag.detection_ids:
                    det = detection_by_id.get(det_id)
                    if det and start_frame <= det.frame_idx <= end_frame:
                        inventory[det.frame_idx].append(frag)
            else:
                # Fallback: use full range (for ghosts or when detection lookup unavailable)
                for frame_idx in range(
                    max(frag.start_frame, start_frame),
                    min(frag.end_frame, end_frame) + 1,
                ):
                    inventory[frame_idx].append(frag)

        return inventory

    def _initialize_level(
        self,
        frame_inventory: Dict[int, List[Fragment]],
        start_frame: int,
    ) -> int:
        """
        Initialize player count level from first N frames.

        Per CLAUDE.md R4:
        - Use first INITIAL_LEVEL_FRAMES frames to estimate level
        - Conservative estimate (may increase later)

        Args:
            frame_inventory: Frame-by-frame fragment inventory
            start_frame: First frame

        Returns:
            Initial level (number of players expected)
        """
        # Count unique track_ids in first N frames
        initial_frames = range(
            start_frame,
            start_frame + const.INITIAL_LEVEL_FRAMES,
        )

        all_track_ids: Set[TrackID] = set()
        for frame_idx in initial_frames:
            fragments = frame_inventory.get(frame_idx, [])
            all_track_ids.update(f.original_track_id for f in fragments)

        initial_level = len(all_track_ids)

        # Cap at DYNAMIC_LEVEL_MAX (futsal regulation)
        return min(initial_level, const.DYNAMIC_LEVEL_MAX)

    def _find_missing_players(
        self,
        player_states: Dict[TrackID, Dict],
        active_track_ids: Set[TrackID],
        current_frame: int,
    ) -> List[TrackID]:
        """
        Find players who were active before but are missing now.

        Per MEMORY.md T3 Principle:
        - Track by original_track_id, not fragment_id
        - Players who were seen recently but not now are missing

        Args:
            player_states: Player state tracking
            active_track_ids: Track IDs active at current frame
            current_frame: Current frame index

        Returns:
            List of missing track_ids (sorted by last_frame, most recent first)
        """
        missing = []

        for track_id, state in player_states.items():
            # Skip if currently active
            if track_id in active_track_ids:
                continue

            # Check if previously active (no MAX_GAP suppression).
            gap = current_frame - state["last_frame"]
            if gap > 0:
                missing.append((track_id, state["last_frame"]))

        # Sort by last_frame (most recent first)
        missing.sort(key=lambda x: x[1], reverse=True)

        return [track_id for track_id, _ in missing]

    def _find_active_ghost(
        self,
        fragments: List[Fragment],
        track_id: TrackID,
        frame_idx: int,
    ) -> Optional[Fragment]:
        """
        Find an existing ghost fragment for this track at this frame.

        Args:
            fragments: All fragments (real + ghosts)
            track_id: Track ID to search for
            frame_idx: Frame index

        Returns:
            Ghost fragment if found, None otherwise
        """
        for frag in fragments:
            if not frag.is_ghost:
                continue
            if frag.original_track_id != track_id:
                continue
            if frag.start_frame <= frame_idx <= frag.end_frame:
                return frag

        return None

    def _create_ghost_fragment(
        self,
        track_id: TrackID,
        player_state: Dict,
        current_frame: int,
        end_frame: int,
        real_fragments: List[Fragment],
        track_detection_frames: Optional[Dict[TrackID, List[int]]] = None,
    ) -> Optional[Fragment]:
        """
        Create a ghost fragment for a missing player.

        Per CLAUDE.md R4:
        - Ghost position: HOLD last known position (no interpolation)
        - Ghost duration: until reappearance or MAX_GAP frames
        - Mark: is_ghost=True, quality="ghost"

        Args:
            track_id: Track ID of missing player
            player_state: Last known state (last_frame, last_bbox, last_centroid)
            current_frame: Frame where ghost starts
            end_frame: Last frame of clip
            real_fragments: All real fragments (to detect reappearance)

        Returns:
            Ghost fragment or None if can't create
        """
        # Find when player reappears (if at all)
        reappearance_frame = self._find_reappearance_frame(
            track_id,
            current_frame,
            end_frame,
            real_fragments,
            track_detection_frames,
        )

        # Determine ghost duration
        if reappearance_frame is not None:
            ghost_end = reappearance_frame - 1  # Ghost ends just before reappearance
        else:
            # No reappearance - ghost continues for MAX_GAP or until end of clip
            gap_limit = current_frame + const.MAX_FRAGMENT_GAP - 1
            ghost_end = min(gap_limit, end_frame)

        # Don't create ghost if duration is too short
        if ghost_end < current_frame:
            return None

        # Create ghost fragment
        ghost_id = f"G{self.ghost_counter:06d}"
        self.ghost_counter += 1

        ghost = ScoredFragment(
            fragment_id=ghost_id,
            original_track_id=track_id,
            start_frame=current_frame,
            end_frame=ghost_end,
            detection_ids=[],  # Ghosts have no real detections
            split_reason=None,
            split_trigger_frame=None,
            split_rule_id=None,
            parent_fragment_id=player_state.get("fragment_id"),
            is_ghost=True,
            quality=FragmentQuality.GHOST,
            quality_score=0.0,  # Ghosts have no quality score
            quality_reasons=["ghost_fragment"],
            avg_confidence=0.0,
            min_confidence=0.0,
            avg_bbox_stability=0.0,
            jersey_consistency=0.0,
            hsv_consistency=0.0,
            ghost_last_known_bbox=player_state.get("last_bbox"),
            ghost_last_known_centroid=player_state.get("last_centroid"),
            ghost_reason="player_occluded" if reappearance_frame else "player_off_screen",
        )

        self.ghost_log.append({
            "ghost_id": ghost_id,
            "track_id": track_id,
            "start_frame": current_frame,
            "end_frame": ghost_end,
            "reason": "player_occluded" if reappearance_frame else "player_off_screen",
            "duration": ghost_end - current_frame + 1,
        })

        logger.debug(
            f"Pass 2C: Created ghost {ghost_id} for track {track_id}, "
            f"frames [{current_frame}, {ghost_end}]"
        )

        return ghost

    def _find_reappearance_frame(
        self,
        track_id: TrackID,
        start_frame: int,
        end_frame: int,
        real_fragments: List[Fragment],
        track_detection_frames: Optional[Dict[TrackID, List[int]]] = None,
    ) -> Optional[int]:
        """
        Find when a player reappears (if at all).

        Args:
            track_id: Track ID to search for
            start_frame: Start searching from this frame
            end_frame: Stop searching at this frame
            real_fragments: All real fragments

        Returns:
            Frame index where player reappears, or None
        """
        # Primary: Pass 1 detection frames (source-of-truth for reappearance)
        if track_detection_frames is not None:
            frames = track_detection_frames.get(track_id, [])
            idx = bisect.bisect_right(frames, start_frame)
            if idx < len(frames):
                candidate = frames[idx]
                if candidate <= end_frame:
                    return candidate

        # Fallback: fragment-boundary heuristic
        for frag in real_fragments:
            if frag.original_track_id != track_id:
                continue
            if frag.start_frame > start_frame and frag.start_frame <= end_frame:
                return frag.start_frame

        return None

    def _get_last_detection_for_frame(
        self,
        fragment: Fragment,
        frame_idx: int,
        detection_by_id: Dict[str, "Detection"],
    ) -> Tuple[Optional[list], Optional[Tuple[float, float]]]:
        """
        Get bbox and centroid from Pass 1 detection for a specific frame.

        Args:
            fragment: Fragment containing detection_ids
            frame_idx: Frame to get detection for
            detection_by_id: Lookup dict (detection_id -> Detection)

        Returns:
            Tuple of (bbox, centroid) or (None, None) if not found
        """
        # Find detection for this frame in fragment's detection_ids
        for det_id in fragment.detection_ids:
            det = detection_by_id.get(det_id)
            if det and det.frame_idx == frame_idx:
                return (det.bbox, det.centroid)

        # Fallback: return None if no detection found for this exact frame
        return (None, None)


def run_pass2c(
    input_dir: Path,
    output_dir: Optional[Path] = None,
) -> Pass2BOutput:
    """
    Execute Pass 2C: Ghost Generation.

    Args:
        input_dir: Directory containing pass2_fragments.json
        output_dir: Directory to write output (default: same as input_dir)

    Returns:
        Pass2BOutput (with ghosts)

    Raises:
        FileNotFoundError if pass2_fragments.json not found
        ValidationError if output fails validation
    """
    if output_dir is None:
        output_dir = input_dir

    pass2b_path = input_dir / const.PASS2_FRAGMENTS_JSON  # Pass 2B writes to same file
    pass1_path = input_dir / const.PASS1_RAW_JSON
    output_path = output_dir / const.PASS2_GHOSTS_JSON

    if not pass2b_path.exists():
        raise FileNotFoundError(f"Pass 2B output not found: {pass2b_path}")

    # Pass 1 is optional (for frame range metadata)
    if not pass1_path.exists():
        pass1_path = None

    generator = Pass2CGhostGenerator()
    return generator.run(pass2b_path, output_path, pass1_path)
