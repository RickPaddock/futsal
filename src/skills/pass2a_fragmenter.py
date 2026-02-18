"""
Pass 2A: Mechanical Fragmentation

Per CLAUDE.md Section 5 (Pass 2A):
- Input: pass1_raw.json
- Output: pass2_fragments.json, pass2_validation.json
- Splits tracks into fragments based on mechanical divergence points
- Keeps ALL fragments (even <10 frames), marks short ones as low_quality
- Merges consecutive short fragments on same track

Per CLAUDE.md Principle P1 (Pass Immutability):
- Reads Pass 1 output (read-only)
- Never modifies Pass 1 artifacts
- Emits new artifacts only

Per CLAUDE.md Rule R1 (No Identity Logic in Fragmentation):
- NO team assignment
- NO player_id assignment
- Only mechanical split triggers (geometry, appearance)
"""

from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import logging

from ..core.data_models import (
    Detection,
    Pass1Output,
    Fragment,
    Pass2AOutput,
)
from ..core.types import FragmentID, TrackID, DetectionID, HSVHistogram
from ..utils.file_utils import load_json, save_json
from ..utils.geometry import centroid_distance
from ..utils.hsv_color import compare_hsv_histograms
from ..validation.validator import Validator
from ..core import constants as const

logger = logging.getLogger(__name__)


class Pass2AFragmenter:
    """
    Pass 2A: Mechanical track fragmentation.

    Responsibilities:
    - Split tracks at divergence points (appearance, geometry, jersey conflicts)
    - Keep ALL fragments (mark short ones as low_quality)
    - Merge consecutive short fragments on same track
    - Validate output (100% coverage, no overlaps)

    Does NOT:
    - Assign teams or player identities
    - Make clustering decisions
    - Modify Pass 1 output
    """

    def __init__(self):
        self.fragment_counter = 0
        self.split_log: List[Dict] = []

    def run(self, pass1_path: Path, output_path: Path) -> Pass2AOutput:
        """
        Execute Pass 2A fragmentation.

        Args:
            pass1_path: Path to pass1_raw.json
            output_path: Path to write pass2_fragments.json

        Returns:
            Pass2AOutput with fragments and split log

        Raises:
            ValidationError if output fails validation
        """
        logger.info(f"Pass 2A: Loading Pass 1 data from {pass1_path}")
        pass1_output = load_json(pass1_path, Pass1Output)

        logger.info(f"Pass 2A: Fragmenting {len(pass1_output.detections)} detections")

        # Group detections by track_id
        tracks = self._group_detections_by_track(pass1_output.detections)

        logger.info(f"Pass 2A: Processing {len(tracks)} tracks")

        # Split each track into fragments
        all_fragments = []
        for track_id, detections in tracks.items():
            fragments = self._split_track(track_id, detections)
            all_fragments.extend(fragments)

        logger.info(f"Pass 2A: Created {len(all_fragments)} fragments (before jersey temporal exclusivity)")

        # Check for jersey temporal exclusivity violations (cross-track conflicts)
        all_fragments = self._split_jersey_temporal_conflicts(all_fragments, pass1_output.detections)
        logger.info(f"Pass 2A: After jersey temporal exclusivity: {len(all_fragments)} fragments")

        # Merge consecutive short fragments
        if const.MERGE_CONSECUTIVE_SHORT:
            all_fragments = self._merge_consecutive_short_fragments(all_fragments)
            logger.info(f"Pass 2A: After merging: {len(all_fragments)} fragments")

        # Create output
        output = Pass2AOutput(
            fragments=all_fragments,
            split_log=self.split_log,
        )

        # Validate BEFORE writing (CRITICAL - fail-fast)
        logger.info("Pass 2A: Validating output")
        validator = Validator()
        validation_result = validator.validate_pass2a(output, pass1_output)

        if not validation_result.passed:
            # Write validation JSON ONLY (no fragments JSON)
            validation_path = output_path.parent / const.PASS2_VALIDATION_JSON
            save_json(validation_result.dict(), str(validation_path))

            # Raise error (fail-fast)
            error_msg = f"Pass 2A validation failed: {len(validation_result.violations)} violations"
            logger.error(error_msg)
            for v in validation_result.violations[:5]:  # Show first 5
                logger.error(f"  - {v.rule}: {v.message}")
            raise ValueError(error_msg)

        # Validation passed - write output JSON
        logger.info(f"Pass 2A: Writing output to {output_path}")
        save_json(output.dict(), str(output_path))

        # Write validation JSON (passed)
        validation_path = output_path.parent / const.PASS2_VALIDATION_JSON
        save_json(validation_result.dict(), str(validation_path))

        logger.info(f"Pass 2A: Complete - {len(all_fragments)} fragments, {len(self.split_log)} splits")

        return output

    def _group_detections_by_track(
        self, detections: List[Detection]
    ) -> Dict[TrackID, List[Detection]]:
        """
        Group detections by track_id.

        Args:
            detections: All Pass 1 detections

        Returns:
            Dict mapping track_id -> detections (sorted by frame_idx)
        """
        tracks: Dict[TrackID, List[Detection]] = {}

        for det in detections:
            if det.track_id not in tracks:
                tracks[det.track_id] = []
            tracks[det.track_id].append(det)

        # Sort detections within each track by frame_idx
        for track_id in tracks:
            tracks[track_id] = sorted(tracks[track_id], key=lambda d: d.frame_idx)

        return tracks

    def _split_track(
        self, track_id: TrackID, detections: List[Detection]
    ) -> List[Fragment]:
        """
        Split a single track into fragments based on divergence points.

        Split triggers (per CLAUDE.md):
        1. Track overlap collision (same track_id, >1 detection per frame)
        2. Appearance drift (HSV histogram change)
        3. Jersey inconsistency (jersey disappears or changes, NOT first appearance)
        4. Jersey temporal exclusivity (handled separately)
        5. Velocity spikes (sudden position jumps)

        Args:
            track_id: Track to split
            detections: Detections for this track (sorted by frame_idx)

        Returns:
            List of fragments (may be 1 if no splits needed)
        """
        if not detections:
            return []

        # Detect split points
        split_points = self._detect_split_points(track_id, detections)

        # Split at detected points
        fragments = self._create_fragments_from_splits(track_id, detections, split_points)

        return fragments

    def _detect_split_points(
        self, track_id: TrackID, detections: List[Detection]
    ) -> List[Tuple[int, str]]:
        """
        Detect split points within a track.

        Returns:
            List of (frame_idx, reason) tuples where splits should occur
        """
        split_points = []

        # Check for track overlap collision (same track_id, multiple detections per frame)
        frame_counts: Dict[int, int] = {}
        for det in detections:
            frame_counts[det.frame_idx] = frame_counts.get(det.frame_idx, 0) + 1

        for frame_idx, count in frame_counts.items():
            if count > 1:
                split_points.append((frame_idx, "track_overlap_collision"))
                self.split_log.append({
                    "track_id": track_id,
                    "frame_idx": frame_idx,
                    "reason": "track_overlap_collision",
                    "details": f"Track {track_id} has {count} detections at frame {frame_idx}",
                })

        # Check for appearance drift, jersey inconsistency, velocity spikes
        for i in range(1, len(detections)):
            prev_det = detections[i - 1]
            curr_det = detections[i]

            # Skip if frames are consecutive (splits only at discontinuities)
            frame_gap = curr_det.frame_idx - prev_det.frame_idx
            if frame_gap == 1:
                # Check appearance drift
                if prev_det.hsv_histogram_jersey and curr_det.hsv_histogram_jersey:
                    correlation = compare_hsv_histograms(
                        prev_det.hsv_histogram_jersey,
                        curr_det.hsv_histogram_jersey,
                    )
                    distance = 1.0 - correlation

                    if distance > const.HSV_DRIFT_THRESHOLD:
                        split_points.append((curr_det.frame_idx, "appearance_drift"))
                        self.split_log.append({
                            "track_id": track_id,
                            "frame_idx": curr_det.frame_idx,
                            "reason": "appearance_drift",
                            "details": f"HSV drift distance={distance:.3f} > {const.HSV_DRIFT_THRESHOLD}",
                        })

                # Check jersey inconsistency (per CLAUDE.md and MEMORY.md)
                prev_jersey = prev_det.jersey_number
                curr_jersey = curr_det.jersey_number

                # ✅ SPLIT: Jersey disappears (#4 → None) - track lost player
                if prev_jersey is not None and curr_jersey is None:
                    split_points.append((curr_det.frame_idx, "jersey_disappeared"))
                    self.split_log.append({
                        "track_id": track_id,
                        "frame_idx": curr_det.frame_idx,
                        "reason": "jersey_disappeared",
                        "details": f"Jersey #{prev_jersey} disappeared",
                    })

                # ✅ SPLIT: Jersey changes (#7 → #4) - track jumped to different player
                elif prev_jersey is not None and curr_jersey is not None and prev_jersey != curr_jersey:
                    split_points.append((curr_det.frame_idx, "jersey_changed"))
                    self.split_log.append({
                        "track_id": track_id,
                        "frame_idx": curr_det.frame_idx,
                        "reason": "jersey_changed",
                        "details": f"Jersey changed from #{prev_jersey} to #{curr_jersey}",
                    })

                # ❌ NO SPLIT: Jersey first appearance (None → #4) - player turned around
                # This is handled by NOT adding a split point here

                # Check velocity spike
                dist = centroid_distance(prev_det.centroid, curr_det.centroid)
                if dist > const.VELOCITY_SPIKE_THRESHOLD:
                    split_points.append((curr_det.frame_idx, "velocity_spike"))
                    self.split_log.append({
                        "track_id": track_id,
                        "frame_idx": curr_det.frame_idx,
                        "reason": "velocity_spike",
                        "details": f"Centroid distance={dist:.1f}px > {const.VELOCITY_SPIKE_THRESHOLD}px",
                    })

        # Sort and deduplicate split points
        split_points = sorted(set(split_points), key=lambda x: x[0])

        return split_points

    def _create_fragments_from_splits(
        self,
        track_id: TrackID,
        detections: List[Detection],
        split_points: List[Tuple[int, str]],
    ) -> List[Fragment]:
        """
        Create fragments by splitting at detected points.

        Args:
            track_id: Original track ID
            detections: All detections for this track
            split_points: List of (frame_idx, reason) split triggers

        Returns:
            List of Fragment objects
        """
        if not detections:
            return []

        fragments = []
        current_segment: List[Detection] = []
        split_reason = "initial"

        for det in detections:
            # Check if we hit a split point
            hit_split = False
            for split_frame, reason in split_points:
                if det.frame_idx == split_frame:
                    # Save current segment as fragment (if not empty)
                    if current_segment:
                        fragments.append(
                            self._create_fragment(track_id, current_segment, split_reason)
                        )

                    # Start new segment
                    current_segment = [det]
                    split_reason = reason
                    hit_split = True
                    break

            if not hit_split:
                current_segment.append(det)

        # Save final segment
        if current_segment:
            fragments.append(
                self._create_fragment(track_id, current_segment, split_reason)
            )

        return fragments

    def _create_fragment(
        self,
        track_id: TrackID,
        detections: List[Detection],
        split_reason: str,
    ) -> Fragment:
        """
        Create a Fragment from a list of detections.

        Args:
            track_id: Original track ID
            detections: Detections for this fragment
            split_reason: Why this fragment was created

        Returns:
            Fragment object
        """
        fragment_id = f"F{self.fragment_counter:06d}"
        self.fragment_counter += 1

        start_frame = min(d.frame_idx for d in detections)
        end_frame = max(d.frame_idx for d in detections)

        # Use actual detection_ids from Pass 1 (format: {frame_idx}_{track_id}_{bbox_hash})
        detection_ids = [d.detection_id for d in detections]

        return Fragment(
            fragment_id=fragment_id,
            original_track_id=track_id,
            start_frame=start_frame,
            end_frame=end_frame,
            detection_ids=detection_ids,
            split_reason=split_reason if split_reason != "initial" else None,
            parent_fragment_id=None,  # Will be set during jersey temporal exclusivity
        )

    def _split_jersey_temporal_conflicts(
        self,
        fragments: List[Fragment],
        all_detections: List[Detection],
    ) -> List[Fragment]:
        """
        Detect and split jersey temporal exclusivity violations.

        Per CLAUDE.md R3 and MEMORY.md:
        - Same jersey CANNOT appear on different tracks simultaneously
        - If jersey #10 appears on Track 5 and Track 8 at overlapping times, split the one where it appears later

        Args:
            fragments: All fragments created so far
            all_detections: All Pass 1 detections (for jersey lookups)

        Returns:
            Fragments with temporal conflicts resolved via splits
        """
        # Build detection lookup by detection_id for fast access
        detection_by_id = {det.detection_id: det for det in all_detections}

        # Build jersey timeline: jersey_number -> [(fragment, start_frame, end_frame, first_jersey_frame)]
        jersey_timeline: Dict[int, List[Tuple[Fragment, int, int, int]]] = {}

        for frag in fragments:
            # Find first frame where jersey appears in this fragment
            first_jersey_frame = None
            jersey_num = None

            for det_id in frag.detection_ids:
                det = detection_by_id.get(det_id)
                if det and det.jersey_number is not None:
                    jersey_num = det.jersey_number
                    first_jersey_frame = det.frame_idx
                    break  # Found first jersey appearance

            if jersey_num is not None and first_jersey_frame is not None:
                if jersey_num not in jersey_timeline:
                    jersey_timeline[jersey_num] = []

                jersey_timeline[jersey_num].append((
                    frag,
                    frag.start_frame,
                    frag.end_frame,
                    first_jersey_frame,
                ))

        # Detect temporal overlaps for each jersey
        new_fragments = []
        fragments_to_split: Set[str] = set()  # fragment_ids to split

        for jersey_num, appearances in jersey_timeline.items():
            if len(appearances) < 2:
                # No conflict - only one fragment has this jersey
                continue

            # Sort by first_jersey_frame (when jersey first appears)
            appearances = sorted(appearances, key=lambda x: x[3])

            # Check for overlaps
            for i in range(len(appearances)):
                for j in range(i + 1, len(appearances)):
                    frag_a, start_a, end_a, first_a = appearances[i]
                    frag_b, start_b, end_b, first_b = appearances[j]

                    # Check if time ranges overlap
                    if not (end_a < start_b or end_b < start_a):
                        # Conflict! Jersey appears on both fragments at overlapping times
                        # Split the one where jersey appears LATER
                        if first_b > first_a:
                            # Fragment B "stole" the jersey - split it at first_b
                            fragments_to_split.add(frag_b.fragment_id)
                            self.split_log.append({
                                "track_id": frag_b.original_track_id,
                                "frame_idx": first_b,
                                "reason": "jersey_temporal_conflict",
                                "details": f"Jersey #{jersey_num} already in use by {frag_a.fragment_id} (track {frag_a.original_track_id})",
                                "conflict_with": frag_a.fragment_id,
                            })
                        else:
                            # Fragment A "stole" the jersey - split it at first_a
                            fragments_to_split.add(frag_a.fragment_id)
                            self.split_log.append({
                                "track_id": frag_a.original_track_id,
                                "frame_idx": first_a,
                                "reason": "jersey_temporal_conflict",
                                "details": f"Jersey #{jersey_num} already in use by {frag_b.fragment_id} (track {frag_b.original_track_id})",
                                "conflict_with": frag_b.fragment_id,
                            })

        # Process fragments: split conflicted ones, keep others
        for frag in fragments:
            if frag.fragment_id in fragments_to_split:
                # Find the split point from split log
                split_frame = None
                for log_entry in self.split_log:
                    if (log_entry.get("reason") == "jersey_temporal_conflict" and
                        log_entry.get("track_id") == frag.original_track_id):
                        split_frame = log_entry.get("frame_idx")
                        break

                if split_frame is not None:
                    # Split fragment at this frame
                    det_before = []
                    det_after = []

                    for det_id in frag.detection_ids:
                        det = detection_by_id.get(det_id)
                        if det:
                            if det.frame_idx < split_frame:
                                det_before.append(det)
                            else:
                                det_after.append(det)

                    # Create two fragments
                    if det_before:
                        new_fragments.append(
                            self._create_fragment(
                                frag.original_track_id,
                                det_before,
                                frag.split_reason or "initial",
                            )
                        )

                    if det_after:
                        new_fragments.append(
                            self._create_fragment(
                                frag.original_track_id,
                                det_after,
                                "jersey_temporal_conflict",
                            )
                        )
                else:
                    # Couldn't find split frame - keep as-is
                    new_fragments.append(frag)
            else:
                # No conflict - keep fragment as-is
                new_fragments.append(frag)

        return new_fragments

    def _merge_consecutive_short_fragments(
        self, fragments: List[Fragment]
    ) -> List[Fragment]:
        """
        Merge consecutive short fragments on the same track.

        Per CLAUDE.md and MEMORY.md:
        - Keep ALL fragments (even <10 frames)
        - Merge consecutive short fragments on same track
        - This balances coverage and clustering quality

        Args:
            fragments: All fragments

        Returns:
            Merged fragments
        """
        # Group by original_track_id
        track_fragments: Dict[TrackID, List[Fragment]] = {}
        for frag in fragments:
            if frag.original_track_id not in track_fragments:
                track_fragments[frag.original_track_id] = []
            track_fragments[frag.original_track_id].append(frag)

        # Sort fragments within each track by start_frame
        for track_id in track_fragments:
            track_fragments[track_id] = sorted(
                track_fragments[track_id], key=lambda f: f.start_frame
            )

        # Merge consecutive short fragments
        merged = []
        for track_id, frags in track_fragments.items():
            merged.extend(self._merge_track_fragments(frags))

        return merged

    def _merge_track_fragments(self, fragments: List[Fragment]) -> List[Fragment]:
        """
        Merge consecutive short fragments on a single track.

        Args:
            fragments: Fragments for one track (sorted by start_frame)

        Returns:
            Merged fragments
        """
        if not fragments:
            return []

        merged = []
        current_group = [fragments[0]]

        for i in range(1, len(fragments)):
            prev_frag = current_group[-1]
            curr_frag = fragments[i]

            # Check if both are short and consecutive
            prev_len = prev_frag.end_frame - prev_frag.start_frame + 1
            curr_len = curr_frag.end_frame - curr_frag.start_frame + 1

            is_consecutive = (curr_frag.start_frame == prev_frag.end_frame + 1)
            both_short = (prev_len < const.MIN_FRAGMENT_LENGTH and
                         curr_len < const.MIN_FRAGMENT_LENGTH)

            if is_consecutive and both_short:
                # Add to current group
                current_group.append(curr_frag)
            else:
                # Save current group and start new one
                if len(current_group) > 1:
                    # Merge group
                    merged_frag = self._merge_fragment_group(current_group)
                    merged.append(merged_frag)
                else:
                    # Keep as-is
                    merged.append(current_group[0])

                current_group = [curr_frag]

        # Save final group
        if len(current_group) > 1:
            merged_frag = self._merge_fragment_group(current_group)
            merged.append(merged_frag)
        else:
            merged.append(current_group[0])

        return merged

    def _merge_fragment_group(self, fragments: List[Fragment]) -> Fragment:
        """
        Merge a group of consecutive fragments into one.

        Args:
            fragments: Fragments to merge (consecutive, same track)

        Returns:
            Merged fragment
        """
        # Combine detection_ids
        all_detection_ids = []
        for frag in fragments:
            all_detection_ids.extend(frag.detection_ids)

        # Create merged fragment
        merged = Fragment(
            fragment_id=fragments[0].fragment_id + "_merged",
            original_track_id=fragments[0].original_track_id,
            start_frame=min(f.start_frame for f in fragments),
            end_frame=max(f.end_frame for f in fragments),
            detection_ids=all_detection_ids,
            split_reason="merged_short_fragments",
            parent_fragment_id=None,
        )

        self.split_log.append({
            "action": "merge",
            "merged_fragment_id": merged.fragment_id,
            "source_fragments": [f.fragment_id for f in fragments],
            "reason": "consecutive_short_fragments",
        })

        return merged


def run_pass2a(
    input_dir: Path,
    output_dir: Optional[Path] = None,
) -> Pass2AOutput:
    """
    Execute Pass 2A: Mechanical Fragmentation.

    Args:
        input_dir: Directory containing pass1_raw.json
        output_dir: Directory to write output (default: same as input_dir)

    Returns:
        Pass2AOutput

    Raises:
        FileNotFoundError if pass1_raw.json not found
        ValidationError if output fails validation
    """
    if output_dir is None:
        output_dir = input_dir

    pass1_path = input_dir / const.PASS1_RAW_JSON
    output_path = output_dir / const.PASS2_FRAGMENTS_JSON

    if not pass1_path.exists():
        raise FileNotFoundError(f"Pass 1 output not found: {pass1_path}")

    fragmenter = Pass2AFragmenter()
    return fragmenter.run(pass1_path, output_path)
