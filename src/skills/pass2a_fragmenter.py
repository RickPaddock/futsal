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
from functools import lru_cache
import hashlib

import cv2
import numpy as np
from tqdm import tqdm

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
from ..utils.video_io import VideoReader
from ..validation.validator import Validator
from ..core import constants as const

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_pass2_intention_lines(max_lines: int = 4) -> List[str]:
    """Load Pass 2 intent lines from validation/pass2_rules.py docstring."""
    try:
        from ..validation import pass2_rules

        doc = pass2_rules.__doc__ or ""
        lines: List[str] = []
        for raw in doc.splitlines():
            line = raw.strip()
            if line.startswith("- "):
                lines.append(line[2:].strip())

        if not lines:
            return ["Pass 2A mechanical fragmentation is active."]

        return lines[:max_lines]
    except Exception:
        return ["Pass 2A mechanical fragmentation is active."]


def _fragment_color(fragment_id: str) -> Tuple[int, int, int]:
    """Deterministic BGR color for fragment overlays."""
    digest = hashlib.md5(fragment_id.encode("utf-8")).digest()
    return (
        60 + (digest[0] % 170),
        60 + (digest[1] % 170),
        60 + (digest[2] % 170),
    )


def _draw_text_with_bg(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float,
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    pad_x = 4
    pad_y = 3
    x1 = max(0, x - pad_x)
    y1 = max(0, y - text_h - pad_y)
    x2 = min(img.shape[1] - 1, x + text_w + pad_x)
    y2 = min(img.shape[0] - 1, y + baseline + pad_y)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_pass2a_debug_overlay_frame(
    frame: np.ndarray,
    frame_idx: int,
    frame_annotations: List[Tuple[Detection, Fragment]],
    split_log_count: int,
) -> np.ndarray:
    """Draw Pass 2A fragment overlays for one frame."""
    overlay = frame.copy()

    active_fragments: Set[str] = set()

    for det, fragment in frame_annotations:
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        color = _fragment_color(fragment.fragment_id)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"{fragment.fragment_id} trk={fragment.original_track_id}"
        _draw_text_with_bg(overlay, label, (x1, max(18, y1 - 8)), 0.42, 1)

        reason = fragment.split_reason or "initial"
        _draw_text_with_bg(overlay, f"reason={reason}", (x1, min(overlay.shape[0] - 6, y2 + 16)), 0.40, 1)
        active_fragments.add(fragment.fragment_id)

    intent_lines = _load_pass2_intention_lines()
    runtime_lines = [
        "Pass 2A: mechanical fragmentation only (no team/player inference)",
        "Split triggers: overlap, appearance drift, jersey inconsistency, velocity spikes",
        f"frame stats: detections={len(frame_annotations)} activeFragments={len(active_fragments)} totalSplits={split_log_count}",
    ]

    title = "PASS 2A RESPONSIBILITIES (ACTIVE)"
    all_lines = intent_lines + runtime_lines

    frame_h, frame_w = overlay.shape[:2]
    top = max(12, int(frame_h * 0.055))
    left = 12
    line_gap = 20
    title_y = top + 14
    first_rule_y = title_y + line_gap
    footer_y = first_rule_y + (len(all_lines) * line_gap) + 8
    panel_bottom = min(frame_h - 8, footer_y + 24)

    panel = overlay.copy()
    panel_top = max(6, top - 12)
    cv2.rectangle(panel, (6, panel_top), (frame_w - 6, panel_bottom), (0, 0, 0), -1)
    overlay = cv2.addWeighted(panel, 0.55, overlay, 0.45, 0)

    _draw_text_with_bg(overlay, title, (left, title_y), 0.50, 1)
    for idx, line in enumerate(all_lines):
        y = first_rule_y + (idx * line_gap)
        _draw_text_with_bg(overlay, f"- {line}", (left + 4, y), 0.46, 1)

    _draw_text_with_bg(
        overlay,
        "Boxes=fragment-colored | Label: Fxxxxxx + original track + split reason",
        (left, footer_y),
        0.45,
        1,
    )
    _draw_text_with_bg(overlay, f"frame={frame_idx}", (left, footer_y + 18), 0.52, 1)

    return overlay


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

        Per CLAUDE.md Section 5 (Pass 2A) - EXHAUSTIVE allowed split triggers:
        1. Track Collision: same track_id, >1 detection per frame (ByteTrack failure)
        2. Jersey Change: #7 → #4 (number to different number, NOT disappearance)
        3. Jersey Temporal Exclusivity: same jersey on different tracks (handled separately)
        4. Hard Appearance Discontinuity: ALL of (jersey visible both sides + HSV + impossible motion)

        FORBIDDEN (loss of observability ≠ identity change):
        - Jersey disappearance (#4 → None)
        - Jersey first appearance (None → #4)
        - Standalone appearance drift
        - Standalone velocity spikes
        - Occlusion, confidence drops, missed detections, normal drift

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

        Per CLAUDE.md Section 5 (Pass 2A) - EXHAUSTIVE split triggers only.

        Returns:
            List of (frame_idx, reason) tuples where splits should occur
        """
        split_points = []

        # ============================================================================
        # TRIGGER 1: Track Collision (ByteTrack failure)
        # Same track_id produces >1 detection in same frame
        # ============================================================================
        frame_counts: Dict[int, int] = {}
        for det in detections:
            frame_counts[det.frame_idx] = frame_counts.get(det.frame_idx, 0) + 1

        for frame_idx, count in frame_counts.items():
            if count > 1:
                split_points.append((frame_idx, "track_collision"))
                self.split_log.append({
                    "track_id": track_id,
                    "frame_idx": frame_idx,
                    "reason": "track_collision",
                    "details": f"Track {track_id} has {count} detections at frame {frame_idx} (ByteTrack failure)",
                })

        # ============================================================================
        # TRIGGER 2: Jersey Change (#7 → #4)
        # Jersey number changes from one NUMBER to another NUMBER (NOT disappearance)
        # ============================================================================
        # Check jersey inconsistency ONLY across jersey-classifier sampled observations.
        # This prevents false splits from unsampled frames where jersey_number is intentionally None.
        jersey_sample_every = max(1, int(const.JERSEY_NUMBER_CLASSIFY_EVERY_N_FRAMES))
        sampled_observations = [d for d in detections if (d.frame_idx % jersey_sample_every) == 0]

        for i in range(1, len(sampled_observations)):
            prev_det = sampled_observations[i - 1]
            curr_det = sampled_observations[i]

            prev_jersey = prev_det.jersey_number
            curr_jersey = curr_det.jersey_number

            # ✅ SPLIT: Jersey changes (#7 → #4) on sampled observations
            # NUMBER → DIFFERENT NUMBER (hard identity proof)
            if prev_jersey is not None and curr_jersey is not None and prev_jersey != curr_jersey:
                split_points.append((curr_det.frame_idx, "jersey_change"))
                self.split_log.append({
                    "track_id": track_id,
                    "frame_idx": curr_det.frame_idx,
                    "reason": "jersey_change",
                    "details": (
                        f"Jersey changed from #{prev_jersey} to #{curr_jersey} "
                        f"between sampled observations ({prev_det.frame_idx}->{curr_det.frame_idx})"
                    ),
                })

            # ❌ NO SPLIT: Jersey disappearance (#4 → None)
            # Per CLAUDE.md: This is loss of observability, NOT identity change
            # Record as metadata only (handled in Pass 2B quality scoring)

            # ❌ NO SPLIT: Jersey first appearance (None → #4)
            # Per CLAUDE.md: Player turned around / became readable
            # This is normal, NOT a track jump

        # ============================================================================
        # TRIGGER 4: Hard Appearance Discontinuity
        # ALL conditions must hold:
        # - Jersey visible on BOTH sides of boundary
        # - Large HSV distance beyond threshold
        # - Incompatible motion (teleport / impossible velocity)
        # ============================================================================
        for i in range(1, len(detections)):
            prev_det = detections[i - 1]
            curr_det = detections[i]

            # Only check consecutive frames (frame_gap == 1)
            frame_gap = curr_det.frame_idx - prev_det.frame_idx
            if frame_gap != 1:
                continue

            # Condition 1: Jersey visible on BOTH sides
            prev_has_jersey = prev_det.jersey_number is not None
            curr_has_jersey = curr_det.jersey_number is not None
            if not (prev_has_jersey and curr_has_jersey):
                continue  # Jersey not visible both sides → NO SPLIT

            # Condition 2: Large HSV distance
            if not (prev_det.hsv_histogram_jersey and curr_det.hsv_histogram_jersey):
                continue  # No HSV data → NO SPLIT

            correlation = compare_hsv_histograms(
                prev_det.hsv_histogram_jersey,
                curr_det.hsv_histogram_jersey,
            )
            hsv_distance = 1.0 - correlation
            if hsv_distance <= const.HSV_DRIFT_THRESHOLD:
                continue  # HSV similar → NO SPLIT

            # Condition 3: Incompatible motion (teleport)
            motion_distance = centroid_distance(prev_det.centroid, curr_det.centroid)
            if motion_distance <= const.VELOCITY_SPIKE_THRESHOLD:
                continue  # Motion reasonable → NO SPLIT

            # ALL 3 CONDITIONS MET → SPLIT
            split_points.append((curr_det.frame_idx, "hard_appearance_discontinuity"))
            self.split_log.append({
                "track_id": track_id,
                "frame_idx": curr_det.frame_idx,
                "reason": "hard_appearance_discontinuity",
                "details": (
                    f"Hard discontinuity: jersey visible both sides, "
                    f"HSV distance={hsv_distance:.3f}, motion={motion_distance:.1f}px"
                ),
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
        split_trigger_frame: Optional[int] = None

        for det in detections:
            # Check if we hit a split point
            hit_split = False
            for split_frame, reason in split_points:
                if det.frame_idx == split_frame:
                    # Save current segment as fragment (if not empty)
                    if current_segment:
                        fragments.append(
                            self._create_fragment(
                                track_id,
                                current_segment,
                                split_reason,
                                split_trigger_frame,
                            )
                        )

                    # Start new segment
                    current_segment = [det]
                    split_reason = reason
                    split_trigger_frame = split_frame
                    hit_split = True
                    break

            if not hit_split:
                current_segment.append(det)

        # Save final segment
        if current_segment:
            fragments.append(
                self._create_fragment(
                    track_id,
                    current_segment,
                    split_reason,
                    split_trigger_frame,
                )
            )

        return fragments

    def _create_fragment(
        self,
        track_id: TrackID,
        detections: List[Detection],
        split_reason: str,
        split_trigger_frame: Optional[int] = None,
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

        # Per CLAUDE.md Section 5 (Pass 2A) - EXHAUSTIVE split rule mapping
        split_rule_by_reason = {
            "track_collision": "TRACK_COLLISION",
            "jersey_change": "JERSEY_CHANGE",
            "jersey_temporal_conflict": "JERSEY_TEMPORAL_CONFLICT",
            "hard_appearance_discontinuity": "HARD_APPEARANCE_DISCONTINUITY",
            "merged_short_fragments": "MERGE_SHORT_FRAGMENTS",
        }

        non_initial = split_reason != "initial"
        effective_trigger_frame = split_trigger_frame if non_initial else None
        if effective_trigger_frame is not None:
            # Secondary split operations (e.g., temporal jersey conflict) can produce
            # sub-fragments where inherited trigger frame sits outside the new bounds.
            # Normalize to this fragment boundary to keep metadata self-consistent.
            if effective_trigger_frame < start_frame or effective_trigger_frame > end_frame:
                effective_trigger_frame = start_frame
        effective_rule_id = split_rule_by_reason.get(split_reason) if non_initial else None

        return Fragment(
            fragment_id=fragment_id,
            original_track_id=track_id,
            start_frame=start_frame,
            end_frame=end_frame,
            detection_ids=detection_ids,
            split_reason=split_reason if split_reason != "initial" else None,
            split_trigger_frame=effective_trigger_frame,
            split_rule_id=effective_rule_id,
            parent_fragment_id=None,  # Will be set during jersey temporal exclusivity
        )

    def _split_jersey_temporal_conflicts(
        self,
        fragments: List[Fragment],
        all_detections: List[Detection],
    ) -> List[Fragment]:
        """
        Detect and split jersey temporal exclusivity violations.

        Per CLAUDE.md R3 and IMPLEMENTATION_PLAN.md:
        - Same jersey CANNOT appear on different tracks simultaneously
        - Uses VOTING/CONSENSUS across fragment, NOT first appearance
        - Jersey must appear ≥3 times with conf ≥0.5 to be considered "owned"
        - Fragment must have ≥50% observations with same jersey to "own" it
        - If two fragments "own" same jersey with overlapping times → split the later one

        Args:
            fragments: All fragments created so far
            all_detections: All Pass 1 detections (for jersey lookups)

        Returns:
            Fragments with temporal conflicts resolved via splits
        """
        from collections import Counter

        # Build detection lookup by detection_id for fast access
        detection_by_id = {det.detection_id: det for det in all_detections}

        # Build jersey timeline: jersey_number -> [(fragment, start_frame, end_frame, first_jersey_frame)]
        jersey_timeline: Dict[int, List[Tuple[Fragment, int, int, int]]] = {}

        for frag in fragments:
            # Count jersey observations across fragment (VOTING/CONSENSUS)
            jersey_observations = []
            first_jersey_frame = None

            for det_id in frag.detection_ids:
                det = detection_by_id.get(det_id)
                if det and det.jersey_number is not None:
                    # Only count high-confidence observations
                    if det.jersey_confidence >= const.JERSEY_MIN_CONFIDENCE:
                        jersey_observations.append(det.jersey_number)
                        if first_jersey_frame is None:
                            first_jersey_frame = det.frame_idx

            # Determine if fragment "owns" a jersey via majority vote
            jersey_num = None
            if len(jersey_observations) >= const.JERSEY_MIN_OBSERVATIONS:
                # Count occurrences
                jersey_counts = Counter(jersey_observations)
                most_common_jersey, most_common_count = jersey_counts.most_common(1)[0]

                # Check if it's a majority (≥50% of observations)
                if most_common_count / len(jersey_observations) >= const.JERSEY_MAJORITY_THRESHOLD:
                    jersey_num = most_common_jersey

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
                                frag.split_trigger_frame,
                            )
                        )

                    if det_after:
                        new_fragments.append(
                            self._create_fragment(
                                frag.original_track_id,
                                det_after,
                                "jersey_temporal_conflict",
                                split_frame,
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
            split_trigger_frame=fragments[0].start_frame,
            split_rule_id="MERGE_SHORT_FRAGMENTS",
            parent_fragment_id=None,
        )

        self.split_log.append({
            "action": "merge",
            "merged_fragment_id": merged.fragment_id,
            "source_fragments": [f.fragment_id for f in fragments],
            "track_id": merged.original_track_id,
            "frame_idx": merged.start_frame,
            "reason": "merged_short_fragments",
            "details": "Merged consecutive short fragments on same track",
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


def render_pass2a_debug_video_from_artifact(
    video_path: str,
    pass1_output_path: str,
    pass2a_output_path: str,
    debug_video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> None:
    """Render Pass 2A debug video using pass1_raw + pass2_fragments artifacts."""
    pass1_output = load_json(Path(pass1_output_path), Pass1Output)
    pass2a_output = load_json(Path(pass2a_output_path), Pass2AOutput)

    detection_by_id: Dict[str, Detection] = {det.detection_id: det for det in pass1_output.detections}

    frame_annotations: Dict[int, List[Tuple[Detection, Fragment]]] = {}
    for fragment in pass2a_output.fragments:
        for detection_id in fragment.detection_ids:
            det = detection_by_id.get(detection_id)
            if det is None:
                continue
            frame_annotations.setdefault(det.frame_idx, []).append((det, fragment))

    reader = VideoReader(video_path)
    writer = None

    try:
        fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
        if fourcc_fn is None:
            fourcc_fn = cv2.VideoWriter.fourcc

        writer = cv2.VideoWriter(
            debug_video_path,
            fourcc_fn(*"mp4v"),
            float(reader.fps),
            (int(reader.width), int(reader.height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open Pass 2A debug video writer: {debug_video_path}")

        artifact_start = pass1_output.processed_start_frame
        artifact_end_exclusive = pass1_output.processed_end_frame_exclusive
        if artifact_end_exclusive is None:
            artifact_end_exclusive = pass1_output.total_frames

        render_start = max(start_frame, artifact_start)
        requested_end = artifact_end_exclusive if end_frame is None else end_frame
        render_end_exclusive = min(requested_end, artifact_end_exclusive)

        if render_end_exclusive <= render_start:
            raise ValueError(
                f"Invalid render window: start={render_start}, end={render_end_exclusive}. "
                f"Artifact range is [{artifact_start}, {artifact_end_exclusive})."
            )

        total_render_frames = render_end_exclusive - render_start

        with tqdm(total=total_render_frames, desc="Pass 2A debug render", unit="frame") as pbar:
            for frame_idx, frame in reader.iter_frames():
                if frame_idx < render_start:
                    continue
                if frame_idx >= render_end_exclusive:
                    break

                annotations = frame_annotations.get(frame_idx, [])
                debug_frame = _draw_pass2a_debug_overlay_frame(
                    frame,
                    frame_idx,
                    annotations,
                    split_log_count=len(pass2a_output.split_log),
                )
                writer.write(debug_frame)
                pbar.update(1)
    finally:
        if writer is not None:
            writer.release()
        reader.close()
