"""
Pass 2B: Fragment Quality Scoring

Per CLAUDE.md Section 5 (Pass 2B):
- Input: pass2_fragments.json, pass1_raw.json
- Output: pass2_fragments.json (with quality scores)
- Computes metadata ONLY (no identity assignment)
- Assigns quality: HIGH, MEDIUM, LOW (GHOST assigned in Pass 2C)

Per CLAUDE.md Principle P1 (Pass Immutability):
- Reads Pass 1 and Pass 2A outputs (read-only)
- Never modifies upstream artifacts
- Emits new artifacts with quality scores added

Quality Metrics:
- Detection confidence stability (avg, min)
- Bbox stability (low jitter = high stability)
- Jersey consistency (how often same jersey appears)
- HSV consistency (histogram similarity across frames)
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

import numpy as np
from tqdm import tqdm

from ..core.data_models import (
    Detection,
    Pass1Output,
    Fragment,
    Pass2AOutput,
    ScoredFragment,
    Pass2BOutput,
)
from ..core.types import FragmentQuality, DetectionID, FragmentID
from ..utils.file_utils import load_json, save_json
from ..utils.geometry import bbox_width, bbox_height, centroid_distance
from ..utils.hsv_color import compare_hsv_histograms
from ..validation.validator import Validator
from ..core import constants as const

logger = logging.getLogger(__name__)


class Pass2BFragmentScorer:
    """
    Pass 2B: Fragment Quality Scoring

    Computes quality scores and metadata for each fragment.
    Quality is metadata only - NO identity decisions.
    """

    def __init__(self, pass1_output: Pass1Output, pass2a_output: Pass2AOutput):
        """
        Initialize scorer with Pass 1 and Pass 2A outputs.

        Args:
            pass1_output: Pass 1 raw detections
            pass2a_output: Pass 2A fragments
        """
        self.pass1_output = pass1_output
        self.pass2a_output = pass2a_output

        # Build detection lookup: detection_id -> Detection
        self.detections: Dict[DetectionID, Detection] = {
            det.detection_id: det for det in pass1_output.detections
        }

        logger.info(f"Pass2BFragmentScorer initialized: {len(pass2a_output.fragments)} fragments")

    def score_fragments(self) -> Pass2BOutput:
        """
        Score all fragments from Pass 2A.

        Returns:
            Pass2BOutput with scored fragments
        """
        logger.info(f"Scoring {len(self.pass2a_output.fragments)} fragments...")

        scored_fragments: List[ScoredFragment] = []
        quality_distribution: Dict[str, int] = {
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        for fragment in tqdm(self.pass2a_output.fragments, desc="Scoring fragments"):
            scored_frag = self._score_fragment(fragment)
            scored_fragments.append(scored_frag)

            # Update quality distribution
            quality_str = scored_frag.quality.value
            if quality_str != "ghost":  # Ghost quality assigned in Pass 2C
                quality_distribution[quality_str] = quality_distribution.get(quality_str, 0) + 1

        logger.info(f"Fragment quality distribution: {quality_distribution}")

        return Pass2BOutput(
            fragments=scored_fragments,
            quality_distribution=quality_distribution,
            split_log=self.pass2a_output.split_log,  # Preserve from Pass 2A
        )

    def _score_fragment(self, fragment: Fragment) -> ScoredFragment:
        """
        Compute quality scores for a single fragment.

        Args:
            fragment: Fragment to score

        Returns:
            ScoredFragment with quality metadata
        """
        # Get detections for this fragment
        fragment_detections = [
            self.detections[det_id]
            for det_id in fragment.detection_ids
            if det_id in self.detections
        ]

        if not fragment_detections:
            # Empty fragment (shouldn't happen, but defensive)
            return self._create_low_quality_fragment(
                fragment,
                quality_score=0.0,
                reasons=["No detections found for fragment"],
            )

        # Compute quality metrics
        avg_confidence = self._compute_avg_confidence(fragment_detections)
        min_confidence = self._compute_min_confidence(fragment_detections)
        avg_bbox_stability = self._compute_bbox_stability(fragment_detections)
        jersey_consistency = self._compute_jersey_consistency(fragment_detections)
        hsv_consistency = self._compute_hsv_consistency(fragment_detections)

        # Compute overall quality score (0-1)
        quality_score, quality_reasons = self._compute_quality_score(
            fragment=fragment,
            avg_confidence=avg_confidence,
            min_confidence=min_confidence,
            avg_bbox_stability=avg_bbox_stability,
            jersey_consistency=jersey_consistency,
            hsv_consistency=hsv_consistency,
        )

        # Assign quality label
        if quality_score >= const.QUALITY_HIGH_THRESHOLD:
            quality = FragmentQuality.HIGH
        elif quality_score >= const.QUALITY_MEDIUM_THRESHOLD:
            quality = FragmentQuality.MEDIUM
        else:
            quality = FragmentQuality.LOW

        # Pass 2B contract: binary fragment classification (identity-agnostic)
        # quality tiers remain metadata only.
        if quality == FragmentQuality.LOW or min_confidence < const.PLAYER_CONF_THRESHOLD:
            presence_class = "occlusion_candidate"
        else:
            presence_class = "real"

        # Create ScoredFragment
        return ScoredFragment(
            # Copy Fragment fields
            fragment_id=fragment.fragment_id,
            original_track_id=fragment.original_track_id,
            start_frame=fragment.start_frame,
            end_frame=fragment.end_frame,
            detection_ids=fragment.detection_ids,
            split_reason=fragment.split_reason,
            split_trigger_frame=fragment.split_trigger_frame,
            split_rule_id=fragment.split_rule_id,
            parent_fragment_id=fragment.parent_fragment_id,
            # Add quality fields
            quality=quality,
            quality_score=quality_score,
            quality_reasons=quality_reasons,
            avg_confidence=avg_confidence,
            min_confidence=min_confidence,
            avg_bbox_stability=avg_bbox_stability,
            jersey_consistency=jersey_consistency,
            hsv_consistency=hsv_consistency,
            presence_class=presence_class,
        )

    def _compute_avg_confidence(self, detections: List[Detection]) -> float:
        """Compute average detection confidence."""
        if not detections:
            return 0.0
        confidences = [det.confidence for det in detections]
        return float(np.mean(confidences))

    def _compute_min_confidence(self, detections: List[Detection]) -> float:
        """Compute minimum detection confidence."""
        if not detections:
            return 0.0
        confidences = [det.confidence for det in detections]
        return float(np.min(confidences))

    def _compute_bbox_stability(self, detections: List[Detection]) -> float:
        """
        Compute bbox stability (low jitter = high stability).

        Measures how stable the bbox size and position are across frames.
        Returns value in [0, 1] where 1 = perfectly stable.
        """
        if len(detections) < 2:
            return 1.0  # Single detection = no jitter

        # Measure bbox size stability
        widths = [bbox_width(det.bbox) for det in detections]
        heights = [bbox_height(det.bbox) for det in detections]

        width_std = float(np.std(widths))
        height_std = float(np.std(heights))

        # Measure centroid motion stability
        centroids = [det.centroid for det in detections]
        centroid_distances = [
            centroid_distance(centroids[i], centroids[i + 1])
            for i in range(len(centroids) - 1)
        ]
        centroid_std = float(np.std(centroid_distances)) if centroid_distances else 0.0

        # Normalize to [0, 1] (lower std = higher stability)
        # Use heuristic normalization based on typical values
        width_stability = max(0.0, 1.0 - width_std / 50.0)
        height_stability = max(0.0, 1.0 - height_std / 50.0)
        centroid_stability = max(0.0, 1.0 - centroid_std / 10.0)

        # Average the three stability metrics
        stability = (width_stability + height_stability + centroid_stability) / 3.0
        return float(np.clip(stability, 0.0, 1.0))

    def _compute_jersey_consistency(self, detections: List[Detection]) -> float:
        """
        Compute jersey consistency (how often same jersey appears).

        Returns value in [0, 1] where 1 = same jersey throughout.
        """
        # Count jersey observations
        jersey_observations = [
            det.jersey_number
            for det in detections
            if det.jersey_number is not None
        ]

        if not jersey_observations:
            return 0.0  # No jersey observed

        # Find most common jersey
        from collections import Counter
        jersey_counts = Counter(jersey_observations)
        most_common_jersey, most_common_count = jersey_counts.most_common(1)[0]

        # Consistency = fraction of observations with most common jersey
        consistency = most_common_count / len(jersey_observations)
        return float(consistency)

    def _compute_hsv_consistency(self, detections: List[Detection]) -> float:
        """
        Compute HSV histogram consistency across frames.

        Returns value in [0, 1] where 1 = identical histograms.
        """
        # Get HSV histograms
        histograms = [
            det.hsv_histogram_jersey
            for det in detections
            if det.hsv_histogram_jersey is not None
        ]

        if len(histograms) < 2:
            return 1.0 if histograms else 0.0  # Single histogram = consistent

        # Compute pairwise histogram similarities
        similarities = []
        for i in range(len(histograms) - 1):
            similarity = compare_hsv_histograms(histograms[i], histograms[i + 1])
            similarities.append(similarity)

        # Average similarity
        avg_similarity = float(np.mean(similarities)) if similarities else 0.0
        return float(np.clip(avg_similarity, 0.0, 1.0))

    def _compute_quality_score(
        self,
        fragment: Fragment,
        avg_confidence: float,
        min_confidence: float,
        avg_bbox_stability: float,
        jersey_consistency: float,
        hsv_consistency: float,
    ) -> Tuple[float, List[str]]:
        """
        Compute overall quality score from individual metrics.

        Returns:
            Tuple of (quality_score, quality_reasons)
        """
        reasons: List[str] = []

        # Fragment length factor (short fragments are lower quality)
        fragment_length = fragment.end_frame - fragment.start_frame + 1
        length_factor = min(1.0, fragment_length / const.MIN_FRAGMENT_LENGTH)

        if fragment_length < const.MIN_FRAGMENT_LENGTH:
            reasons.append(f"Short fragment ({fragment_length} frames < {const.MIN_FRAGMENT_LENGTH})")

        # Confidence factor
        confidence_factor = (avg_confidence + min_confidence) / 2.0

        if min_confidence < const.PLAYER_CONF_THRESHOLD:
            reasons.append(f"Low min confidence ({min_confidence:.2f})")

        # Stability factor
        stability_factor = avg_bbox_stability

        if avg_bbox_stability < 0.5:
            reasons.append(f"Low bbox stability ({avg_bbox_stability:.2f})")

        # Jersey factor
        jersey_factor = jersey_consistency

        if jersey_consistency < 0.5:
            reasons.append(f"Low jersey consistency ({jersey_consistency:.2f})")

        # HSV factor
        hsv_factor = hsv_consistency

        if hsv_consistency < 0.5:
            reasons.append(f"Low HSV consistency ({hsv_consistency:.2f})")

        # Weighted average (prioritize confidence and stability)
        quality_score = (
            0.25 * confidence_factor +
            0.20 * stability_factor +
            0.20 * length_factor +
            0.15 * jersey_factor +
            0.20 * hsv_factor
        )

        quality_score = float(np.clip(quality_score, 0.0, 1.0))

        if not reasons:
            reasons.append(f"Good quality (score={quality_score:.2f})")

        return quality_score, reasons

    def _create_low_quality_fragment(
        self,
        fragment: Fragment,
        quality_score: float,
        reasons: List[str],
    ) -> ScoredFragment:
        """Create a low-quality ScoredFragment."""
        return ScoredFragment(
            # Copy Fragment fields
            fragment_id=fragment.fragment_id,
            original_track_id=fragment.original_track_id,
            start_frame=fragment.start_frame,
            end_frame=fragment.end_frame,
            detection_ids=fragment.detection_ids,
            split_reason=fragment.split_reason,
            split_trigger_frame=fragment.split_trigger_frame,
            split_rule_id=fragment.split_rule_id,
            parent_fragment_id=fragment.parent_fragment_id,
            # Add quality fields
            quality=FragmentQuality.LOW,
            quality_score=quality_score,
            quality_reasons=reasons,
            presence_class="occlusion_candidate",
            avg_confidence=0.0,
            min_confidence=0.0,
            avg_bbox_stability=0.0,
            jersey_consistency=0.0,
            hsv_consistency=0.0,
        )


def run_pass2b(
    pass1_json_path: Path,
    pass2a_json_path: Path,
    output_dir: Path,
) -> Path:
    """
    Run Pass 2B: Fragment Quality Scoring.

    Args:
        pass1_json_path: Path to pass1_raw.json
        pass2a_json_path: Path to pass2_fragments.json
        output_dir: Output directory for Pass 2B artifacts

    Returns:
        Path to pass2_fragments.json (with quality scores)
    """
    logger.info("=" * 80)
    logger.info("PASS 2B: FRAGMENT QUALITY SCORING")
    logger.info("=" * 80)

    # Load Pass 1 output
    logger.info(f"Loading Pass 1 output: {pass1_json_path}")
    pass1_output = Pass1Output(**load_json(pass1_json_path))

    # Load Pass 2A output
    logger.info(f"Loading Pass 2A output: {pass2a_json_path}")
    pass2a_output = Pass2AOutput(**load_json(pass2a_json_path))

    # Score fragments
    scorer = Pass2BFragmentScorer(pass1_output, pass2a_output)
    pass2b_output = scorer.score_fragments()

    # Validate output BEFORE writing
    logger.info("Validating Pass 2B output...")
    validator = Validator()
    validation_result = validator.validate_pass2b(pass2b_output)

    # Write validation report
    validation_path = output_dir / const.PASS2_VALIDATION_JSON
    save_json(validation_result.model_dump(), str(validation_path))
    logger.info(f"Validation report: {validation_path}")

    if not validation_result.passed:
        logger.error("Pass 2B validation FAILED")
        for violation in validation_result.violations:
            logger.error(f"  - {violation.rule}: {violation.message}")
        raise ValueError("Pass 2B validation failed (fail-fast)")

    logger.info("Pass 2B validation PASSED ✓")

    # Write Pass 2B output (overwrite pass2_fragments.json with scored fragments)
    output_path = output_dir / const.PASS2_FRAGMENTS_JSON
    save_json(pass2b_output.model_dump(), str(output_path))
    logger.info(f"Pass 2B output: {output_path}")

    logger.info("=" * 80)
    logger.info(f"PASS 2B COMPLETE: {len(pass2b_output.fragments)} scored fragments")
    logger.info(f"Quality distribution: {pass2b_output.quality_distribution}")
    logger.info("=" * 80)

    return output_path
