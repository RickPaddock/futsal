"""
Pass 2B: Fragment Quality Scoring

Per CLAUDE.md Section 5 (Pass 2B):
- Input: pass2_fragments.json, pass1_raw.json
- Output: pass2b_scored_fragments.json (new artifact; pass2_fragments.json unchanged)
- Computes metadata ONLY (no identity assignment)
- Assigns quality: HIGH, MEDIUM, LOW (GHOST assigned in Pass 2C)

Per CLAUDE.md Principle P1 (Pass Immutability):
- Reads Pass 1 and Pass 2A outputs (read-only)
- Never modifies upstream artifacts
- Emits new artifacts with quality scores added

Quality Metrics:
- Detection confidence stability (avg, min)
- Occlusion ratio (jersey ROI invalid fraction)
- Jersey visible ratio (consistency of jersey observations)
- Appearance stability score (HSV histogram similarity)
- Mean velocity (centroid displacement per frame)
- Motion smoothness score (inverse velocity variance)
"""

from typing import List, Dict, Tuple
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
from ..core.types import FragmentQuality, DetectionID
from ..utils.file_utils import load_json, save_json
from ..utils.geometry import centroid_distance
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

        # Compute Pass 2B contract metrics
        avg_confidence = self._compute_avg_confidence(fragment_detections)
        min_confidence = self._compute_min_confidence(fragment_detections)
        occlusion_ratio = self._compute_occlusion_ratio(fragment_detections)
        jersey_visible_ratio = self._compute_jersey_visible_ratio(fragment_detections)
        jersey_observability_score = jersey_visible_ratio
        appearance_stability_score = self._compute_hsv_consistency(fragment_detections)
        mean_velocity, motion_smoothness_score = self._compute_motion_metrics(fragment_detections)

        quality = self._assign_quality_tier(
            appearance_stability_score=appearance_stability_score,
            occlusion_ratio=occlusion_ratio,
        )

        # Keep a continuous score for ranking/audit while tiering remains rule-based.
        quality_score, quality_reasons = self._compute_quality_score(
            quality=quality,
            appearance_stability_score=appearance_stability_score,
            occlusion_ratio=occlusion_ratio,
            jersey_observability_score=jersey_observability_score,
            motion_smoothness_score=motion_smoothness_score,
        )

        # Pass 2B contract: binary fragment classification (identity-agnostic)
        # quality tiers remain metadata only.
        if quality == FragmentQuality.LOW or occlusion_ratio > 0.5:
            presence_class = "occlusion_candidate"
        else:
            presence_class = "real"

        # Create ScoredFragment
        return ScoredFragment(
            # Copy Fragment fields
            fragment_id=fragment.fragment_id,
            track_id=fragment.track_id,
            start_frame=fragment.start_frame,
            end_frame=fragment.end_frame,
            detection_ids=fragment.detection_ids,
            split_reason=fragment.split_reason,
            split_trigger_frame=fragment.split_trigger_frame,
            split_rule_id=fragment.split_rule_id,
            parent_fragment_id=fragment.parent_fragment_id,
            dominant_jersey_number=fragment.dominant_jersey_number,
            # Add quality fields
            quality=quality,
            quality_score=quality_score,
            quality_reasons=quality_reasons,
            avg_confidence=avg_confidence,
            min_confidence=min_confidence,
            occlusion_ratio=occlusion_ratio,
            jersey_visible_ratio=jersey_visible_ratio,
            jersey_observability_score=jersey_observability_score,
            appearance_stability_score=appearance_stability_score,
            mean_velocity=mean_velocity,
            motion_smoothness_score=motion_smoothness_score,
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

    def _compute_occlusion_ratio(self, detections: List[Detection]) -> float:
        """
        Compute occlusion ratio as fraction of frames where jersey ROI is invalid.
        """
        if not detections:
            return 0.0
        invalid = sum(1 for det in detections if not det.jersey_roi_valid)
        return float(invalid / len(detections))

    def _compute_motion_metrics(self, detections: List[Detection]) -> Tuple[float, float]:
        """
        Compute mean velocity and motion smoothness.

        motion_smoothness_score is inverse velocity variance in [0, 1].
        """
        if len(detections) < 2:
            return 0.0, 1.0

        ordered = sorted(detections, key=lambda d: d.frame_idx)
        velocities: List[float] = []

        for i in range(len(ordered) - 1):
            current = ordered[i]
            nxt = ordered[i + 1]
            frame_delta = max(1, nxt.frame_idx - current.frame_idx)
            displacement = centroid_distance(current.centroid, nxt.centroid)
            velocities.append(displacement / frame_delta)

        if not velocities:
            return 0.0, 1.0

        mean_velocity = float(np.mean(velocities))
        variance = float(np.var(velocities))
        motion_smoothness_score = float(np.clip(1.0 / (1.0 + variance), 0.0, 1.0))
        return mean_velocity, motion_smoothness_score

    def _compute_jersey_visible_ratio(self, detections: List[Detection]) -> float:
        """
        Compute jersey observability as fraction of detections with visible jersey number.
        """
        if not detections:
            return 0.0
        visible = sum(1 for det in detections if det.jersey_number is not None)
        return float(visible / len(detections))

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

        # Compute pairwise histogram similarities.
        max_samples = 40
        if len(histograms) > max_samples:
            idx = np.linspace(0, len(histograms) - 1, max_samples, dtype=int)
            histograms = [histograms[i] for i in idx]

        similarities = []
        for i in range(len(histograms) - 1):
            for j in range(i + 1, len(histograms)):
                corr = compare_hsv_histograms(histograms[i], histograms[j])
                similarities.append(float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0)))

        # Average similarity
        avg_similarity = float(np.mean(similarities)) if similarities else 0.0
        return float(np.clip(avg_similarity, 0.0, 1.0))

    def _assign_quality_tier(
        self,
        appearance_stability_score: float,
        occlusion_ratio: float,
    ) -> FragmentQuality:
        """
        Assign quality tier using strict Pass 2B contract rules.

        HIGH   : appearance_stability_score >= 0.7 and occlusion_ratio <= 0.2
        MEDIUM : appearance_stability_score >= 0.4
        LOW    : otherwise
        """
        if (
            appearance_stability_score >= const.QUALITY_HIGH_THRESHOLD
            and occlusion_ratio <= 0.2
        ):
            return FragmentQuality.HIGH

        if appearance_stability_score >= const.QUALITY_MEDIUM_THRESHOLD:
            return FragmentQuality.MEDIUM

        return FragmentQuality.LOW

    def _compute_quality_score(
        self,
        quality: FragmentQuality,
        appearance_stability_score: float,
        occlusion_ratio: float,
        jersey_observability_score: float,
        motion_smoothness_score: float,
    ) -> Tuple[float, List[str]]:
        """
        Compute a continuous quality score for diagnostics/ranking.

        Returns:
            Tuple of (quality_score, quality_reasons)
        """
        reasons: List[str] = []

        quality_score = (
            0.65 * appearance_stability_score
            + 0.20 * motion_smoothness_score
            + 0.15 * jersey_observability_score
        ) * (1.0 - 0.30 * occlusion_ratio)

        quality_score = float(np.clip(quality_score, 0.0, 1.0))

        reasons.append(f"tier={quality.value}")
        reasons.append(f"appearance={appearance_stability_score:.2f}")
        reasons.append(f"occlusion={occlusion_ratio:.2f}")
        reasons.append(f"jersey_observability={jersey_observability_score:.2f}")
        reasons.append(f"motion_smoothness={motion_smoothness_score:.2f}")

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
            track_id=fragment.track_id,
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
            occlusion_ratio=0.0,
            jersey_visible_ratio=0.0,
            jersey_observability_score=0.0,
            appearance_stability_score=0.0,
            mean_velocity=0.0,
            motion_smoothness_score=1.0,
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
        Path to pass2b_scored_fragments.json (new artifact; pass2_fragments.json unchanged)
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
    validation_result = validator.validate_pass2b(pass2b_output, pass2a_output)

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

    # Write Pass 2B output to new artifact (P1: pass2_fragments.json remains untouched)
    output_path = output_dir / const.PASS2B_SCORED_FRAGMENTS_JSON
    save_json(pass2b_output.model_dump(), str(output_path))
    logger.info(f"Pass 2B output (scored): {output_path}")

    logger.info("=" * 80)
    logger.info(f"PASS 2B COMPLETE: {len(pass2b_output.fragments)} scored fragments")
    logger.info(f"Quality distribution: {pass2b_output.quality_distribution}")
    logger.info("=" * 80)

    return output_path
