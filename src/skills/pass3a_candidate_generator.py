"""
Pass 3A: Identity Candidate Generation

Per CLAUDE.md Section 5 (Pass 3A):
- Input: pass2_ghosts.json (required), pass1_raw.json (optional evidence enrichment)
- Output: pass3_candidates.json
- Generate candidate evidence only (NO identity locking)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..core import constants as const
from ..core.data_models import (
    Detection,
    IdentityCandidate,
    Pass1Output,
    Pass2COutput,
    Pass3AOutput,
    ScoredFragment,
    ValidationResult,
)
from ..core.types import TeamID
from ..utils.file_utils import load_json, save_json
from ..utils.logging_utils import get_logger
from ..validation.pass3_rules import validate_pass3a_candidates

logger = get_logger("pass3a_candidate_generator")
ALLOWED_JERSEY_NUMBERS = set(const.JERSEY_NUMBERS)


class Pass3ACandidateGenerator:
    """Generate identity candidates from Pass 2 fragments/ghosts."""

    def __init__(self, pass2c_output: Pass2COutput, pass1_output: Optional[Pass1Output] = None):
        self.pass2c_output = pass2c_output
        self.pass1_output = pass1_output

        self._detections_by_id: Dict[str, Detection] = {}
        if pass1_output is not None:
            self._detections_by_id = {
                detection.detection_id: detection
                for detection in pass1_output.detections
            }

        self._fragments_by_track: Dict[int, List[ScoredFragment]] = defaultdict(list)
        for fragment in self.pass2c_output.fragments:
            self._fragments_by_track[fragment.original_track_id].append(fragment)
        for fragments in self._fragments_by_track.values():
            fragments.sort(key=lambda fragment: (fragment.start_frame, fragment.end_frame))

    def generate(self) -> Pass3AOutput:
        """Generate Pass 3A candidate records for all fragments."""
        candidates: List[IdentityCandidate] = []

        for fragment in self.pass2c_output.fragments:
            detections = self._resolve_fragment_detections(fragment)
            team_evidence = self._build_team_evidence(fragment, detections)
            jersey_evidence = self._build_jersey_evidence(detections)
            player_evidence = self._build_player_evidence(fragment)

            candidate_jersey = self._select_candidate_jersey(jersey_evidence)
            candidate_team = self._select_candidate_team(team_evidence)

            candidates.append(
                IdentityCandidate(
                    fragment_id=fragment.fragment_id,
                    candidate_team=candidate_team,
                    candidate_jersey=candidate_jersey,
                    candidate_player_id=None,
                    team_evidence=team_evidence,
                    jersey_evidence=jersey_evidence,
                    player_evidence=player_evidence,
                )
            )

        return Pass3AOutput(candidates=candidates)

    def _resolve_fragment_detections(self, fragment: ScoredFragment) -> List[Detection]:
        if not self._detections_by_id:
            return []
        return [
            self._detections_by_id[detection_id]
            for detection_id in fragment.detection_ids
            if detection_id in self._detections_by_id
        ]

    def _build_team_evidence(self, fragment: ScoredFragment, detections: List[Detection]) -> Dict[str, float]:
        if getattr(fragment, "is_ghost", False):
            return {
                TeamID.TEAM_A.value: 0.5,
                TeamID.TEAM_B.value: 0.5,
                "ghost_fragment": 1.0,
            }

        observability_values: List[float] = []
        if detections:
            roi_valid_count = sum(1 for detection in detections if detection.jersey_roi_valid)
            observability_values.append(roi_valid_count / len(detections))

            sampled_count = sum(1 for detection in detections if detection.jersey_color_sampled)
            observability_values.append(sampled_count / len(detections))

        if hasattr(fragment, "hsv_consistency"):
            observability_values.append(float(max(0.0, min(1.0, fragment.hsv_consistency))))

        if observability_values:
            hsv_observability = float(sum(observability_values) / len(observability_values))
        else:
            hsv_observability = 0.0

        return {
            TeamID.TEAM_A.value: 0.5,
            TeamID.TEAM_B.value: 0.5,
            "hsv_observability": hsv_observability,
        }

    def _build_jersey_evidence(self, detections: List[Detection]) -> Dict[str, object]:
        """
        Build jersey evidence as {jersey_key: {"count": int, "ratio": float}}.

        Uses raw detection counts (not normalized by max) so evidence strength is
        preserved for conflict resolution in Pass 3C.

        Filters:
          - MIN_JERSEY_DETECTIONS: suppress fragments with too few hits
          - MIN_JERSEY_RATIO: suppress accidental crops of background players
        """
        if not detections:
            return {}

        jersey_counts: Dict[int, int] = defaultdict(int)
        total_detections = len(detections)

        for detection in detections:
            if (
                detection.jersey_number is not None
                and detection.jersey_number in ALLOWED_JERSEY_NUMBERS
                and detection.jersey_confidence >= const.JERSEY_CONF_THRESHOLD
            ):
                jersey_counts[detection.jersey_number] += 1
            elif detection.jersey_probs:
                # Fallback: use top-scoring jersey from probability distribution
                best_jersey: Optional[int] = None
                best_score = 0.0
                for jersey_key, score in detection.jersey_probs.items():
                    try:
                        jersey_number = int(jersey_key)
                    except (TypeError, ValueError):
                        continue
                    if jersey_number in ALLOWED_JERSEY_NUMBERS and float(score) > best_score:
                        best_jersey = jersey_number
                        best_score = float(score)
                if best_jersey is not None:
                    jersey_counts[best_jersey] += 1

        result: Dict[str, object] = {}
        for jersey_number, count in sorted(jersey_counts.items()):
            ratio = count / total_detections
            if count >= const.MIN_JERSEY_DETECTIONS and ratio >= const.MIN_JERSEY_RATIO:
                result[f"{jersey_number:02d}"] = {
                    "count": count,
                    "ratio": float(ratio),
                }

        return result

    def _build_player_evidence(self, fragment: ScoredFragment) -> Dict[str, float]:
        track_fragments = self._fragments_by_track.get(fragment.original_track_id, [])

        previous_exists = False
        next_exists = False
        for index, track_fragment in enumerate(track_fragments):
            if track_fragment.fragment_id != fragment.fragment_id:
                continue
            previous_exists = index > 0
            next_exists = index < len(track_fragments) - 1
            break

        return {
            "same_track_previous": 1.0 if previous_exists else 0.0,
            "same_track_next": 1.0 if next_exists else 0.0,
            "is_ghost": 1.0 if getattr(fragment, "is_ghost", False) else 0.0,
            "quality_score": float(max(0.0, min(1.0, getattr(fragment, "quality_score", 0.0)))),
        }

    @staticmethod
    def _select_candidate_jersey(jersey_evidence: Dict[str, object]) -> Optional[int]:
        if not jersey_evidence:
            return None

        # Select jersey with highest detection count
        best_key = max(
            jersey_evidence,
            key=lambda k: jersey_evidence[k].get("count", 0) if isinstance(jersey_evidence[k], dict) else 0,
        )

        try:
            jersey_number = int(best_key)
        except (ValueError, TypeError):
            return None

        return jersey_number if jersey_number in ALLOWED_JERSEY_NUMBERS else None

    @staticmethod
    def _select_candidate_team(team_evidence: Dict[str, float]) -> Optional[TeamID]:
        team_a = team_evidence.get(TeamID.TEAM_A.value)
        team_b = team_evidence.get(TeamID.TEAM_B.value)

        if team_a is None or team_b is None:
            return None
        if team_a > team_b:
            return TeamID.TEAM_A
        if team_b > team_a:
            return TeamID.TEAM_B
        return None


def run_pass3a(
    input_dir: Path,
    output_dir: Optional[Path] = None,
) -> Pass3AOutput:
    """Execute Pass 3A candidate generation with fail-fast validation."""
    if output_dir is None:
        output_dir = input_dir

    pass2c_path = input_dir / const.PASS2_GHOSTS_JSON
    pass1_path = input_dir / const.PASS1_RAW_JSON
    output_path = output_dir / const.PASS3_CANDIDATES_JSON
    validation_path = output_dir / const.PASS3_VALIDATION_JSON

    if not pass2c_path.exists():
        raise FileNotFoundError(f"Pass 2C output not found: {pass2c_path}")

    pass2c_output = load_json(str(pass2c_path), Pass2COutput)

    pass1_output: Optional[Pass1Output] = None
    if pass1_path.exists():
        pass1_output = load_json(str(pass1_path), Pass1Output)
    else:
        logger.warning("Pass 1 output not found; Pass 3A will run with reduced evidence")

    generator = Pass3ACandidateGenerator(pass2c_output=pass2c_output, pass1_output=pass1_output)
    pass3a_output = generator.generate()

    violations = validate_pass3a_candidates(pass3a_output)
    errors = [violation for violation in violations if violation.severity == "error"]
    warnings = [violation for violation in violations if violation.severity == "warning"]

    validation_result = ValidationResult(
        passed=len(errors) == 0,
        violations=errors,
        warnings=warnings,
        timestamp=datetime.utcnow().isoformat() + "Z",
        pass_name="pass3a",
    )

    if not validation_result.passed:
        save_json(validation_result.model_dump(), str(validation_path))
        raise ValueError(
            f"Pass 3A validation failed with {len(validation_result.violations)} error(s). "
            f"See {validation_path}"
        )

    save_json(pass3a_output.model_dump(), str(output_path))
    save_json(validation_result.model_dump(), str(validation_path))

    logger.info(
        f"Pass 3A complete: candidates={len(pass3a_output.candidates)}, "
        f"warnings={len(validation_result.warnings)}, output={output_path}"
    )

    return pass3a_output
