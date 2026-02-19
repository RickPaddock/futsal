"""
Pass 3B: Constraint Graph Construction

Per CLAUDE.md Section 5 (Pass 3B):
- Input: pass3_candidates.json, pass2_ghosts.json
- Output: pass3_constraints.json
- Build MUST_SAME / CANNOT_SAME / SOFT_SAME constraints only
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core import constants as const
from ..core.data_models import (
    Constraint,
    IdentityCandidate,
    Pass2COutput,
    Pass3AOutput,
    Pass3BOutput,
    ScoredFragment,
    ValidationResult,
)
from ..core.types import ConstraintType
from ..utils.file_utils import load_json, save_json
from ..utils.logging_utils import get_logger
from ..validation.pass3_rules import validate_pass3b_constraints

logger = get_logger("pass3b_constraint_builder")


class Pass3BConstraintBuilder:
    """Construct identity constraints from Pass 3A candidates and Pass 2C fragments."""

    def __init__(self, pass3a_output: Pass3AOutput, pass2c_output: Pass2COutput):
        self.pass3a_output = pass3a_output
        self.pass2c_output = pass2c_output

        self._fragments_by_id: Dict[str, ScoredFragment] = {
            fragment.fragment_id: fragment
            for fragment in pass2c_output.fragments
        }
        self._candidates_by_fragment: Dict[str, IdentityCandidate] = {
            candidate.fragment_id: candidate
            for candidate in pass3a_output.candidates
        }

        self._constraints: List[Constraint] = []
        self._constraint_counter = 0

    def build(self) -> Pass3BOutput:
        self._add_must_same_track_adjacency()
        self._add_must_same_ghost_continuity()
        self._add_cannot_same_jersey_conflicts()
        self._add_soft_same_track_preferences()

        graph = self._build_constraint_graph()
        return Pass3BOutput(constraints=self._constraints, constraint_graph=graph)

    def _next_constraint_id(self) -> str:
        self._constraint_counter += 1
        return f"C{self._constraint_counter:06d}"

    def _add_constraint(
        self,
        constraint_type: ConstraintType,
        fragment_ids: Tuple[str, str],
        reason: str,
        value: Optional[object] = None,
        weight: float = 1.0,
    ) -> None:
        frag_a, frag_b = fragment_ids
        if frag_a == frag_b:
            return

        normalized_pair = tuple(sorted((frag_a, frag_b)))
        for existing in self._constraints:
            if existing.constraint_type != constraint_type:
                continue
            if tuple(sorted(existing.fragment_ids)) == normalized_pair:
                return

        self._constraints.append(
            Constraint(
                constraint_id=self._next_constraint_id(),
                constraint_type=constraint_type,
                fragment_ids=list(normalized_pair),
                value=value,
                weight=weight,
                reason=reason,
            )
        )

    def _add_must_same_track_adjacency(self) -> None:
        hard_discontinuity_rules = {
            "TRACK_COLLISION",
            "JERSEY_CHANGE",
            "JERSEY_TEMPORAL_EXCLUSIVITY",
            "HARD_APPEARANCE_DISCONTINUITY",
        }

        fragments_by_track: Dict[int, List[ScoredFragment]] = defaultdict(list)
        for fragment in self.pass2c_output.fragments:
            fragments_by_track[fragment.original_track_id].append(fragment)

        for track_id, fragments in fragments_by_track.items():
            ordered = sorted(fragments, key=lambda fragment: (fragment.start_frame, fragment.end_frame))
            for index in range(len(ordered) - 1):
                left = ordered[index]
                right = ordered[index + 1]

                split_rule = (right.split_rule_id or "").upper()
                if split_rule in hard_discontinuity_rules:
                    self._add_constraint(
                        constraint_type=ConstraintType.CANNOT_SAME,
                        fragment_ids=(left.fragment_id, right.fragment_id),
                        reason=(
                            f"Hard discontinuity split on track {track_id}: {split_rule}"
                        ),
                        value={"kind": "split_discontinuity", "split_rule_id": split_rule, "track_id": track_id},
                    )
                    continue

                self._add_constraint(
                    constraint_type=ConstraintType.MUST_SAME,
                    fragment_ids=(left.fragment_id, right.fragment_id),
                    reason=f"Track adjacency continuity on track {track_id}",
                    value={"kind": "track_adjacency", "track_id": track_id},
                )

    def _add_must_same_ghost_continuity(self) -> None:
        fragments_by_track: Dict[int, List[ScoredFragment]] = defaultdict(list)
        for fragment in self.pass2c_output.fragments:
            fragments_by_track[fragment.original_track_id].append(fragment)

        for track_id, fragments in fragments_by_track.items():
            ghosts = [fragment for fragment in fragments if getattr(fragment, "is_ghost", False)]
            reals = [fragment for fragment in fragments if not getattr(fragment, "is_ghost", False)]
            if not ghosts or not reals:
                continue

            reals_sorted = sorted(reals, key=lambda fragment: (fragment.start_frame, fragment.end_frame))
            for ghost in ghosts:
                anchor = self._nearest_fragment(ghost, reals_sorted)
                if anchor is None:
                    continue
                self._add_constraint(
                    constraint_type=ConstraintType.MUST_SAME,
                    fragment_ids=(ghost.fragment_id, anchor.fragment_id),
                    reason=f"Ghost continuity on track {track_id}",
                    value={"kind": "ghost_continuity", "track_id": track_id},
                )

    def _add_cannot_same_jersey_conflicts(self) -> None:
        for left, right in combinations(self.pass2c_output.fragments, 2):
            if not self._time_overlaps(left, right):
                continue

            jersey_left = self._candidate_jersey(left.fragment_id)
            jersey_right = self._candidate_jersey(right.fragment_id)
            if jersey_left is None or jersey_right is None:
                continue
            if jersey_left != jersey_right:
                continue

            self._add_constraint(
                constraint_type=ConstraintType.CANNOT_SAME,
                fragment_ids=(left.fragment_id, right.fragment_id),
                reason=(
                    f"Temporal jersey exclusivity conflict: jersey {jersey_left} overlaps in time"
                ),
                value={"jersey": jersey_left},
            )

    def _add_soft_same_track_preferences(self) -> None:
        fragments_by_track: Dict[int, List[ScoredFragment]] = defaultdict(list)
        for fragment in self.pass2c_output.fragments:
            fragments_by_track[fragment.original_track_id].append(fragment)

        for track_id, fragments in fragments_by_track.items():
            ordered = sorted(fragments, key=lambda fragment: (fragment.start_frame, fragment.end_frame))
            for index in range(len(ordered) - 1):
                left = ordered[index]
                right = ordered[index + 1]
                frame_gap = max(0, right.start_frame - left.end_frame - 1)
                weight = max(0.1, 1.0 / (1.0 + frame_gap / 10.0))
                self._add_constraint(
                    constraint_type=ConstraintType.SOFT_SAME,
                    fragment_ids=(left.fragment_id, right.fragment_id),
                    reason=f"Track continuity preference on track {track_id} (gap={frame_gap})",
                    weight=weight,
                )

    def _build_constraint_graph(self) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = {
            fragment.fragment_id: []
            for fragment in self.pass2c_output.fragments
        }

        for constraint in self._constraints:
            for fragment_id in constraint.fragment_ids:
                if fragment_id not in graph:
                    graph[fragment_id] = []
                graph[fragment_id].append(constraint.constraint_id)

        for fragment_id in graph:
            graph[fragment_id].sort()

        return graph

    def _candidate_jersey(self, fragment_id: str) -> Optional[int]:
        candidate = self._candidates_by_fragment.get(fragment_id)
        if candidate is None:
            return None
        return candidate.candidate_jersey

    @staticmethod
    def _time_overlaps(left: ScoredFragment, right: ScoredFragment) -> bool:
        return not (left.end_frame < right.start_frame or right.end_frame < left.start_frame)

    @staticmethod
    def _nearest_fragment(target: ScoredFragment, candidates: List[ScoredFragment]) -> Optional[ScoredFragment]:
        best: Optional[ScoredFragment] = None
        best_distance: Optional[int] = None

        for candidate in candidates:
            if candidate.end_frame < target.start_frame:
                distance = target.start_frame - candidate.end_frame
            elif target.end_frame < candidate.start_frame:
                distance = candidate.start_frame - target.end_frame
            else:
                distance = 0

            if best is None or distance < best_distance:
                best = candidate
                best_distance = distance

        return best


def run_pass3b(
    input_dir: Path,
    output_dir: Optional[Path] = None,
) -> Pass3BOutput:
    """Execute Pass 3B constraint graph construction with fail-fast validation."""
    if output_dir is None:
        output_dir = input_dir

    pass3a_path = input_dir / const.PASS3_CANDIDATES_JSON
    pass2c_path = input_dir / const.PASS2_GHOSTS_JSON
    output_path = output_dir / const.PASS3_CONSTRAINTS_JSON
    validation_path = output_dir / const.PASS3_VALIDATION_JSON

    if not pass3a_path.exists():
        raise FileNotFoundError(f"Pass 3A output not found: {pass3a_path}")
    if not pass2c_path.exists():
        raise FileNotFoundError(f"Pass 2C output not found: {pass2c_path}")

    pass3a_output = load_json(str(pass3a_path), Pass3AOutput)
    pass2c_output = load_json(str(pass2c_path), Pass2COutput)

    builder = Pass3BConstraintBuilder(pass3a_output=pass3a_output, pass2c_output=pass2c_output)
    pass3b_output = builder.build()

    violations = validate_pass3b_constraints(pass3b_output)
    errors = [violation for violation in violations if violation.severity == "error"]
    warnings = [violation for violation in violations if violation.severity == "warning"]

    validation_result = ValidationResult(
        passed=len(errors) == 0,
        violations=errors,
        warnings=warnings,
        timestamp=datetime.utcnow().isoformat() + "Z",
        pass_name="pass3b",
    )

    if not validation_result.passed:
        save_json(validation_result.model_dump(), str(validation_path))
        raise ValueError(
            f"Pass 3B validation failed with {len(validation_result.violations)} error(s). "
            f"See {validation_path}"
        )

    save_json(pass3b_output.model_dump(), str(output_path))
    save_json(validation_result.model_dump(), str(validation_path))

    logger.info(
        f"Pass 3B complete: constraints={len(pass3b_output.constraints)}, "
        f"warnings={len(validation_result.warnings)}, output={output_path}"
    )

    return pass3b_output
