"""
Pass 3B: Constraint Graph Construction

Per CLAUDE.md Section 5 (Pass 3B):
- Input: pass3a_candidates.json, pass2_ghosts.json
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
    IdentityCandidateEdge,
    IdentityCandidate,
    Pass2COutput,
    Pass3AEdgesOutput,
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
PASS3B_SOFT_THRESHOLD = float(getattr(const, "PASS3B_SOFT_THRESHOLD", 0.35))
PASS3B_MUST_SAME_THRESHOLD = float(getattr(const, "PASS3B_MUST_SAME_THRESHOLD", 0.85))


class Pass3BConstraintBuilder:
    """Construct identity constraints from Pass 3A candidates and Pass 2C fragments."""

    def __init__(
        self,
        pass3a_edges_output: Pass3AEdgesOutput,
        pass2c_output: Pass2COutput,
        pass3a_legacy_output: Optional[Pass3AOutput] = None,
    ):
        self.pass3a_edges_output = pass3a_edges_output
        self.pass3a_legacy_output = pass3a_legacy_output
        self.pass2c_output = pass2c_output

        self._fragments_by_id: Dict[str, ScoredFragment] = {
            fragment.fragment_id: fragment
            for fragment in pass2c_output.fragments
        }
        self._candidates_by_fragment: Dict[str, IdentityCandidate] = {}
        if pass3a_legacy_output is not None:
            self._candidates_by_fragment = {
                candidate.fragment_id: candidate
                for candidate in pass3a_legacy_output.candidates
            }

        self._edge_by_pair: Dict[Tuple[str, str], IdentityCandidateEdge] = {}
        for edge in pass3a_edges_output.candidates:
            pair = tuple(sorted((edge.fragment_a, edge.fragment_b)))
            existing = self._edge_by_pair.get(pair)
            if existing is None or edge.overall_candidate_score > existing.overall_candidate_score:
                self._edge_by_pair[pair] = edge

        self._constraints: List[Constraint] = []
        self._constraint_counter = 0

    def build(self) -> Pass3BOutput:
        self._add_must_same_track_adjacency()
        self._add_must_same_ghost_continuity()
        self._add_cannot_same_jersey_conflicts()
        self._add_must_same_edge_constraints()
        self._add_soft_same_edge_constraints()
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

    def _has_constraint(self, constraint_type: ConstraintType, fragment_ids: Tuple[str, str]) -> bool:
        normalized_pair = tuple(sorted(fragment_ids))
        for existing in self._constraints:
            if existing.constraint_type != constraint_type:
                continue
            if tuple(sorted(existing.fragment_ids)) == normalized_pair:
                return True
        return False

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

    def _add_must_same_edge_constraints(self) -> None:
        all_fragment_ids = [fragment.fragment_id for fragment in self.pass2c_output.fragments]
        parent: Dict[str, str] = {fragment_id: fragment_id for fragment_id in all_fragment_ids}
        members: Dict[str, set] = {fragment_id: {fragment_id} for fragment_id in all_fragment_ids}

        def find(fragment_id: str) -> str:
            root = fragment_id
            while parent[root] != root:
                root = parent[root]
            while parent[fragment_id] != fragment_id:
                next_id = parent[fragment_id]
                parent[fragment_id] = root
                fragment_id = next_id
            return root

        def union(fragment_a: str, fragment_b: str) -> None:
            root_a = find(fragment_a)
            root_b = find(fragment_b)
            if root_a == root_b:
                return
            if len(members[root_a]) < len(members[root_b]):
                root_a, root_b = root_b, root_a
            parent[root_b] = root_a
            members[root_a].update(members[root_b])
            members.pop(root_b, None)

        cannot_pairs = {
            tuple(sorted((constraint.fragment_ids[0], constraint.fragment_ids[1])))
            for constraint in self._constraints
            if constraint.constraint_type == ConstraintType.CANNOT_SAME and len(constraint.fragment_ids) >= 2
        }

        for constraint in self._constraints:
            if constraint.constraint_type != ConstraintType.MUST_SAME or len(constraint.fragment_ids) < 2:
                continue
            left, right = constraint.fragment_ids[0], constraint.fragment_ids[1]
            if left in parent and right in parent:
                union(left, right)

        sorted_edges = sorted(
            self._edge_by_pair.items(),
            key=lambda item: item[1].overall_candidate_score,
            reverse=True,
        )

        for pair, edge in sorted_edges:
            if edge.overall_candidate_score < PASS3B_MUST_SAME_THRESHOLD:
                continue
            if self._has_constraint(ConstraintType.CANNOT_SAME, pair):
                continue

            left, right = pair
            if left not in parent or right not in parent:
                continue

            root_left = find(left)
            root_right = find(right)
            if root_left == root_right:
                continue

            has_transitive_conflict = any(
                tuple(sorted((fragment_left, fragment_right))) in cannot_pairs
                for fragment_left in members[root_left]
                for fragment_right in members[root_right]
            )
            if has_transitive_conflict:
                continue

            self._add_constraint(
                constraint_type=ConstraintType.MUST_SAME,
                fragment_ids=pair,
                reason=(
                    f"Pass 3A edge score >= MUST threshold "
                    f"({edge.overall_candidate_score:.3f} >= {PASS3B_MUST_SAME_THRESHOLD:.3f})"
                ),
                value={
                    "kind": "pass3a_edge_must",
                    "overall_candidate_score": float(edge.overall_candidate_score),
                    "temporal_gap": int(edge.temporal_gap),
                    "spatial_distance": float(edge.spatial_distance),
                },
            )
            union(left, right)

    def _add_soft_same_edge_constraints(self) -> None:
        for pair, edge in self._edge_by_pair.items():
            score = float(edge.overall_candidate_score)
            if score < PASS3B_SOFT_THRESHOLD or score >= PASS3B_MUST_SAME_THRESHOLD:
                continue
            if self._has_constraint(ConstraintType.CANNOT_SAME, pair):
                continue

            self._add_constraint(
                constraint_type=ConstraintType.SOFT_SAME,
                fragment_ids=pair,
                reason=(
                    f"Pass 3A edge score in SOFT range "
                    f"({score:.3f} in [{PASS3B_SOFT_THRESHOLD:.3f}, {PASS3B_MUST_SAME_THRESHOLD:.3f}))"
                ),
                value={
                    "kind": "pass3a_edge_soft",
                    "overall_candidate_score": score,
                    "temporal_gap": int(edge.temporal_gap),
                    "spatial_distance": float(edge.spatial_distance),
                },
                weight=max(0.1, score),
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

    pass3a_edges_path = input_dir / const.PASS3A_CANDIDATES_JSON
    pass3a_legacy_path = input_dir / const.PASS3_CANDIDATES_JSON
    pass2c_path = input_dir / const.PASS2_GHOSTS_JSON
    output_path = output_dir / const.PASS3_CONSTRAINTS_JSON
    validation_path = output_dir / const.PASS3_VALIDATION_JSON

    if not pass3a_edges_path.exists():
        raise FileNotFoundError(f"Pass 3A edge output not found: {pass3a_edges_path}")
    if not pass2c_path.exists():
        raise FileNotFoundError(f"Pass 2C output not found: {pass2c_path}")

    pass3a_edges_output = load_json(str(pass3a_edges_path), Pass3AEdgesOutput)
    pass3a_legacy_output: Optional[Pass3AOutput] = None
    if pass3a_legacy_path.exists():
        pass3a_legacy_output = load_json(str(pass3a_legacy_path), Pass3AOutput)
    pass2c_output = load_json(str(pass2c_path), Pass2COutput)

    builder = Pass3BConstraintBuilder(
        pass3a_edges_output=pass3a_edges_output,
        pass2c_output=pass2c_output,
        pass3a_legacy_output=pass3a_legacy_output,
    )
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
