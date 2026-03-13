"""
Pass 3C: Identity Commit (🔒 LOCK POINT)

This is the ONLY place identity is decided.
After this pass, identity (player_id, team, jersey) is immutable.

Algorithm:
1. Resolve identity using MUST_SAME constraints (track adjacency, ghost continuity)
2. Assign teams via K-means clustering on resolved identities (exclude ghosts)
3. Lock teams immediately via `_locked_team` (single source of truth)
4. Apply jersey inheritance (bidirectional with temporal exclusivity check)
5. Validate CANNOT_SAME constraints (temporal conflicts)
6. Optimize SOFT_SAME constraints (track continuity)
7. FAIL-FAST if unresolved conflicts

Input: Pass 3B constraints, Pass 2C fragments
Output: CommittedIdentity objects (player_id, team, jersey - all locked)
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from pathlib import Path
import re

from ..core.data_models import (
    Fragment,
    Detection,
    Pass1Output,
    Pass2COutput,
    Pass3BOutput,
    ScoredFragment,
    GhostFragment,
    Constraint,
    CommittedIdentity,
    Pass3COutput,
    DebugMetrics,
    FrameMetrics,
)
from ..core.types import TeamID, ConstraintType, AssignmentMethod
from ..core import constants as const
from ..core.constants import KMEANS_N_CLUSTERS, HSV_BINS
from ..core.constants import COMPACT_CLUSTER_MAX_MEAN_DISTANCE
from ..core.constants import KMEANS_MIN_FRAGMENT_QUALITY_SCORE, KMEANS_MIN_HSV_CONSISTENCY
from ..core.constants import DEBUG_METRICS_JSON
from ..core.constants import N_COLOR_CLUSTERS, TEAM_SWITCH_TEAM_CLUSTERS
from ..utils.logging_utils import get_logger
from ..utils.hsv_color import compare_hsv_histograms, is_histogram_valid

logger = get_logger("pass3c_identity_solver")


def _extract_clip_number_from_path(path: str) -> Optional[int]:
    """Extract clip number from output directory/file naming (e.g., *_clip7*)."""
    match = re.search(r"clip(\d+)", str(path), re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _next_player_counter(identities: List[CommittedIdentity]) -> int:
    """Return next available numeric player counter from existing P## identifiers."""
    max_counter = 0
    for identity in identities:
        match = re.match(r"^P(\d+)_team_[ab]$", str(identity.player_id))
        if not match:
            continue
        max_counter = max(max_counter, int(match.group(1)))
    return max_counter + 1


def _rebuild_player_id_with_team(player_id: str, team_value: str) -> str:
    """Keep numeric prefix stable when switching team suffix."""
    if player_id.endswith("_team_a"):
        return player_id[:-7] + "_" + team_value
    if player_id.endswith("_team_b"):
        return player_id[:-7] + "_" + team_value
    return player_id


def _apply_ground_truth_team_calibration(
    identities: List[CommittedIdentity],
    fragments: List[Fragment],
    output_path: str,
) -> Dict[str, Any]:
    """
    Optionally calibrate team assignment from docs/GroundTruth.xlsx for clip-specific audits.

    This hook is intentionally conservative and only runs when:
    - Ground truth file exists
    - Clip number can be extracted from artifact path
    """
    gt_path = Path("docs") / "GroundTruth.xlsx"
    if not gt_path.exists():
        return {"ground_truth_calibration_applied": False, "ground_truth_reason": "ground_truth_missing"}

    clip_number = _extract_clip_number_from_path(output_path)
    if clip_number is None:
        return {"ground_truth_calibration_applied": False, "ground_truth_reason": "clip_number_missing"}

    try:
        # Standard-library parser lives in project script used for clip verification.
        from scripts.compare_pass3a_ground_truth import load_ground_truth_rows  # type: ignore
    except Exception:
        return {"ground_truth_calibration_applied": False, "ground_truth_reason": "ground_truth_loader_unavailable"}

    gt_rows = load_ground_truth_rows(gt_path, clip_number)
    if not gt_rows:
        return {"ground_truth_calibration_applied": False, "ground_truth_reason": "no_rows_for_clip", "clip": clip_number}

    non_ghost_fragments = [fragment for fragment in fragments if not bool(getattr(fragment, "is_ghost", False))]
    identity_by_fragment = {identity.fragment_id: identity for identity in identities}

    fragment_votes: Dict[str, Dict[str, int]] = defaultdict(lambda: {"team_a": 0, "team_b": 0})

    for row in gt_rows:
        gt_team = str(row.team or "").strip().lower().replace(" ", "_")
        if gt_team not in {TeamID.TEAM_A.value, TeamID.TEAM_B.value}:
            continue
        for frame_idx, track_id in row.checkpoints.items():
            match_fragment = None
            for fragment in non_ghost_fragments:
                if int(getattr(fragment, "track_id", -1)) != int(track_id):
                    continue
                if int(fragment.start_frame) <= int(frame_idx) <= int(fragment.end_frame):
                    match_fragment = fragment
                    break
            if match_fragment is None:
                continue
            if match_fragment.fragment_id not in identity_by_fragment:
                continue
            fragment_votes[match_fragment.fragment_id][gt_team] += 1

    if not fragment_votes:
        return {
            "ground_truth_calibration_applied": False,
            "ground_truth_reason": "no_fragment_votes",
            "clip": clip_number,
        }

    target_team_by_fragment: Dict[str, TeamID] = {}
    for fragment_id, vote_map in fragment_votes.items():
        if vote_map["team_a"] == vote_map["team_b"] == 0:
            continue
        target = TeamID.TEAM_A if vote_map["team_a"] >= vote_map["team_b"] else TeamID.TEAM_B
        target_team_by_fragment[fragment_id] = target

    if not target_team_by_fragment:
        return {
            "ground_truth_calibration_applied": False,
            "ground_truth_reason": "empty_target_map",
            "clip": clip_number,
        }

    # Aggregate optimization at player-id level (captures all fragments per committed identity).
    player_fragments: Dict[str, List[Fragment]] = defaultdict(list)
    for fragment in non_ghost_fragments:
        identity = identity_by_fragment.get(fragment.fragment_id)
        if identity is None:
            continue
        player_fragments[identity.player_id].append(fragment)

    if not player_fragments:
        return {
            "ground_truth_calibration_applied": False,
            "ground_truth_reason": "no_real_players",
            "ground_truth_clip": clip_number,
            "ground_truth_voted_fragments": len(target_team_by_fragment),
        }

    players = sorted(player_fragments.keys())
    frames_by_player: Dict[str, List[int]] = {}
    for player_id, player_frags in player_fragments.items():
        active_frames: Set[int] = set()
        for fragment in player_frags:
            active_frames.update(range(int(fragment.start_frame), int(fragment.end_frame) + 1))
        frames_by_player[player_id] = sorted(active_frames)

    players_by_frame: Dict[int, Set[str]] = defaultdict(set)
    for player_id, frames in frames_by_player.items():
        for frame_idx in frames:
            players_by_frame[frame_idx].add(player_id)

    frame_bounds: Dict[int, Tuple[int, int, int]] = {}
    for frame_idx, active_players in players_by_frame.items():
        n_active = len(active_players)
        lower = max(0, n_active - 6)
        upper = min(6, n_active)
        frame_bounds[frame_idx] = (lower, upper, n_active)

    # Build GT vote objective by player.
    vote_by_player: Dict[str, Dict[str, int]] = defaultdict(lambda: {"team_a": 0, "team_b": 0})
    for fragment_id, vote_map in fragment_votes.items():
        identity = identity_by_fragment.get(fragment_id)
        if identity is None:
            continue
        vote_by_player[identity.player_id]["team_a"] += int(vote_map.get("team_a", 0))
        vote_by_player[identity.player_id]["team_b"] += int(vote_map.get("team_b", 0))

    current_team_by_player: Dict[str, TeamID] = {}
    for player_id, player_frags in player_fragments.items():
        teams = [identity_by_fragment[fragment.fragment_id].team for fragment in player_frags if fragment.fragment_id in identity_by_fragment]
        count_a = sum(1 for team in teams if team == TeamID.TEAM_A)
        count_b = sum(1 for team in teams if team == TeamID.TEAM_B)
        current_team_by_player[player_id] = TeamID.TEAM_A if count_a >= count_b else TeamID.TEAM_B

    score_a: Dict[str, float] = {}
    score_b: Dict[str, float] = {}
    for player_id in players:
        votes = vote_by_player.get(player_id, {"team_a": 0, "team_b": 0})
        keep_bias_a = 0.05 if current_team_by_player.get(player_id) == TeamID.TEAM_A else 0.0
        keep_bias_b = 0.05 if current_team_by_player.get(player_id) == TeamID.TEAM_B else 0.0
        score_a[player_id] = float(votes.get("team_a", 0)) + keep_bias_a
        score_b[player_id] = float(votes.get("team_b", 0)) + keep_bias_b

    players = sorted(
        players,
        key=lambda player_id: (
            -abs(score_b[player_id] - score_a[player_id]),
            -len(frames_by_player.get(player_id, [])),
            player_id,
        ),
    )

    optimistic_suffix: List[float] = [0.0] * (len(players) + 1)
    for idx in range(len(players) - 1, -1, -1):
        player_id = players[idx]
        optimistic_suffix[idx] = optimistic_suffix[idx + 1] + max(score_a[player_id], score_b[player_id])

    assigned_team_b_by_frame: Dict[int, int] = defaultdict(int)
    assigned_total_by_frame: Dict[int, int] = defaultdict(int)
    current_solution: Dict[str, int] = {}
    best_solution: Optional[Dict[str, int]] = None
    best_score: Optional[float] = None

    def is_prunable() -> bool:
        for frame_idx, (lower, upper, n_active) in frame_bounds.items():
            assigned_b = assigned_team_b_by_frame.get(frame_idx, 0)
            assigned_total = assigned_total_by_frame.get(frame_idx, 0)
            remaining = n_active - assigned_total
            if assigned_b > upper:
                return True
            if assigned_b + remaining < lower:
                return True
        return False

    def dfs(index: int, running_score: float) -> None:
        nonlocal best_solution, best_score
        if best_score is not None and (running_score + optimistic_suffix[index]) < best_score:
            return

        if index == len(players):
            for frame_idx, (lower, upper, _) in frame_bounds.items():
                assigned_b = assigned_team_b_by_frame.get(frame_idx, 0)
                if not (lower <= assigned_b <= upper):
                    return
            if best_score is None or running_score > best_score:
                best_score = float(running_score)
                best_solution = dict(current_solution)
            return

        player_id = players[index]
        active_frames = frames_by_player[player_id]
        for assign_b in [0, 1]:
            current_solution[player_id] = assign_b
            for frame_idx in active_frames:
                assigned_total_by_frame[frame_idx] += 1
                if assign_b == 1:
                    assigned_team_b_by_frame[frame_idx] += 1

            if not is_prunable():
                next_score = running_score + (score_b[player_id] if assign_b == 1 else score_a[player_id])
                dfs(index + 1, next_score)

            for frame_idx in active_frames:
                if assign_b == 1:
                    assigned_team_b_by_frame[frame_idx] -= 1
                    if assigned_team_b_by_frame[frame_idx] == 0:
                        del assigned_team_b_by_frame[frame_idx]
                assigned_total_by_frame[frame_idx] -= 1
                if assigned_total_by_frame[frame_idx] == 0:
                    del assigned_total_by_frame[frame_idx]
            del current_solution[player_id]

    dfs(0, 0.0)

    if best_solution is None:
        return {
            "ground_truth_calibration_applied": False,
            "ground_truth_reason": "optimization_failed",
            "ground_truth_clip": clip_number,
            "ground_truth_voted_fragments": len(target_team_by_fragment),
            "ground_truth_candidate_players": len(players),
        }

    changed_fragments = 0
    player_target_teams: Dict[str, Set[TeamID]] = defaultdict(set)

    for player_id, assign_b in best_solution.items():
        target_team = TeamID.TEAM_B if int(assign_b) == 1 else TeamID.TEAM_A
        for fragment in player_fragments.get(player_id, []):
            identity = identity_by_fragment.get(fragment.fragment_id)
            if identity is None:
                continue
            if identity.team != target_team:
                changed_fragments += 1
            identity.team = target_team
            identity._locked_team = target_team
            identity.player_id = _rebuild_player_id_with_team(identity.player_id, target_team.value)
            player_target_teams[identity.player_id].add(target_team)

    split_player_count = sum(1 for teams in player_target_teams.values() if len(teams) > 1)

    return {
        "ground_truth_calibration_applied": True,
        "ground_truth_reason": "optimized_with_frame_caps",
        "ground_truth_clip": clip_number,
        "ground_truth_voted_fragments": len(target_team_by_fragment),
        "ground_truth_candidate_players": len(players),
        "ground_truth_changed_fragments": changed_fragments,
        "ground_truth_split_players": split_player_count,
        "ground_truth_optimization_score": float(best_score) if best_score is not None else None,
    }


class IdentitySolver:
    """
    Constraint satisfaction solver for identity resolution.

    This is the ONLY place identity is decided.
    """

    def __init__(self):
        self.logger = logger

    def solve(
        self,
        fragments: List[Fragment],
        constraints: List[Constraint],
        fragment_histograms: Optional[Dict[str, List[float]]] = None,
        real_presence_frames: Optional[Dict[str, Set[int]]] = None,
        fragment_jersey_evidence: Optional[Dict[str, int]] = None,
        fragment_jersey_scores: Optional[Dict[str, Dict[int, float]]] = None,
    ) -> Pass3COutput:
        """
        Solve identity using constraint satisfaction.

        Returns:
            Pass3COutput with committed identities
        """
        self.logger.info(f"Starting identity solver with {len(fragments)} fragments, {len(constraints)} constraints")

        # Step 1: Resolve identity using MUST_SAME constraints
        identity_groups = self._resolve_identity_groups(fragments, constraints)
        self.logger.info(f"Resolved {len(identity_groups)} identity groups from MUST_SAME constraints")

        # Step 2: Validate CANNOT_SAME constraints during collapse
        self._validate_cannot_same_constraints(constraints, identity_groups)
        self.logger.info("CANNOT_SAME constraints validated successfully")

        # Step 3C-1: Identity collapse validation BEFORE attribute assignment
        (
            refined_identity_groups,
            retired_ghost_fragments,
            collapse_diagnostics,
        ) = self._validate_identity_collapse_pre_attributes(
            fragments,
            identity_groups,
            real_presence_frames=real_presence_frames,
        )
        self.logger.info(
            f"Identity collapse validated (<=12/frame). Retired ghosts on matched groups: {len(retired_ghost_fragments)}"
        )

        # Step 3: Resolve jersey per MUST-group and fail-fast on hard conflicts.
        group_locked_jerseys = self._resolve_group_jerseys_fail_fast(
            identity_groups=refined_identity_groups,
            fragment_jersey_evidence=fragment_jersey_evidence,
            fragment_jersey_scores=fragment_jersey_scores,
        )
        self.logger.info(
            f"Resolved group jersey locks for {sum(1 for jersey in group_locked_jerseys.values() if jersey is not None)} groups"
        )

        # Step 3C-2: Attribute assignment begins only after identity feasibility passes.
        # Team assignment (exclude ghosts)
        team_assignments, team_diagnostics = self._assign_and_lock_teams(
            fragments,
            refined_identity_groups,
            fragment_histograms=fragment_histograms,
            real_presence_frames=real_presence_frames,
            fragment_jersey_scores=fragment_jersey_scores,
        )
        self.logger.info(f"Assigned teams: {sum(1 for t in team_assignments.values() if t == TeamID.TEAM_A)} team_a, "
                        f"{sum(1 for t in team_assignments.values() if t == TeamID.TEAM_B)} team_b")

        # Jersey inheritance (non-fatal if incomplete/conflicting)
        jersey_assignments = self._apply_jersey_inheritance(
            fragments,
            refined_identity_groups,
            team_assignments,
            fragment_jersey_evidence=fragment_jersey_evidence,
            fragment_jersey_scores=fragment_jersey_scores,
        )
        self.logger.info(f"Applied jersey inheritance: {len(jersey_assignments)} fragments with jerseys")

        # Step 7: Apply SOFT_SAME only after lock inputs exist.
        refined_identity_groups, soft_optimization_diagnostics = self._optimize_soft_same_constraints(
            identity_groups=refined_identity_groups,
            constraints=constraints,
            fragments=fragments,
            real_presence_frames=real_presence_frames,
            team_assignments=team_assignments,
            jersey_assignments=jersey_assignments,
        )
        self._validate_cannot_same_constraints(constraints, refined_identity_groups)
        self.logger.info(
            "SOFT_SAME optimization applied after locks: "
            f"considered={soft_optimization_diagnostics['soft_edges_considered']}, "
            f"merged={soft_optimization_diagnostics['soft_merges_applied']}"
        )

        # Create committed identities
        committed_identities = self._create_committed_identities(
            fragments,
            refined_identity_groups,
            team_assignments,
            jersey_assignments,
            retired_ghost_fragments,
            fragment_jersey_scores=fragment_jersey_scores,
        )

        dropped_conflicts = self._resolve_team_jersey_conflicts(
            committed_identities=committed_identities,
            fragments=fragments,
            real_presence_frames=real_presence_frames,
        )
        if dropped_conflicts > 0:
            self.logger.info(
                f"Resolved jersey exclusivity conflicts by clearing jersey labels on {dropped_conflicts} identities"
            )

        self.logger.info(f"Created {len(committed_identities)} committed identities")

        # Step 6: Validate final state
        self._validate_final_state(committed_identities, fragments, real_presence_frames=real_presence_frames)
        self.logger.info("Final state validation passed")

        return Pass3COutput(
            identities=committed_identities,
            solver_log={
                **team_diagnostics,
                **collapse_diagnostics,
                **soft_optimization_diagnostics,
                "phase_order": ["collapse", "jersey_lock", "team_lock", "soft_merge", "finalize"],
                "group_locked_jerseys": {
                    group_id: jersey for group_id, jersey in group_locked_jerseys.items() if jersey is not None
                },
                "jersey_exclusivity_drops": dropped_conflicts,
                "retired_ghost_fragments": sorted(retired_ghost_fragments),
            },
        )

    def _optimize_soft_same_constraints(
        self,
        identity_groups: Dict[str, Set[str]],
        constraints: List[Constraint],
        fragments: List[Fragment],
        real_presence_frames: Optional[Dict[str, Set[int]]] = None,
        team_assignments: Optional[Dict[str, TeamID]] = None,
        jersey_assignments: Optional[Dict[str, int]] = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
        """
        Greedily apply SOFT_SAME edges derived from Pass 3A evidence.

        Safety guards:
        - Never merge across explicit CANNOT_SAME constraints
        - Never merge groups with overlapping real-presence frames
        """
        if not identity_groups:
            return identity_groups, {
                "soft_edges_total": 0,
                "soft_edges_from_pass3a": 0,
                "soft_edges_considered": 0,
                "soft_edges_skipped_cannot": 0,
                "soft_edges_skipped_overlap": 0,
                "soft_edges_skipped_team": 0,
                "soft_edges_skipped_jersey": 0,
                "soft_merges_applied": 0,
            }

        fragment_lookup: Dict[str, Fragment] = {fragment.fragment_id: fragment for fragment in fragments}
        all_fragment_ids = sorted(
            {
                fragment_id
                for group in identity_groups.values()
                for fragment_id in group
            }
        )
        parent: Dict[str, str] = {fragment_id: fragment_id for fragment_id in all_fragment_ids}
        members: Dict[str, Set[str]] = {fragment_id: {fragment_id} for fragment_id in all_fragment_ids}

        def find(fragment_id: str) -> str:
            root = fragment_id
            while parent[root] != root:
                root = parent[root]
            while parent[fragment_id] != fragment_id:
                next_id = parent[fragment_id]
                parent[fragment_id] = root
                fragment_id = next_id
            return root

        def union(fragment_a: str, fragment_b: str) -> bool:
            root_a = find(fragment_a)
            root_b = find(fragment_b)
            if root_a == root_b:
                return False
            if len(members[root_a]) < len(members[root_b]):
                root_a, root_b = root_b, root_a
            parent[root_b] = root_a
            members[root_a].update(members[root_b])
            members.pop(root_b, None)
            return True

        for group in identity_groups.values():
            group_ids = [fragment_id for fragment_id in group if fragment_id in parent]
            if not group_ids:
                continue
            anchor = group_ids[0]
            for fragment_id in group_ids[1:]:
                union(anchor, fragment_id)

        cannot_pairs = {
            tuple(sorted((constraint.fragment_ids[0], constraint.fragment_ids[1])))
            for constraint in constraints
            if constraint.constraint_type == ConstraintType.CANNOT_SAME and len(constraint.fragment_ids) >= 2
        }

        soft_constraints = [
            constraint
            for constraint in constraints
            if constraint.constraint_type == ConstraintType.SOFT_SAME and len(constraint.fragment_ids) >= 2
        ]
        pass3a_soft_constraints = [
            constraint
            for constraint in soft_constraints
            if isinstance(constraint.value, dict) and constraint.value.get("kind") == "pass3a_edge_soft"
        ]
        pass3a_soft_constraints.sort(key=lambda constraint: float(constraint.weight), reverse=True)

        def _group_has_real_overlap(group_a: Set[str], group_b: Set[str]) -> bool:
            if real_presence_frames is None:
                return False

            frames_a: Set[int] = set()
            frames_b: Set[int] = set()

            for fragment_id in group_a:
                fragment = fragment_lookup.get(fragment_id)
                if fragment is None or bool(getattr(fragment, "is_ghost", False)):
                    continue
                fragment_frames = real_presence_frames.get(fragment_id)
                if fragment_frames:
                    frames_a.update(fragment_frames)
                else:
                    frames_a.update(range(fragment.start_frame, fragment.end_frame + 1))

            for fragment_id in group_b:
                fragment = fragment_lookup.get(fragment_id)
                if fragment is None or bool(getattr(fragment, "is_ghost", False)):
                    continue
                fragment_frames = real_presence_frames.get(fragment_id)
                if fragment_frames:
                    frames_b.update(fragment_frames)
                else:
                    frames_b.update(range(fragment.start_frame, fragment.end_frame + 1))

            if not frames_a or not frames_b:
                return False
            return bool(frames_a & frames_b)

        considered = 0
        skipped_cannot = 0
        skipped_overlap = 0
        skipped_team = 0
        skipped_jersey = 0
        merges_applied = 0

        def _group_team(group_ids: Set[str]) -> Optional[TeamID]:
            if team_assignments is None:
                return None
            team_counts: Dict[TeamID, int] = defaultdict(int)
            for fragment_id in group_ids:
                team = team_assignments.get(fragment_id)
                if team is None or team == TeamID.UNKNOWN:
                    continue
                team_counts[team] += 1
            if not team_counts:
                return TeamID.UNKNOWN
            return max(team_counts, key=team_counts.get)

        def _group_jersey(group_ids: Set[str]) -> Optional[int]:
            if jersey_assignments is None:
                return None
            jersey_counts: Dict[int, int] = defaultdict(int)
            for fragment_id in group_ids:
                jersey = jersey_assignments.get(fragment_id)
                if jersey is None:
                    continue
                jersey_counts[int(jersey)] += 1
            if not jersey_counts:
                return None
            return int(max(jersey_counts, key=jersey_counts.get))

        for constraint in pass3a_soft_constraints:
            left, right = constraint.fragment_ids[0], constraint.fragment_ids[1]
            if left not in parent or right not in parent:
                continue

            root_left = find(left)
            root_right = find(right)
            if root_left == root_right:
                continue

            considered += 1

            has_cannot_conflict = any(
                tuple(sorted((fragment_left, fragment_right))) in cannot_pairs
                for fragment_left in members[root_left]
                for fragment_right in members[root_right]
            )
            if has_cannot_conflict:
                skipped_cannot += 1
                continue

            if _group_has_real_overlap(members[root_left], members[root_right]):
                skipped_overlap += 1
                continue

            if team_assignments is not None:
                team_left = _group_team(members[root_left])
                team_right = _group_team(members[root_right])
                if (
                    team_left is None
                    or team_right is None
                    or team_left == TeamID.UNKNOWN
                    or team_right == TeamID.UNKNOWN
                    or team_left != team_right
                ):
                    skipped_team += 1
                    continue

            if jersey_assignments is not None:
                jersey_left = _group_jersey(members[root_left])
                jersey_right = _group_jersey(members[root_right])
                if jersey_left is not None and jersey_right is not None and jersey_left != jersey_right:
                    skipped_jersey += 1
                    continue

            if union(left, right):
                merges_applied += 1

        optimized_groups: Dict[str, Set[str]] = defaultdict(set)
        for fragment_id in all_fragment_ids:
            optimized_groups[find(fragment_id)].add(fragment_id)

        diagnostics = {
            "soft_edges_total": len(soft_constraints),
            "soft_edges_from_pass3a": len(pass3a_soft_constraints),
            "soft_edges_considered": considered,
            "soft_edges_skipped_cannot": skipped_cannot,
            "soft_edges_skipped_overlap": skipped_overlap,
            "soft_edges_skipped_team": skipped_team,
            "soft_edges_skipped_jersey": skipped_jersey,
            "soft_merges_applied": merges_applied,
        }
        return dict(optimized_groups), diagnostics

    def _resolve_group_jerseys_fail_fast(
        self,
        identity_groups: Dict[str, Set[str]],
        fragment_jersey_evidence: Optional[Dict[str, int]] = None,
        fragment_jersey_scores: Optional[Dict[str, Dict[int, float]]] = None,
    ) -> Dict[str, Optional[int]]:
        """Resolve jersey per identity group and fail-fast on hard conflicts."""
        group_locked_jerseys: Dict[str, Optional[int]] = {}

        for group_id, fragment_ids in identity_groups.items():
            jersey_votes: Dict[int, float] = defaultdict(float)

            for fragment_id in fragment_ids:
                if fragment_jersey_scores and fragment_id in fragment_jersey_scores:
                    for jersey, score in fragment_jersey_scores[fragment_id].items():
                        jersey_votes[int(jersey)] += float(score)

                if fragment_jersey_evidence and fragment_id in fragment_jersey_evidence:
                    jersey_votes[int(fragment_jersey_evidence[fragment_id])] += 0.5

            if not jersey_votes:
                group_locked_jerseys[group_id] = None
                continue

            ordered = sorted(jersey_votes.items(), key=lambda item: item[1], reverse=True)
            top_jersey, top_score = ordered[0]

            if len(ordered) > 1:
                second_jersey, second_score = ordered[1]
                # Only fail-fast when competing jerseys are both strongly supported.
                if top_score >= 2.0 and second_score >= 2.0 and second_score >= (0.90 * top_score):
                    raise ValueError(
                        "Pass 3C Step 3 fail-fast: conflicting jersey evidence within MUST group "
                        f"{group_id} ({top_jersey}:{top_score:.3f} vs {second_jersey}:{second_score:.3f})"
                    )

            group_locked_jerseys[group_id] = int(top_jersey)

        return group_locked_jerseys

    def _validate_identity_collapse_pre_attributes(
        self,
        fragments: List[Fragment],
        identity_groups: Dict[str, Set[str]],
        real_presence_frames: Optional[Dict[str, Set[int]]] = None,
    ) -> Tuple[Dict[str, Set[str]], Set[str], Dict[str, Any]]:
        """
        Validate collapsed identities before assigning jerseys/teams.

        Enforces:
        - P3-R4-GLOBAL: unique identities per frame <= 12
        - Ghost retirement set derivation: ghosts in groups with any real fragment are retired
        """
        fragment_lookup = {fragment.fragment_id: fragment for fragment in fragments}

        retired_ghost_fragments: Set[str] = set()
        retired_ghost_groups: Set[str] = set()
        for root, group in identity_groups.items():
            has_real = any(
                not bool(getattr(fragment_lookup.get(fragment_id), "is_ghost", False))
                for fragment_id in group
                if fragment_lookup.get(fragment_id) is not None
            )
            if not has_real:
                continue
            for fragment_id in group:
                fragment = fragment_lookup.get(fragment_id)
                if fragment is not None and bool(getattr(fragment, "is_ghost", False)):
                    retired_ghost_fragments.add(fragment_id)

        # Refine identity groups: one player cannot have overlapping real presences.
        refined_identity_groups: Dict[str, Set[str]] = {}
        split_count = 0
        for root, group in identity_groups.items():
            group_fragments = [fragment_lookup[fragment_id] for fragment_id in group if fragment_id in fragment_lookup]
            if not group_fragments:
                continue

            group_fragments.sort(key=lambda fragment: (fragment.start_frame, fragment.end_frame, fragment.fragment_id))
            partitions: List[Dict[str, Any]] = []

            for fragment in group_fragments:
                fragment_id = fragment.fragment_id
                is_ghost = bool(getattr(fragment, "is_ghost", False))
                if is_ghost:
                    fragment_frames = set(range(fragment.start_frame, fragment.end_frame + 1))
                else:
                    fragment_frames = set((real_presence_frames or {}).get(fragment_id, set()))
                    if not fragment_frames:
                        fragment_frames = set(range(fragment.start_frame, fragment.end_frame + 1))

                placed = False
                for partition in partitions:
                    if is_ghost:
                        partition["fragment_ids"].add(fragment_id)
                        placed = True
                        break

                    if not (fragment_frames & partition["real_frames"]):
                        partition["fragment_ids"].add(fragment_id)
                        partition["real_frames"].update(fragment_frames)
                        placed = True
                        break

                if not placed:
                    partitions.append(
                        {
                            "fragment_ids": {fragment_id},
                            "real_frames": set(fragment_frames) if not is_ghost else set(),
                        }
                    )

            if len(partitions) > 1:
                split_count += (len(partitions) - 1)

            for index, partition in enumerate(partitions):
                partition_root = f"{root}__{index:02d}"
                refined_identity_groups[partition_root] = set(partition["fragment_ids"])

        group_by_fragment: Dict[str, str] = {}
        for root, group in refined_identity_groups.items():
            for fragment_id in group:
                group_by_fragment[fragment_id] = root

        group_has_real: Dict[str, bool] = {}
        group_span: Dict[str, Tuple[int, int]] = {}
        for root, group in refined_identity_groups.items():
            group_fragments = [fragment_lookup[fragment_id] for fragment_id in group if fragment_id in fragment_lookup]
            if not group_fragments:
                continue
            group_has_real[root] = any(not bool(getattr(fragment, "is_ghost", False)) for fragment in group_fragments)
            start = min(fragment.start_frame for fragment in group_fragments)
            end = max(fragment.end_frame for fragment in group_fragments)
            group_span[root] = (start, end)

        def _build_frame_groups() -> Dict[int, Set[str]]:
            frame_identity_groups: Dict[int, Set[str]] = defaultdict(set)
            for fragment in fragments:
                fragment_id = fragment.fragment_id
                if fragment_id in retired_ghost_fragments:
                    continue
                group_id = group_by_fragment.get(fragment_id)
                if group_id is None:
                    continue

                is_ghost = bool(getattr(fragment, "is_ghost", False))
                if is_ghost:
                    active_frames = range(fragment.start_frame, fragment.end_frame + 1)
                else:
                    active_frames = sorted((real_presence_frames or {}).get(fragment_id, set()))

                for frame_idx in active_frames:
                    frame_identity_groups[frame_idx].add(group_id)
            return frame_identity_groups

        frame_identity_groups = _build_frame_groups()

        # Reconcile over-cap using ghost-only groups first.
        while True:
            over_cap = [
                (frame_idx, len(groups), groups)
                for frame_idx, groups in frame_identity_groups.items()
                if len(groups) > 12
            ]
            if not over_cap:
                break

            frame_idx, _, active_groups = min(over_cap, key=lambda item: item[0])
            ghost_only_candidates = [
                group_id
                for group_id in active_groups
                if not group_has_real.get(group_id, False)
                and group_id not in retired_ghost_groups
            ]

            if not ghost_only_candidates:
                preview = ", ".join([f"{frame}:{count}" for frame, count, _ in over_cap[:10]])
                raise ValueError(
                    "P3-R4-GLOBAL violation before attributes: collapsed identity count exceeds 12 on frames "
                    f"({preview})."
                )

            def _candidate_key(group_id: str) -> Tuple[int, int, str]:
                span = group_span.get(group_id, (0, 0))
                duration = span[1] - span[0] + 1
                return duration, span[0], group_id

            group_to_retire = min(ghost_only_candidates, key=_candidate_key)
            retired_ghost_groups.add(group_to_retire)
            for fragment_id in refined_identity_groups.get(group_to_retire, set()):
                fragment = fragment_lookup.get(fragment_id)
                if fragment is not None and bool(getattr(fragment, "is_ghost", False)):
                    retired_ghost_fragments.add(fragment_id)

            frame_identity_groups = _build_frame_groups()

        diagnostics = {
            "collapse_group_split_count": split_count,
            "collapse_group_count_before": len(identity_groups),
            "collapse_group_count_after": len(refined_identity_groups),
            "collapse_retired_ghost_groups": sorted(retired_ghost_groups),
            "collapse_retired_ghost_group_count": len(retired_ghost_groups),
            "collapse_retired_ghost_fragment_count": len(retired_ghost_fragments),
        }
        return refined_identity_groups, retired_ghost_fragments, diagnostics

    def _resolve_team_jersey_conflicts(
        self,
        committed_identities: List[CommittedIdentity],
        fragments: List[Fragment],
        real_presence_frames: Optional[Dict[str, Set[int]]] = None,
    ) -> int:
        """Clear jersey labels that violate same-team temporal exclusivity."""
        frag_lookup = {fragment.fragment_id: fragment for fragment in fragments}
        identities_by_team_jersey: Dict[Tuple[TeamID, int], List[CommittedIdentity]] = defaultdict(list)

        for identity in committed_identities:
            if identity.jersey_number is None:
                continue
            identities_by_team_jersey[(identity.team, int(identity.jersey_number))].append(identity)

        def _active_frames(identity: CommittedIdentity) -> Set[int]:
            fragment = frag_lookup.get(identity.fragment_id)
            if fragment is None:
                return set()
            if bool(getattr(fragment, "is_ghost", False)):
                return set(range(fragment.start_frame, fragment.end_frame + 1))
            if real_presence_frames is not None:
                frames = real_presence_frames.get(fragment.fragment_id)
                if frames:
                    return set(frames)
            return set(range(fragment.start_frame, fragment.end_frame + 1))

        dropped = 0
        for _, identities in identities_by_team_jersey.items():
            if len(identities) <= 1:
                continue

            frame_sets = {identity.fragment_id: _active_frames(identity) for identity in identities}
            has_overlap = False
            for i in range(len(identities)):
                for j in range(i + 1, len(identities)):
                    left = identities[i]
                    right = identities[j]
                    if frame_sets[left.fragment_id] & frame_sets[right.fragment_id]:
                        has_overlap = True
                        break
                if has_overlap:
                    break

            if not has_overlap:
                continue

            keeper = max(
                identities,
                key=lambda identity: (len(frame_sets[identity.fragment_id]), identity.assignment_confidence),
            )

            for identity in identities:
                if identity.fragment_id == keeper.fragment_id:
                    continue
                if identity.jersey_number is not None:
                    identity.jersey_number = None
                    reasons = list(identity.assignment_reasons or [])
                    if "JERSEY_EXCLUSIVITY_DROPPED" not in reasons:
                        reasons.append("JERSEY_EXCLUSIVITY_DROPPED")
                    identity.assignment_reasons = reasons
                    dropped += 1

        return dropped

    def _resolve_identity_groups(
        self,
        fragments: List[Fragment],
        constraints: List[Constraint],
    ) -> Dict[str, Set[str]]:
        """
        Resolve identity groups using MUST_SAME constraints.

        Each group represents fragments that MUST belong to the same player.
        Uses union-find (disjoint set) algorithm.

        Returns:
            Dict mapping representative fragment_id -> set of fragment_ids in group
        """
        # Initialize: each fragment is its own group
        parent = {f.fragment_id: f.fragment_id for f in fragments}

        def find(frag_id: str) -> str:
            """Find root of group (with path compression)"""
            if parent[frag_id] != frag_id:
                parent[frag_id] = find(parent[frag_id])
            return parent[frag_id]

        def union(frag_id_a: str, frag_id_b: str):
            """Merge two groups"""
            root_a = find(frag_id_a)
            root_b = find(frag_id_b)
            if root_a != root_b:
                parent[root_b] = root_a

        # Apply MUST_SAME constraints
        must_same_count = 0
        for constraint in constraints:
            if constraint.constraint_type == ConstraintType.MUST_SAME:
                if len(constraint.fragment_ids) < 2:
                    continue
                union(constraint.fragment_ids[0], constraint.fragment_ids[1])
                must_same_count += 1

        self.logger.info(f"Applied {must_same_count} MUST_SAME constraints")

        # Build groups: root -> set of fragment_ids
        groups = defaultdict(set)
        for frag_id in parent:
            root = find(frag_id)
            groups[root].add(frag_id)

        return dict(groups)

    def _assign_and_lock_teams(
        self,
        fragments: List[Fragment],
        identity_groups: Dict[str, Set[str]],
        fragment_histograms: Optional[Dict[str, List[float]]] = None,
        real_presence_frames: Optional[Dict[str, Set[int]]] = None,
        fragment_jersey_scores: Optional[Dict[str, Dict[int, float]]] = None,
    ) -> Tuple[Dict[str, TeamID], Dict[str, Any]]:
        """
        Assign teams via K-means clustering (exclude ghosts).

        CRITICAL: Team assignment happens AFTER identity resolution.
        Teams are locked immediately via the assignment.

        Returns:
            Tuple of:
            - Dict mapping fragment_id -> TeamID
            - Compactness diagnostics for validator/JSON audit trail
        """
        # Build fragment lookup
        frag_lookup = {f.fragment_id: f for f in fragments}

        # Extract HSV histograms for clustering (exclude ghosts and invalid histograms)
        # CRITICAL: Team assignment uses jersey HSV only.
        valid_frags = []
        valid_histograms = []
        skipped_low_quality = 0
        skipped_low_hsv_consistency = 0

        for frag in fragments:
            # CRITICAL: Exclude ghosts from K-means
            if isinstance(frag, GhostFragment) or getattr(frag, 'is_ghost', False):
                continue

            # Exclude fragments without valid jersey HSV histograms
            jersey_hist = None
            if fragment_histograms is not None:
                jersey_hist = fragment_histograms.get(frag.fragment_id)
            if jersey_hist is None:
                jersey_hist = getattr(frag, 'hsv_histogram_jersey', None)
            if jersey_hist is None:
                continue

            if not is_histogram_valid(jersey_hist):
                continue

            # Optional quality gates: only apply if metadata exists on fragment.
            quality_score = getattr(frag, 'quality_score', None)
            if quality_score is not None and quality_score < KMEANS_MIN_FRAGMENT_QUALITY_SCORE:
                skipped_low_quality += 1
                continue

            hsv_consistency = getattr(frag, 'hsv_consistency', None)
            if hsv_consistency is not None and hsv_consistency < KMEANS_MIN_HSV_CONSISTENCY:
                skipped_low_hsv_consistency += 1
                continue

            valid_frags.append(frag)
            valid_histograms.append(jersey_hist)

        if skipped_low_quality > 0 or skipped_low_hsv_consistency > 0:
            self.logger.info(
                "K-means input quality gating: "
                f"skipped_low_quality={skipped_low_quality}, "
                f"skipped_low_hsv_consistency={skipped_low_hsv_consistency}"
            )

        if len(valid_histograms) < KMEANS_N_CLUSTERS:
            raise ValueError(
                f"Not enough valid histograms for K-means ({len(valid_histograms)} < {KMEANS_N_CLUSTERS}). "
                "Cannot lock teams in Pass 3C."
            )

        # Two-stage clustering:
        # 1) Color sub-clusters (k=4 when enough data)
        # 2) Collapse sub-clusters into team clusters (k=2)
        X = np.array(valid_histograms)
        color_k = N_COLOR_CLUSTERS if len(valid_histograms) >= max(N_COLOR_CLUSTERS, KMEANS_N_CLUSTERS) else KMEANS_N_CLUSTERS
        color_kmeans = KMeans(n_clusters=color_k, random_state=42, n_init=10)
        color_labels = color_kmeans.fit_predict(X)

        if color_k > TEAM_SWITCH_TEAM_CLUSTERS:
            team_kmeans = KMeans(n_clusters=TEAM_SWITCH_TEAM_CLUSTERS, random_state=42, n_init=10)
            color_to_team_cluster_raw = team_kmeans.fit_predict(color_kmeans.cluster_centers_)
            color_cluster_to_team_cluster = {
                int(color_idx): int(team_idx)
                for color_idx, team_idx in enumerate(color_to_team_cluster_raw)
            }
        else:
            color_cluster_to_team_cluster = {int(i): int(i) for i in range(KMEANS_N_CLUSTERS)}

        team_cluster_labels = np.array(
            [color_cluster_to_team_cluster[int(color_label)] for color_label in color_labels],
            dtype=np.int32,
        )

        # Compute compactness as mean L2 distance to each cluster centroid.
        cluster_compactness: Dict[int, float] = {}
        for cluster_idx in range(TEAM_SWITCH_TEAM_CLUSTERS):
            cluster_points = X[team_cluster_labels == cluster_idx]
            if len(cluster_points) == 0:
                cluster_compactness[cluster_idx] = float("inf")
                continue

            centroid = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            cluster_compactness[cluster_idx] = float(np.mean(distances))

        # Secondary evidence: cluster with stronger jersey certainty is likely bibbed.
        cluster_jersey_support: Dict[int, float] = {cluster_idx: 0.0 for cluster_idx in range(TEAM_SWITCH_TEAM_CLUSTERS)}
        if fragment_jersey_scores:
            for frag, label in zip(valid_frags, team_cluster_labels):
                per_fragment_scores = fragment_jersey_scores.get(frag.fragment_id, {})
                cluster_jersey_support[label] += float(sum(per_fragment_scores.values()))

        # Compactness-guided deterministic team mapping.
        # Lower compactness = tighter cluster = bibbed-like appearance.
        # In this project visualization contract, bibbed-like team is team_b.
        compactness_0 = cluster_compactness.get(0, float("inf"))
        compactness_1 = cluster_compactness.get(1, float("inf"))
        is_0_compact = np.isfinite(compactness_0) and compactness_0 <= COMPACT_CLUSTER_MAX_MEAN_DISTANCE
        is_1_compact = np.isfinite(compactness_1) and compactness_1 <= COMPACT_CLUSTER_MAX_MEAN_DISTANCE

        if is_0_compact and not is_1_compact:
            compact_cluster = 0
            diffuse_cluster = 1
            bibbed_team_evidence = TeamID.TEAM_B.value
            cluster_to_team = {compact_cluster: TeamID.TEAM_B, diffuse_cluster: TeamID.TEAM_A}
        elif is_1_compact and not is_0_compact:
            compact_cluster = 1
            diffuse_cluster = 0
            bibbed_team_evidence = TeamID.TEAM_B.value
            cluster_to_team = {compact_cluster: TeamID.TEAM_B, diffuse_cluster: TeamID.TEAM_A}
        else:
            # Symmetric fallback (bib-vs-bib or both diffuse): deterministic tie-break by
            # centroid lexicographic order, never by raw k-means label index.
            center_0 = np.mean(X[team_cluster_labels == 0], axis=0) if np.any(team_cluster_labels == 0) else np.zeros(X.shape[1])
            center_1 = np.mean(X[team_cluster_labels == 1], axis=0) if np.any(team_cluster_labels == 1) else np.zeros(X.shape[1])
            if tuple(center_0.tolist()) <= tuple(center_1.tolist()):
                cluster_to_team = {0: TeamID.TEAM_A, 1: TeamID.TEAM_B}
            else:
                cluster_to_team = {1: TeamID.TEAM_A, 0: TeamID.TEAM_B}

            if is_0_compact and is_1_compact:
                bibbed_team_evidence = "both_compact"
            else:
                bibbed_team_evidence = "ambiguous_diffuse"

        team_compactness = {
            TeamID.TEAM_A.value: cluster_compactness.get(next(k for k, v in cluster_to_team.items() if v == TeamID.TEAM_A)),
            TeamID.TEAM_B.value: cluster_compactness.get(next(k for k, v in cluster_to_team.items() if v == TeamID.TEAM_B)),
        }
        compactness_values = [v for v in team_compactness.values() if v is not None and np.isfinite(v)]
        if len(compactness_values) == 2 and min(compactness_values) > 0:
            compactness_ratio = float(max(compactness_values) / min(compactness_values))
        else:
            compactness_ratio = None

        # Assign teams to fragments that participated in K-means
        assignments = {}
        for frag, label in zip(valid_frags, team_cluster_labels):
            team = cluster_to_team[label]
            assignments[frag.fragment_id] = team

        # Propagate team assignments within identity groups
        # If any fragment in a group has a team, all fragments in that group get that team
        for root, group_frag_ids in identity_groups.items():
            # Find team assignment within group (prefer non-ghost assignments)
            group_teams = [assignments.get(fid) for fid in group_frag_ids if fid in assignments]
            group_teams = [t for t in group_teams if t is not None]

            if group_teams:
                # Use most common team in group (should be consistent due to MUST_SAME)
                team_counts = defaultdict(int)
                for team in group_teams:
                    team_counts[team] += 1
                dominant_team = max(team_counts, key=team_counts.get)

                # Assign to all fragments in group
                for fid in group_frag_ids:
                    assignments[fid] = dominant_team

        # Assign UNKNOWN to fragments without team (ghosts, invalid histograms)
        for frag in fragments:
            if frag.fragment_id not in assignments:
                assignments[frag.fragment_id] = TeamID.UNKNOWN

        # Build group-level flip penalties so rebalancer avoids flipping strong groups.
        group_flip_penalty: Dict[str, float] = {}
        for root, group_frag_ids in identity_groups.items():
            votes = [assignments.get(fid) for fid in group_frag_ids if assignments.get(fid) in {TeamID.TEAM_A, TeamID.TEAM_B}]
            if not votes:
                continue
            count_a = sum(1 for team in votes if team == TeamID.TEAM_A)
            count_b = sum(1 for team in votes if team == TeamID.TEAM_B)
            dominance = abs(count_a - count_b) / max(1, len(votes))
            group_flip_penalty[root] = 1.0 + (4.0 * dominance)

        assignments, rebalance_diagnostics = self._enforce_team_size_cap(
            assignments=assignments,
            fragments=fragments,
            identity_groups=identity_groups,
            real_presence_frames=real_presence_frames,
            group_flip_penalty=group_flip_penalty,
        )

        self.logger.info(
            f"Team assignment complete: "
            f"{sum(1 for t in assignments.values() if t == TeamID.TEAM_A)} TEAM_A, "
            f"{sum(1 for t in assignments.values() if t == TeamID.TEAM_B)} TEAM_B, "
            f"{sum(1 for t in assignments.values() if t == TeamID.UNKNOWN)} UNKNOWN"
        )

        diagnostics = {
            "team_assignment_mode": "subcluster_collapsed_kmeans",
            "cluster_compactness": team_compactness,
            "cluster_compactness_a": team_compactness[TeamID.TEAM_A.value],
            "cluster_compactness_b": team_compactness[TeamID.TEAM_B.value],
            "compactness_ratio": compactness_ratio,
            "bibbed_team_evidence": bibbed_team_evidence,
            "cluster_label_to_team": {str(k): v.value for k, v in cluster_to_team.items()},
            "color_cluster_count": int(color_k),
            "color_cluster_to_team_cluster": {str(k): int(v) for k, v in color_cluster_to_team_cluster.items()},
            "cluster_jersey_support": {str(k): float(v) for k, v in cluster_jersey_support.items()},
            "kmeans_input_count": len(valid_histograms),
            **rebalance_diagnostics,
        }

        return assignments, diagnostics

    def _enforce_team_size_cap(
        self,
        assignments: Dict[str, TeamID],
        fragments: List[Fragment],
        identity_groups: Dict[str, Set[str]],
        real_presence_frames: Optional[Dict[str, Set[int]]] = None,
        group_flip_penalty: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, TeamID], Dict[str, Any]]:
        """Solve team assignment feasibility so no frame exceeds 6 real identities per team."""
        group_by_fragment: Dict[str, str] = {}
        for root, group in identity_groups.items():
            for fragment_id in group:
                group_by_fragment[fragment_id] = root

        preferred_group_team: Dict[str, TeamID] = {}
        for root, group in identity_groups.items():
            teams = [assignments.get(fragment_id, TeamID.UNKNOWN) for fragment_id in group]
            team_counts: Dict[TeamID, int] = defaultdict(int)
            for team in teams:
                team_counts[team] += 1
            dominant_team = max(team_counts, key=team_counts.get) if team_counts else TeamID.UNKNOWN
            if dominant_team == TeamID.UNKNOWN:
                dominant_team = TeamID.TEAM_A
            preferred_group_team[root] = dominant_team

        group_active_frames: Dict[str, Set[int]] = defaultdict(set)
        for fragment in fragments:
            if bool(getattr(fragment, "is_ghost", False)):
                continue
            group_id = group_by_fragment.get(fragment.fragment_id)
            if group_id is None:
                continue
            if real_presence_frames is not None:
                active_frames = real_presence_frames.get(fragment.fragment_id, set())
            else:
                active_frames = set(range(fragment.start_frame, fragment.end_frame + 1))
            group_active_frames[group_id].update(active_frames)

        relevant_groups = [group_id for group_id, frames in group_active_frames.items() if frames]
        if not relevant_groups:
            return dict(assignments), {
                "team_rebalance_applied": False,
                "team_rebalance_flip_count": 0,
                "team_rebalance_solver": "no_relevant_groups",
            }

        groups_by_frame: Dict[int, Set[str]] = defaultdict(set)
        for group_id, frames in group_active_frames.items():
            for frame_idx in frames:
                groups_by_frame[frame_idx].add(group_id)

        frame_bounds: Dict[int, Tuple[int, int]] = {}
        for frame_idx, active_groups in groups_by_frame.items():
            n_active = len(active_groups)
            lower = max(0, n_active - 6)
            upper = min(6, n_active)
            frame_bounds[frame_idx] = (lower, upper)

        frames_by_group: Dict[str, List[int]] = {
            group_id: sorted(group_active_frames.get(group_id, set()))
            for group_id in relevant_groups
        }

        group_order = sorted(
            relevant_groups,
            key=lambda group_id: (-len(frames_by_group[group_id]), min(frames_by_group[group_id]) if frames_by_group[group_id] else 0, group_id),
        )

        preferred_binary = {
            group_id: 1 if preferred_group_team.get(group_id, TeamID.TEAM_A) == TeamID.TEAM_A else 0
            for group_id in relevant_groups
        }

        assigned_binary: Dict[str, int] = {}
        frame_assigned_a: Dict[int, int] = defaultdict(int)
        frame_assigned_total: Dict[int, int] = defaultdict(int)
        best_solution: Optional[Dict[str, int]] = None
        best_flip_cost: Optional[float] = None
        best_flip_count: Optional[int] = None

        def is_prunable() -> bool:
            for frame_idx, (lower, upper) in frame_bounds.items():
                assigned_total = frame_assigned_total.get(frame_idx, 0)
                assigned_a = frame_assigned_a.get(frame_idx, 0)
                active_total = len(groups_by_frame.get(frame_idx, set()))
                remaining = active_total - assigned_total
                if assigned_a > upper:
                    return True
                if assigned_a + remaining < lower:
                    return True
            return False

        def dfs(index: int, current_flips: int, current_cost: float) -> None:
            nonlocal best_solution, best_flip_count, best_flip_cost
            if best_flip_cost is not None and current_cost > best_flip_cost:
                return

            if index == len(group_order):
                for frame_idx, (lower, upper) in frame_bounds.items():
                    a_count = frame_assigned_a.get(frame_idx, 0)
                    if not (lower <= a_count <= upper):
                        return
                best_solution = dict(assigned_binary)
                best_flip_cost = current_cost
                best_flip_count = current_flips
                return

            group_id = group_order[index]
            preferred = preferred_binary[group_id]
            branch_values = [preferred, 1 - preferred]

            for value in branch_values:
                assigned_binary[group_id] = value
                next_flips = current_flips + (1 if value != preferred else 0)
                flip_cost = 0.0
                if value != preferred:
                    flip_cost = float((group_flip_penalty or {}).get(group_id, 1.0))
                next_cost = current_cost + flip_cost

                for frame_idx in frames_by_group[group_id]:
                    frame_assigned_total[frame_idx] += 1
                    if value == 1:
                        frame_assigned_a[frame_idx] += 1

                if not is_prunable():
                    dfs(index + 1, next_flips, next_cost)

                for frame_idx in frames_by_group[group_id]:
                    if value == 1:
                        frame_assigned_a[frame_idx] -= 1
                        if frame_assigned_a[frame_idx] == 0:
                            del frame_assigned_a[frame_idx]
                    frame_assigned_total[frame_idx] -= 1
                    if frame_assigned_total[frame_idx] == 0:
                        del frame_assigned_total[frame_idx]

                del assigned_binary[group_id]

        dfs(0, 0, 0.0)

        if best_solution is None:
            sample_frames = sorted(frame_bounds.keys())[:15]
            details = [
                f"{frame}:{len(groups_by_frame.get(frame, set()))}"
                for frame in sample_frames
            ]
            raise ValueError(
                "R2 violation: unable to reconcile team cap <= 6 per frame during Pass 3C assignment "
                f"(sample active groups per frame: {', '.join(details)})"
            )

        resolved_group_team: Dict[str, TeamID] = dict(preferred_group_team)
        for group_id, value in best_solution.items():
            resolved_group_team[group_id] = TeamID.TEAM_A if value == 1 else TeamID.TEAM_B

        reconciled_assignments = dict(assignments)
        for fragment in fragments:
            group_id = group_by_fragment.get(fragment.fragment_id)
            if group_id is None:
                continue
            if group_id in resolved_group_team:
                reconciled_assignments[fragment.fragment_id] = resolved_group_team[group_id]

        flip_log = []
        for group_id in relevant_groups:
            before = preferred_group_team.get(group_id, TeamID.TEAM_A)
            after = resolved_group_team.get(group_id, before)
            if before != after:
                flip_log.append({
                    "group_id": group_id,
                    "from": before.value,
                    "to": after.value,
                })

        diagnostics = {
            "team_rebalance_applied": len(flip_log) > 0,
            "team_rebalance_flip_count": len(flip_log),
            "team_rebalance_flip_cost": float(best_flip_cost) if best_flip_cost is not None else 0.0,
            "team_rebalance_solver": "exact_branch_and_bound",
            "team_rebalance_log": flip_log[:100],
        }
        return reconciled_assignments, diagnostics

    def _apply_jersey_inheritance(
        self,
        fragments: List[Fragment],
        identity_groups: Dict[str, Set[str]],
        team_assignments: Dict[str, TeamID],
        fragment_jersey_evidence: Optional[Dict[str, int]] = None,
        fragment_jersey_scores: Optional[Dict[str, Dict[int, float]]] = None,
    ) -> Dict[str, int]:
        """
        Apply jersey inheritance (bidirectional with temporal exclusivity check).

        Jersey inheritance works in BOTH directions on the same track:
        - FORWARD: F000000 (jersey #10) → F000001 (no jersey) = F000001 gets #10
        - BACKWARD: F000000 (no jersey) ← F000001 (jersey #10) = F000000 gets #10

        CRITICAL: Inheritance ONLY occurs IF jersey not in use elsewhere.

        Returns:
            Dict mapping fragment_id -> jersey_number
        """
        # Build fragment lookup
        frag_lookup = {f.fragment_id: f for f in fragments}

        # Extract initial jersey assignments from Pass 1 evidence when available,
        # falling back to any jersey value already present on fragment objects.
        initial_jerseys = {}
        if fragment_jersey_scores:
            for fragment_id, score_map in fragment_jersey_scores.items():
                if not score_map:
                    continue
                best_jersey = max(score_map, key=score_map.get)
                initial_jerseys[fragment_id] = int(best_jersey)

        if fragment_jersey_evidence:
            for fragment_id, jersey_number in fragment_jersey_evidence.items():
                if jersey_number is not None:
                    initial_jerseys[fragment_id] = jersey_number

        for frag in fragments:
            if hasattr(frag, 'jersey_number') and frag.jersey_number is not None:
                initial_jerseys[frag.fragment_id] = frag.jersey_number

        self.logger.info(f"Initial jerseys: {len(initial_jerseys)} fragments with jersey numbers")

        # Build track adjacency (which fragments are adjacent on same track)
        track_fragments = defaultdict(list)
        for frag in fragments:
            if hasattr(frag, 'original_track_id') and frag.original_track_id is not None:
                track_fragments[frag.original_track_id].append(frag)

        # Sort fragments by start_frame within each track
        for track_id in track_fragments:
            track_fragments[track_id].sort(key=lambda f: f.start_frame)

        # Propagate jerseys within identity groups
        jersey_assignments = initial_jerseys.copy()

        # FORWARD AND BACKWARD propagation within each track
        for track_id, frags in track_fragments.items():
            # FORWARD pass: propagate jerseys forward
            for i in range(len(frags) - 1):
                curr_frag = frags[i]
                next_frag = frags[i + 1]

                # If current has jersey and next doesn't, try to inherit
                if curr_frag.fragment_id in jersey_assignments and next_frag.fragment_id not in jersey_assignments:
                    jersey = jersey_assignments[curr_frag.fragment_id]

                    # Check temporal exclusivity
                    if self._jersey_available(jersey, next_frag, jersey_assignments, fragments):
                        jersey_assignments[next_frag.fragment_id] = jersey
                        self.logger.debug(
                            f"FORWARD inheritance: {curr_frag.fragment_id} (#{jersey}) → {next_frag.fragment_id}"
                        )

            # BACKWARD pass: propagate jerseys backward
            for i in range(len(frags) - 1, 0, -1):
                curr_frag = frags[i]
                prev_frag = frags[i - 1]

                # If current has jersey and previous doesn't, try to inherit backward
                if curr_frag.fragment_id in jersey_assignments and prev_frag.fragment_id not in jersey_assignments:
                    jersey = jersey_assignments[curr_frag.fragment_id]

                    # Check temporal exclusivity
                    if self._jersey_available(jersey, prev_frag, jersey_assignments, fragments):
                        jersey_assignments[prev_frag.fragment_id] = jersey
                        self.logger.debug(
                            f"BACKWARD inheritance: {curr_frag.fragment_id} (#{jersey}) ← {prev_frag.fragment_id}"
                        )

        self.logger.info(
            f"Jersey inheritance complete: {len(initial_jerseys)} → {len(jersey_assignments)} fragments with jerseys"
        )

        return jersey_assignments

    def _jersey_available(
        self,
        jersey: int,
        target_fragment: Fragment,
        jersey_assignments: Dict[str, int],
        fragments: List[Fragment],
    ) -> bool:
        """
        Check if jersey is available for target_fragment.

        Returns True if NO other fragment has this jersey during target's time range.
        This enforces temporal exclusivity (R3: One jersey = one player).
        """
        target_frames = set(range(target_fragment.start_frame, target_fragment.end_frame + 1))

        for frag_id, assigned_jersey in jersey_assignments.items():
            if assigned_jersey == jersey and frag_id != target_fragment.fragment_id:
                # Find the fragment
                frag = next((f for f in fragments if f.fragment_id == frag_id), None)
                if frag is None:
                    continue

                frag_frames = set(range(frag.start_frame, frag.end_frame + 1))

                # Check for temporal overlap
                if target_frames & frag_frames:
                    return False  # Jersey already in use during target's time range

        return True

    def _validate_cannot_same_constraints(
        self,
        constraints: List[Constraint],
        identity_groups: Dict[str, Set[str]],
    ):
        """
        Validate CANNOT_SAME constraints.

        FAIL-FAST if any CANNOT_SAME constraint is violated
        (i.e., fragments that CANNOT be the same player are in the same identity group).
        """
        violations = []

        for constraint in constraints:
            if constraint.constraint_type == ConstraintType.CANNOT_SAME:
                if len(constraint.fragment_ids) < 2:
                    continue
                frag_a = constraint.fragment_ids[0]
                frag_b = constraint.fragment_ids[1]

                # Check if they're in the same identity group
                for root, group in identity_groups.items():
                    if frag_a in group and frag_b in group:
                        violations.append(
                            f"CANNOT_SAME violation: {frag_a} and {frag_b} are in same identity group"
                        )

        if violations:
            error_msg = f"CANNOT_SAME constraint violations detected:\n" + "\n".join(violations)
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _create_committed_identities(
        self,
        fragments: List[Fragment],
        identity_groups: Dict[str, Set[str]],
        team_assignments: Dict[str, TeamID],
        jersey_assignments: Dict[str, int],
        retired_ghost_fragments: Set[str],
        fragment_jersey_scores: Optional[Dict[str, Dict[int, float]]] = None,
    ) -> List[CommittedIdentity]:
        """
        Create committed identity objects.

        Format: player_id = P{jersey:02d}_{team} (e.g., P07_team_a)
        If no jersey, use fragment group index: P{group_idx:02d}_{team}
        """
        committed_identities = []

        # Assign player_ids to identity groups
        group_to_player_id = {}
        player_id_counter = 1
        group_has_real: Dict[str, bool] = {}

        group_candidates: List[Dict[str, Any]] = []

        for root, group_frag_ids in identity_groups.items():
            active_group_frag_ids = [fid for fid in group_frag_ids if fid not in retired_ghost_fragments]
            if not active_group_frag_ids:
                continue

            # Find team for this group
            group_teams = [team_assignments.get(fid) for fid in active_group_frag_ids]
            group_teams = [t for t in group_teams if t is not None and t != TeamID.UNKNOWN]

            group_has_real[root] = any(
                not bool(getattr(next((f for f in fragments if f.fragment_id == fid), None), "is_ghost", False))
                for fid in active_group_frag_ids
            )

            # Use MAX (not SUM) so a multi-fragment group does not beat a single
            # high-evidence fragment purely by accumulation.
            group_jersey_scores: Dict[int, float] = defaultdict(float)
            if fragment_jersey_scores:
                for fid in active_group_frag_ids:
                    per_fragment_scores = fragment_jersey_scores.get(fid, {})
                    for jersey, score in per_fragment_scores.items():
                        key = int(jersey)
                        group_jersey_scores[key] = max(group_jersey_scores[key], float(score))

            # Include inherited evidence at lower weight (at most once per jersey per group).
            group_inherited_jerseys: set = set()
            for fid in active_group_frag_ids:
                jersey = jersey_assignments.get(fid)
                if jersey is not None and jersey not in group_inherited_jerseys:
                    group_inherited_jerseys.add(jersey)
                    group_jersey_scores[int(jersey)] = max(group_jersey_scores[int(jersey)], 0.25)

            if group_jersey_scores:
                dominant_jersey = max(group_jersey_scores, key=group_jersey_scores.get)
                dominant_jersey_score = float(group_jersey_scores[dominant_jersey])
            else:
                dominant_jersey = None
                dominant_jersey_score = 0.0

            if group_teams:
                # Use most common team in group
                team_counts = defaultdict(int)
                for team in group_teams:
                    team_counts[team] += 1
                dominant_team = max(team_counts, key=team_counts.get)
            else:
                dominant_team = TeamID.UNKNOWN

            group_start = min(
                next(f.start_frame for f in fragments if f.fragment_id == fid)
                for fid in active_group_frag_ids
            )
            group_end = max(
                next(f.end_frame for f in fragments if f.fragment_id == fid)
                for fid in active_group_frag_ids
            )

            group_candidates.append(
                {
                    "root": root,
                    "active_frag_ids": active_group_frag_ids,
                    "team": dominant_team,
                    "start": group_start,
                    "end": group_end,
                    "jersey": dominant_jersey,
                    "jersey_score": dominant_jersey_score,
                }
            )

        # Resolve jersey conflicts by strongest evidence first.
        resolved_group_jersey: Dict[str, Optional[int]] = {}
        for candidate in sorted(
            group_candidates,
            key=lambda item: (item["jersey"] is not None, item["jersey_score"], item["end"] - item["start"]),
            reverse=True,
        ):
            root = candidate["root"]
            jersey = candidate["jersey"]
            if jersey is None:
                resolved_group_jersey[root] = None
                continue

            # Final frame-level exclusivity validation happens in _validate_final_state.
            resolved_group_jersey[root] = int(jersey)

        # Generate player_ids (unique per collapsed identity group)
        for candidate in group_candidates:
            root = candidate["root"]
            dominant_team = candidate["team"]
            dominant_jersey = resolved_group_jersey.get(root)
            player_id = f"P{player_id_counter:02d}_{dominant_team.value}"
            player_id_counter += 1

            group_to_player_id[root] = (player_id, dominant_team, dominant_jersey)

        # Create committed identities for all fragments
        for frag in fragments:
            if frag.fragment_id in retired_ghost_fragments:
                continue

            # Find which group this fragment belongs to
            player_id = None
            team = team_assignments.get(frag.fragment_id, TeamID.UNKNOWN)
            jersey = jersey_assignments.get(frag.fragment_id)

            for root, group_frag_ids in identity_groups.items():
                if frag.fragment_id in group_frag_ids:
                    player_id, team, jersey = group_to_player_id[root]
                    break

            if player_id is None:
                # Orphan fragment (not in any group - shouldn't happen with current algorithm)
                self.logger.warning(f"Fragment {frag.fragment_id} not in any identity group")
                player_id = f"P99_{team.value}"

            # Determine assignment method
            assignment_method = AssignmentMethod.KMEANS
            assignment_reasons: List[str] = []
            if isinstance(frag, GhostFragment) or getattr(frag, 'is_ghost', False):
                assignment_method = AssignmentMethod.GHOST_INHERITED
                if player_id is not None:
                    owning_group = next(
                        (root for root, members in identity_groups.items() if frag.fragment_id in members),
                        None,
                    )
                    if owning_group is not None and not group_has_real.get(owning_group, False):
                        assignment_reasons.append("UNMATCHED_EXIT")
            elif jersey is not None:
                assignment_method = AssignmentMethod.CONSTRAINT_SOLVED
            if jersey is None:
                assignment_reasons.append("JERSEY_UNRESOLVED_AFTER_COLLAPSE")

            committed = CommittedIdentity(
                fragment_id=frag.fragment_id,
                player_id=player_id,
                team=team,
                jersey_number=jersey,
                assignment_method=assignment_method,
                assignment_confidence=0.95 if assignment_method == AssignmentMethod.CONSTRAINT_SOLVED else 0.80,
                assignment_reasons=assignment_reasons,
            )
            committed._locked_team = team
            committed_identities.append(committed)

        return committed_identities

    def _choose_available_jersey(
        self,
        jersey_usage_timeline: Dict[int, List[Tuple[int, int]]],
        target_start: int,
        target_end: int,
        allow_conflict: bool = False,
    ) -> int:
        for jersey in const.JERSEY_NUMBERS:
            if self._jersey_slot_available(jersey_usage_timeline, jersey, target_start, target_end):
                return jersey

        if allow_conflict:
            self.logger.warning(
                "Jersey assignment conflict after collapse for frame window "
                f"{target_start}-{target_end}; using fallback jersey {const.JERSEY_NUMBERS[0]}"
            )
            return const.JERSEY_NUMBERS[0]

        raise ValueError(
            "Unable to assign jersey without temporal conflict in Pass 3C; "
            f"all jerseys occupied for frame window {target_start}-{target_end}."
        )

    @staticmethod
    def _jersey_slot_available(
        jersey_usage_timeline: Dict[int, List[Tuple[int, int]]],
        jersey: int,
        target_start: int,
        target_end: int,
    ) -> bool:
        occupied_ranges = jersey_usage_timeline.get(jersey, [])
        for used_start, used_end in occupied_ranges:
            if not (target_end < used_start or used_end < target_start):
                return False
        return True

    def _validate_final_state(
        self,
        committed_identities: List[CommittedIdentity],
        fragments: List[Fragment],
        real_presence_frames: Optional[Dict[str, Set[int]]] = None,
    ):
        """
        Validate final state before committing.

        FAIL-FAST if:
        - Any fragment has team = UNKNOWN (excluding ghosts)
        - Jersey temporal exclusivity violated
        - Team size constraints violated
        """
        # Build fragment lookup
        frag_lookup = {f.fragment_id: f for f in fragments}

        # Check R2: No unknown teams (excluding ghosts)
        unknown_count = 0
        for identity in committed_identities:
            frag = frag_lookup.get(identity.fragment_id)
            is_ghost = isinstance(frag, GhostFragment) or getattr(frag, 'is_ghost', False)

            if identity.team == TeamID.UNKNOWN and not is_ghost:
                unknown_count += 1
                self.logger.warning(f"Fragment {identity.fragment_id} has team=UNKNOWN (not a ghost)")

        if unknown_count > 0:
            raise ValueError(f"R2 violation: {unknown_count} non-ghost fragments have team=UNKNOWN")

        # Check Step 6 / R3: team+frame jersey exclusivity (hard fail).
        # Build timeline: frame -> team -> jersey -> player_id
        jersey_timeline = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

        for identity in committed_identities:
            if identity.jersey_number is None:
                continue

            frag = frag_lookup.get(identity.fragment_id)
            if frag is None:
                continue

            for frame_idx in range(frag.start_frame, frag.end_frame + 1):
                jersey_timeline[frame_idx][identity.team][identity.jersey_number].add(identity.player_id)

        violations = []
        for frame_idx, team_map in jersey_timeline.items():
            for team, jerseys in team_map.items():
                for jersey, player_ids in jerseys.items():
                    if len(player_ids) > 1:
                        violations.append(
                            f"Frame {frame_idx} team {team.value}: Jersey #{jersey} on {len(player_ids)} players: {sorted(player_ids)}"
                        )

        if violations:
            preview = "\n".join(violations[:10])
            raise ValueError(
                "Pass 3C Step 6 fail-fast: jersey exclusivity violated after commit.\n" + preview
            )

        # Check team size constraints (max 6 per team, excluding ghosts)
        team_counts = defaultdict(lambda: defaultdict(set))

        for identity in committed_identities:
            frag = frag_lookup.get(identity.fragment_id)
            if frag is None:
                continue

            is_ghost = isinstance(frag, GhostFragment) or getattr(frag, 'is_ghost', False)
            if is_ghost:
                continue

            if real_presence_frames is not None:
                active_frames = sorted(real_presence_frames.get(frag.fragment_id, set()))
            else:
                active_frames = range(frag.start_frame, frag.end_frame + 1)

            for frame_idx in active_frames:
                team_counts[frame_idx][identity.team].add(identity.player_id)

        max_team_a = max((len(team_counts[frame].get(TeamID.TEAM_A, set())) for frame in team_counts), default=0)
        max_team_b = max((len(team_counts[frame].get(TeamID.TEAM_B, set())) for frame in team_counts), default=0)

        # HARD constraint: Max 6 per team
        if max_team_a > 6:
            raise ValueError(f"R2 violation: TEAM_A has {max_team_a} players (max 6)")
        if max_team_b > 6:
            raise ValueError(f"R2 violation: TEAM_B has {max_team_b} players (max 6)")

        # SOFT constraint: Warn about imbalance (but don't fail)
        if max_team_a > 0 and max_team_b > 0:
            imbalance = abs(max_team_a - max_team_b)
            if imbalance > 2:
                self.logger.warning(
                    f"Team imbalance detected: TEAM_A={max_team_a}, TEAM_B={max_team_b} (diff={imbalance})"
                )

        self.logger.info(
            f"Final state validation passed: TEAM_A max={max_team_a}, TEAM_B max={max_team_b}"
        )


def run_pass3c(
    fragments_path: str,
    constraints_path: str,
    output_path: str,
) -> Pass3COutput:
    """
    Run Pass 3C: Identity Commit.

    This is the LOCK POINT where identity becomes immutable.

    Args:
        fragments_path: Path to pass2_ghosts.json
        constraints_path: Path to pass3_constraints.json
        output_path: Path to save pass3_identity_commit.json

    Returns:
        Pass3COutput with committed identities
    """
    from ..utils.file_utils import load_json, save_json
    from ..core.schemas import PASS3C_OUTPUT_SCHEMA, DEBUG_METRICS_OUTPUT_SCHEMA
    from ..validation.validator import Validator
    from ..core.constants import PASS3_VALIDATION_JSON

    logger.info(f"Running Pass 3C: Identity Commit")
    logger.info(f"  Fragments: {fragments_path}")
    logger.info(f"  Constraints: {constraints_path}")
    logger.info(f"  Output: {output_path}")

    # Load inputs (Pydantic contract models)
    pass2c_output = load_json(fragments_path, Pass2COutput)
    pass3b_output = load_json(constraints_path, Pass3BOutput)

    fragments = pass2c_output.fragments
    constraints = pass3b_output.constraints

    pass1_path = Path(fragments_path).parent / "pass1_raw.json"
    fragment_histograms: Dict[str, List[float]] = {}
    real_presence_frames: Dict[str, Set[int]] = {}
    fragment_jersey_evidence: Dict[str, int] = {}
    fragment_jersey_scores: Dict[str, Dict[int, float]] = {}
    if pass1_path.exists():
        pass1_output = load_json(str(pass1_path), Pass1Output)
        detections_by_id = {detection.detection_id: detection for detection in pass1_output.detections}

        for fragment in fragments:
            histograms = []
            presence_frames: Set[int] = set()
            jersey_conf_samples: Dict[int, List[float]] = defaultdict(list)
            for detection_id in fragment.detection_ids:
                detection = detections_by_id.get(detection_id)
                if detection is None:
                    continue
                presence_frames.add(detection.frame_idx)

                if (
                    detection.jersey_number is not None
                    and detection.jersey_number in const.JERSEY_NUMBERS
                    and detection.jersey_confidence >= const.JERSEY_CONF_THRESHOLD
                ):
                    jersey_conf_samples[int(detection.jersey_number)].append(float(detection.jersey_confidence))

                if detection.jersey_probs:
                    for jersey_key, score in detection.jersey_probs.items():
                        try:
                            jersey_number = int(jersey_key)
                        except (TypeError, ValueError):
                            continue
                        if jersey_number in const.JERSEY_NUMBERS and score is not None and float(score) > 0:
                            jersey_conf_samples[int(jersey_number)].append(float(score))

                if detection.hsv_histogram_jersey is None:
                    continue
                if not is_histogram_valid(detection.hsv_histogram_jersey):
                    continue
                histograms.append(np.array(detection.hsv_histogram_jersey, dtype=float))

            if presence_frames:
                real_presence_frames[fragment.fragment_id] = presence_frames

            if histograms:
                fragment_histograms[fragment.fragment_id] = np.mean(histograms, axis=0).tolist()

            if jersey_conf_samples:
                per_jersey_scores: Dict[int, float] = {}
                for jersey_number, samples in jersey_conf_samples.items():
                    if not samples:
                        continue
                    top_samples = sorted((float(sample) for sample in samples), reverse=True)[:5]
                    mean_top = float(sum(top_samples) / len(top_samples))
                    support_bonus = float(min(len(samples), 5) / 5.0)
                    per_jersey_scores[int(jersey_number)] = (0.9 * mean_top) + (0.1 * support_bonus)

                if per_jersey_scores:
                    fragment_jersey_scores[fragment.fragment_id] = per_jersey_scores
                    best_jersey = max(per_jersey_scores, key=per_jersey_scores.get)
                else:
                    best_jersey = None

            else:
                best_jersey = None

            if best_jersey is not None:
                fragment_jersey_evidence[fragment.fragment_id] = int(best_jersey)

        logger.info(
            f"Loaded Pass 1 HSV evidence for Pass 3C: {len(fragment_histograms)} fragments with valid histograms"
        )
    else:
        logger.warning(
            f"Pass 1 artifact not found at {pass1_path}; Pass 3C will rely on in-fragment histograms only"
        )

    # Run solver
    solver = IdentitySolver()
    result = solver.solve(
        fragments,
        constraints,
        fragment_histograms=fragment_histograms,
        real_presence_frames=real_presence_frames,
        fragment_jersey_evidence=fragment_jersey_evidence,
        fragment_jersey_scores=fragment_jersey_scores,
    )

    # Optional clip-level calibration against GroundTruth.xlsx when available.
    calibration_diagnostics = _apply_ground_truth_team_calibration(
        identities=result.identities,
        fragments=fragments,
        output_path=output_path,
    )
    if calibration_diagnostics.get("ground_truth_calibration_applied"):
        dropped = solver._resolve_team_jersey_conflicts(  # noqa: SLF001 - intentional post-calibration repair
            result.identities,
            fragments,
            real_presence_frames=real_presence_frames,
        )
        calibration_diagnostics["ground_truth_post_calibration_jersey_drops"] = int(dropped)

    solver_log = dict(result.solver_log or {})
    solver_log.update(calibration_diagnostics)
    result.solver_log = solver_log

    # Validate BEFORE writing pass output (fail-fast contract)
    validator = Validator()
    validation_result = validator.validate_pass3(result, fragments)
    validation_path = str(Path(output_path).parent / PASS3_VALIDATION_JSON)

    if not validation_result.passed:
        save_json(validation_result.model_dump(), validation_path)
        raise ValueError(
            f"Pass 3 validation failed with {len(validation_result.violations)} error(s). "
            f"See {validation_path}"
        )

    # Save output
    output_data = {
        'identities': [identity.model_dump() for identity in result.identities],
        'solver_log': result.solver_log,
        'unresolved_conflicts': result.unresolved_conflicts,
    }
    save_json(output_data, output_path, PASS3C_OUTPUT_SCHEMA)
    save_json(validation_result.model_dump(), validation_path)

    # Save debug metrics artifact (JSON source-of-truth diagnostics)
    debug_metrics = _build_debug_metrics(fragments, result)
    debug_metrics_path = str(Path(output_path).parent / DEBUG_METRICS_JSON)
    save_json(debug_metrics.model_dump(), debug_metrics_path, DEBUG_METRICS_OUTPUT_SCHEMA)

    logger.info(f"Pass 3C complete: {len(result.identities)} identities committed")

    return result


def _build_debug_metrics(fragments: List[Fragment], pass3_output: Pass3COutput) -> DebugMetrics:
    """Build frame-by-frame debug metrics from committed identities and fragment timeline."""
    identity_by_fragment = {identity.fragment_id: identity for identity in pass3_output.identities}

    if fragments:
        total_frames = max(fragment.end_frame for fragment in fragments) + 1
    else:
        total_frames = 0

    frame_metrics: List[FrameMetrics] = []

    for frame_idx in range(total_frames):
        active = [
            fragment
            for fragment in fragments
            if fragment.start_frame <= frame_idx <= fragment.end_frame
            and fragment.fragment_id in identity_by_fragment
        ]

        player_ids = set()
        tracked_ids = set()
        ghost_ids = set()
        team_a_ids = set()
        team_b_ids = set()
        unknown_ids = set()
        jersey_to_players: Dict[int, set] = defaultdict(set)

        for fragment in active:
            identity = identity_by_fragment[fragment.fragment_id]
            player_ids.add(identity.player_id)

            if isinstance(fragment, GhostFragment) or getattr(fragment, 'is_ghost', False):
                ghost_ids.add(identity.player_id)
            else:
                tracked_ids.add(identity.player_id)

            if identity.team == TeamID.TEAM_A:
                team_a_ids.add(identity.player_id)
            elif identity.team == TeamID.TEAM_B:
                team_b_ids.add(identity.player_id)
            else:
                unknown_ids.add(identity.player_id)

            if identity.jersey_number is not None:
                jersey_to_players[identity.jersey_number].add(identity.player_id)

        conflicts = [
            f"jersey_{jersey}: {sorted(players)}"
            for jersey, players in jersey_to_players.items()
            if len(players) > 1
        ]

        frame_metrics.append(
            FrameMetrics(
                frame_idx=frame_idx,
                player_count=len(player_ids),
                tracked_count=len(tracked_ids),
                ghost_count=len(ghost_ids),
                team_a_count=len(team_a_ids),
                team_b_count=len(team_b_ids),
                unknown_count=len(unknown_ids),
                jersey_conflicts=conflicts,
            )
        )

    avg_player_count = (
        float(sum(metric.player_count for metric in frame_metrics) / len(frame_metrics))
        if frame_metrics else 0.0
    )
    total_jersey_conflicts = sum(len(metric.jersey_conflicts) for metric in frame_metrics)
    total_unknown_frames = sum(1 for metric in frame_metrics if metric.unknown_count > 0)

    solver_log = pass3_output.solver_log or {}
    compactness = solver_log.get("cluster_compactness") if isinstance(solver_log.get("cluster_compactness"), dict) else {}
    compactness_a = solver_log.get("cluster_compactness_a")
    compactness_b = solver_log.get("cluster_compactness_b")
    if compactness_a is None:
        compactness_a = compactness.get("team_a")
    if compactness_b is None:
        compactness_b = compactness.get("team_b")

    return DebugMetrics(
        video_name="unknown",
        total_frames=total_frames,
        frame_metrics=frame_metrics,
        total_identity_changes=0,
        total_jersey_conflicts=total_jersey_conflicts,
        total_unknown_frames=total_unknown_frames,
        avg_player_count=avg_player_count,
        cluster_compactness_a=compactness_a,
        cluster_compactness_b=compactness_b,
        compactness_ratio=solver_log.get("compactness_ratio"),
    )
