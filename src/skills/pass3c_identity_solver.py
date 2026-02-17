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

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

from ..core.data_models import (
    Fragment,
    ScoredFragment,
    GhostFragment,
    Constraint,
    CommittedIdentity,
    Pass3COutput,
)
from ..core.types import TeamID, ConstraintType
from ..core.constants import KMEANS_N_CLUSTERS, HSV_BINS
from ..utils.logging_utils import get_logger
from ..utils.hsv_color import compare_hsv_histograms, is_histogram_valid

logger = get_logger("pass3c_identity_solver")


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

        # Step 2: Assign teams via K-means (exclude ghosts)
        team_assignments = self._assign_and_lock_teams(fragments, identity_groups)
        self.logger.info(f"Assigned teams: {sum(1 for t in team_assignments.values() if t == TeamID.TEAM_A)} team_a, "
                        f"{sum(1 for t in team_assignments.values() if t == TeamID.TEAM_B)} team_b")

        # Step 3: Apply jersey inheritance (bidirectional with temporal exclusivity)
        jersey_assignments = self._apply_jersey_inheritance(fragments, identity_groups, team_assignments)
        self.logger.info(f"Applied jersey inheritance: {len(jersey_assignments)} fragments with jerseys")

        # Step 4: Validate CANNOT_SAME constraints
        self._validate_cannot_same_constraints(constraints, identity_groups)
        self.logger.info("CANNOT_SAME constraints validated successfully")

        # Step 5: Create committed identities
        committed_identities = self._create_committed_identities(
            fragments,
            identity_groups,
            team_assignments,
            jersey_assignments,
        )
        self.logger.info(f"Created {len(committed_identities)} committed identities")

        # Step 6: Validate final state
        self._validate_final_state(committed_identities, fragments)
        self.logger.info("Final state validation passed")

        return Pass3COutput(identities=committed_identities)

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
                union(constraint.fragment_id_a, constraint.fragment_id_b)
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
    ) -> Dict[str, TeamID]:
        """
        Assign teams via K-means clustering (exclude ghosts).

        CRITICAL: Team assignment happens AFTER identity resolution.
        Teams are locked immediately via the assignment.

        Returns:
            Dict mapping fragment_id -> TeamID
        """
        # Build fragment lookup
        frag_lookup = {f.fragment_id: f for f in fragments}

        # Extract HSV histograms for clustering (exclude ghosts and invalid histograms)
        valid_frags = []
        valid_histograms = []

        for frag in fragments:
            # CRITICAL: Exclude ghosts from K-means
            if isinstance(frag, GhostFragment) or getattr(frag, 'is_ghost', False):
                continue

            # Exclude fragments without valid HSV histograms
            if not hasattr(frag, 'hsv_histogram') or frag.hsv_histogram is None:
                continue

            if not is_histogram_valid(frag.hsv_histogram):
                continue

            valid_frags.append(frag)
            valid_histograms.append(frag.hsv_histogram)

        if len(valid_histograms) < KMEANS_N_CLUSTERS:
            self.logger.warning(
                f"Not enough valid histograms for K-means ({len(valid_histograms)} < {KMEANS_N_CLUSTERS}). "
                "All fragments will be assigned to UNKNOWN."
            )
            return {f.fragment_id: TeamID.UNKNOWN for f in fragments}

        # K-means clustering
        X = np.array(valid_histograms)
        kmeans = KMeans(n_clusters=KMEANS_N_CLUSTERS, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Map cluster labels to team IDs
        # Cluster 0 -> TEAM_A, Cluster 1 -> TEAM_B
        cluster_to_team = {0: TeamID.TEAM_A, 1: TeamID.TEAM_B}

        # Assign teams to fragments that participated in K-means
        assignments = {}
        for frag, label in zip(valid_frags, labels):
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

        self.logger.info(
            f"Team assignment complete: "
            f"{sum(1 for t in assignments.values() if t == TeamID.TEAM_A)} TEAM_A, "
            f"{sum(1 for t in assignments.values() if t == TeamID.TEAM_B)} TEAM_B, "
            f"{sum(1 for t in assignments.values() if t == TeamID.UNKNOWN)} UNKNOWN"
        )

        return assignments

    def _apply_jersey_inheritance(
        self,
        fragments: List[Fragment],
        identity_groups: Dict[str, Set[str]],
        team_assignments: Dict[str, TeamID],
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

        # Extract initial jersey assignments from detections
        initial_jerseys = {}
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
                frag_a = constraint.fragment_id_a
                frag_b = constraint.fragment_id_b

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

        for root, group_frag_ids in identity_groups.items():
            # Find jersey for this group (if any)
            group_jerseys = [jersey_assignments.get(fid) for fid in group_frag_ids]
            group_jerseys = [j for j in group_jerseys if j is not None]

            # Find team for this group
            group_teams = [team_assignments.get(fid) for fid in group_frag_ids]
            group_teams = [t for t in group_teams if t is not None and t != TeamID.UNKNOWN]

            # Determine player_id
            if group_jerseys:
                # Use most common jersey in group
                jersey_counts = defaultdict(int)
                for jersey in group_jerseys:
                    jersey_counts[jersey] += 1
                dominant_jersey = max(jersey_counts, key=jersey_counts.get)
            else:
                dominant_jersey = None

            if group_teams:
                # Use most common team in group
                team_counts = defaultdict(int)
                for team in group_teams:
                    team_counts[team] += 1
                dominant_team = max(team_counts, key=team_counts.get)
            else:
                dominant_team = TeamID.UNKNOWN

            # Generate player_id
            if dominant_jersey is not None:
                player_id = f"P{dominant_jersey:02d}_{dominant_team.value}"
            else:
                player_id = f"P{player_id_counter:02d}_{dominant_team.value}"
                player_id_counter += 1

            group_to_player_id[root] = (player_id, dominant_team, dominant_jersey)

        # Create committed identities for all fragments
        for frag in fragments:
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
            assignment_method = "kmeans"
            if isinstance(frag, GhostFragment) or getattr(frag, 'is_ghost', False):
                assignment_method = "ghost_inheritance"
            elif jersey is not None:
                assignment_method = "jersey_detection"

            committed_identities.append(
                CommittedIdentity(
                    fragment_id=frag.fragment_id,
                    player_id=player_id,
                    team=team,
                    jersey_number=jersey,
                    assignment_method=assignment_method,
                    assignment_confidence=0.95 if assignment_method == "jersey_detection" else 0.80,
                )
            )

        return committed_identities

    def _validate_final_state(
        self,
        committed_identities: List[CommittedIdentity],
        fragments: List[Fragment],
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

        # Check R3: Jersey temporal exclusivity
        # Build timeline: frame -> jersey -> fragment_id
        jersey_timeline = defaultdict(lambda: defaultdict(set))

        for identity in committed_identities:
            if identity.jersey_number is None:
                continue

            frag = frag_lookup.get(identity.fragment_id)
            if frag is None:
                continue

            for frame_idx in range(frag.start_frame, frag.end_frame + 1):
                jersey_timeline[frame_idx][identity.jersey_number].add(identity.fragment_id)

        violations = []
        for frame_idx, jerseys in jersey_timeline.items():
            for jersey, frag_ids in jerseys.items():
                if len(frag_ids) > 1:
                    violations.append(f"Frame {frame_idx}: Jersey #{jersey} on {len(frag_ids)} fragments: {frag_ids}")

        if violations:
            error_msg = f"R3 violation: Jersey temporal exclusivity violated:\n" + "\n".join(violations[:10])
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Check team size constraints (max 6 per team, excluding ghosts)
        team_counts = defaultdict(lambda: defaultdict(set))

        for identity in committed_identities:
            frag = frag_lookup.get(identity.fragment_id)
            if frag is None:
                continue

            is_ghost = isinstance(frag, GhostFragment) or getattr(frag, 'is_ghost', False)
            if is_ghost:
                continue

            for frame_idx in range(frag.start_frame, frag.end_frame + 1):
                team_counts[frame_idx][identity.team].add(identity.player_id)

        max_team_a = max((len(players) for players in team_counts[frame].get(TeamID.TEAM_A, set()) for frame in team_counts), default=0)
        max_team_b = max((len(players) for players in team_counts[frame].get(TeamID.TEAM_B, set()) for frame in team_counts), default=0)

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
    from ..core.schemas import PASS2C_OUTPUT_SCHEMA, PASS3B_OUTPUT_SCHEMA, PASS3C_OUTPUT_SCHEMA

    logger.info(f"Running Pass 3C: Identity Commit")
    logger.info(f"  Fragments: {fragments_path}")
    logger.info(f"  Constraints: {constraints_path}")
    logger.info(f"  Output: {output_path}")

    # Load inputs
    fragments_data = load_json(fragments_path, PASS2C_OUTPUT_SCHEMA)
    constraints_data = load_json(constraints_path, PASS3B_OUTPUT_SCHEMA)

    # Parse to Pydantic models
    fragments = [Fragment(**f) for f in fragments_data.get('fragments', [])]
    constraints = [Constraint(**c) for c in constraints_data.get('constraints', [])]

    # Run solver
    solver = IdentitySolver()
    result = solver.solve(fragments, constraints)

    # Save output
    output_data = {
        'identities': [identity.model_dump() for identity in result.identities]
    }
    save_json(output_path, output_data, PASS3C_OUTPUT_SCHEMA)

    logger.info(f"Pass 3C complete: {len(result.identities)} identities committed")

    return result
