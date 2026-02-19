"""
Pass 3 validation rules.

Per CLAUDE.md Section 5 (Pass 3: Identity Resolution):
- Pass 3A: Candidate generation validity
- Pass 3B: Constraint graph validity
- Pass 3C: Identity commit validity (LOCK POINT)
"""

from typing import List, Dict, Set
import math
from ..core.data_models import (
    IdentityCandidate,
    Constraint,
    CommittedIdentity,
    ScoredFragment,
    Pass3AOutput,
    Pass3BOutput,
    Pass3COutput,
    ValidationViolation,
)
from ..core.types import TeamID, ConstraintType
from ..core.constants import COMPACT_CLUSTER_MAX_MEAN_DISTANCE, COMPACTNESS_DIFF_MIN
from ..core import constants as const


def validate_pass3a_candidates(pass3a_output: Pass3AOutput) -> List[ValidationViolation]:
    """
    Validate Pass 3A candidate generation.

    Checks:
    - Every fragment has a candidate
    - Candidate evidence scores are valid
    - NO LOCKING (candidates only)

    Args:
        pass3a_output: Pass 3A output data

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    for candidate in pass3a_output.candidates:
        # Check evidence scores are non-negative
        for source, score in candidate.team_evidence.items():
            if score < 0:
                violations.append(
                    ValidationViolation(
                        rule="PASS3A_NEGATIVE_SCORE",
                        severity="warning",
                        message=f"Candidate {candidate.fragment_id} has negative team evidence score for '{source}': {score}",
                        fragment_id=candidate.fragment_id,
                        details={
                            "fragment_id": candidate.fragment_id,
                            "source": source,
                            "score": score,
                        },
                    )
                )

        for source, score in candidate.jersey_evidence.items():
            if score < 0:
                violations.append(
                    ValidationViolation(
                        rule="PASS3A_NEGATIVE_SCORE",
                        severity="warning",
                        message=f"Candidate {candidate.fragment_id} has negative jersey evidence score for '{source}': {score}",
                        fragment_id=candidate.fragment_id,
                        details={
                            "fragment_id": candidate.fragment_id,
                            "source": source,
                            "score": score,
                        },
                    )
                )

    return violations


def validate_pass3b_constraints(pass3b_output: Pass3BOutput) -> List[ValidationViolation]:
    """
    Validate Pass 3B constraint graph.

    Checks:
    - Constraint types are valid (MUST_SAME, CANNOT_SAME, SOFT_SAME)
    - Fragments referenced in constraints exist
    - MUST_SAME constraints have value (team or jersey)
    - SOFT constraints have weight

    Args:
        pass3b_output: Pass 3B output data

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    # Get all fragment IDs from constraint graph
    all_fragment_ids = set(pass3b_output.constraint_graph.keys())

    for constraint in pass3b_output.constraints:
        # Check constraint type
        constraint_type_str = constraint.constraint_type.value if isinstance(constraint.constraint_type, ConstraintType) else constraint.constraint_type
        if constraint_type_str not in ["must_same", "cannot_same", "soft_same"]:
            violations.append(
                ValidationViolation(
                    rule="PASS3B_INVALID_CONSTRAINT_TYPE",
                    severity="error",
                    message=f"Constraint {constraint.constraint_id} has invalid type '{constraint_type_str}'",
                    details={
                        "constraint_id": constraint.constraint_id,
                        "constraint_type": constraint_type_str,
                    },
                )
            )

        # Check fragments exist
        for fragment_id in constraint.fragment_ids:
            if fragment_id not in all_fragment_ids:
                violations.append(
                    ValidationViolation(
                        rule="PASS3B_FRAGMENT_NOT_FOUND",
                        severity="error",
                        message=f"Constraint {constraint.constraint_id} references unknown fragment {fragment_id}",
                        details={
                            "constraint_id": constraint.constraint_id,
                            "fragment_id": fragment_id,
                        },
                    )
                )

        # Check MUST_SAME constraints have value
        if constraint_type_str == "must_same":
            if constraint.value is None:
                violations.append(
                    ValidationViolation(
                        rule="PASS3B_MUST_SAME_NO_VALUE",
                        severity="error",
                        message=f"MUST_SAME constraint {constraint.constraint_id} has no value",
                        details={"constraint_id": constraint.constraint_id},
                    )
                )

        # Check SOFT constraints have weight
        if constraint_type_str == "soft_same":
            if constraint.weight <= 0:
                violations.append(
                    ValidationViolation(
                        rule="PASS3B_SOFT_INVALID_WEIGHT",
                        severity="warning",
                        message=f"SOFT_SAME constraint {constraint.constraint_id} has weight={constraint.weight} (expected > 0)",
                        details={
                            "constraint_id": constraint.constraint_id,
                            "weight": constraint.weight,
                        },
                    )
                )

    return violations


def validate_pass3c_identity_commit(
    pass3c_output: Pass3COutput,
    fragments: List[ScoredFragment],
) -> List[ValidationViolation]:
    """
    Validate Pass 3C identity commit (LOCK POINT).

    Checks:
    - Every non-ghost fragment has committed identity
    - R2: No "unknown" teams
    - R3: Jersey temporal exclusivity
    - player_id format valid (P##_team)
    - assignment_confidence in [0, 1]
    - Unresolved conflicts = 0 (fail-fast)

    Args:
        pass3c_output: Pass 3C output data
        fragments: List of fragments (for R2/R3 validation)

    Returns:
        List of violations (empty if valid)
    """
    violations = []

    # Check for unresolved conflicts
    if len(pass3c_output.unresolved_conflicts) > 0:
        violations.append(
            ValidationViolation(
                rule="PASS3C_UNRESOLVED_CONFLICTS",
                severity="error",
                message=f"{len(pass3c_output.unresolved_conflicts)} unresolved conflicts (FAIL-FAST)",
                details={
                    "unresolved_count": len(pass3c_output.unresolved_conflicts),
                    "conflicts": pass3c_output.unresolved_conflicts,
                },
            )
        )

    # Check each identity
    identity_map = {i.fragment_id: i for i in pass3c_output.identities}

    for identity in pass3c_output.identities:
        # Check player_id format (P##_team)
        import re
        if not re.match(r'^P\d{2}_(team_a|team_b)$', identity.player_id):
            violations.append(
                ValidationViolation(
                    rule="PASS3C_INVALID_PLAYER_ID",
                    severity="error",
                    message=f"Identity {identity.fragment_id} has invalid player_id format: '{identity.player_id}'",
                    fragment_id=identity.fragment_id,
                    details={
                        "fragment_id": identity.fragment_id,
                        "player_id": identity.player_id,
                    },
                )
            )

        # Check assignment confidence
        if not (0 <= identity.assignment_confidence <= 1):
            violations.append(
                ValidationViolation(
                    rule="PASS3C_INVALID_CONFIDENCE",
                    severity="warning",
                    message=f"Identity {identity.fragment_id} assignment_confidence={identity.assignment_confidence} out of range [0, 1]",
                    fragment_id=identity.fragment_id,
                    details={
                        "fragment_id": identity.fragment_id,
                        "assignment_confidence": identity.assignment_confidence,
                    },
                )
            )

        # Check jersey number range
        if not (1 <= identity.jersey_number <= 12):
            violations.append(
                ValidationViolation(
                    rule="PASS3C_INVALID_JERSEY",
                    severity="error",
                    message=f"Identity {identity.fragment_id} jersey_number={identity.jersey_number} out of range [1, 12]",
                    fragment_id=identity.fragment_id,
                    details={
                        "fragment_id": identity.fragment_id,
                        "jersey_number": identity.jersey_number,
                    },
                )
            )

    # Check every non-ghost fragment has identity
    for fragment in fragments:
        if getattr(fragment, 'is_ghost', False):
            continue  # Ghosts can skip identity (inherit from source)

        if fragment.fragment_id not in identity_map:
            violations.append(
                ValidationViolation(
                    rule="PASS3C_MISSING_IDENTITY",
                    severity="error",
                    message=f"Fragment {fragment.fragment_id} has no committed identity",
                    fragment_id=fragment.fragment_id,
                    details={"fragment_id": fragment.fragment_id},
                )
            )

    # R2: No unknown teams
    from .global_rules import validate_r2_no_unknown_teams
    violations.extend(validate_r2_no_unknown_teams(pass3c_output.identities, fragments))

    # R3: Jersey temporal exclusivity
    from .global_rules import validate_r3_jersey_temporal_exclusivity
    violations.extend(validate_r3_jersey_temporal_exclusivity(pass3c_output.identities, fragments))

    # Compactness-aware team validation (contract extension)
    resolved_teams = {
        identity.team.value if isinstance(identity.team, TeamID) else identity.team
        for identity in pass3c_output.identities
    }
    expected_teams = {TeamID.TEAM_A.value, TeamID.TEAM_B.value}
    if resolved_teams != expected_teams:
        violations.append(
            ValidationViolation(
                rule="PASS3C_TEAM_COUNT_INVALID",
                severity="error",
                message=(
                    "Pass 3C must resolve exactly 2 teams (team_a, team_b) "
                    f"but got {sorted(list(resolved_teams))}"
                ),
                details={
                    "resolved_teams": sorted(list(resolved_teams)),
                    "expected_teams": sorted(list(expected_teams)),
                },
            )
        )

    solver_log = pass3c_output.solver_log or {}
    cluster_compactness = solver_log.get("cluster_compactness")
    compactness_ratio = solver_log.get("compactness_ratio")

    if cluster_compactness is None:
        violations.append(
            ValidationViolation(
                rule="PASS3C_MISSING_CLUSTER_COMPACTNESS",
                severity="error",
                message="Pass 3C solver_log missing required 'cluster_compactness' diagnostics",
            )
        )
    if compactness_ratio is None:
        violations.append(
            ValidationViolation(
                rule="PASS3C_MISSING_COMPACTNESS_RATIO",
                severity="error",
                message="Pass 3C solver_log missing required 'compactness_ratio' diagnostics",
            )
        )

    compactness_a = None
    compactness_b = None
    if isinstance(cluster_compactness, dict):
        compactness_a = cluster_compactness.get(TeamID.TEAM_A.value)
        compactness_b = cluster_compactness.get(TeamID.TEAM_B.value)
    elif cluster_compactness is not None:
        violations.append(
            ValidationViolation(
                rule="PASS3C_INVALID_CLUSTER_COMPACTNESS",
                severity="error",
                message="Pass 3C cluster_compactness must be an object with team_a/team_b values",
                details={"cluster_compactness_type": type(cluster_compactness).__name__},
            )
        )

    if compactness_a is not None and (not isinstance(compactness_a, (int, float)) or not math.isfinite(compactness_a)):
        violations.append(
            ValidationViolation(
                rule="PASS3C_INVALID_CLUSTER_COMPACTNESS_VALUE",
                severity="error",
                message="Pass 3C cluster_compactness.team_a must be a finite number",
                details={"value": compactness_a},
            )
        )
        compactness_a = None

    if compactness_b is not None and (not isinstance(compactness_b, (int, float)) or not math.isfinite(compactness_b)):
        violations.append(
            ValidationViolation(
                rule="PASS3C_INVALID_CLUSTER_COMPACTNESS_VALUE",
                severity="error",
                message="Pass 3C cluster_compactness.team_b must be a finite number",
                details={"value": compactness_b},
            )
        )
        compactness_b = None

    if compactness_ratio is not None and (not isinstance(compactness_ratio, (int, float)) or not math.isfinite(compactness_ratio) or compactness_ratio <= 0):
        violations.append(
            ValidationViolation(
                rule="PASS3C_INVALID_COMPACTNESS_RATIO",
                severity="error",
                message="Pass 3C compactness_ratio must be a finite number > 0",
                details={"compactness_ratio": compactness_ratio},
            )
        )
        compactness_ratio = None

    if compactness_a is not None and compactness_b is not None:
        is_a_compact = compactness_a <= COMPACT_CLUSTER_MAX_MEAN_DISTANCE
        is_b_compact = compactness_b <= COMPACT_CLUSTER_MAX_MEAN_DISTANCE

        # Bib-vs-random tolerant: both diffuse is allowed only if clearly separable.
        if not (is_a_compact or is_b_compact):
            compactness_diff = abs(compactness_a - compactness_b)
            if compactness_diff < COMPACTNESS_DIFF_MIN:
                violations.append(
                    ValidationViolation(
                        rule="PASS3C_AMBIGUOUS_DIFFUSE_CLUSTERS",
                        severity="error",
                        message=(
                            "Both clusters are diffuse and compactness difference is below threshold "
                            "(FAIL-FAST)."
                        ),
                        details={
                            "cluster_compactness": {
                                TeamID.TEAM_A.value: compactness_a,
                                TeamID.TEAM_B.value: compactness_b,
                            },
                            "compactness_diff": compactness_diff,
                            "compactness_diff_min": COMPACTNESS_DIFF_MIN,
                            "compact_cluster_max_mean_distance": COMPACT_CLUSTER_MAX_MEAN_DISTANCE,
                        },
                    )
                )

    # Pass 3 global reality validations (first stage allowed to enforce physical reality).
    frag_lookup = {f.fragment_id: f for f in fragments}
    identity_by_fragment = {i.fragment_id: i for i in pass3c_output.identities}

    frame_player_sources: Dict[int, Dict[str, List[Dict[str, object]]]] = {}

    for fragment_id, identity in identity_by_fragment.items():
        fragment = frag_lookup.get(fragment_id)
        if fragment is None:
            continue

        player_id = identity.player_id
        is_ghost = bool(getattr(fragment, "is_ghost", False))

        for frame_idx in range(fragment.start_frame, fragment.end_frame + 1):
            if frame_idx not in frame_player_sources:
                frame_player_sources[frame_idx] = {}
            if player_id not in frame_player_sources[frame_idx]:
                frame_player_sources[frame_idx][player_id] = []

            frame_player_sources[frame_idx][player_id].append(
                {
                    "fragment_id": fragment_id,
                    "is_ghost": is_ghost,
                }
            )

    if frame_player_sources:
        start_frame = min(frame_player_sources.keys())
        end_frame = max(frame_player_sources.keys())

        init_end = min(end_frame, start_frame + const.INITIAL_LEVEL_FRAMES - 1)
        level = 0
        for frame_idx in range(start_frame, init_end + 1):
            level = max(level, len(frame_player_sources.get(frame_idx, {})))
        level = min(level, const.DYNAMIC_LEVEL_MAX)

        over_cap_frames: List[Dict[str, int]] = []
        deficit_frames: List[Dict[str, int]] = []

        for frame_idx in range(start_frame, end_frame + 1):
            player_sources = frame_player_sources.get(frame_idx, {})
            frame_count = len(player_sources)

            # Physical cap: <= 12 identities per frame.
            if frame_count > const.DYNAMIC_LEVEL_MAX:
                over_cap_frames.append({"frame": frame_idx, "count": frame_count})

            level = min(max(level, frame_count), const.DYNAMIC_LEVEL_MAX)

            # Global R4 after reconciliation: exactly level identities per frame.
            if frame_count != level:
                deficit_frames.append({
                    "frame": frame_idx,
                    "count": frame_count,
                    "expected_level": level,
                })

            # No identity splits / duplicate same-player presences at same frame.
            for player_id, sources in player_sources.items():
                if len(sources) > 1:
                    violations.append(
                        ValidationViolation(
                            rule="PASS3C_PLAYER_OVERLAP_SAME_FRAME",
                            severity="error",
                            message=(
                                f"Frame {frame_idx}: player {player_id} has overlapping presences "
                                f"across {len(sources)} fragments"
                            ),
                            frame_idx=frame_idx,
                            details={
                                "frame": frame_idx,
                                "player_id": player_id,
                                "sources": sources,
                            },
                        )
                    )

                # Ghost retirement: matched real and ghost cannot overlap.
                has_real = any(not bool(s["is_ghost"]) for s in sources)
                has_ghost = any(bool(s["is_ghost"]) for s in sources)
                if has_real and has_ghost:
                    violations.append(
                        ValidationViolation(
                            rule="PASS3C_GHOST_NOT_RETIRED",
                            severity="error",
                            message=(
                                f"Frame {frame_idx}: player {player_id} has both real and ghost presence. "
                                "Ghost must retire once matched to real."
                            ),
                            frame_idx=frame_idx,
                            details={
                                "frame": frame_idx,
                                "player_id": player_id,
                                "sources": sources,
                            },
                        )
                    )

        if over_cap_frames:
            violations.append(
                ValidationViolation(
                    rule="PASS3C_PHYSICAL_CAP_EXCEEDED",
                    severity="error",
                    message=(
                        f"{len(over_cap_frames)} frames exceed physical cap of {const.DYNAMIC_LEVEL_MAX} identities."
                    ),
                    details={
                        "violation_count": len(over_cap_frames),
                        "sample_violations": over_cap_frames[:20],
                    },
                )
            )

        if deficit_frames:
            violations.append(
                ValidationViolation(
                    rule="PASS3C_R4_GLOBAL_MISMATCH",
                    severity="error",
                    message=(
                        f"Global R4 mismatch on {len(deficit_frames)} frames: reconciled identities != dynamic level."
                    ),
                    details={
                        "violation_count": len(deficit_frames),
                        "sample_violations": deficit_frames[:20],
                    },
                )
            )

    return violations


def validate_pass3(
    pass3c_output: Pass3COutput,
    fragments: List[ScoredFragment],
) -> List[ValidationViolation]:
    """
    Run all Pass 3 validations.

    Args:
        pass3c_output: Pass 3C output data
        fragments: List of fragments (for R2/R3 validation)

    Returns:
        List of all violations
    """
    violations = []

    # Pass 3C checks (identity commit - LOCK POINT)
    violations.extend(validate_pass3c_identity_commit(pass3c_output, fragments))

    return violations
