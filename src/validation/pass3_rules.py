"""
Pass 3 validation rules.

Per CLAUDE.md Section 5 (Pass 3: Identity Resolution):
- Pass 3A: Candidate generation validity
- Pass 3B: Constraint graph validity
- Pass 3C: Identity commit validity (LOCK POINT)
"""

from typing import List, Dict, Set
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
        if fragment.get('is_ghost', False):
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
