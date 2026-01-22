# PROV: FUSBAL.PIPELINE.TEAM.COLORS.01
# REQ: FUSBAL-V1-TEAM-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Convert lightweight bib color observations into team evidence scores (no raw pixels persisted).

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TeamColorConfig:
    team_a_label: str = "yellow"
    team_b_label: str = "blue"


def color_label_to_team_evidence(
    *,
    color_label: str | None,
    cfg: TeamColorConfig | None = None,
) -> tuple[float, float, dict[str, object]]:
    """Return (score_a, score_b, diagnostics) from a simple color label observation.

    This is intentionally conservative: unknown/empty labels yield no evidence.
    """

    c = cfg or TeamColorConfig()
    label = (color_label or "").strip().lower()
    if not label:
        return 0.0, 0.0, {"color_label": "unknown", "reason": "missing_label"}

    if label == c.team_a_label.strip().lower():
        return 1.0, 0.0, {"color_label": label, "mapped_team": "A"}
    if label == c.team_b_label.strip().lower():
        return 0.0, 1.0, {"color_label": label, "mapped_team": "B"}

    return 0.0, 0.0, {"color_label": label, "reason": "unmapped_label"}

