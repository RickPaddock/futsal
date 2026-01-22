# PROV: FUSBAL.PIPELINE.TESTS.TEAM_ASSIGNMENT.01
# REQ: SYS-ARCH-15, FUSBAL-V1-TEAM-001, FUSBAL-V1-TRUST-001
# WHY: Lock in trust-first Unknown behavior and smoothing stability.

from __future__ import annotations

from fusbal_pipeline.team.assignment import TeamAssigner, TeamSmoothingConfig


def test_team_assigner_unknown_when_no_evidence() -> None:
    a = TeamAssigner(cfg=TeamSmoothingConfig(window_frames=3, min_confidence=0.65, hysteresis=0.10))
    team, conf, diag = a.update(track_id="t1", score_a=0.0, score_b=0.0)
    assert team == "unknown"
    assert conf == 0.0
    assert diag.get("unknown_reason") == "no_color_evidence"


def test_team_assigner_hysteresis_prevents_flip_near_tie() -> None:
    a = TeamAssigner(cfg=TeamSmoothingConfig(window_frames=3, min_confidence=0.0, hysteresis=0.20))

    # Establish Team A.
    team1, _, _ = a.update(track_id="t1", score_a=1.0, score_b=0.0)
    assert team1 == "A"

    # Near-tie should keep last team due to hysteresis.
    team2, _, diag2 = a.update(track_id="t1", score_a=0.51, score_b=0.49)
    assert team2 == "A"
    assert "smoothing" in diag2
