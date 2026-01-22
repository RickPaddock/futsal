# PROV: FUSBAL.PIPELINE.TESTS.INT_030_FIXTURE_RUN.01
# REQ: SYS-ARCH-15, FUSBAL-V1-PLAYER-001, FUSBAL-V1-TEAM-001, FUSBAL-V1-TRUST-001
# WHY: Smoke test the deterministic fixture run path for INT-030.

from __future__ import annotations

from pathlib import Path

from fusbal_pipeline.cli import main


def test_int_030_fixture_run_creates_valid_bundle(tmp_path: Path) -> None:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "player_tracking"
        / "fixture_int_030_mvp.json"
    )
    bundle = tmp_path / "bundle"
    
    # Run the fixture-based pipeline
    rc = main(
        [
            "run-fixture",
            "--fixture",
            str(fixture),
            "--match-id",
            "INT030_FIXTURE",
            "--out",
            str(bundle),
        ]
    )
    assert rc == 0

    # Validate the bundle using the CLI validator
    validate_rc = main(["validate", str(bundle)])
    assert validate_rc == 0, "Bundle validation should pass"

    # Check required outputs exist
    tracks = bundle / "tracks.jsonl"
    events = bundle / "events.json"
    assert tracks.is_file(), "tracks.jsonl should be generated"
    assert events.is_file(), "events.json should be generated"
    
    # Parse and verify tracks content
    tracks_content = tracks.read_text(encoding="utf8").strip()
    assert tracks_content, "tracks.jsonl should not be empty"
    
    import json
    track_lines = [json.loads(line) for line in tracks_content.split("\n") if line.strip()]
    assert len(track_lines) > 0, "At least one track record should be emitted"
    
    # Verify player records are present
    player_records = [r for r in track_lines if r.get("entity_type") == "player"]
    assert len(player_records) > 0, "At least one player record should be present"
    
    # Verify break/team semantics appear when appropriate
    has_break_reason = any(r.get("break_reason") for r in track_lines)
    has_team_assignment = any(r.get("team") for r in track_lines)
    
    # Fixture should be designed to produce some break reasons and team assignments
    # This validates the pipeline is actually exercising the MOT and team components
    assert has_break_reason or has_team_assignment, (
        "Fixture should produce either break_reason or team assignment semantics to validate pipeline components"
    )
