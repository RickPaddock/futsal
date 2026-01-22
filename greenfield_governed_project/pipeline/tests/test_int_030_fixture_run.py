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

    tracks = bundle / "tracks.jsonl"
    assert tracks.is_file()
    content = tracks.read_text(encoding="utf8").strip()
    assert content
