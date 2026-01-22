# PROV: FUSBAL.PIPELINE.TESTS.CLI.RENDER_OVERLAY.01
# REQ: SYS-ARCH-15, FUSBAL-V1-OUT-001, FUSBAL-V1-BALL-001, FUSBAL-V1-TRUST-001
# WHY: Ensure render-overlay probes fps deterministically when omitted and wires report outputs.

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fusbal_pipeline import cli
from fusbal_pipeline.overlay.render import OverlayPlan, OverlayRenderResult


def _ok_result(*, out_mp4: Path, fps: float, vf: str = "null") -> OverlayRenderResult:
    plan = OverlayPlan(
        vf=vf,
        vf_sha256="0" * 64,
        enable_expr="n",
        fps=float(fps),
        width_px=1920,
        height_px=1080,
        eligible_boxes=0,
        selected_boxes=0,
        draw_ops=0,
    )
    return OverlayRenderResult(
        ok=True,
        out_mp4=out_mp4,
        plan=plan,
        ffmpeg_path="/usr/bin/ffmpeg",
        ffmpeg_version_first_line="ffmpeg version test",
        stderr_excerpt=None,
        error=None,
    )


def test_cmd_render_overlay_uses_provided_fps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "overlay.mp4"
    rep = tmp_path / "overlay_render_report.json"

    # Probe returns a different fps, but should not override provided --fps.
    probe = Mock(return_value=SimpleNamespace(fps=999.0, width_px=640, height_px=480))
    monkeypatch.setattr(cli, "probe_video_metadata", probe)

    def _render_side_effect(**kwargs):
        return _ok_result(out_mp4=out, fps=float(kwargs.get("fps") or 0.0))

    render = Mock(side_effect=_render_side_effect)
    monkeypatch.setattr(cli, "render_overlay_mp4", render)

    args = SimpleNamespace(
        video=str(tmp_path / "in.mp4"),
        tracks=str(tmp_path / "tracks.jsonl"),
        out=str(out),
        fps=30.0,
        max_ops=10,
        timeout_s=1.0,
        report=str(rep),
    )

    rc = cli.cmd_render_overlay(args)
    assert rc == 0
    assert render.call_args.kwargs["fps"] == 30.0

    payload = json.loads(rep.read_text(encoding="utf8"))
    assert payload["ok"] is True
    assert payload["fps"] == 30.0


def test_cmd_render_overlay_probes_fps_when_omitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "overlay.mp4"
    rep = tmp_path / "overlay_render_report.json"

    probe = Mock(return_value=SimpleNamespace(fps=29.97, width_px=640, height_px=480))
    monkeypatch.setattr(cli, "probe_video_metadata", probe)

    def _render_side_effect(**kwargs):
        return _ok_result(out_mp4=out, fps=float(kwargs.get("fps") or 0.0))

    render = Mock(side_effect=_render_side_effect)
    monkeypatch.setattr(cli, "render_overlay_mp4", render)

    args = SimpleNamespace(
        video=str(tmp_path / "in.mp4"),
        tracks=str(tmp_path / "tracks.jsonl"),
        out=str(out),
        fps=None,
        max_ops=10,
        timeout_s=1.0,
        report=str(rep),
    )

    rc = cli.cmd_render_overlay(args)
    assert rc == 0
    assert probe.call_count >= 1
    assert render.call_args.kwargs["fps"] == pytest.approx(29.97)

    payload = json.loads(rep.read_text(encoding="utf8"))
    assert payload["ok"] is True
    assert payload["fps"] == pytest.approx(29.97)
