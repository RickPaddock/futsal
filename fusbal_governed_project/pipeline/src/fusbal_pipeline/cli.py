# PROV: FUSBAL.PIPELINE.CLI.01
# REQ: FUSBAL-V1-OUT-001, FUSBAL-V1-DATA-001, SYS-ARCH-15
# WHY: Provide a minimal CLI to initialize and validate Fusbal match bundles and manifests.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, NotRequired, TypedDict

from .bundle import (
    BUNDLE_ARTIFACT_SPECS_V1,
    BUNDLE_LAYOUT_VERSION,
    MatchBundlePaths,
    ensure_bundle_layout,
    ensure_placeholder_outputs_v1,
    validate_bundle_layout,
)
from .contract import validate_events_json, validate_tracks_jsonl
from .errors import ERROR, ValidationError, make_error
from .overlay.render import OverlayError, render_overlay_mp4

from .player.detections import DetectionGatingConfig, emit_player_detections_v1
from .player.mot import SwapAvoidantMOT
from .team.assignment import TeamAssigner, TeamSmoothingConfig, annotate_track_with_team
from .team.colors import TeamColorConfig, color_label_to_team_evidence
from .ball.detections import DetectionGatingConfig as BallDetectionGatingConfig, emit_ball_detections_v1
from .ball.detector import BaselineBallDetector, BaselineBallDetectorConfig, NullBallDetector
from .ball.tracker import BallTracker, compute_ball_quality_metrics_v1
from .events.shots_goals import ShotsGoalsConfig, infer_shots_goals_v1
from .video.ingest import VideoIngestError, VideoMetadata, VideoFrame, iter_video_frames_rgb24, probe_video_metadata

MANIFEST_SCHEMA_VERSION = 1


class ManifestArtifactV1(TypedDict):
    artifact_id: str
    rel_path: str
    kind: Literal["file", "dir"]
    requirement: Literal["required", "optional", "required_if_sensors_present"]
    description: str


class ManifestInputsV1(TypedDict):
    video_paths: list[str]
    sensor_paths: list[str]


class MatchManifestV1(TypedDict):
    schema_version: Literal[1]
    bundle_layout_version: int
    match_id: str
    inputs: ManifestInputsV1
    artifacts: list[ManifestArtifactV1]
    notes: NotRequired[str]


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf8")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    lines = [json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf8")


def cmd_init(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).expanduser().resolve()
    paths: MatchBundlePaths = ensure_bundle_layout(out_dir)
    video_paths = _normalize_paths(args.video, base_dir=out_dir)
    sensor_paths = _normalize_paths(args.sensor or [], base_dir=out_dir)
    notes = (
        str(args.notes)
        if args.notes
        else ("placeholder bundle (no real outputs)" if args.with_placeholders else None)
    )
    manifest: MatchManifestV1 = _build_manifest_v1(
        match_id=str(args.match_id),
        video_paths=video_paths,
        sensor_paths=sensor_paths,
        notes=notes,
    )
    _write_json(paths.manifest_json, manifest)

    if args.with_placeholders:
        ensure_placeholder_outputs_v1(out_dir, sensors_present=bool(sensor_paths))
    return 0


def cmd_run_fixture(args: argparse.Namespace) -> int:
    """Run a deterministic fixture and write a non-placeholder bundle.

    PROV: FUSBAL.PIPELINE.CLI.RUN_FIXTURE.01
    REQ: FUSBAL-V1-PLAYER-001, FUSBAL-V1-TEAM-001, FUSBAL-V1-BALL-001, FUSBAL-V1-EVENT-001, FUSBAL-V1-TRUST-001, FUSBAL-V1-OUT-001, SYS-ARCH-15
    WHY: Provide a repeatable, audit-friendly execution path for INT-030/INT-040 pipeline components.
    """

    out_dir = Path(args.out).expanduser().resolve()
    fixture_path = Path(args.fixture).expanduser().resolve()
    match_id = str(args.match_id)

    fixture_obj: Any = json.loads(fixture_path.read_text(encoding="utf8"))
    source = str(fixture_obj.get("source") or "fixture")
    frames = fixture_obj.get("frames")
    if not isinstance(frames, list):
        raise ValueError("fixture.frames must be a list")

    paths: MatchBundlePaths = ensure_bundle_layout(out_dir)
    manifest: MatchManifestV1 = _build_manifest_v1(
        match_id=match_id,
        video_paths=_normalize_paths([], base_dir=out_dir),
        sensor_paths=_normalize_paths([], base_dir=out_dir),
        notes="fixture run (INT-030)",
    )
    _write_json(paths.manifest_json, manifest)

    # Create required placeholders for non-tracks artifacts so `validate` focuses on contract.
    ensure_placeholder_outputs_v1(out_dir, sensors_present=False)

    gating = DetectionGatingConfig(min_confidence=float(args.min_confidence))
    ball_gating = BallDetectionGatingConfig(min_confidence=float(args.min_confidence))
    mot = SwapAvoidantMOT(source=source)
    ball_tracker = BallTracker(source=source)
    team_cfg = TeamSmoothingConfig(
        window_frames=int(args.team_window_frames),
        min_confidence=float(args.team_min_confidence),
        hysteresis=float(args.team_hysteresis),
    )
    team_assigner = TeamAssigner(cfg=team_cfg)
    team_color_cfg = TeamColorConfig(team_a_label=str(args.team_a_label), team_b_label=str(args.team_b_label))

    all_records: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_index = int(frame.get("frame_index") or 0)
        t_ms = int(frame.get("t_ms") or 0)
        raw_dets = frame.get("detections")
        if not isinstance(raw_dets, list):
            raw_dets = []

        detections, _frame_diag = emit_player_detections_v1(
            frame_index=frame_index,
            t_ms=t_ms,
            source=source,
            candidates=raw_dets,
            gating=gating,
        )

        tracked = mot.update(t_ms=t_ms, detections=detections)
        # Optional INT-040 fixture fields:
        # - ball_detections: list of {bbox_xyxy_px:[x1,y1,x2,y2], confidence:0..1, diagnostics?:{}}
        # - ball_unevaluable_reason: "frame_unavailable" | "detector_error"
        raw_ball = frame.get("ball_detections")
        if not isinstance(raw_ball, list):
            raw_ball = []
        ball_unevaluable_reason = frame.get("ball_unevaluable_reason")
        if not isinstance(ball_unevaluable_reason, str) or not ball_unevaluable_reason.strip():
            ball_unevaluable_reason = None
        if ball_unevaluable_reason is None:
            ball_dets, _ball_diag = emit_ball_detections_v1(
                frame_index=frame_index,
                t_ms=t_ms,
                source=source,
                candidates=raw_ball,
                gating=ball_gating,
            )
            ball_rec = ball_tracker.update(t_ms=t_ms, frame_index=frame_index, detections=ball_dets)
        else:
            ball_rec = ball_tracker.update(
                t_ms=t_ms,
                frame_index=frame_index,
                detections=[],
                unevaluable_reason=ball_unevaluable_reason,
            )
        all_records.append(dict(ball_rec))

        for rec in tracked:
            if rec.get("entity_type") != "player" or rec.get("pos_state") != "present":
                all_records.append(dict(rec))
                continue

            diag = rec.get("diagnostics") if isinstance(rec.get("diagnostics"), dict) else {}
            color_label = diag.get("color_label") if isinstance(diag, dict) else None
            score_a, score_b, color_diag = color_label_to_team_evidence(
                color_label=str(color_label) if color_label is not None else None,
                cfg=team_color_cfg,
            )
            team, team_confidence, team_diag = team_assigner.update(
                track_id=str(rec.get("track_id") or ""),
                score_a=score_a,
                score_b=score_b,
            )
            merged = dict(color_diag)
            merged.update(team_diag)
            all_records.append(
                dict(
                    annotate_track_with_team(
                        record=rec,
                        team=team,
                        team_confidence=team_confidence,
                        diagnostics=merged,
                    )
                )
            )

    all_records.sort(
        key=lambda r: (
            int(r.get("t_ms", 0)),
            str(r.get("entity_type", "")),
            str(r.get("track_id", "")),
            str(r.get("segment_id", "")),
            str(r.get("pos_state", "")),
        )
    )
    _write_jsonl(paths.tracks_jsonl, all_records)
    events = infer_shots_goals_v1(tracks=all_records, source="shots_goals_v1")
    _write_json(paths.events_json, events)
    _write_json(
        paths.diagnostics_quality_summary_json,
        {
            "schema_version": 1,
            "ball": compute_ball_quality_metrics_v1(tracks=all_records, cfg=ball_tracker.cfg),
        },
    )

    # Validate what we wrote.
    validate_args = argparse.Namespace(bundle=str(out_dir), format="text")
    rc = cmd_validate(validate_args)
    if rc != 0:
        return rc
    sys.stdout.write("[run-fixture] ok\n")
    return 0


def _normalize_paths(values: list[str], *, base_dir: Path) -> list[str]:
    # Prefer portable, deterministic paths rooted at the bundle directory.
    # We intentionally avoid `.resolve()` to prevent machine-specific absolute paths.
    normalized: list[str] = []
    for raw in values:
        p = Path(str(raw)).expanduser()
        try:
            if p.is_absolute():
                rel = p.relative_to(base_dir)
                normalized.append(rel.as_posix())
            else:
                normalized.append(p.as_posix())
        except Exception:
            # If not relative to the base (external absolute path), store only a portable basename.
            # This prevents machine-specific absolute paths from leaking into manifests/outputs.
            if p.is_absolute():
                base = p.name.strip()
                normalized.append(base if base else "external_video")
            else:
                normalized.append(p.as_posix())
    normalized = [s.replace("\\", "/") for s in normalized]
    return sorted(dict.fromkeys(normalized))


def _sorted_unique_strings(values: list[str]) -> list[str]:
    cleaned = [str(v).strip().replace("\\", "/") for v in values]
    return sorted(dict.fromkeys(cleaned))


def _expected_manifest_artifacts_v1() -> list[ManifestArtifactV1]:
    out: list[ManifestArtifactV1] = []
    for spec in BUNDLE_ARTIFACT_SPECS_V1:
        out.append(
            {
                "artifact_id": spec.artifact_id,
                "rel_path": spec.rel_path,
                "kind": spec.kind,
                "requirement": spec.requirement,
                "description": spec.description,
            }
        )
    out.sort(key=lambda a: a["artifact_id"])
    return out


def _build_manifest_v1(
    *,
    match_id: str,
    video_paths: list[str],
    sensor_paths: list[str],
    notes: str | None,
) -> MatchManifestV1:
    manifest: MatchManifestV1 = {
        "schema_version": 1,
        "bundle_layout_version": BUNDLE_LAYOUT_VERSION,
        "match_id": match_id,
        "inputs": {"video_paths": video_paths, "sensor_paths": sensor_paths},
        "artifacts": _expected_manifest_artifacts_v1(),
    }
    if notes:
        manifest["notes"] = notes
    return manifest


def cmd_ingest_video(args: argparse.Namespace) -> int:
    """Probe a local video and print deterministic metadata.

    PROV: FUSBAL.PIPELINE.CLI.INGEST_VIDEO.01
    REQ: REQ-V1-VIDEO-INGEST-001, SYS-ARCH-15
    WHY: Provide a smoke-able ingest entrypoint for UAT and diagnostics without producing a bundle.
    """

    try:
        video_path = Path(args.video).expanduser()
        out_dir = Path(args.out).expanduser().resolve() if args.out else Path(".").resolve()
        meta = probe_video_metadata(video_path=video_path, source_rel_path=_normalize_paths([str(video_path)], base_dir=out_dir)[0])
    except VideoIngestError as e:
        sys.stderr.write(f"[ingest-video:error] {e}\n")
        return int(getattr(e, "exit_code", 2))
    payload = {
        "fps": meta.fps,
        "width_px": meta.width_px,
        "height_px": meta.height_px,
        "nb_frames": meta.nb_frames,
        "duration_s": meta.duration_s,
        "source_rel_path": meta.source_rel_path,
        "diagnostics": meta.diagnostics,
    }
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _write_quality_summary_v1(paths: MatchBundlePaths, *, tracks: list[dict[str, Any]], ball_tracker: BallTracker) -> None:
    _write_json(
        paths.diagnostics_quality_summary_json,
        {
            "schema_version": 1,
            "ball": compute_ball_quality_metrics_v1(tracks=tracks, cfg=ball_tracker.cfg),
        },
    )


def _run_video_write_bundle(
    *,
    out_dir: Path,
    match_id: str,
    meta: VideoMetadata,
    frames: Iterable[VideoFrame],
    ball_detector_enabled: bool,
    ball_detector_cfg: BaselineBallDetectorConfig,
    min_confidence: float,
    shots_cfg: ShotsGoalsConfig | None = None,
) -> None:
    paths: MatchBundlePaths = ensure_bundle_layout(out_dir)

    manifest: MatchManifestV1 = _build_manifest_v1(
        match_id=str(match_id),
        video_paths=[str(meta.source_rel_path)],
        sensor_paths=[],
        notes="run-video (INT-050)",
    )
    _write_json(paths.manifest_json, manifest)

    ensure_placeholder_outputs_v1(out_dir, sensors_present=False)

    ball_gating = BallDetectionGatingConfig(min_confidence=float(min_confidence))
    ball_tracker = BallTracker(source="video")

    it = iter(frames)
    try:
        first = next(it)
    except StopIteration as e:
        raise VideoIngestError("no frames decoded (empty frames iterator)", exit_code=5) from e

    detector = (
        BaselineBallDetector(
            decode_width_px=int(first.width_px),
            decode_height_px=int(first.height_px),
            src_width_px=int(meta.width_px),
            src_height_px=int(meta.height_px),
            cfg=ball_detector_cfg,
        )
        if bool(ball_detector_enabled)
        else NullBallDetector()
    )

    records: list[dict[str, Any]] = []
    det_diag_rows: list[dict[str, Any]] = []
    frames_decoded = 0

    def iter_all() -> Iterator[VideoFrame]:
        yield first
        yield from it

    for f in iter_all():
        frames_decoded += 1
        raw_candidates = detector.detect(frame_index=int(f.frame_index), t_ms=int(f.t_ms), image_bytes=f.rgb24)
        det_records, _det_diag = emit_ball_detections_v1(
            frame_index=int(f.frame_index),
            t_ms=int(f.t_ms),
            source="video",
            candidates=raw_candidates,
            gating=ball_gating,
        )
        det_diag_rows.append(
            {
                "schema_version": 1,
                "frame_index": int(f.frame_index),
                "t_ms": int(f.t_ms),
                "detector_enabled": bool(ball_detector_enabled),
                "num_raw_candidates": int(len(raw_candidates)),
                "diagnostics": dict(_det_diag),
            }
        )
        ball_rec = ball_tracker.update(t_ms=int(f.t_ms), frame_index=int(f.frame_index), detections=det_records)
        records.append(dict(ball_rec))

    records.sort(
        key=lambda r: (
            int(r.get("t_ms", 0)),
            str(r.get("entity_type", "")),
            str(r.get("track_id", "")),
            str(r.get("segment_id", "")),
            str(r.get("pos_state", "")),
        )
    )
    _write_jsonl(paths.tracks_jsonl, records)
    events = infer_shots_goals_v1(tracks=records, source="shots_goals_v1", cfg=shots_cfg)
    _write_json(paths.events_json, events)
    _write_jsonl(paths.diagnostics_dir / "ball_detections.jsonl", det_diag_rows)
    _write_quality_summary_v1(paths, tracks=records, ball_tracker=ball_tracker)

    report = {
        "schema_version": 1,
        "placeholder": False,
        "notes": "run-video plumbing proof",
        "video": {
            "fps": meta.fps,
            "width_px": meta.width_px,
            "height_px": meta.height_px,
            "nb_frames": meta.nb_frames,
            "duration_s": meta.duration_s,
            "source_rel_path": meta.source_rel_path,
        },
        "counts": {
            "frames_decoded": int(frames_decoded),
            "ball_present": sum(1 for r in records if r.get("pos_state") == "present"),
            "ball_missing": sum(1 for r in records if r.get("pos_state") == "missing"),
            "ball_unknown": sum(1 for r in records if r.get("pos_state") == "unknown"),
            "events": len(events),
        },
    }
    _write_json(paths.report_json, report)

    # Validate what we wrote.
    validate_args = argparse.Namespace(bundle=str(out_dir), format="text")
    rc = cmd_validate(validate_args)
    if rc != 0:
        raise VideoIngestError(
            "bundle validation failed (run `fusbal-pipeline validate` for details)",
            exit_code=6,
        )


def cmd_run_video(args: argparse.Namespace) -> int:
    """Ingest a local video and export a valid V1 bundle (INT-050 plumbing proof).

    PROV: FUSBAL.PIPELINE.CLI.RUN_VIDEO.01
    REQ: REQ-V1-VIDEO-RUNNER-001, REQ-V1-VIDEO-INGEST-001, REQ-V1-BALL-DETECT-BASELINE-001, FUSBAL-V1-BALL-001, FUSBAL-V1-EVENT-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
    WHY: Provide an end-to-end local runner for UAT that remains deterministic and trust-first.
    """

    try:
        video_path = Path(args.video).expanduser()
        match_id = str(args.match_id)
        out_dir = Path(args.out).expanduser().resolve() if args.out else (Path("output") / match_id).resolve()

        source_rel = _normalize_paths([str(video_path)], base_dir=out_dir)[0]
        it = iter_video_frames_rgb24(
            video_path=video_path,
            source_rel_path=source_rel,
            max_decode_width_px=int(args.max_decode_width_px),
            max_frames=int(args.max_frames) if args.max_frames is not None else None,
        )
        try:
            meta, first_frame = next(it)
        except StopIteration:
            raise VideoIngestError(f"no frames decoded from video: {video_path}", exit_code=5)

        det_cfg = BaselineBallDetectorConfig(
            sample_step_px=int(args.det_sample_step_px),
            min_luma_0_to_255=int(args.det_min_luma),
            max_saturation_0_to_255=int(args.det_max_saturation),
            bbox_radius_px=int(args.det_bbox_radius_px),
            edge_margin_ratio_0_to_1=float(args.det_edge_margin_ratio),
            min_abs_delta_luma_0_to_255=int(args.det_min_abs_delta_luma),
        )
        shots_cfg = ShotsGoalsConfig(
            min_shot_speed_px_per_s=float(args.shot_min_speed_px_per_s),
            min_dt_ms=int(args.shot_min_dt_ms),
            goal_missing_ms=int(args.goal_missing_ms),
        )

        def frames_iter() -> Iterator[VideoFrame]:
            yield first_frame
            for _m, frame in it:
                yield frame

        _run_video_write_bundle(
            out_dir=out_dir,
            match_id=match_id,
            meta=meta,
            frames=frames_iter(),
            ball_detector_enabled=not bool(args.disable_ball_detector),
            ball_detector_cfg=det_cfg,
            min_confidence=float(args.min_confidence),
            shots_cfg=shots_cfg,
        )
    except VideoIngestError as e:
        sys.stderr.write(f"[run-video:error] {e}\n")
        return int(getattr(e, "exit_code", 2))

    sys.stdout.write("[run-video] ok\n")
    return 0


def _validate_manifest_v1(obj: object) -> tuple[MatchManifestV1 | None, list[str]]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return None, ["manifest must be a JSON object"]

    schema_version = obj.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        errors.append(f"manifest.schema_version must be {MANIFEST_SCHEMA_VERSION}")

    bundle_layout_version = obj.get("bundle_layout_version")
    if bundle_layout_version != BUNDLE_LAYOUT_VERSION:
        errors.append(f"manifest.bundle_layout_version must be {BUNDLE_LAYOUT_VERSION}")

    match_id = obj.get("match_id")
    if not isinstance(match_id, str) or not match_id.strip():
        errors.append("manifest.match_id must be a non-empty string")

    inputs = obj.get("inputs")
    if not isinstance(inputs, dict):
        errors.append("manifest.inputs must be an object")
        inputs = {}

    video_paths = inputs.get("video_paths")
    if not isinstance(video_paths, list) or not all(isinstance(x, str) for x in video_paths):
        errors.append("manifest.inputs.video_paths must be a list of strings")
        video_paths = []

    sensor_paths = inputs.get("sensor_paths")
    if not isinstance(sensor_paths, list) or not all(isinstance(x, str) for x in sensor_paths):
        errors.append("manifest.inputs.sensor_paths must be a list of strings")
        sensor_paths = []

    if isinstance(video_paths, list):
        normalized = _sorted_unique_strings(video_paths)
        if video_paths != normalized:
            errors.append("manifest.inputs.video_paths must be sorted and unique (deterministic)")

    if isinstance(sensor_paths, list):
        normalized = _sorted_unique_strings(sensor_paths)
        if sensor_paths != normalized:
            errors.append("manifest.inputs.sensor_paths must be sorted and unique (deterministic)")

    artifacts = obj.get("artifacts")
    if not isinstance(artifacts, list) or not all(isinstance(x, dict) for x in artifacts):
        errors.append("manifest.artifacts must be a list of objects")
        artifacts = []

    expected = _expected_manifest_artifacts_v1()
    if artifacts != expected:
        errors.append(
            "manifest.artifacts must exactly match the expected V1 artifact list "
            "(stable ids + ordering)"
        )

    if errors:
        return None, errors

    return obj, []


def cmd_validate(args: argparse.Namespace) -> int:
    bundle_root = Path(args.bundle).expanduser().resolve()
    paths = MatchBundlePaths(root=bundle_root)
    out_format: Literal["text", "json"] = getattr(args, "format", "text")

    errors: list[ValidationError] = []

    def emit_and_exit(code: int) -> int:
        if out_format == "json":
            payload = {"ok": code == 0, "errors": errors}
            sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            return code
        for e in errors:
            p = e.get("path")
            msg = e.get("message", "")
            c = e.get("code", "")
            ctx = ""
            if "line" in e:
                ctx = f" line={e['line']}"
            elif "index" in e:
                ctx = f" index={e['index']}"
            elif "field" in e:
                ctx = f" field={e['field']}"
            prefix = f"[validate:error] {c}"
            if p:
                sys.stderr.write(f"{prefix} {p}: {msg}{ctx}\n")
            else:
                sys.stderr.write(f"{prefix} {msg}{ctx}\n")
        if code == 0:
            sys.stdout.write("[validate] ok\n")
        return code

    def add_contract_error(kind: str, raw: str) -> None:
        # Try to parse the structured prefix added by contract validation.
        # Examples:
        #   <path>: line 12: track.confidence ...
        #   <path>: index 3: event.confidence ...
        msg = str(raw)
        if ": line " in msg:
            pfx, rest = msg.split(": line ", 1)
            line_str, detail = rest.split(": ", 1) if ": " in rest else (rest, "")
            try:
                line_no = int(line_str)
            except ValueError:
                line_no = None
            errors.append(
                make_error(
                    ERROR.TRACKS_INVALID if kind == "tracks" else ERROR.EVENTS_INVALID,
                    detail or msg,
                    path=pfx,
                    line=line_no,
                )
            )
            return
        if ": index " in msg:
            pfx, rest = msg.split(": index ", 1)
            idx_str, detail = rest.split(": ", 1) if ": " in rest else (rest, "")
            try:
                idx_no = int(idx_str)
            except ValueError:
                idx_no = None
            errors.append(make_error(ERROR.EVENTS_INVALID, detail or msg, path=pfx, index=idx_no))
            return
        errors.append(
            make_error(
                ERROR.TRACKS_INVALID if kind == "tracks" else ERROR.EVENTS_INVALID,
                msg,
            )
        )

    if not paths.manifest_json.exists():
        errors.append(
            make_error(ERROR.MANIFEST_MISSING, "missing manifest", path=str(paths.manifest_json))
        )
        return emit_and_exit(2)
    try:
        manifest_raw: Any = json.loads(paths.manifest_json.read_text(encoding="utf8"))
    except json.JSONDecodeError as e:
        errors.append(
            make_error(
                ERROR.MANIFEST_INVALID_JSON,
                f"invalid JSON: {e}",
                path=str(paths.manifest_json),
            )
        )
        return emit_and_exit(2)

    manifest, manifest_errors = _validate_manifest_v1(manifest_raw)
    if manifest_errors:
        for err in manifest_errors:
            errors.append(make_error(ERROR.MANIFEST_INVALID, err, path=str(paths.manifest_json)))
        return emit_and_exit(2)

    sensors_present = bool(manifest["inputs"]["sensor_paths"])
    layout_errors = validate_bundle_layout(bundle_root, sensors_present=sensors_present)
    if layout_errors:
        for err in layout_errors:
            errors.append(make_error(ERROR.BUNDLE_LAYOUT_INVALID, err, path=str(bundle_root)))
        return emit_and_exit(2)

    data_errors: list[str] = []
    data_errors.extend(validate_tracks_jsonl(paths.tracks_jsonl))
    data_errors.extend(validate_events_json(paths.events_json))
    if data_errors:
        for err in data_errors:
            kind = "events" if "/events.json" in err.replace("\\", "/") else "tracks"
            add_contract_error(kind, err)
        return emit_and_exit(2)

    return emit_and_exit(0)


def cmd_render_overlay(args: argparse.Namespace) -> int:
    """Render overlay.mp4 from an input video and tracks.jsonl.

    PROV: FUSBAL.PIPELINE.CLI.RENDER_OVERLAY.01
    REQ: FUSBAL-V1-OUT-001, FUSBAL-V1-BALL-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
    WHY: Provide a practical, human-useful overlay for UAT from contract-valid tracks.
    """

    try:
        render_overlay_mp4(
            video_path=Path(args.video).expanduser(),
            tracks_jsonl=Path(args.tracks).expanduser(),
            out_mp4=Path(args.out).expanduser(),
            fps=float(args.fps),
            max_ops=int(args.max_ops),
        )
    except OverlayError as e:
        sys.stderr.write(f"[render-overlay:error] {e}\n")
        return 2
    sys.stdout.write("[render-overlay] ok\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fusbal-pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest-video", help="Probe a local video and print deterministic metadata")
    p_ingest.add_argument("--video", required=True, help="Input video path (local)")
    p_ingest.add_argument(
        "--out",
        help="Optional bundle dir used to normalize paths (absolute paths outside this are reduced to basenames)",
    )
    p_ingest.set_defaults(func=cmd_ingest_video)

    p_run_video = sub.add_parser("run-video", help="Ingest a local video and export a valid V1 bundle (INT-050)")
    p_run_video.add_argument("--video", required=True, help="Input video path (local)")
    p_run_video.add_argument("--match-id", required=True, help="Match id (used for manifest + default output dir)")
    p_run_video.add_argument(
        "--out",
        help="Bundle root directory (default: output/<match_id>)",
    )
    p_run_video.add_argument("--max-frames", type=int, help="Optional cap for decoded frames (debug/UAT)")
    p_run_video.add_argument("--max-decode-width-px", type=int, default=640, help="Decode width cap (performance)")
    p_run_video.add_argument("--min-confidence", type=float, default=0.6, help="Min confidence for present ball points")
    p_run_video.add_argument(
        "--disable-ball-detector",
        action="store_true",
        help="Disable baseline ball detector (trust-first: expect only missing/unknown).",
    )
    p_run_video.add_argument("--det-sample-step-px", type=int, default=3)
    p_run_video.add_argument("--det-min-luma", type=int, default=220)
    p_run_video.add_argument("--det-max-saturation", type=int, default=40)
    p_run_video.add_argument("--det-bbox-radius-px", type=int, default=6)
    p_run_video.add_argument("--det-edge-margin-ratio", type=float, default=0.08)
    p_run_video.add_argument("--det-min-abs-delta-luma", type=int, default=3)
    p_run_video.add_argument(
        "--shot-min-speed-px-per-s",
        type=float,
        default=ShotsGoalsConfig.min_shot_speed_px_per_s,
        help="Min inferred shot speed in px/s (conservative default).",
    )
    p_run_video.add_argument(
        "--shot-min-dt-ms",
        type=int,
        default=ShotsGoalsConfig.min_dt_ms,
        help="Min dt between ball presents to infer a shot (ms).",
    )
    p_run_video.add_argument(
        "--goal-missing-ms",
        type=int,
        default=ShotsGoalsConfig.goal_missing_ms,
        help="Conservative goal candidate: missing duration after a shot (ms).",
    )
    p_run_video.set_defaults(func=cmd_run_video)

    p_init = sub.add_parser("init", help="Initialize a match bundle directory + manifest")
    p_init.add_argument("--match-id", required=True)
    p_init.add_argument(
        "--out",
        required=True,
        help="Bundle root directory (e.g. output/MATCH_001)",
    )
    p_init.add_argument(
        "--video",
        action="append",
        required=True,
        help="Video file path (repeatable)",
    )
    p_init.add_argument("--sensor", action="append", help="Optional sensor log path (repeatable)")
    p_init.add_argument(
        "--with-placeholders",
        action="store_true",
        help="Also create minimal placeholder outputs (for demos/tests)",
    )
    p_init.add_argument("--notes")
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run-fixture", help="Run a deterministic fixture and write a valid bundle")
    p_run.add_argument("--fixture", required=True, help="Path to a fixture JSON (deterministic inputs)")
    p_run.add_argument("--match-id", required=True)
    p_run.add_argument(
        "--out",
        required=True,
        help="Bundle root directory (e.g. output/MATCH_001)",
    )
    p_run.add_argument("--min-confidence", type=float, default=0.5)
    p_run.add_argument("--team-window-frames", type=int, default=15)
    p_run.add_argument("--team-min-confidence", type=float, default=0.65)
    p_run.add_argument("--team-hysteresis", type=float, default=0.10)
    p_run.add_argument("--team-a-label", type=str, default="yellow")
    p_run.add_argument("--team-b-label", type=str, default="blue")
    p_run.set_defaults(func=cmd_run_fixture)

    p_validate = sub.add_parser("validate", help="Validate a match bundle manifest")
    p_validate.add_argument("bundle", help="Bundle root directory (e.g. output/MATCH_001)")
    p_validate.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    p_validate.set_defaults(func=cmd_validate)

    p_overlay = sub.add_parser("render-overlay", help="Render overlay.mp4 from video + tracks.jsonl")
    p_overlay.add_argument("--video", required=True, help="Input video path (local)")
    p_overlay.add_argument("--tracks", required=True, help="tracks.jsonl path (contract-valid)")
    p_overlay.add_argument("--out", required=True, help="Output overlay.mp4 path")
    p_overlay.add_argument("--fps", type=float, default=30.0, help="FPS used to map t_ms to frame windows")
    p_overlay.add_argument("--max-ops", type=int, default=5000, help="Max draw operations to include (safety cap)")
    p_overlay.set_defaults(func=cmd_render_overlay)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
