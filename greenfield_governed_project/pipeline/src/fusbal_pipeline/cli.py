# PROV: FUSBAL.PIPELINE.CLI.01
# REQ: FUSBAL-V1-OUT-001, FUSBAL-V1-DATA-001, SYS-ARCH-15
# WHY: Provide a minimal CLI to initialize and validate Fusbal match bundles and manifests.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

from .bundle import BUNDLE_ARTIFACT_SPECS_V1, BUNDLE_LAYOUT_VERSION, MatchBundlePaths, ensure_bundle_layout, validate_bundle_layout
from .contract import validate_events_json, validate_tracks_jsonl


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


def cmd_init(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).expanduser().resolve()
    paths: MatchBundlePaths = ensure_bundle_layout(out_dir)
    video_paths = _normalize_paths(args.video)
    sensor_paths = _normalize_paths(args.sensor or [])
    manifest: MatchManifestV1 = _build_manifest_v1(
        match_id=str(args.match_id),
        video_paths=video_paths,
        sensor_paths=sensor_paths,
        notes=str(args.notes) if args.notes else None,
    )
    _write_json(paths.manifest_json, manifest)
    return 0


def _normalize_paths(values: list[str]) -> list[str]:
    resolved = [str(Path(p).expanduser().resolve()) for p in values]
    return sorted(dict.fromkeys(resolved))


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


def _build_manifest_v1(*, match_id: str, video_paths: list[str], sensor_paths: list[str], notes: str | None) -> MatchManifestV1:
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
        normalized = _normalize_paths(video_paths)
        if video_paths != normalized:
            errors.append("manifest.inputs.video_paths must be sorted and unique (deterministic)")

    if isinstance(sensor_paths, list):
        normalized = _normalize_paths(sensor_paths)
        if sensor_paths != normalized:
            errors.append("manifest.inputs.sensor_paths must be sorted and unique (deterministic)")

    artifacts = obj.get("artifacts")
    if not isinstance(artifacts, list) or not all(isinstance(x, dict) for x in artifacts):
        errors.append("manifest.artifacts must be a list of objects")
        artifacts = []

    expected = _expected_manifest_artifacts_v1()
    if artifacts != expected:
        errors.append("manifest.artifacts must exactly match the expected V1 artifact list (stable ids + ordering)")

    if errors:
        return None, errors

    return obj, []


def cmd_validate(args: argparse.Namespace) -> int:
    bundle_root = Path(args.bundle).expanduser().resolve()
    paths = MatchBundlePaths(root=bundle_root)
    if not paths.manifest_json.exists():
        sys.stderr.write(f"[validate:error] missing manifest: {paths.manifest_json}\n")
        return 2
    try:
        manifest_raw: Any = json.loads(paths.manifest_json.read_text(encoding="utf8"))
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[validate:error] invalid JSON in {paths.manifest_json}: {e}\n")
        return 2

    manifest, manifest_errors = _validate_manifest_v1(manifest_raw)
    if manifest_errors:
        for err in manifest_errors:
            sys.stderr.write(f"[validate:error] {paths.manifest_json}: {err}\n")
        return 2

    sensors_present = bool(manifest["inputs"]["sensor_paths"])
    layout_errors = validate_bundle_layout(bundle_root, sensors_present=sensors_present)
    if layout_errors:
        for err in layout_errors:
            sys.stderr.write(f"[validate:error] {bundle_root}: {err}\n")
        return 2

    data_errors: list[str] = []
    data_errors.extend(validate_tracks_jsonl(paths.tracks_jsonl))
    data_errors.extend(validate_events_json(paths.events_json))
    if data_errors:
        for err in data_errors:
            sys.stderr.write(f"[validate:error] {err}\n")
        return 2

    sys.stdout.write("[validate] ok\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fusbal-pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Initialize a match bundle directory + manifest")
    p_init.add_argument("--match-id", required=True)
    p_init.add_argument("--out", required=True, help="Bundle root directory (e.g. output/MATCH_001)")
    p_init.add_argument("--video", action="append", required=True, help="Video file path (repeatable)")
    p_init.add_argument("--sensor", action="append", help="Optional sensor log path (repeatable)")
    p_init.add_argument("--notes")
    p_init.set_defaults(func=cmd_init)

    p_validate = sub.add_parser("validate", help="Validate a match bundle manifest")
    p_validate.add_argument("bundle", help="Bundle root directory (e.g. output/MATCH_001)")
    p_validate.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
