# PROV: FUSBAL.PIPELINE.CLI.01
# REQ: FUSBAL-V1-OUT-001, FUSBAL-V1-DATA-001
# WHY: Provide a minimal CLI to initialize and validate Fusbal match bundles and manifests.

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .bundle import MatchBundlePaths, ensure_bundle_layout


@dataclass(frozen=True)
class MatchManifest:
    match_id: str
    video_paths: list[str]
    sensor_paths: list[str]
    notes: str | None = None


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf8")


def cmd_init(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).expanduser().resolve()
    paths: MatchBundlePaths = ensure_bundle_layout(out_dir)
    manifest = MatchManifest(
        match_id=args.match_id,
        video_paths=[str(Path(p).expanduser().resolve()) for p in args.video],
        sensor_paths=[str(Path(p).expanduser().resolve()) for p in (args.sensor or [])],
        notes=args.notes,
    )
    _write_json(paths.manifest_json, asdict(manifest))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    bundle_root = Path(args.bundle).expanduser().resolve()
    paths = MatchBundlePaths(root=bundle_root)
    if not paths.manifest_json.exists():
        sys.stderr.write(f"[validate:error] missing manifest: {paths.manifest_json}\n")
        return 2
    try:
        manifest = json.loads(paths.manifest_json.read_text(encoding="utf8"))
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[validate:error] invalid JSON in {paths.manifest_json}: {e}\n")
        return 2

    missing = []
    for key in ("match_id", "video_paths", "sensor_paths"):
        if key not in manifest:
            missing.append(key)
    if missing:
        sys.stderr.write(f"[validate:error] manifest missing keys: {', '.join(missing)}\n")
        return 2

    if not isinstance(manifest["video_paths"], list) or not all(isinstance(x, str) for x in manifest["video_paths"]):
        sys.stderr.write("[validate:error] manifest.video_paths must be a list of strings\n")
        return 2

    if not isinstance(manifest["sensor_paths"], list) or not all(isinstance(x, str) for x in manifest["sensor_paths"]):
        sys.stderr.write("[validate:error] manifest.sensor_paths must be a list of strings\n")
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
