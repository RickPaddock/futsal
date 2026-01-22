"""fusbal_pipeline.calibration.inputs

PROV: FUSBAL.PIPELINE.TASK_CAL_001.SUB_003
REQ: FUSBAL-V1-CAL-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
WHY: Load and validate per-venue/pitch calibration input records with deterministic, actionable errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from fusbal_pipeline.errors import ERROR, ValidationError, make_error

CALIBRATION_INPUT_SCHEMA_VERSION = 1


class PitchTemplateRefV1(TypedDict):
    pitch_template_id: str


class CalibrationInputV1(TypedDict):
    schema_version: Literal[1]
    venue_id: str
    pitch_id: str
    pitch_template_ref: PitchTemplateRefV1
    camera_id: str
    image_pre_undistorted: bool
    image_size_px: NotRequired[list[int]]  # [width, height]
    source_video_path: NotRequired[str]  # provenance only (no implied existence)


def _validate_non_empty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_image_size_px(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    w, h = value
    if not isinstance(w, int) or isinstance(w, bool):
        return False
    if not isinstance(h, int) or isinstance(h, bool):
        return False
    return w > 0 and h > 0


def validate_calibration_input_v1(obj: object, *, path: str | None = None) -> list[ValidationError]:
    errors: list[ValidationError] = []
    if not isinstance(obj, dict):
        return [make_error(ERROR.CAL_INPUT_INVALID, "calibration_input must be a JSON object", path=path)]

    if obj.get("schema_version") != CALIBRATION_INPUT_SCHEMA_VERSION:
        errors.append(
            make_error(
                ERROR.CAL_INPUT_INVALID,
                f"calibration_input.schema_version must be {CALIBRATION_INPUT_SCHEMA_VERSION}",
                path=path,
                field="schema_version",
                value=obj.get("schema_version"),
            )
        )

    for key in ("venue_id", "pitch_id", "camera_id"):
        if not _validate_non_empty_str(obj.get(key)):
            errors.append(
                make_error(
                    ERROR.CAL_INPUT_INVALID,
                    f"calibration_input.{key} must be a non-empty string",
                    path=path,
                    field=key,
                    value=obj.get(key),
                )
            )

    if not isinstance(obj.get("image_pre_undistorted"), bool):
        errors.append(
            make_error(
                ERROR.CAL_INPUT_INVALID,
                "calibration_input.image_pre_undistorted must be a boolean",
                path=path,
                field="image_pre_undistorted",
                value=obj.get("image_pre_undistorted"),
            )
        )

    if "image_size_px" in obj and not _validate_image_size_px(obj.get("image_size_px")):
        errors.append(
            make_error(
                ERROR.CAL_INPUT_INVALID,
                "calibration_input.image_size_px must be [width, height] positive integers",
                path=path,
                field="image_size_px",
                value=obj.get("image_size_px"),
            )
        )

    if "source_video_path" in obj and not _validate_non_empty_str(obj.get("source_video_path")):
        errors.append(
            make_error(
                ERROR.CAL_INPUT_INVALID,
                "calibration_input.source_video_path must be a non-empty string when present",
                path=path,
                field="source_video_path",
                value=obj.get("source_video_path"),
            )
        )

    ref = obj.get("pitch_template_ref")
    if not isinstance(ref, dict):
        errors.append(
            make_error(
                ERROR.CAL_INPUT_INVALID,
                "calibration_input.pitch_template_ref must be an object",
                path=path,
                field="pitch_template_ref",
                value=ref,
            )
        )
        ref = {}
    if not _validate_non_empty_str(ref.get("pitch_template_id")):
        errors.append(
            make_error(
                ERROR.CAL_INPUT_INVALID,
                "calibration_input.pitch_template_ref.pitch_template_id must be a non-empty string",
                path=path,
                field="pitch_template_ref.pitch_template_id",
                value=ref.get("pitch_template_id"),
            )
        )

    return errors


def load_calibration_input(path: Path) -> tuple[CalibrationInputV1 | None, list[ValidationError]]:
    try:
        obj = json.loads(path.read_text(encoding="utf8"))
    except FileNotFoundError:
        return None, [make_error(ERROR.CAL_INPUT_MISSING, f"missing calibration input: {path}", path=str(path))]
    except json.JSONDecodeError as e:
        return None, [make_error(ERROR.CAL_INPUT_INVALID_JSON, f"invalid JSON: {e}", path=str(path))]

    errors = validate_calibration_input_v1(obj, path=str(path))
    if errors:
        return None, errors
    return obj, []


def _main_validate(path: Path) -> int:
    _, errors = load_calibration_input(path)
    payload = {"ok": not errors, "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fusbal_pipeline.calibration.inputs")
    parser.add_argument(
        "--validate",
        metavar="PATH",
        help="Validate a calibration_input.json record and emit JSON {ok, errors}.",
    )
    args = parser.parse_args(argv)
    if not args.validate:
        parser.error("missing required argument: --validate PATH")
    return _main_validate(Path(args.validate))


if __name__ == "__main__":
    raise SystemExit(main())

