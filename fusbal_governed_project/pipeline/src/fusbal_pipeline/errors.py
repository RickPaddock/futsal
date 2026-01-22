"""fusbal_pipeline.errors

PROV: FUSBAL.PIPELINE.ERRORS.01
REQ: FUSBAL-V1-TRUST-001, SYS-ARCH-15
WHY: Provide stable, machine-readable validation error codes and a small JSON-safe error shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


class ValidationError(TypedDict, total=False):
    code: str
    message: str
    path: str
    line: int
    index: int
    field: str
    value: Any


@dataclass(frozen=True)
class ErrorCodes:
    # Manifest
    MANIFEST_MISSING: str = "E_MANIFEST_MISSING"
    MANIFEST_INVALID_JSON: str = "E_MANIFEST_INVALID_JSON"
    MANIFEST_INVALID: str = "E_MANIFEST_INVALID"

    # Bundle layout
    BUNDLE_LAYOUT_INVALID: str = "E_BUNDLE_LAYOUT_INVALID"

    # Data files
    TRACKS_INVALID: str = "E_TRACKS_INVALID"
    EVENTS_INVALID: str = "E_EVENTS_INVALID"

    # Calibration (V1)
    CAL_PITCH_TEMPLATE_MISSING: str = "E_CAL_PITCH_TEMPLATE_MISSING"
    CAL_PITCH_TEMPLATE_INVALID_JSON: str = "E_CAL_PITCH_TEMPLATE_INVALID_JSON"
    CAL_PITCH_TEMPLATE_INVALID: str = "E_CAL_PITCH_TEMPLATE_INVALID"

    CAL_INPUT_MISSING: str = "E_CAL_INPUT_MISSING"
    CAL_INPUT_INVALID_JSON: str = "E_CAL_INPUT_INVALID_JSON"
    CAL_INPUT_INVALID: str = "E_CAL_INPUT_INVALID"

    CAL_MARKINGS_MISSING: str = "E_CAL_MARKINGS_MISSING"
    CAL_MARKINGS_INVALID_JSON: str = "E_CAL_MARKINGS_INVALID_JSON"
    CAL_MARKINGS_INVALID: str = "E_CAL_MARKINGS_INVALID"

    CAL_MANUAL_MISSING: str = "E_CAL_MANUAL_MISSING"
    CAL_MANUAL_INVALID_JSON: str = "E_CAL_MANUAL_INVALID_JSON"
    CAL_MANUAL_INVALID: str = "E_CAL_MANUAL_INVALID"

    CAL_RESULT_MISSING: str = "E_CAL_RESULT_MISSING"
    CAL_RESULT_INVALID_JSON: str = "E_CAL_RESULT_INVALID_JSON"
    CAL_RESULT_INVALID: str = "E_CAL_RESULT_INVALID"


ERROR = ErrorCodes()


def make_error(code: str, message: str, **context: Any) -> ValidationError:
    err: ValidationError = {"code": code, "message": message}
    for k, v in context.items():
        if v is None:
            continue
        err[k] = v
    return err
