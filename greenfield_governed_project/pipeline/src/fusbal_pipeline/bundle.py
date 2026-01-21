# PROV: FUSBAL.PIPELINE.BUNDLE.01
# REQ: FUSBAL-V1-OUT-001, SYS-ARCH-15
# WHY: Define the deterministic output bundle layout and helper utilities.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ArtifactKind = Literal["file", "dir"]
ArtifactRequirement = Literal["required", "optional", "required_if_sensors_present"]

BUNDLE_LAYOUT_VERSION = 1


@dataclass(frozen=True)
class BundleArtifactSpec:
    artifact_id: str
    rel_path: str
    kind: ArtifactKind
    requirement: ArtifactRequirement
    description: str


BUNDLE_ARTIFACT_SPECS_V1: tuple[BundleArtifactSpec, ...] = (
    BundleArtifactSpec(
        artifact_id="manifest_json",
        rel_path="manifest.json",
        kind="file",
        requirement="required",
        description="Bundle manifest (schema + inputs + expected artifacts).",
    ),
    BundleArtifactSpec(
        artifact_id="overlay_mp4",
        rel_path="overlay.mp4",
        kind="file",
        requirement="required",
        description="Rendered video overlay with boxes/IDs/labels.",
    ),
    BundleArtifactSpec(
        artifact_id="bev_mp4",
        rel_path="bev.mp4",
        kind="file",
        requirement="optional",
        description="Best-effort birds-eye-view render (only when calibration passes).",
    ),
    BundleArtifactSpec(
        artifact_id="tracks_jsonl",
        rel_path="tracks.jsonl",
        kind="file",
        requirement="required",
        description="Canonical track records (JSON Lines).",
    ),
    BundleArtifactSpec(
        artifact_id="events_json",
        rel_path="events.json",
        kind="file",
        requirement="required",
        description="Canonical events list (JSON).",
    ),
    BundleArtifactSpec(
        artifact_id="report_json",
        rel_path="report.json",
        kind="file",
        requirement="required",
        description="Report summary and diagnostics (JSON).",
    ),
    BundleArtifactSpec(
        artifact_id="report_html",
        rel_path="report.html",
        kind="file",
        requirement="required",
        description="Rendered human report (HTML).",
    ),
    BundleArtifactSpec(
        artifact_id="diagnostics_dir",
        rel_path="diagnostics",
        kind="dir",
        requirement="required",
        description="Diagnostics directory (calibration/sync/quality summaries).",
    ),
    BundleArtifactSpec(
        artifact_id="diagnostics_calibration_json",
        rel_path="diagnostics/calibration.json",
        kind="file",
        requirement="required",
        description="Calibration diagnostics JSON.",
    ),
    BundleArtifactSpec(
        artifact_id="diagnostics_sync_json",
        rel_path="diagnostics/sync.json",
        kind="file",
        requirement="required_if_sensors_present",
        description="Sync diagnostics JSON (required when sensors/multicam are used).",
    ),
    BundleArtifactSpec(
        artifact_id="diagnostics_quality_summary_json",
        rel_path="diagnostics/quality_summary.json",
        kind="file",
        requirement="required",
        description="Quality summary diagnostics JSON.",
    ),
)


@dataclass(frozen=True)
class MatchBundlePaths:
    root: Path

    @property
    def overlay_mp4(self) -> Path:
        return self.root / "overlay.mp4"

    @property
    def bev_mp4(self) -> Path:
        return self.root / "bev.mp4"

    @property
    def tracks_jsonl(self) -> Path:
        return self.root / "tracks.jsonl"

    @property
    def events_json(self) -> Path:
        return self.root / "events.json"

    @property
    def report_json(self) -> Path:
        return self.root / "report.json"

    @property
    def report_html(self) -> Path:
        return self.root / "report.html"

    @property
    def diagnostics_dir(self) -> Path:
        return self.root / "diagnostics"

    @property
    def diagnostics_calibration_json(self) -> Path:
        return self.diagnostics_dir / "calibration.json"

    @property
    def diagnostics_sync_json(self) -> Path:
        return self.diagnostics_dir / "sync.json"

    @property
    def diagnostics_quality_summary_json(self) -> Path:
        return self.diagnostics_dir / "quality_summary.json"

    @property
    def manifest_json(self) -> Path:
        return self.root / "manifest.json"


def required_artifacts_v1(*, sensors_present: bool) -> tuple[BundleArtifactSpec, ...]:
    required: list[BundleArtifactSpec] = []
    for spec in BUNDLE_ARTIFACT_SPECS_V1:
        if spec.requirement == "required":
            required.append(spec)
        elif spec.requirement == "required_if_sensors_present" and sensors_present:
            required.append(spec)
    return tuple(required)


def validate_bundle_layout(bundle_root: Path, *, sensors_present: bool) -> list[str]:
    errors: list[str] = []
    for spec in required_artifacts_v1(sensors_present=sensors_present):
        abs_path = bundle_root / spec.rel_path
        if spec.kind == "dir":
            if not abs_path.is_dir():
                errors.append(f"missing required directory '{spec.rel_path}' ({spec.artifact_id})")
        else:
            if not abs_path.is_file():
                errors.append(f"missing required file '{spec.rel_path}' ({spec.artifact_id})")
    return errors


def ensure_bundle_layout(bundle_root: Path) -> MatchBundlePaths:
    bundle_root.mkdir(parents=True, exist_ok=True)
    paths = MatchBundlePaths(root=bundle_root)
    paths.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    return paths
