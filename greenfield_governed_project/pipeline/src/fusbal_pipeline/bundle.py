# PROV: FUSBAL.PIPELINE.BUNDLE.01
# REQ: FUSBAL-V1-OUT-001
# WHY: Define the deterministic output bundle layout and helper utilities.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


def ensure_bundle_layout(bundle_root: Path) -> MatchBundlePaths:
    bundle_root.mkdir(parents=True, exist_ok=True)
    paths = MatchBundlePaths(root=bundle_root)
    paths.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    return paths
