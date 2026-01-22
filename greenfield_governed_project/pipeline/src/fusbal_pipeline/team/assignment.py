# PROV: FUSBAL.PIPELINE.TEAM.ASSIGNMENT.01
# REQ: FUSBAL-V1-TEAM-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Assign team labels (A/B/unknown) with explicit temporal smoothing and trust-first Unknown.

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..contract import TeamLabel, TrackRecordV1


@dataclass(frozen=True)
class TeamSmoothingConfig:
    window_frames: int = 15
    min_confidence: float = 0.65
    hysteresis: float = 0.10


def _clamp_0_1(value: float) -> float:
    if value <= 0:
        return 0.0
    if value >= 1:
        return 1.0
    return float(value)


@dataclass
class TeamAssigner:
    cfg: TeamSmoothingConfig = field(default_factory=TeamSmoothingConfig)
    _history: dict[str, deque[tuple[float, float]]] = field(default_factory=dict)
    _last_team: dict[str, TeamLabel] = field(default_factory=dict)

    def update(
        self,
        *,
        track_id: str,
        score_a: float,
        score_b: float,
    ) -> tuple[TeamLabel, float, dict[str, object]]:
        wa = _clamp_0_1(float(score_a))
        wb = _clamp_0_1(float(score_b))
        h = self._history.get(track_id)
        if h is None:
            h = deque(maxlen=max(1, int(self.cfg.window_frames)))
            self._history[track_id] = h
        h.append((wa, wb))

        mean_a = sum(x[0] for x in h) / len(h)
        mean_b = sum(x[1] for x in h) / len(h)
        denom = mean_a + mean_b
        if denom <= 1e-9:
            team: TeamLabel = "unknown"
            conf = 0.0
            diag = {
                "unknown_reason": "no_color_evidence",
                "smoothing": {"window_frames": int(self.cfg.window_frames), "hysteresis": float(self.cfg.hysteresis)},
            }
            self._last_team[track_id] = team
            return team, conf, diag

        p_a = mean_a / denom
        p_b = mean_b / denom
        preferred: TeamLabel = "A" if p_a >= p_b else "B"
        conf = float(max(p_a, p_b))

        last = self._last_team.get(track_id, "unknown")
        if last != "unknown" and preferred != last and abs(p_a - p_b) < float(self.cfg.hysteresis):
            preferred = last

        if conf < float(self.cfg.min_confidence):
            team = "unknown"
            diag = {
                "unknown_reason": "low_confidence",
                "smoothing": {"window_frames": int(self.cfg.window_frames), "hysteresis": float(self.cfg.hysteresis)},
                "color_evidence": {"p_a": float(p_a), "p_b": float(p_b)},
            }
        else:
            team = preferred
            diag = {
                "smoothing": {"window_frames": int(self.cfg.window_frames), "hysteresis": float(self.cfg.hysteresis)},
                "color_evidence": {"p_a": float(p_a), "p_b": float(p_b)},
            }

        self._last_team[track_id] = team
        return team, _clamp_0_1(conf), diag


def annotate_track_with_team(
    *,
    record: TrackRecordV1,
    team: TeamLabel,
    team_confidence: float,
    diagnostics: dict[str, object],
) -> TrackRecordV1:
    out = dict(record)
    out["team"] = team
    out["team_confidence"] = _clamp_0_1(float(team_confidence))
    merged_diag: dict[str, object] = {}
    if isinstance(record.get("diagnostics"), dict):
        merged_diag.update(record["diagnostics"])  # type: ignore[index]
    merged_diag.update(diagnostics)
    out["diagnostics"] = merged_diag
    return out  # type: ignore[return-value]
