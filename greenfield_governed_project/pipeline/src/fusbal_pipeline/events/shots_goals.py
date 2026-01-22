# PROV: FUSBAL.PIPELINE.EVENTS.SHOTS_GOALS.01
# REQ: FUSBAL-V1-EVENT-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Provide conservative (high precision) shots/goals inference with evidence pointers.

from __future__ import annotations

from dataclasses import dataclass

from ..bundle import BUNDLE_ARTIFACT_SPECS_V1
from ..contract import EventRecordV1, TrackRecordV1


def _bbox_center_xy(bbox: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _artifact_ids() -> set[str]:
    return {s.artifact_id for s in BUNDLE_ARTIFACT_SPECS_V1}


@dataclass(frozen=True)
class ShotsGoalsConfig:
    min_shot_speed_px_per_s: float = 900.0
    min_dt_ms: int = 10
    goal_missing_ms: int = 1500


def infer_shots_goals_v1(
    *,
    tracks: list[TrackRecordV1],
    source: str = "shots_goals_v1",
    cfg: ShotsGoalsConfig | None = None,
) -> list[EventRecordV1]:
    """Infer shots/goals conservatively from ball track records.

    This is intentionally conservative: it emits only `candidate` / `unknown` states unless
    a higher-confidence signal is introduced in a later intent.
    """

    config = cfg or ShotsGoalsConfig()
    ball = [
        r
        for r in tracks
        if r.get("entity_type") == "ball"
        and r.get("frame") == "image_px"
        and isinstance(r.get("t_ms"), int)
    ]
    ball.sort(key=lambda r: (int(r.get("t_ms", 0)), str(r.get("pos_state"))))

    allowed_artifacts = _artifact_ids()
    evidence_artifact = "tracks_jsonl" if "tracks_jsonl" in allowed_artifacts else "tracks"

    last_present: tuple[int, tuple[float, float]] | None = None
    last_present_t_ms: int | None = None
    last_shot_t_ms: int | None = None
    missing_run_start_ms: int | None = None

    out: list[EventRecordV1] = []

    for rec in ball:
        t_ms = int(rec.get("t_ms", 0))
        pos_state = rec.get("pos_state")
        if pos_state == "present" and isinstance(rec.get("bbox_xyxy_px"), list):
            bbox = rec.get("bbox_xyxy_px")
            if isinstance(bbox, list) and len(bbox) == 4:
                cxy = _bbox_center_xy([int(x) for x in bbox])
                if last_present and last_present_t_ms is not None:
                    dt_ms = t_ms - last_present_t_ms
                    if dt_ms >= config.min_dt_ms:
                        speed = _dist(last_present[1], cxy) / (dt_ms / 1000.0)
                        if speed >= config.min_shot_speed_px_per_s:
                            last_shot_t_ms = t_ms
                            out.append(
                                {
                                    "schema_version": 1,
                                    "t_ms": t_ms,
                                    "event_type": "shot",
                                    "event_state": "candidate",
                                    "confidence": 0.6,
                                    "source": str(source),
                                    "evidence": [
                                        {
                                            "artifact_id": evidence_artifact,
                                            "time_range_ms": {
                                                "start_ms": max(0, t_ms - 300),
                                                "end_ms": t_ms + 300,
                                            },
                                        }
                                    ],
                                    "diagnostics": {"speed_px_per_s": float(speed)},
                                }
                            )
                last_present = (t_ms, cxy)
                last_present_t_ms = t_ms
                missing_run_start_ms = None
            continue

        if pos_state == "missing":
            if missing_run_start_ms is None:
                missing_run_start_ms = t_ms
            if (
                last_shot_t_ms is not None
                and missing_run_start_ms is not None
                and (t_ms - missing_run_start_ms) >= config.goal_missing_ms
            ):
                # Very conservative: prolonged missing after a shot is at most a goal *candidate*.
                out.append(
                    {
                        "schema_version": 1,
                        "t_ms": int(missing_run_start_ms),
                        "event_type": "goal",
                        "event_state": "unknown",
                        "confidence": 0.4,
                        "source": str(source),
                        "evidence": [
                            {
                                "artifact_id": evidence_artifact,
                                "time_range_ms": {
                                    "start_ms": max(0, int(missing_run_start_ms) - 500),
                                    "end_ms": int(missing_run_start_ms) + int(config.goal_missing_ms),
                                },
                            }
                        ],
                        "diagnostics": {
                            "goal_missing_ms": int(config.goal_missing_ms),
                            "missing_run_ms": int(t_ms - missing_run_start_ms),
                        },
                    }
                )
                last_shot_t_ms = None
                missing_run_start_ms = None

    out.sort(key=lambda e: (int(e.get("t_ms", 0)), str(e.get("event_type"))))
    return out

