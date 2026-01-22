# PROV: FUSBAL.PIPELINE.OVERLAY.RENDER.01
# REQ: FUSBAL-V1-OUT-001, FUSBAL-V1-DATA-001, FUSBAL-V1-BALL-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Render a human-useful overlay.mp4 from an input video and contract-valid tracks.jsonl.

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class OverlayError(RuntimeError):
    pass


@dataclass(frozen=True)
class BallBox:
    t_ms: int
    bbox_xyxy_px: list[int]
    confidence: float


def _clamp_0_1(x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return float(x)


def _parse_ball_boxes_from_tracks_jsonl(tracks_jsonl: Path) -> list[BallBox]:
    boxes: list[BallBox] = []
    if not tracks_jsonl.is_file():
        raise OverlayError(f"tracks file missing: {tracks_jsonl}")

    with tracks_jsonl.open("r", encoding="utf8") as f:
        for idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise OverlayError(f"{tracks_jsonl}: line {idx}: invalid JSON: {e}") from e
            if not isinstance(rec, dict):
                continue
            if rec.get("entity_type") != "ball":
                continue
            if rec.get("frame") != "image_px":
                continue
            if rec.get("pos_state") != "present":
                continue
            t_ms = rec.get("t_ms")
            bbox = rec.get("bbox_xyxy_px")
            conf = rec.get("confidence", 0.0)
            if not isinstance(t_ms, int) or isinstance(t_ms, bool) or t_ms < 0:
                continue
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in bbox)
            ):
                continue
            x1, y1, x2, y2 = bbox
            if x2 < x1 or y2 < y1:
                continue
            if not isinstance(conf, (int, float)) or isinstance(conf, bool):
                conf = 0.0
            boxes.append(BallBox(t_ms=int(t_ms), bbox_xyxy_px=[x1, y1, x2, y2], confidence=_clamp_0_1(float(conf))))

    boxes.sort(key=lambda b: (b.t_ms, b.bbox_xyxy_px[0], b.bbox_xyxy_px[1], -b.confidence))
    return boxes


def build_ball_overlay_filter(*, tracks_jsonl: Path, fps: float, max_ops: int = 5000) -> str:
    """Build a deterministic ffmpeg filter string for ball overlay (bbox + confidence).

    Note: output mp4 bytes may vary by ffmpeg build; this function is deterministic for a given
    tracks input.
    """

    if fps <= 0:
        raise OverlayError(f"fps must be > 0 (got {fps})")
    if max_ops <= 0:
        raise OverlayError(f"max_ops must be > 0 (got {max_ops})")

    dt_s = 1.0 / float(fps)
    boxes = _parse_ball_boxes_from_tracks_jsonl(tracks_jsonl)

    if not boxes:
        return "null"

    parts: list[str] = []
    count = 0
    for b in boxes:
        if count >= max_ops:
            break
        x1, y1, x2, y2 = b.bbox_xyxy_px
        w = x2 - x1
        h = y2 - y1
        start_s = b.t_ms / 1000.0
        end_s = start_s + dt_s
        enable = f"between(t\\,{start_s:.3f}\\,{end_s:.3f})"

        parts.append(
            "drawbox="
            + f"x={x1}:y={y1}:w={w}:h={h}"
            + ":color=lime@0.75:t=2"
            + f":enable='{enable}'"
        )
        # Draw confidence label just above the bbox (best-effort).
        label = f"ball {b.confidence:.2f}"
        text_x = x1
        text_y = max(0, y1 - 24)
        parts.append(
            "drawtext="
            + f"text={shlex.quote(label)}"
            + f":x={text_x}:y={text_y}:fontsize=22"
            + ":fontcolor=white:borderw=2:bordercolor=black"
            + f":enable='{enable}'"
        )
        count += 1

    return ",".join(parts)


def _require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise OverlayError(
            "ffmpeg is required to render overlay.mp4 but was not found on PATH. "
            "Install ffmpeg (e.g., `brew install ffmpeg`) and retry."
        )
    return exe


def build_ffmpeg_overlay_args(
    *,
    video_path: Path,
    tracks_jsonl: Path,
    out_mp4: Path,
    fps: float,
    max_ops: int = 5000,
) -> list[str]:
    ffmpeg = _require_ffmpeg()
    vf = build_ball_overlay_filter(tracks_jsonl=tracks_jsonl, fps=fps, max_ops=max_ops)
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "-preset",
        "veryfast",
        str(out_mp4),
    ]


def render_overlay_mp4(
    *,
    video_path: Path,
    tracks_jsonl: Path,
    out_mp4: Path,
    fps: float,
    max_ops: int = 5000,
) -> None:
    if not video_path.is_file():
        raise OverlayError(f"video file missing: {video_path}")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    args = build_ffmpeg_overlay_args(
        video_path=video_path,
        tracks_jsonl=tracks_jsonl,
        out_mp4=out_mp4,
        fps=fps,
        max_ops=max_ops,
    )
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise OverlayError(f"ffmpeg failed (exit {proc.returncode}): {msg or 'unknown error'}")

