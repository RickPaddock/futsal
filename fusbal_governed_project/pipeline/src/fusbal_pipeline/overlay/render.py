# PROV: FUSBAL.PIPELINE.OVERLAY.RENDER.01
# REQ: FUSBAL-V1-OUT-001, FUSBAL-V1-DATA-001, FUSBAL-V1-BALL-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Render a human-useful overlay.mp4 from an input video and contract-valid tracks.jsonl.

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class OverlayError(RuntimeError):
    pass


@dataclass(frozen=True)
class OverlayPlan:
    vf: str
    vf_sha256: str
    enable_expr: str
    fps: float
    width_px: int | None
    height_px: int | None
    eligible_boxes: int
    selected_boxes: int
    draw_ops: int


@dataclass(frozen=True)
class OverlayRenderResult:
    ok: bool
    out_mp4: Path
    plan: OverlayPlan
    ffmpeg_path: str
    ffmpeg_version_first_line: str | None
    stderr_excerpt: str | None
    error: str | None


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


def _stderr_excerpt(text: str, *, max_lines: int = 24, max_chars: int = 2400) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    lines = raw.splitlines()
    tail = "\n".join(lines[-max_lines:])
    if len(tail) <= max_chars:
        return tail
    # Keep the end (often has the most relevant ffmpeg error).
    return tail[-max_chars:]


def _sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf8"))
    return h.hexdigest()


def _frame_index_from_t_ms(*, t_ms: int, fps: float) -> int:
    # Deterministic mapping used for n-based enable expressions.
    # Note: uses float fps, but avoids per-op float time windows in the ffmpeg filter.
    return int(round((float(t_ms) * float(fps)) / 1000.0))


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
    return boxes


def _ball_box_sort_key(b: BallBox) -> tuple[int, int, int, int, int, float]:
    x1, y1, x2, y2 = b.bbox_xyxy_px
    # Sort by time ascending, then geometry, then confidence descending.
    return (b.t_ms, x1, y1, x2, y2, -float(b.confidence))


def _select_best_per_frame(
    boxes: list[BallBox],
    *,
    fps: float,
    width_px: int | None,
    height_px: int | None,
) -> list[BallBox]:
    # Deduplicate by computed frame index: highest confidence wins; stable tie-break by geometry.
    best_by_frame: dict[int, BallBox] = {}

    def normalize_bbox(b: BallBox) -> BallBox | None:
        x1, y1, x2, y2 = b.bbox_xyxy_px
        if width_px is not None and height_px is not None:
            if width_px <= 0 or height_px <= 0:
                return None
            x1 = max(0, min(int(x1), int(width_px) - 1))
            x2 = max(0, min(int(x2), int(width_px) - 1))
            y1 = max(0, min(int(y1), int(height_px) - 1))
            y2 = max(0, min(int(y2), int(height_px) - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return BallBox(t_ms=b.t_ms, bbox_xyxy_px=[x1, y1, x2, y2], confidence=b.confidence)

    def rank(b: BallBox) -> tuple[float, int, int, int, int, int]:
        x1, y1, x2, y2 = b.bbox_xyxy_px
        # Higher confidence preferred, then deterministic tie-break.
        return (-float(b.confidence), x1, y1, x2, y2, int(b.t_ms))

    for b in boxes:
        nb = normalize_bbox(b)
        if nb is None:
            continue
        frame_index = _frame_index_from_t_ms(t_ms=nb.t_ms, fps=fps)
        prev = best_by_frame.get(frame_index)
        if prev is None or rank(nb) < rank(prev):
            best_by_frame[frame_index] = nb

    # Return deterministic order by frame index, then tie-break.
    out = list(best_by_frame.values())
    out.sort(key=_ball_box_sort_key)
    return out


def build_ball_overlay_filter(
    *,
    tracks_jsonl: Path,
    fps: float,
    max_ops: int = 5000,
    width_px: int | None = None,
    height_px: int | None = None,
) -> str:
    """Build a deterministic ffmpeg filter string for ball overlay (bbox + confidence).

    Note: output mp4 bytes may vary by ffmpeg build; this function is deterministic for a given
    tracks input.
    """

    if fps <= 0:
        raise OverlayError(f"fps must be > 0 (got {fps})")
    if max_ops <= 0:
        raise OverlayError(f"max_ops must be > 0 (got {max_ops})")

    boxes_raw = _parse_ball_boxes_from_tracks_jsonl(tracks_jsonl)
    boxes = _select_best_per_frame(boxes_raw, fps=fps, width_px=width_px, height_px=height_px)

    if not boxes:
        return "null"

    parts: list[str] = []
    # Each selected box produces 2 draw ops.
    max_boxes = int(max_ops) // 2
    if max_boxes <= 0:
        return "null"
    if len(boxes) > max_boxes:
        # Avoid sorting the full eligible set when truncating heavily.
        # Determinism comes from the explicit key.
        import heapq

        boxes = heapq.nsmallest(max_boxes, boxes, key=_ball_box_sort_key)
        boxes.sort(key=_ball_box_sort_key)

    for b in boxes:
        x1, y1, x2, y2 = b.bbox_xyxy_px
        w = x2 - x1
        h = y2 - y1
        # n-based (frame index) enable expression to avoid float drift.
        frame_index = _frame_index_from_t_ms(t_ms=b.t_ms, fps=fps)
        enable = f"eq(n\\,{frame_index})"

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

    return ",".join(parts)


def _require_ffmpeg() -> str:
    override = str(os.environ.get("FUSBAL_FFMPEG_PATH", "")).strip()
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return str(p)
        exe = shutil.which(override)
        if exe:
            return exe
        raise OverlayError(
            "ffmpeg override was set but could not be resolved. "
            "Check FUSBAL_FFMPEG_PATH and ensure the file exists or is on PATH."
        )

    exe = shutil.which("ffmpeg")
    if not exe:
        raise OverlayError(
            "ffmpeg is required to render overlay.mp4 but was not found on PATH. "
            "Install ffmpeg (e.g., `brew install ffmpeg`) and retry. "
            "(You can also set FUSBAL_FFMPEG_PATH to an explicit ffmpeg binary.)"
        )
    return exe


def _probe_ffmpeg_version_first_line(ffmpeg_path: str) -> str | None:
    try:
        res = subprocess.run(
            [ffmpeg_path, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    out = (res.stdout or "").splitlines()
    if out:
        line = out[0].strip()
        return line or None
    err = (res.stderr or "").splitlines()
    if err:
        line = err[0].strip()
        return line or None
    return None


def build_ffmpeg_overlay_args(
    *,
    ffmpeg_path: str | None = None,
    video_path: Path,
    tracks_jsonl: Path,
    out_mp4: Path,
    fps: float,
    max_ops: int = 5000,
    width_px: int | None = None,
    height_px: int | None = None,
) -> list[str]:
    ffmpeg = str(ffmpeg_path).strip() if ffmpeg_path else _require_ffmpeg()
    vf = build_ball_overlay_filter(
        tracks_jsonl=tracks_jsonl,
        fps=fps,
        max_ops=max_ops,
        width_px=width_px,
        height_px=height_px,
    )
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
    timeout_s: float | None = None,
    width_px: int | None = None,
    height_px: int | None = None,
) -> OverlayRenderResult:
    if not video_path.is_file():
        raise OverlayError(f"video file missing: {video_path}")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Build a deterministic plan/report regardless of ffmpeg runtime behavior.
    vf = build_ball_overlay_filter(
        tracks_jsonl=tracks_jsonl,
        fps=fps,
        max_ops=max_ops,
        width_px=width_px,
        height_px=height_px,
    )
    plan = OverlayPlan(
        vf=vf,
        vf_sha256=_sha256_text(vf),
        enable_expr="n",
        fps=float(fps),
        width_px=int(width_px) if width_px is not None else None,
        height_px=int(height_px) if height_px is not None else None,
        eligible_boxes=len(_parse_ball_boxes_from_tracks_jsonl(tracks_jsonl)),
        selected_boxes=0 if vf == "null" else vf.count("drawbox="),
        draw_ops=0 if vf == "null" else (vf.count("drawbox=") + vf.count("drawtext=")),
    )

    ffmpeg = _require_ffmpeg()
    ffmpeg_version = _probe_ffmpeg_version_first_line(ffmpeg)

    args = build_ffmpeg_overlay_args(
        ffmpeg_path=ffmpeg,
        video_path=video_path,
        tracks_jsonl=tracks_jsonl,
        out_mp4=out_mp4,
        fps=fps,
        max_ops=max_ops,
        width_px=width_px,
        height_px=height_px,
    )
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        err = _stderr_excerpt(str(e))
        return OverlayRenderResult(
            ok=False,
            out_mp4=out_mp4,
            plan=plan,
            ffmpeg_path=ffmpeg,
            ffmpeg_version_first_line=ffmpeg_version,
            stderr_excerpt=err or None,
            error=f"ffmpeg timed out after {timeout_s}s" + (f": {err}" if err else ""),
        )

    stderr_excerpt = _stderr_excerpt(proc.stderr or proc.stdout or "")
    if proc.returncode != 0:
        msg = stderr_excerpt or "unknown error"
        return OverlayRenderResult(
            ok=False,
            out_mp4=out_mp4,
            plan=plan,
            ffmpeg_path=ffmpeg,
            ffmpeg_version_first_line=ffmpeg_version,
            stderr_excerpt=stderr_excerpt or None,
            error=f"ffmpeg failed (exit {proc.returncode}): {msg}",
        )

    return OverlayRenderResult(
        ok=True,
        out_mp4=out_mp4,
        plan=plan,
        ffmpeg_path=ffmpeg,
        ffmpeg_version_first_line=ffmpeg_version,
        stderr_excerpt=stderr_excerpt or None,
        error=None,
    )

