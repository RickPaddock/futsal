# PROV: FUSBAL.PIPELINE.VIDEO.INGEST.01
# REQ: REQ-V1-VIDEO-INGEST-001, FUSBAL-V1-TRUST-001, SYS-ARCH-15
# WHY: Deterministically decode local videos using explicit ffprobe/ffmpeg tooling and a defined timebase rule.

from __future__ import annotations

import json
import math
import os
import re
import select
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


class VideoIngestError(RuntimeError):
    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = int(exit_code)


EXIT_CODE_MISSING_INPUT = 3
EXIT_CODE_MISSING_TOOL = 4
EXIT_CODE_DECODE_FAILED = 5


_MIN_FFMPEGISH_VERSION = (4, 0, 0)


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    width_px: int
    height_px: int
    nb_frames: int | None
    duration_s: float | None
    source_rel_path: str
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class VideoFrame:
    frame_index: int
    t_ms: int
    width_px: int
    height_px: int
    pix_fmt: str
    rgb24: bytes


def _is_finite_positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0


def _parse_fraction(text: str) -> float | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    if "/" in raw:
        a, b = raw.split("/", 1)
        try:
            num = float(a)
            den = float(b)
        except ValueError:
            return None
        if den == 0:
            return None
        return float(num / den)
    try:
        return float(raw)
    except ValueError:
        return None


def compute_t_ms_from_frame_index(*, frame_index: int, fps: float) -> int:
    """Deterministic timebase rule for V1: t_ms = round(frame_index * 1000 / fps)."""
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise VideoIngestError(f"invalid frame_index (expected int>=0): {frame_index!r}")
    if not _is_finite_positive_number(fps):
        raise VideoIngestError(f"invalid fps (expected finite > 0): {fps!r}")
    return int(round(frame_index * 1000.0 / float(fps)))


def _require_tool(name: str) -> str:
    env_key = "FUSBAL_FFPROBE_PATH" if name == "ffprobe" else "FUSBAL_FFMPEG_PATH" if name == "ffmpeg" else None
    if env_key:
        override = os.environ.get(env_key)
        if override:
            p = Path(str(override)).expanduser()
            if p.is_file():
                exe = str(p)
                if name in {"ffmpeg", "ffprobe"}:
                    vline = _probe_tool_version(exe)
                    parsed = _parse_ffmpegish_version(vline or "") if vline else None
                    if parsed and parsed < _MIN_FFMPEGISH_VERSION:
                        raise VideoIngestError(
                            f"{name} too old: {vline!r} (need >= {_MIN_FFMPEGISH_VERSION[0]}.{_MIN_FFMPEGISH_VERSION[1]})",
                            exit_code=EXIT_CODE_MISSING_TOOL,
                        )
                return exe
            raise VideoIngestError(
                f"configured tool path does not exist for {name}: {override} (via {env_key})",
                exit_code=EXIT_CODE_MISSING_TOOL,
            )

    exe = shutil.which(name)
    if exe:
        if name in {"ffmpeg", "ffprobe"}:
            vline = _probe_tool_version(exe)
            parsed = _parse_ffmpegish_version(vline or "") if vline else None
            if parsed and parsed < _MIN_FFMPEGISH_VERSION:
                raise VideoIngestError(
                    f"{name} too old: {vline!r} (need >= {_MIN_FFMPEGISH_VERSION[0]}.{_MIN_FFMPEGISH_VERSION[1]})",
                    exit_code=EXIT_CODE_MISSING_TOOL,
                )
        return exe
    raise VideoIngestError(
        f"missing required tool '{name}'. Install ffmpeg/ffprobe and ensure it is on PATH "
        f"(or set {env_key}).",
        exit_code=EXIT_CODE_MISSING_TOOL,
    )


def _stderr_excerpt(stderr: str, *, max_chars: int = 1000) -> str:
    s = (stderr or "").strip()
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[-max_chars:]


def _probe_tool_version(exe: str, *, timeout_s: float = 2.0) -> str | None:
    # Best-effort only; never fail ingest due to version probing.
    try:
        res = _run_checked([str(exe), "-version"], timeout_s=timeout_s)
    except VideoIngestError:
        return None
    if res.returncode != 0:
        return None
    out = res.stdout.decode("utf8", errors="replace").strip().splitlines()
    if not out:
        return None
    # Keep deterministic, compact.
    return out[0].strip()[:200] if out[0].strip() else None


def _parse_ffmpegish_version(first_line: str) -> tuple[int, int, int] | None:
    # Accept common formats:
    #   ffmpeg version 6.1.1 ...
    #   ffprobe version 6.1.1 ...
    s = (first_line or "").strip()
    if not s:
        return None
    m = re.search(r"\bff(?:mpeg|probe)\s+version\s+([0-9]+)(?:\.([0-9]+))?(?:\.([0-9]+))?\b", s)
    if not m:
        return None
    try:
        major = int(m.group(1))
        minor = int(m.group(2) or 0)
        patch = int(m.group(3) or 0)
    except ValueError:
        return None
    return major, minor, patch


def _run_checked(args: list[str], *, timeout_s: float | None = None) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        raise VideoIngestError(
            f"command timed out after {timeout_s}s: {args[0]}",
            exit_code=EXIT_CODE_DECODE_FAILED,
        ) from e
    except OSError as e:
        raise VideoIngestError(
            f"failed to execute command: {args[0]} ({e})",
            exit_code=EXIT_CODE_DECODE_FAILED,
        ) from e


def probe_video_metadata(*, video_path: Path, source_rel_path: str) -> VideoMetadata:
    """Probe a local video file using ffprobe (deterministic JSON output)."""
    if not isinstance(video_path, Path):
        video_path = Path(str(video_path))
    if not video_path.is_file():
        raise VideoIngestError(
            f"video file not found: {video_path}",
            exit_code=EXIT_CODE_MISSING_INPUT,
        )

    ffprobe = _require_tool("ffprobe")
    timeout_s = float(os.environ.get("FUSBAL_FFPROBE_TIMEOUT_S", "10.0"))
    res = _run_checked(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,width,height,nb_frames,duration",
            "-of",
            "json",
            str(video_path),
        ],
        timeout_s=timeout_s,
    )
    if res.returncode != 0:
        msg = res.stderr.decode("utf8", errors="replace")
        raise VideoIngestError(
            f"ffprobe failed for {video_path}: {_stderr_excerpt(msg) or 'unknown_error'}",
            exit_code=EXIT_CODE_DECODE_FAILED,
        )
    try:
        obj = json.loads(res.stdout.decode("utf8", errors="replace"))
    except json.JSONDecodeError as e:
        raise VideoIngestError(
            f"ffprobe returned invalid JSON for {video_path}: {e}",
            exit_code=EXIT_CODE_DECODE_FAILED,
        ) from e

    streams = obj.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise VideoIngestError(
            f"ffprobe missing stream info for {video_path}",
            exit_code=EXIT_CODE_DECODE_FAILED,
        )
    s0 = streams[0]
    width = s0.get("width")
    height = s0.get("height")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise VideoIngestError(
            f"invalid video width from ffprobe for {video_path}: {width!r}",
            exit_code=EXIT_CODE_DECODE_FAILED,
        )
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise VideoIngestError(
            f"invalid video height from ffprobe for {video_path}: {height!r}",
            exit_code=EXIT_CODE_DECODE_FAILED,
        )

    fps = _parse_fraction(str(s0.get("avg_frame_rate") or "")) or _parse_fraction(str(s0.get("r_frame_rate") or ""))
    if fps is None or not _is_finite_positive_number(fps):
        raise VideoIngestError(
            f"invalid fps from ffprobe for {video_path} (avg_frame_rate={s0.get('avg_frame_rate')!r}, r_frame_rate={s0.get('r_frame_rate')!r})",
            exit_code=EXIT_CODE_DECODE_FAILED,
        )

    nb_frames_raw = s0.get("nb_frames")
    nb_frames: int | None = None
    if isinstance(nb_frames_raw, str) and nb_frames_raw.strip().isdigit():
        nb_frames = int(nb_frames_raw.strip())
    duration_raw = s0.get("duration")
    duration_s: float | None = None
    if isinstance(duration_raw, str):
        try:
            duration_s = float(duration_raw)
        except ValueError:
            duration_s = None
    elif isinstance(duration_raw, (int, float)) and not isinstance(duration_raw, bool):
        duration_s = float(duration_raw)

    diagnostics: dict[str, object] = {
        "ffprobe_path": str(ffprobe),
        "ffprobe_version": _probe_tool_version(str(ffprobe)) or "unknown",
        "ffprobe_stream_0": s0,
    }
    return VideoMetadata(
        fps=float(fps),
        width_px=int(width),
        height_px=int(height),
        nb_frames=nb_frames,
        duration_s=duration_s,
        source_rel_path=str(source_rel_path),
        diagnostics=diagnostics,
    )


def iter_video_frames_rgb24(
    *,
    video_path: Path,
    source_rel_path: str,
    max_decode_width_px: int = 640,
    max_frames: int | None = None,
) -> Iterator[tuple[VideoMetadata, VideoFrame]]:
    """Yield (metadata, frame) pairs with rgb24 bytes for each decoded frame.

    Implementation is explicit (ffmpeg pipe) and timebase is derived from frame_index and probed fps.
    """

    meta = probe_video_metadata(video_path=video_path, source_rel_path=source_rel_path)

    ffmpeg = _require_tool("ffmpeg")

    # Attach ffmpeg info to diagnostics (best-effort).
    meta = VideoMetadata(
        fps=meta.fps,
        width_px=meta.width_px,
        height_px=meta.height_px,
        nb_frames=meta.nb_frames,
        duration_s=meta.duration_s,
        source_rel_path=meta.source_rel_path,
        diagnostics={
            **dict(meta.diagnostics or {}),
            "ffmpeg_path": str(ffmpeg),
            "ffmpeg_version": _probe_tool_version(str(ffmpeg)) or "unknown",
            "ffmpeg_read_timeout_s": float(os.environ.get("FUSBAL_FFMPEG_READ_TIMEOUT_S", "10.0")),
        },
    )

    if not isinstance(max_decode_width_px, int) or isinstance(max_decode_width_px, bool) or max_decode_width_px <= 0:
        raise VideoIngestError(f"invalid max_decode_width_px: {max_decode_width_px!r}")

    src_w = meta.width_px
    src_h = meta.height_px
    if src_w <= max_decode_width_px:
        dec_w = src_w
    else:
        dec_w = int(max_decode_width_px)
    dec_h = int(round(src_h * (dec_w / float(src_w))))
    # ffmpeg scale expects even dimensions for many codecs; keep deterministic.
    if dec_h % 2 == 1:
        dec_h += 1
    if dec_h <= 0:
        raise VideoIngestError(f"invalid derived decode height: {dec_h}")

    vf = f"scale={dec_w}:{dec_h}"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as e:
        raise VideoIngestError(
            f"failed to start ffmpeg for {video_path}: {e}",
            exit_code=EXIT_CODE_DECODE_FAILED,
        ) from e

    assert proc.stdout is not None
    assert proc.stderr is not None
    frame_size = int(dec_w * dec_h * 3)
    frame_index = 0
    stopped_early = False
    read_timeout_s = float(os.environ.get("FUSBAL_FFMPEG_READ_TIMEOUT_S", "10.0"))
    try:
        while True:
            if max_frames is not None and frame_index >= int(max_frames):
                stopped_early = True
                break
            ready, _w, _x = select.select([proc.stdout], [], [], read_timeout_s)
            if not ready:
                raise VideoIngestError(
                    f"ffmpeg decode timed out after {read_timeout_s}s while reading frame {frame_index} for {video_path}",
                    exit_code=EXIT_CODE_DECODE_FAILED,
                )
            buf = proc.stdout.read(frame_size)
            if not buf:
                break
            if len(buf) != frame_size:
                raise VideoIngestError(
                    f"decode truncated for {video_path}: expected {frame_size} bytes, got {len(buf)}",
                    exit_code=EXIT_CODE_DECODE_FAILED,
                )
            t_ms = compute_t_ms_from_frame_index(frame_index=frame_index, fps=meta.fps)
            yield meta, VideoFrame(
                frame_index=int(frame_index),
                t_ms=int(t_ms),
                width_px=int(dec_w),
                height_px=int(dec_h),
                pix_fmt="rgb24",
                rgb24=bytes(buf),
            )
            frame_index += 1
    finally:
        if stopped_early:
            try:
                proc.terminate()
            except Exception:
                pass
        proc.stdout.close()
        stderr = proc.stderr.read().decode("utf8", errors="replace")
        proc.stderr.close()
        rc = proc.wait()
        if rc != 0 and "Broken pipe" not in stderr:
            raise VideoIngestError(
                f"ffmpeg decode failed for {video_path}: {_stderr_excerpt(stderr) or 'unknown_error'}",
                exit_code=EXIT_CODE_DECODE_FAILED,
            )
