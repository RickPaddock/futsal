"""Audit all ball detector candidates over a frame window.

Usage:
  c:/.../.venv/Scripts/python.exe scripts/audit_ball_candidates.py \
      --input videos/input/GoPro_Futsal_part1_CLEANED_clip7.mp4 \
      --output-dir videos/output/GoPro_Futsal_part1_CLEANED_clip7 \
      --start-frame 859 \
      --end-frame 1025 \
      --conf-threshold 0.0 \
      --debug-video

This is a diagnostic-only tool. It reruns only the ball model and writes every
candidate kept by the detector wrapper for each frame in the requested window.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detectors.ball_detector import BallDetector
from src.utils.file_utils import save_json
from src.utils.video_io import VideoReader


def _window_stem(start_frame: int, end_frame: int) -> str:
    return f"ball_candidates_{start_frame}_{end_frame}"


def _candidate_to_dict(rank: int, bbox: List[float], confidence: float) -> Dict[str, object]:
    centroid_x = (float(bbox[0]) + float(bbox[2])) / 2.0
    centroid_y = (float(bbox[1]) + float(bbox[3])) / 2.0
    return {
        "rank": rank,
        "bbox": [float(value) for value in bbox],
        "centroid": [centroid_x, centroid_y],
        "confidence": float(confidence),
    }


def _candidate_color(rank: int, confidence: float) -> Tuple[int, int, int]:
    if rank == 0:
        return (0, 255, 255)
    if confidence >= 0.5:
        return (80, 220, 80)
    if confidence >= 0.2:
        return (0, 165, 255)
    return (0, 0, 255)


def _draw_text_with_bg(
    frame,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.55,
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    x1 = max(0, x - 4)
    y1 = max(0, y - text_h - 4)
    x2 = min(frame.shape[1] - 1, x + text_w + 4)
    y2 = min(frame.shape[0] - 1, y + baseline + 4)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_candidate_overlay(frame, frame_idx: int, candidates: List[Dict[str, object]]) -> None:
    _draw_text_with_bg(
        frame,
        f"frame {frame_idx} | candidates {len(candidates)}",
        (16, 28),
        (255, 255, 255),
        font_scale=0.7,
        thickness=2,
    )

    for candidate in candidates:
        rank = int(candidate["rank"])
        bbox = candidate["bbox"]
        centroid = candidate["centroid"]
        confidence = float(candidate["confidence"])
        color = _candidate_color(rank, confidence)

        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        cx, cy = [int(round(value)) for value in centroid]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 4, color, -1)
        _draw_text_with_bg(
            frame,
            f"#{rank + 1} {confidence:.3f}",
            (x1, max(18, y1 - 6)),
            color,
        )


def _select_render_candidates(
    candidates: List[Dict[str, object]],
    render_threshold: float,
    max_render_candidates: int,
) -> Tuple[List[Dict[str, object]], int]:
    visible = [candidate for candidate in candidates if float(candidate["confidence"]) >= render_threshold]

    if not visible and candidates:
        visible = [candidates[0]]

    limited = visible[:max(1, max_render_candidates)]
    suppressed_count = max(0, len(candidates) - len(limited))
    return limited, suppressed_count


def _draw_render_summary(
    frame,
    total_candidates: int,
    shown_candidates: int,
    suppressed_count: int,
    render_threshold: float,
) -> None:
    _draw_text_with_bg(
        frame,
        (
            f"shown {shown_candidates}/{total_candidates} "
            f"candidates | suppressed {suppressed_count} | render_threshold {render_threshold:.3f}"
        ),
        (16, 54),
        (255, 255, 255),
        font_scale=0.55,
        thickness=1,
    )


def _report_progress(processed_frames: int, total_frames: int, last_reported_percent: int) -> int:
    if total_frames <= 0:
        return last_reported_percent

    percent_complete = int((processed_frames * 100) / total_frames)
    while percent_complete >= last_reported_percent + 5 and last_reported_percent < 100:
        last_reported_percent += 5
        print(f"Progress: {last_reported_percent}% ({processed_frames}/{total_frames} frames)")

    return last_reported_percent


def render_ball_candidate_video_from_json(
    audit_json_path: str,
    input_path: Optional[str],
    output_video_path: Optional[str],
    render_threshold: float,
    max_render_candidates: int,
) -> Dict[str, object]:
    json_path = Path(audit_json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    resolved_input = input_path or str(payload.get("video_path") or "")
    if not resolved_input:
        raise ValueError("input video path is required for render-only mode")

    resolved_output = output_video_path
    if resolved_output is None:
        resolved_output = str(json_path.with_name(f"{json_path.stem}_debug.mp4"))

    frame_map = {int(frame["frame_idx"]): frame for frame in payload.get("frames", [])}
    start_frame = int(payload.get("start_frame", 0))
    end_frame = int(payload.get("end_frame", start_frame - 1))
    total_frames = max(0, end_frame - start_frame + 1)
    processed_frames = 0
    last_reported_percent = 0

    with VideoReader(str(resolved_input)) as reader:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(resolved_output),
            fourcc,
            reader.fps,
            (reader.width, reader.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open debug video writer: {resolved_output}")

        try:
            for frame_idx, frame in reader.iter_frames():
                if frame_idx < start_frame:
                    continue
                if frame_idx > end_frame:
                    break

                frame_payload = frame_map.get(frame_idx, {"candidates": []})
                candidates = list(frame_payload.get("candidates", []))
                render_candidates, suppressed_count = _select_render_candidates(
                    candidates,
                    render_threshold=render_threshold,
                    max_render_candidates=max_render_candidates,
                )

                overlay_frame = frame.copy()
                _draw_candidate_overlay(overlay_frame, frame_idx, render_candidates)
                _draw_render_summary(
                    overlay_frame,
                    total_candidates=len(candidates),
                    shown_candidates=len(render_candidates),
                    suppressed_count=suppressed_count,
                    render_threshold=render_threshold,
                )
                writer.write(overlay_frame)
                processed_frames += 1
                last_reported_percent = _report_progress(
                    processed_frames,
                    total_frames,
                    last_reported_percent,
                )
        finally:
            writer.release()

    if total_frames > 0 and last_reported_percent < 100:
        print(f"Progress: 100% ({total_frames}/{total_frames} frames)")

    payload["debug_video_path"] = str(resolved_output)
    payload["render_threshold"] = render_threshold
    payload["max_render_candidates"] = max_render_candidates
    return payload


def audit_ball_candidates(
    input_path: str,
    output_dir: str,
    start_frame: int,
    end_frame: int,
    ball_model_path: str,
    conf_threshold: float,
    debug_video: bool,
    use_inference_slicer: bool,
    render_threshold: float,
    max_render_candidates: int,
) -> Dict[str, object]:
    video_path = Path(input_path)
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    detector = BallDetector(
        model_path=ball_model_path,
        use_inference_slicer=use_inference_slicer,
    )

    artifact_stem = _window_stem(start_frame, end_frame)
    json_path = destination_dir / f"{artifact_stem}.json"
    debug_video_path = destination_dir / f"{artifact_stem}_debug.mp4"

    frames_payload: List[Dict[str, object]] = []
    total_candidates = 0
    max_candidates_in_frame = 0
    writer = None

    with VideoReader(str(video_path)) as reader:
        if start_frame < 0:
            raise ValueError("start_frame must be >= 0")

        effective_end = min(end_frame, reader.total_frames - 1)
        if effective_end < start_frame:
            raise ValueError(
                f"Invalid frame window [{start_frame}, {end_frame}] for video with {reader.total_frames} frames"
            )
        total_window_frames = effective_end - start_frame + 1
        processed_frames = 0
        last_reported_percent = 0

        if debug_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(debug_video_path),
                fourcc,
                reader.fps,
                (reader.width, reader.height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open debug video writer: {debug_video_path}")

        try:
            for frame_idx, frame in reader.iter_frames():
                if frame_idx < start_frame:
                    continue
                if frame_idx > effective_end:
                    break

                detections = detector.detect_all(frame, conf_threshold=conf_threshold)
                candidate_rows = [
                    _candidate_to_dict(rank=rank, bbox=bbox, confidence=confidence)
                    for rank, (bbox, confidence) in enumerate(detections)
                ]

                total_candidates += len(candidate_rows)
                max_candidates_in_frame = max(max_candidates_in_frame, len(candidate_rows))
                frames_payload.append(
                    {
                        "frame_idx": frame_idx,
                        "candidate_count": len(candidate_rows),
                        "candidates": candidate_rows,
                    }
                )

                if writer is not None:
                    overlay_frame = frame.copy()
                    render_candidates, suppressed_count = _select_render_candidates(
                        candidate_rows,
                        render_threshold=render_threshold,
                        max_render_candidates=max_render_candidates,
                    )
                    _draw_candidate_overlay(overlay_frame, frame_idx, render_candidates)
                    _draw_render_summary(
                        overlay_frame,
                        total_candidates=len(candidate_rows),
                        shown_candidates=len(render_candidates),
                        suppressed_count=suppressed_count,
                        render_threshold=render_threshold,
                    )
                    writer.write(overlay_frame)

                processed_frames += 1
                last_reported_percent = _report_progress(
                    processed_frames,
                    total_window_frames,
                    last_reported_percent,
                )
        finally:
            if writer is not None:
                writer.release()

        if total_window_frames > 0 and last_reported_percent < 100:
            print(f"Progress: 100% ({total_window_frames}/{total_window_frames} frames)")

        payload: Dict[str, object] = {
            "video_name": video_path.stem,
            "video_path": str(video_path),
            "ball_model_path": ball_model_path,
            "start_frame": start_frame,
            "end_frame": effective_end,
            "frame_count": len(frames_payload),
            "fps": reader.fps,
            "width": reader.width,
            "height": reader.height,
            "conf_threshold": conf_threshold,
            "use_inference_slicer": use_inference_slicer,
            "render_threshold": render_threshold,
            "max_render_candidates": max_render_candidates,
            "total_candidates": total_candidates,
            "max_candidates_in_frame": max_candidates_in_frame,
            "frames": frames_payload,
        }

    save_json(payload, str(json_path))

    if debug_video:
        payload["debug_video_path"] = str(debug_video_path)
    payload["json_path"] = str(json_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit all ball detector candidates over a frame window")
    parser.add_argument("--input", help="Path to input video")
    parser.add_argument("--output-dir", help="Directory for audit outputs")
    parser.add_argument("--start-frame", type=int, help="Inclusive start frame")
    parser.add_argument("--end-frame", type=int, help="Inclusive end frame")
    parser.add_argument(
        "--audit-json",
        help="Existing candidate audit JSON to use for render-only mode",
    )
    parser.add_argument(
        "--ball-model",
        default="models/BALL_MODEL_best_v2.pt",
        help="Path to ball detection model",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.05,
        help="Confidence threshold passed to the model (default: 0.05)",
    )
    parser.add_argument(
        "--debug-video",
        action="store_true",
        help="Render a debug video with every candidate overlaid",
    )
    parser.add_argument(
        "--render-threshold",
        type=float,
        default=0.05,
        help="Minimum confidence drawn in the debug video; JSON still keeps all candidates above --conf-threshold (default: 0.05)",
    )
    parser.add_argument(
        "--max-render-candidates",
        type=int,
        default=8,
        help="Maximum number of candidates drawn per frame in the debug video (default: 8)",
    )
    parser.add_argument(
        "--no-inference-slicer",
        action="store_true",
        help="Disable tiled inference for the audit run",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip inference and render a filtered debug video from --audit-json",
    )
    parser.add_argument(
        "--output-video",
        help="Override debug video path for render-only mode",
    )

    args = parser.parse_args()

    if args.render_only:
        if not args.audit_json:
            raise ValueError("--render-only requires --audit-json")

        result = render_ball_candidate_video_from_json(
            audit_json_path=args.audit_json,
            input_path=args.input,
            output_video_path=args.output_video,
            render_threshold=args.render_threshold,
            max_render_candidates=args.max_render_candidates,
        )
        print(f"Wrote candidate audit debug video: {result['debug_video_path']}")
        print(
            "Render-only mode: "
            f"frames {result['start_frame']}-{result['end_frame']} | "
            f"render_threshold={result['render_threshold']} | "
            f"max_render_candidates={result['max_render_candidates']}"
        )
        return 0

    if not args.input:
        raise ValueError("--input is required unless --render-only is used")
    if not args.output_dir:
        raise ValueError("--output-dir is required unless --render-only is used")
    if args.start_frame is None or args.end_frame is None:
        raise ValueError("--start-frame and --end-frame are required unless --render-only is used")

    result = audit_ball_candidates(
        input_path=args.input,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        ball_model_path=args.ball_model,
        conf_threshold=args.conf_threshold,
        debug_video=args.debug_video,
        use_inference_slicer=not args.no_inference_slicer,
        render_threshold=args.render_threshold,
        max_render_candidates=args.max_render_candidates,
    )

    print(f"Wrote candidate audit JSON: {result['json_path']}")
    if "debug_video_path" in result:
        print(f"Wrote candidate audit debug video: {result['debug_video_path']}")
    print(
        "Frames audited: "
        f"{result['start_frame']}-{result['end_frame']} "
        f"({result['frame_count']} frames), total candidates={result['total_candidates']}, "
        f"max candidates/frame={result['max_candidates_in_frame']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())