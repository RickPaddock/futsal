"""
Visualization Pass: Generate annotated video from Pass 3 output.

Loads Pass 1 (bboxes), Pass 2 (divergences), and Pass 3 (team/jersey) to create
frame-by-frame annotated video output.
"""

from pathlib import Path
import json
import numpy as np
import cv2
import supervision as sv
from tqdm import tqdm

from src.utils.video_io import VideoReader, VideoWriter
from src.utils.data_models import BoundingBox


# Team colors (BGR format for OpenCV)
TEAM_COLORS = {
    "team_a": (255, 153, 51),    # Orange (bibbed team)
    "team_b": (0, 0, 0),  # Black (non-bibbed team)
    "unknown": (255, 0, 255), # Pink
}

class BallAnnotator:
    """Draws ball with a trailing jet-colormap tail showing recent positions."""

    def __init__(
        self,
        radius: int = 12,
        buffer_size: int = 30,
        thickness: int = 2,
        max_age_seconds: float = 2.0,
        fps: float | None = None,
    ):
        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer: list[tuple[tuple[int, int], int | None]] = []
        self.buffer_size = buffer_size
        self.radius = radius
        self.thickness = thickness
        self.max_age_frames = int(max_age_seconds * fps) if fps and max_age_seconds > 0 else None

    def interpolate_radius(self, i: int, max_i: int) -> int:
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def _prune_old(self, frame_idx: int | None) -> None:
        if self.max_age_frames is None or frame_idx is None:
            return
        cutoff = frame_idx - self.max_age_frames
        self.buffer = [(pos, idx) for pos, idx in self.buffer if idx is None or idx >= cutoff]

    def annotate(
        self,
        frame: np.ndarray,
        center: tuple[int, int] | None = None,
        frame_idx: int | None = None,
    ) -> None:
        if center is not None:
            self.buffer.append((center, frame_idx))
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)

        self._prune_old(frame_idx)

        for i, (pos, _) in enumerate(self.buffer):
            color = self.color_palette.by_idx(i).as_bgr()
            radius = self.interpolate_radius(i, len(self.buffer))
            cv2.circle(frame, pos, radius, color, self.thickness)


def visualize_run(
    run_dir: Path,
    clip_filter: str | None,
    input_dir: Path,
    output_scale: float,
    config: dict,
):
    """
    Visualize all clips (or a specific clip) from a run directory.

    Automatically detects clips from Pass 1 JSON files.

    Args:
        run_dir: Run output directory (output/run_DDMMYY_HHMMSS)
        clip_filter: Optional clip name to filter (None = all clips)
        input_dir: Directory with original video files
        output_scale: Output video scale (0.5 = half size)
        config: Configuration dictionary
    """
    run_dir = Path(run_dir)
    pass1_dir = run_dir / "pass1_raw"

    # Get all Pass 1 JSON files
    pass1_files = list(pass1_dir.glob("*.json"))
    if not pass1_files:
        print(f"ERROR: No Pass 1 JSON files found in {pass1_dir}")
        return

    # Extract clip names from Pass 1 JSONs
    clips_to_visualize = []
    for pass1_file in pass1_files:
        with open(pass1_file, 'r', encoding='utf-8') as f:
            pass1_data = json.load(f)
        clip_name = pass1_data.get("clip_name")
        if not clip_name:
            print(f"WARNING: No clip_name in {pass1_file.name}, skipping")
            continue

        # Apply filter if specified
        if clip_filter is None or Path(clip_name).stem == Path(clip_filter).stem:
            clips_to_visualize.append(clip_name)

    if not clips_to_visualize:
        if clip_filter:
            print(f"ERROR: No clips matched filter: {clip_filter}")
        else:
            print(f"ERROR: No clips found in {pass1_dir}")
        return

    print(f"\nFound {len(clips_to_visualize)} clip(s) to visualize:")
    for clip_name in clips_to_visualize:
        print(f"  - {clip_name}")

    # Visualize each clip
    for clip_name in clips_to_visualize:
        print(f"\n{'='*60}")
        print(f"Visualizing: {clip_name}")
        print(f"{'='*60}")
        visualize_clip(
            run_dir=run_dir,
            clip_name=clip_name,
            input_dir=input_dir,
            output_scale=output_scale,
            config=config,
        )

    print(f"\n{'='*60}")
    print(f"All visualizations complete!")
    print(f"{'='*60}")


def visualize_clip(
    run_dir: Path,
    clip_name: str,
    input_dir: Path,
    output_scale: float,
    config: dict,
):
    """
    Generate annotated video from Pass 3 output.

    Args:
        run_dir: Run output directory (output/run_DDMMYY_HHMMSS)
        clip_name: Video clip filename (e.g., "clip_001.mp4")
        input_dir: Directory with original video files
        output_scale: Output video scale (0.5 = half size)
        config: Configuration dictionary
    """
    run_dir = Path(run_dir)
    input_dir = Path(input_dir)

    # Find original video file
    clip_path = input_dir / clip_name
    if not clip_path.exists():
        # Try with different extensions
        stem = Path(clip_name).stem
        candidates = list(input_dir.glob(f"{stem}.*"))
        if candidates:
            clip_path = candidates[0]
        else:
            print(f"ERROR: Video file not found: {clip_path}")
            return

    # Load Pass 1 JSON (bboxes per track)
    pass1_dir = run_dir / "pass1_raw"
    pass1_file = pass1_dir / f"{Path(clip_name).stem}.json"
    if not pass1_file.exists():
        print(f"ERROR: Pass 1 file not found: {pass1_file}")
        return

    with open(pass1_file, 'r', encoding='utf-8') as f:
        pass1_data = json.load(f)

    # Load ball positions from Pass 1 JSON
    ball_data = pass1_data.get("ball", {})
    ball_frames = ball_data.get("frames", [])
    ball_positions = ball_data.get("positions", [])
    ball_by_frame: dict[int, tuple[int, int]] = {}
    for frame_idx, pos in zip(ball_frames, ball_positions):
        if pos and len(pos) == 2:
            ball_by_frame[int(frame_idx)] = (int(pos[0]), int(pos[1]))

    # Load Pass 2 JSON (divergence markers)
    pass2_dir = run_dir / "pass2_identity"
    pass2_file = pass2_dir / f"{Path(clip_name).stem}_fragments.json"
    divergence_markers = {}
    pass2_data = None
    if pass2_file.exists():
        with open(pass2_file, 'r', encoding='utf-8') as f:
            pass2_data = json.load(f)
        print(f"  Pass 2: {len(pass2_data.get('fragments', []))} fragments")
        # Build divergence marker lookup: frame_idx → split_reason
        for fragment in pass2_data.get("fragments", []):
            split_reason = fragment.get("split_reason")
            if split_reason and "frame_idx" in split_reason:
                frame_idx = split_reason["frame_idx"]
                if frame_idx not in divergence_markers:
                    divergence_markers[frame_idx] = []
                divergence_markers[frame_idx].append(split_reason)
    else:
        print(f"  WARNING: Pass 2 file not found: {pass2_file}")

    # Load Pass 3 JSON (team/jersey assignments)
    pass3_dir = run_dir / "pass3_final"
    pass3_file = pass3_dir / f"{Path(clip_name).stem}.json"
    name_map = config.get("jersey", {}).get("name_map", {}) if config else {}
    visualize_cfg = config.get("visualize", {}) if config else {}
    debug_mode = bool(visualize_cfg.get("debug", False))
    fragment_identity_map: dict[str, dict[str, object]] = {}  # fragment_id → {team, jersey, confidence}
    if pass3_file.exists():
        with open(pass3_file, 'r', encoding='utf-8') as f:
            pass3_data = json.load(f)
        print(f"  Pass 3: {len(pass3_data.get('identities', []))} identities")

        # Build identity map: fragment_id → {team, jersey, confidence}
        # Pass 3 outputs ONE identity per fragment (not one per player)
        identities = pass3_data.get("identities", [])
        for identity in identities:
            frag_id = identity.get("fragment_id")  # Single fragment ID
            if frag_id:
                team = identity.get("team", "unknown")
                jersey = identity.get("assigned_jersey_number", identity.get("jersey_number"))
                confidence = identity.get("confidence", 0.0)
                fragment_identity_map[frag_id] = {
                    "team": team,
                    "jersey": jersey,
                    "confidence": confidence,
                }

        print(f"  Fragment→Identity mappings: {len(fragment_identity_map)}")

        # Count team assignments
        team_breakdown = {"team_a": 0, "team_b": 0, "unknown": 0}
        jersey_breakdown = {}
        for identity in fragment_identity_map.values():
            team = identity.get("team", "unknown")
            jersey = identity.get("jersey")
            team_breakdown[team] = team_breakdown.get(team, 0) + 1
            if jersey is not None:
                jersey_breakdown[jersey] = jersey_breakdown.get(jersey, 0) + 1
        print(f"  Fragments by team: {team_breakdown}")
        if jersey_breakdown:
            print(f"  Fragments by jersey: {jersey_breakdown}")
    else:
        print(f"  WARNING: Pass 3 file not found: {pass3_file}")

    # Build fragment lookup: (original_track_id, frame_idx) → fragment_id
    track_frame_to_fragment = {}
    if pass2_data:
        for fragment in pass2_data.get("fragments", []):
            frag_id = fragment["fragment_id"]
            track_id = str(fragment["original_track_id"])
            start_frame = fragment["start_frame"]
            end_frame = fragment["end_frame"]
            for frame_idx in range(start_frame, end_frame + 1):
                track_frame_to_fragment[(track_id, frame_idx)] = frag_id
        print(f"  Track→Fragment mappings: {len(track_frame_to_fragment)} frame entries")

    # Build frame-by-frame annotation data
    # Structure: {frame_idx: {track_id: {"bbox": [...], "team": "...", "jersey": ...}}}
    frame_annotations = {}

    tracks = pass1_data.get("tracks", {})

    # Debug: Track team/jersey assignment stats
    team_counts = {"team_a": 0, "team_b": 0, "unknown": 0}
    jersey_counts = {}

    for track_id, track_data in tracks.items():
        frames = track_data.get("frames", [])
        bboxes = track_data.get("bboxes", [])
        for i, frame_idx in enumerate(frames):
            bbox = bboxes[i]

            # Lookup fragment and identity
            frag_id = track_frame_to_fragment.get((str(track_id), frame_idx))
            team = "unknown"
            jersey = None
            jersey_confidence = 0.0
            if frag_id and frag_id in fragment_identity_map:
                identity = fragment_identity_map[frag_id]
                team = identity.get("team", "unknown")
                jersey = identity.get("jersey")
                jersey_confidence = float(identity.get("confidence", 0.0))

            # Track stats
            team_counts[team] = team_counts.get(team, 0) + 1
            if jersey is not None:
                jersey_counts[jersey] = jersey_counts.get(jersey, 0) + 1

            if frame_idx not in frame_annotations:
                frame_annotations[frame_idx] = {}

            frame_annotations[frame_idx][track_id] = {
                "bbox": bbox,
                "team": team,
                "jersey": jersey,
                "jersey_confidence": jersey_confidence,
                "fragment_id": frag_id,
            }

    print(f"  Team assignments: {team_counts}")
    print(f"  Jersey assignments: {jersey_counts}")

    # Open video and create output
    reader = VideoReader(str(clip_path))
    fps = reader.fps
    output_width = int(reader.width * output_scale)
    output_height = int(reader.height * output_scale)

    # Create output directory
    pass3_dir = run_dir / "pass3_final"
    pass3_dir.mkdir(parents=True, exist_ok=True)
    output_path = pass3_dir / f"{Path(clip_name).stem}_annotated.mp4"

    writer = VideoWriter(
        str(output_path),
        fps=fps,
        width=output_width,
        height=output_height,
        codec="h264",
    )

    print(f"\nGenerating annotated video: {output_path}")
    print(f"  Input: {clip_path}")
    print(f"  Output size: {output_width}x{output_height} @ {fps} FPS")
    print(f"  Tracks: {len(tracks)}")
    print(f"  Divergence markers: {len(divergence_markers)}")

    # Process frames
    trail_radius = max(2, int(9 * output_scale))
    trail_thickness = max(1, int(2 * output_scale))
    ball_annotator = BallAnnotator(fps=fps, radius=trail_radius, thickness=trail_thickness)
    fragment_debug_shown: set[str] = set()
    for frame_idx, frame in tqdm(reader.frames(), total=reader.total_frames, desc="Rendering"):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize frame
        frame_bgr = cv2.resize(frame_bgr, (output_width, output_height))

        # Draw divergence markers (if any at this frame)
        if frame_idx in divergence_markers:
            for split_reason in divergence_markers[frame_idx]:
                reason_type = split_reason.get("reason", "unknown")
                metric = split_reason.get("metric", 0.0)
                # Draw warning banner at top
                text = f"DIVERGENCE: {reason_type} (metric={metric:.2f})"
                cv2.rectangle(frame_bgr, (0, 0), (output_width, 50), (0, 0, 255), -1)
                cv2.putText(
                    frame_bgr,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        # Draw ball trail (from Pass 1 ball positions)
        if frame_idx in ball_by_frame:
            bx, by = ball_by_frame[frame_idx]
            ball_annotator.annotate(
                frame_bgr,
                center=(int(bx * output_scale), int(by * output_scale)),
                frame_idx=frame_idx,
            )
        else:
            ball_annotator.annotate(frame_bgr, center=None, frame_idx=frame_idx)

        # Draw player annotations
        annotations = frame_annotations.get(frame_idx, {})
        rendered_jerseys: set[int] = set()
        for track_id, data in annotations.items():
            bbox = data["bbox"]
            team = data["team"]
            jersey = data["jersey"]
            jersey_confidence = float(data.get("jersey_confidence", 0.0))
            frag_id = data.get("fragment_id")

            # Scale bbox to output size
            x1 = int(bbox[0] * output_scale)
            y1 = int(bbox[1] * output_scale)
            x2 = int(bbox[2] * output_scale)
            y2 = int(bbox[3] * output_scale)

            # Get team color
            color = TEAM_COLORS.get(team, TEAM_COLORS["unknown"])

            # Draw ellipse around player (archive style)
            det = sv.Detections(
                xyxy=np.array([[x1, y1, x2, y2]]),
                class_id=np.array([0]),
            )
            ellipse_color = sv.Color(color[2], color[1], color[0])
            frame_bgr = sv.EllipseAnnotator(color=ellipse_color, thickness=2).annotate(frame_bgr, det)

            # Build label (always show track id; jersey overlay is Pass 3 assigned ONLY)
            label_parts = [f"T{track_id}"]
            if frag_id:
                frag_suffix = frag_id.replace("frag_", "F")
                label_parts.append(f": {frag_suffix}")
            if jersey is not None:
                jersey_int = int(jersey)
                if jersey_int not in rendered_jerseys:
                    rendered_jerseys.add(jersey_int)
                    jersey_name = name_map.get(jersey_int)
                    if jersey_name is None:
                        jersey_name = name_map.get(str(jersey_int))
                    if debug_mode and frag_id and frag_id not in fragment_debug_shown:
                        label_parts.append(f"(assigned={jersey_int}, conf={jersey_confidence:.2f})")
                        fragment_debug_shown.add(frag_id)
            elif debug_mode:
                label_parts.append("?")

            label = " ".join(label_parts)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.35, 0.5 * output_scale)
            thickness = max(1, int(2 * output_scale))

            top_y = y1

            jersey_label = None
            if jersey is not None:
                jersey_int = int(jersey)
                if jersey_int in rendered_jerseys:
                    jersey_name = name_map.get(jersey_int)
                    if jersey_name is None:
                        jersey_name = name_map.get(str(jersey_int))
                    jersey_label = f"#{jersey_int}"
                    if jersey_name is not None:
                        jersey_label = f"{jersey_label} : {jersey_name}"

            if jersey_label:
                (j_text_w, j_text_h), j_base = cv2.getTextSize(
                    jersey_label, font, font_scale, thickness
                )
                cv2.rectangle(
                    frame_bgr,
                    (x1, top_y - j_text_h - 8),
                    (x1 + j_text_w + 4, top_y),
                    color,
                    -1,
                )
                cv2.putText(
                    frame_bgr,
                    jersey_label,
                    (x1 + 2, top_y - 5),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                )
                top_y = top_y - j_text_h - 8

            if label:
                # Draw top label background (track + fragment)
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                cv2.rectangle(
                    frame_bgr,
                    (x1, top_y - text_height - 8),
                    (x1 + text_width + 4, top_y),
                    color,
                    -1,
                )
                cv2.putText(
                    frame_bgr,
                    label,
                    (x1 + 2, top_y - 5),
                    font,
                    font_scale,
                    (0, 0, 0),  # Black text
                    thickness,
                )

        # Draw frame counter
        cv2.putText(
            frame_bgr,
            f"Frame: {frame_idx}",
            (10, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, 0.7 * output_scale),
            (255, 255, 255),
            max(1, int(2 * output_scale)),
        )

        # Convert back to RGB for writer
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Write frame
        writer.write_frame(frame_rgb)

    writer.close()
    print(f"\nVisualization saved: {output_path}")
