"""
Task 1: Standalone 2D Pitch - Player Positions

Renders top-down 2D pitch video with player positions, team colors, and jersey numbers.

CRITICAL CONSTRAINTS:
- Read-only visualization (does NOT modify Pass 1/2/3 logic or JSON)
- Reuses ARCHIVE homography and pitch drawing utilities
- Jersey numbers shown ONLY if locked for that fragment at current frame
- No temporal smoothing, deterministic output

Input:
- Pass 1: centroids (pixel coordinates)
- Pass 2: fragment ↔ frame mapping, court coordinates
- Pass 3: team assignment, jersey number (with lock status)

Output:
- 2D pitch view with player circles (team colored) and jersey numbers (when locked)
"""

from pathlib import Path
import json
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, List, Any

# Import geometry utilities (copied from ARCHIVE to src/)
from src.geometry.homography import CourtHomography, create_homography_from_config
from src.geometry.pitch_drawing import create_court_view


# Team colors (BGR for OpenCV) - matching pass_visualize.py
TEAM_COLORS = {
    "team_a": (0, 140, 255),      # Orange (bibbed team)
    "team_b": (40, 40, 40),       # Black (diverse team)
    "unknown": (200, 200, 200),   # Light gray
}


class PitchRenderer2D:
    """
    Renders 2D tactical pitch view from Pass 1-3 JSON outputs.

    Reuses ARCHIVE pitch geometry and homography logic.
    """

    def __init__(
        self,
        pass1_data: Dict[str, Any],
        pass2_data: Dict[str, Any],
        pass3_data: Dict[str, Any],
        homography: CourtHomography,
    ):
        """
        Initialize pitch renderer.

        Args:
            pass1_data: Pass 1 JSON data (centroids)
            pass2_data: Pass 2 JSON data (fragments with court positions)
            pass3_data: Pass 3 JSON data (team + jersey assignments)
            homography: CourtHomography instance
        """
        self.pass1_data = pass1_data
        self.pass2_data = pass2_data
        self.pass3_data = pass3_data
        self.homography = homography

        # Use homography's built-in output dimensions (matches ARCHIVE exactly)
        self.pitch_width_px = homography.output_w
        self.pitch_height_px = homography.output_h

        # Build fragment lookup index for fast access
        self._build_fragment_index()

        # Build identity lookup (fragment_id -> identity info)
        self._build_identity_index()

        # Build ball position index (frame -> position)
        self._build_ball_index()

    def _build_fragment_index(self):
        """Build index: frame -> list of active fragments with positions."""
        self.fragment_by_frame: Dict[int, List[Dict[str, Any]]] = {}

        fragments = self.pass2_data.get("fragments", [])

        for fragment in fragments:
            fragment_id = fragment.get("fragment_id")
            start_frame = fragment.get("start_frame")
            end_frame = fragment.get("end_frame")

            # Get court positions from spatial_footprint
            spatial_footprint = fragment.get("spatial_footprint", {})
            court_positions = spatial_footprint.get("court_positions", [])

            if start_frame is None or end_frame is None:
                continue

            if not court_positions:
                continue

            # Index each frame (court_positions are sampled positions across fragment duration)
            for i in range(len(court_positions)):
                # Map position index to frame number
                # Assuming court_positions are sampled at regular intervals
                frame_idx = start_frame + i

                if frame_idx > end_frame:
                    break  # Don't exceed end_frame

                if frame_idx not in self.fragment_by_frame:
                    self.fragment_by_frame[frame_idx] = []

                court_pos = court_positions[i]
                if court_pos and len(court_pos) == 2:
                    self.fragment_by_frame[frame_idx].append({
                        "fragment_id": fragment_id,
                        "court_position": court_pos,
                    })


    def _build_identity_index(self):
        """Build index: fragment_id -> identity info (team, jersey_number)."""
        self.identity_by_fragment: Dict[str, Dict[str, Any]] = {}

        identities = self.pass3_data.get("identities", [])

        for identity in identities:
            fragment_id = identity.get("fragment_id")
            self.identity_by_fragment[fragment_id] = {
                "team": identity.get("team", "unknown"),
                "jersey_number": identity.get("jersey_number"),
                "start_frame": identity.get("start_frame"),
                "end_frame": identity.get("end_frame"),
            }

    def _build_ball_index(self):
        """Build index: frame -> ball pixel position."""
        self.ball_by_frame: Dict[int, Tuple[float, float]] = {}

        ball_data = self.pass1_data.get("ball", {})
        ball_frames = ball_data.get("frames", [])
        ball_positions = ball_data.get("positions", [])

        for frame_idx, pos in zip(ball_frames, ball_positions):
            if pos and len(pos) == 2:
                self.ball_by_frame[int(frame_idx)] = (float(pos[0]), float(pos[1]))

    def render_pitch_frame(
        self,
        frame_idx: int,
        draw_voronoi: bool = False,
    ) -> np.ndarray:
        """
        Render a single 2D pitch frame.

        Args:
            frame_idx: Frame index to render
            draw_voronoi: Whether to draw Voronoi regions (Task 2)

        Returns:
            BGR image (numpy array) of 2D pitch with player positions
        """
        # Create blank pitch background using ARCHIVE utility
        pitch = create_court_view(
            width=self.pitch_width_px,
            height=self.pitch_height_px,
            court_length=self.homography.pitch_width_m,
            court_width=self.homography.pitch_height_m,
        )

        # Get active fragments for this frame
        active_fragments = self.fragment_by_frame.get(frame_idx, [])

        # Collect player positions for Voronoi (if enabled)
        player_positions_court = []
        player_metadata = []

        for fragment_data in active_fragments:
            fragment_id = fragment_data["fragment_id"]
            court_position = fragment_data["court_position"]

            if court_position is None:
                continue

            # Get identity info (team + jersey number)
            identity = self.identity_by_fragment.get(fragment_id, {})
            team = identity.get("team", "unknown")
            jersey_number = identity.get("jersey_number")

            # Get team color
            team_color = TEAM_COLORS.get(team, TEAM_COLORS["unknown"])

            # Convert court coordinates to 2D pitch pixels
            court_x, court_y = court_position
            pitch_x, pitch_y = self.homography.court_to_pixel_2d(court_x, court_y)
            # Flip Y so camera-near is at bottom (consistent with ARCHIVE minimap)
            pitch_y = self.pitch_height_px - pitch_y

            # Store for Voronoi (if enabled)
            if draw_voronoi:
                player_positions_court.append((court_x, court_y))
                player_metadata.append({
                    "team": team,
                    "team_color": team_color,
                    "pitch_x": pitch_x,
                    "pitch_y": pitch_y,
                })

            # Draw player circle (team colored)
            radius = 16
            cv2.circle(pitch, (pitch_x, pitch_y), radius, team_color, -1)  # Filled
            cv2.circle(pitch, (pitch_x, pitch_y), radius, (255, 255, 255), 2)  # White outline

            # Draw jersey number ONLY if locked for this fragment
            # CRITICAL: Jersey number is shown ONLY if Pass 3 says it's locked
            if jersey_number is not None:
                # Draw jersey number in center of circle
                text = str(jersey_number)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2

                # Get text size for centering
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = pitch_x - text_width // 2
                text_y = pitch_y + text_height // 2

                # Draw text with black outline for visibility
                cv2.putText(pitch, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 2)
                cv2.putText(pitch, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

        # Draw Voronoi regions (Task 2)
        if draw_voronoi and len(player_positions_court) >= 3:
            pitch = self._draw_voronoi_regions(
                pitch,
                player_positions_court,
                player_metadata,
            )

        # Draw ball
        ball_pixel_pos = self.ball_by_frame.get(frame_idx)
        if ball_pixel_pos:
            ball_px, ball_py = ball_pixel_pos
            # Convert pixel to court coordinates
            ball_court_x, ball_court_y = self.homography.pixel_to_court(ball_px, ball_py)
            # Convert court to pitch pixels
            ball_pitch_x, ball_pitch_y = self.homography.court_to_pixel_2d(ball_court_x, ball_court_y)
            # Flip Y
            ball_pitch_y = self.pitch_height_px - ball_pitch_y

            # Draw ball as white circle with orange outline
            ball_radius = 10
            cv2.circle(pitch, (ball_pitch_x, ball_pitch_y), ball_radius, (255, 255, 255), -1)  # White fill
            cv2.circle(pitch, (ball_pitch_x, ball_pitch_y), ball_radius, (0, 165, 255), 2)  # Orange outline

        # Draw frame index overlay
        cv2.putText(
            pitch,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        return pitch

    def _draw_voronoi_regions(
        self,
        pitch: np.ndarray,
        player_positions_court: List[Tuple[float, float]],
        player_metadata: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Draw Voronoi regions on pitch (Task 2).

        CRITICAL: Voronoi computed in court coordinates (meters), NOT pixels.

        Args:
            pitch: Pitch image (BGR)
            player_positions_court: List of (court_x, court_y) in meters
            player_metadata: List of player metadata (team, color, pitch position)

        Returns:
            Pitch image with Voronoi overlay
        """
        from scipy.spatial import Voronoi

        # Convert to numpy array
        points = np.array(player_positions_court)

        # Compute Voronoi in court coordinates
        vor = Voronoi(points)

        # Create overlay for alpha blending
        overlay = pitch.copy()

        # Draw each Voronoi region
        for region_idx, region in enumerate(vor.regions):
            # Skip empty regions
            if not region or -1 in region:
                continue

            # Find which point this region belongs to
            point_idx = None
            for i, point_region in enumerate(vor.point_region):
                if point_region == region_idx:
                    point_idx = i
                    break

            if point_idx is None or point_idx >= len(player_metadata):
                continue

            # Get team color for this region
            metadata = player_metadata[point_idx]
            team_color = metadata["team_color"]

            # Convert Voronoi vertices (court coords) to pitch pixels
            vertices_court = vor.vertices[region]
            vertices_pitch = []
            for vx, vy in vertices_court:
                px, py = self.homography.court_to_pixel_2d(vx, vy)
                py = self.pitch_height_px - py  # Flip Y
                vertices_pitch.append([px, py])

            # Convert to integer pixel coordinates
            vertices_pitch = np.array(vertices_pitch, dtype=np.int32)

            # Clip to pitch boundary
            # Create pitch boundary polygon
            pitch_boundary = np.array([
                [0, 0],
                [self.pitch_width_px - 1, 0],
                [self.pitch_width_px - 1, self.pitch_height_px - 1],
                [0, self.pitch_height_px - 1],
            ], dtype=np.int32)

            # Use polygon intersection (simple clipping to image bounds)
            vertices_clipped = []
            for vx, vy in vertices_pitch:
                vx = max(0, min(vx, self.pitch_width_px - 1))
                vy = max(0, min(vy, self.pitch_height_px - 1))
                vertices_clipped.append([vx, vy])
            vertices_clipped = np.array(vertices_clipped, dtype=np.int32)

            # Draw filled polygon on overlay
            cv2.fillPoly(overlay, [vertices_clipped], team_color)

        # Blend overlay with pitch (alpha = 0.25)
        alpha = 0.25
        pitch = cv2.addWeighted(overlay, alpha, pitch, 1 - alpha, 0)

        return pitch


def load_pass_data(
    run_dir: Path,
    clip_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load Pass 1, 2, 3 JSON data for a clip.

    Args:
        run_dir: Run output directory (e.g., output/run_DDMMYY_HHMMSS)
        clip_name: Clip name (without .json extension)

    Returns:
        (pass1_data, pass2_data, pass3_data) tuple
    """
    run_dir = Path(run_dir)

    # Load Pass 1
    pass1_file = run_dir / "pass1_raw" / f"{clip_name}.json"
    with open(pass1_file, 'r', encoding='utf-8') as f:
        pass1_data = json.load(f)

    # Load Pass 2
    pass2_file = run_dir / "pass2_identity" / f"{clip_name}_fragments.json"
    with open(pass2_file, 'r', encoding='utf-8') as f:
        pass2_data = json.load(f)

    # Load Pass 3
    pass3_file = run_dir / "pass3_final" / f"{clip_name}.json"
    with open(pass3_file, 'r', encoding='utf-8') as f:
        pass3_data = json.load(f)

    return pass1_data, pass2_data, pass3_data


def render_pitch_frame(
    frame_idx: int,
    pass1_data: Dict[str, Any],
    pass2_data: Dict[str, Any],
    pass3_data: Dict[str, Any],
    homography: CourtHomography,
    draw_voronoi: bool = False,
) -> np.ndarray:
    """
    Render a single 2D pitch frame.

    Main entry point for rendering a single frame.

    Args:
        frame_idx: Frame index to render
        pass1_data: Pass 1 JSON data
        pass2_data: Pass 2 JSON data
        pass3_data: Pass 3 JSON data
        homography: CourtHomography instance
        draw_voronoi: Whether to draw Voronoi regions (Task 2)

    Returns:
        BGR image of 2D pitch
    """
    renderer = PitchRenderer2D(
        pass1_data=pass1_data,
        pass2_data=pass2_data,
        pass3_data=pass3_data,
        homography=homography,
    )
    return renderer.render_pitch_frame(frame_idx, draw_voronoi=draw_voronoi)


def export_frame_png(
    output_path: Path,
    frame_idx: int,
    pass1_data: Dict[str, Any],
    pass2_data: Dict[str, Any],
    pass3_data: Dict[str, Any],
    homography: CourtHomography,
    draw_voronoi: bool = False,
):
    """
    Export a single frame as PNG.

    Args:
        output_path: Output PNG file path
        frame_idx: Frame index to render
        pass1_data: Pass 1 JSON data
        pass2_data: Pass 2 JSON data
        pass3_data: Pass 3 JSON data
        homography: CourtHomography instance
        draw_voronoi: Whether to draw Voronoi regions
    """
    pitch_frame = render_pitch_frame(
        frame_idx=frame_idx,
        pass1_data=pass1_data,
        pass2_data=pass2_data,
        pass3_data=pass3_data,
        homography=homography,
        draw_voronoi=draw_voronoi,
    )

    cv2.imwrite(str(output_path), pitch_frame)
    print(f"Saved: {output_path}")


def export_mp4(
    output_path: Path,
    start_frame: int,
    end_frame: int,
    pass1_data: Dict[str, Any],
    pass2_data: Dict[str, Any],
    pass3_data: Dict[str, Any],
    homography: CourtHomography,
    fps: float = 30.0,
    draw_voronoi: bool = False,
):
    """
    Export frame range as MP4 video.

    Args:
        output_path: Output MP4 file path
        start_frame: Start frame index (inclusive)
        end_frame: End frame index (inclusive)
        pass1_data: Pass 1 JSON data
        pass2_data: Pass 2 JSON data
        pass3_data: Pass 3 JSON data
        homography: CourtHomography instance
        fps: Output video FPS
        draw_voronoi: Whether to draw Voronoi regions
    """
    # Initialize renderer
    renderer = PitchRenderer2D(
        pass1_data=pass1_data,
        pass2_data=pass2_data,
        pass3_data=pass3_data,
        homography=homography,
    )

    # Get first frame to determine video dimensions
    first_frame = renderer.render_pitch_frame(start_frame, draw_voronoi=draw_voronoi)
    height, width = first_frame.shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Render and write frames
    from tqdm import tqdm
    for frame_idx in tqdm(range(start_frame, end_frame + 1), desc="Rendering pitch video"):
        pitch_frame = renderer.render_pitch_frame(frame_idx, draw_voronoi=draw_voronoi)
        out.write(pitch_frame)

    out.release()
    print(f"Saved: {output_path}")


def main():
    """Example usage / CLI driver."""
    import argparse

    parser = argparse.ArgumentParser(description="Render 2D pitch visualization")
    parser.add_argument("run_dir", type=Path, help="Run directory (e.g., output/run_DDMMYY_HHMMSS)")
    parser.add_argument("clip_name", type=str, help="Clip name (without .json)")
    parser.add_argument("--config", type=Path, default="config/default.yaml", help="Config file")
    parser.add_argument("--frame", type=int, help="Export single frame as PNG")
    parser.add_argument("--start", type=int, help="Start frame for MP4 export")
    parser.add_argument("--end", type=int, help="End frame for MP4 export")
    parser.add_argument("--fps", type=float, default=30.0, help="Output FPS for MP4")
    parser.add_argument("--voronoi", action="store_true", help="Draw Voronoi regions (Task 2)")
    parser.add_argument("--output", type=Path, help="Output file path")

    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Create homography
    from src.geometry.homography import create_homography_from_config
    homography = create_homography_from_config(config)
    if homography is None:
        print("ERROR: Homography not configured")
        return

    # Load pass data
    pass1_data, pass2_data, pass3_data = load_pass_data(args.run_dir, args.clip_name)

    # Export single frame
    if args.frame is not None:
        output_path = args.output or args.run_dir / f"pitch_frame_{args.frame:06d}.png"
        export_frame_png(
            output_path=output_path,
            frame_idx=args.frame,
            pass1_data=pass1_data,
            pass2_data=pass2_data,
            pass3_data=pass3_data,
            homography=homography,
            draw_voronoi=args.voronoi,
        )

    # Export MP4
    elif args.start is not None and args.end is not None:
        output_path = args.output or args.run_dir / f"pitch_{args.clip_name}.mp4"
        export_mp4(
            output_path=output_path,
            start_frame=args.start,
            end_frame=args.end,
            pass1_data=pass1_data,
            pass2_data=pass2_data,
            pass3_data=pass3_data,
            homography=homography,
            fps=args.fps,
            draw_voronoi=args.voronoi,
        )

    else:
        print("ERROR: Must specify either --frame or --start/--end")


if __name__ == "__main__":
    main()
