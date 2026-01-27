"""
Visualization utilities for debug output.

Draws bounding boxes, track IDs, ball positions, and trajectories on frames.
"""

from typing import Optional
from collections import deque
import numpy as np
import cv2
import supervision as sv

from src.utils.data_models import MatchData, TeamID, BoundingBox
from src.geometry.homography import CourtHomography


# Color palette for tracks (BGR format for OpenCV)
TRACK_COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 255),    # Purple
    (255, 128, 0),    # Orange
    (0, 128, 255),    # Light blue
    (128, 255, 0),    # Lime
]

TEAM_COLORS = {
    TeamID.TEAM_A: (255, 100, 100),    # Blue-ish
    TeamID.TEAM_B: (100, 100, 255),    # Red-ish
    TeamID.UNKNOWN: (200, 200, 200),   # Gray
}

BALL_COLOR = (0, 165, 255)  # Orange


def get_track_color(track_id: int) -> tuple[int, int, int]:
    """Get color for a track ID."""
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def draw_bbox(
    frame: np.ndarray,
    bbox: BoundingBox,
    color: tuple[int, int, int],
    thickness: int = 2,
    label: Optional[str] = None,
    dotted: bool = False,
) -> np.ndarray:
    """
    Draw a bounding box on the frame.

    Args:
        frame: BGR numpy array (will be modified in place)
        bbox: BoundingBox to draw
        color: BGR color tuple
        thickness: Line thickness
        label: Optional text label
        dotted: If True, draw dotted box instead of solid

    Returns:
        Modified frame
    """
    x1, y1 = int(bbox.x1), int(bbox.y1)
    x2, y2 = int(bbox.x2), int(bbox.y2)

    if dotted:
        # Draw dotted box by drawing dashes
        dash_len = 10
        gap_len = 5
        # Top and bottom
        for x in range(x1, x2, dash_len + gap_len):
            cv2.line(frame, (x, y1), (min(x + dash_len, x2), y1), color, thickness)
            cv2.line(frame, (x, y2), (min(x + dash_len, x2), y2), color, thickness)
        # Left and right
        for y in range(y1, y2, dash_len + gap_len):
            cv2.line(frame, (x1, y), (x1, min(y + dash_len, y2)), color, thickness)
            cv2.line(frame, (x2, y), (x2, min(y + dash_len, y2)), color, thickness)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if label:
        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )

    return frame


def draw_mask_contour(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    alpha: float = 0.2,
) -> np.ndarray:
    """Draw the contour of a binary mask and optional translucent fill."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # Ensure mask is 0/1
    mask = (mask > 0).astype(np.uint8)

    # Compute contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame

    # Draw translucent fill
    if alpha > 0:
        overlay = frame.copy()
        cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Draw contour lines
    cv2.drawContours(frame, contours, -1, color, thickness)
    return frame


def draw_sam_mask_prominent(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 255, 0),  # Cyan default
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw a SAM mask with high visibility - filled overlay with thick contour.

    Used for T2 (SAM recovered) detections to make them very obvious.
    """
    if mask is None:
        return frame
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame

    # Strong filled overlay
    overlay = frame.copy()
    cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Thick bright contour
    cv2.drawContours(frame, contours, -1, color, thickness=3)

    return frame


def draw_ball_trajectory(
    frame: np.ndarray,
    positions: list[tuple[float, float]],
    max_points: int = 30,
    fade: bool = True,
) -> np.ndarray:
    """
    Draw ball trajectory trail.

    Args:
        frame: BGR numpy array
        positions: List of (x, y) positions, newest last
        max_points: Maximum trail length
        fade: Whether to fade older points

    Returns:
        Modified frame
    """
    positions = positions[-max_points:]

    for i, (x, y) in enumerate(positions):
        if fade:
            alpha = (i + 1) / len(positions)
            color = tuple(int(c * alpha) for c in BALL_COLOR)
            radius = max(2, int(6 * alpha))
        else:
            color = BALL_COLOR
            radius = 4

        cv2.circle(frame, (int(x), int(y)), radius, color, -1)

    # Draw connecting lines
    if len(positions) >= 2:
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]
            if fade:
                alpha = (i + 1) / len(positions)
                color = tuple(int(c * alpha) for c in BALL_COLOR)
            else:
                color = BALL_COLOR
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

    return frame


class BallAnnotator:
    """Draws ball with a trailing jet-colormap tail showing recent positions."""

    def __init__(
        self,
        radius: int = 12,
        buffer_size: int = 30,
        thickness: int = 2,
        max_age_seconds: float = 2.0,
        fps: Optional[float] = None,
    ):
        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer: deque[tuple[tuple[int, int], Optional[int]]] = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness
        self.max_age_frames = int(max_age_seconds * fps) if fps and max_age_seconds > 0 else None

    def interpolate_radius(self, i: int, max_i: int) -> int:
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def _prune_old(self, frame_idx: Optional[int]) -> None:
        if self.max_age_frames is None or frame_idx is None:
            return
        cutoff = frame_idx - self.max_age_frames
        while self.buffer and self.buffer[0][1] is not None and self.buffer[0][1] < cutoff:
            self.buffer.popleft()

    def annotate(
        self,
        frame: np.ndarray,
        center: Optional[tuple[int, int]] = None,
        frame_idx: Optional[int] = None,
    ) -> np.ndarray:
        if center is not None:
            self.buffer.append((center, frame_idx))

        self._prune_old(frame_idx)

        for i, (pos, _) in enumerate(self.buffer):
            color = self.color_palette.by_idx(i).as_bgr()
            radius = self.interpolate_radius(i, len(self.buffer))
            cv2.circle(frame, pos, radius, color, self.thickness)

        return frame


def draw_frame_annotations(
    frame: np.ndarray,
    frame_idx: int,
    match_data: MatchData,
    draw_tracks: bool = True,
    draw_ball: bool = True,
    draw_trajectory: bool = True,
    trajectory_length: int = 30,
    pitch_top_y: int = None,
    pitch_bottom_y: int = None,
    draw_minimap: bool = True,
    minimap_size: tuple[int, int] = (600, 300),
    homography: CourtHomography = None,
    ball_annotator: Optional[BallAnnotator] = None,
    team_colors: Optional[dict[TeamID, tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Draw all annotations for a frame.

    Args:
        frame: RGB numpy array
        frame_idx: Current frame number
        match_data: Match tracking data
        draw_tracks: Whether to draw player tracks
        draw_ball: Whether to draw ball
        draw_trajectory: Whether to draw ball trajectory
        trajectory_length: Trail length for trajectory
        pitch_top_y: Top Y boundary of pitch (for SAM filtering and minimap mapping)
        pitch_bottom_y: Bottom Y boundary of pitch (for SAM filtering and minimap mapping)
        draw_minimap: Whether to draw bird's eye view minimap in top-right corner
        minimap_size: Size of minimap (width, height) in pixels

    Returns:
        Annotated RGB frame
    """
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Resolve palette (allows runtime override from pipeline)
    palette = team_colors or TEAM_COLORS

    # Draw pitch boundary lines for SAM filtering (if provided)
    # if pitch_top_y is not None and pitch_bottom_y is not None:
    #     frame_width = frame_bgr.shape[1]
    #     # Draw top boundary in yellow
    #     cv2.line(frame_bgr, (0, pitch_top_y), (frame_width, pitch_top_y), (0, 255, 255), 3)
    #     cv2.putText(frame_bgr, f"SAM Top Y={pitch_top_y}", (10, pitch_top_y - 10), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    #     # Draw bottom boundary in yellow
    #     cv2.line(frame_bgr, (0, pitch_bottom_y), (frame_width, pitch_bottom_y), (0, 255, 255), 3)
    #     cv2.putText(frame_bgr, f"SAM Bottom Y={pitch_bottom_y}", (10, pitch_bottom_y + 30), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Draw low-confidence detections as dotted boxes (filtered out from tracking)
    if frame_idx in match_data.low_confidence_detections:
        gray_color = (100, 100, 100)  # Gray for low-confidence
        for det in match_data.low_confidence_detections[frame_idx]:
            conf_text = f"UNTRACKED {det.bbox.confidence:.2f}"
            draw_bbox(frame_bgr, det.bbox, gray_color, thickness=1, label=conf_text, dotted=True)

    # Draw player tracks
    # =============================================================
    # TRACKING TIER HIERARCHY (always trying to use lowest tier):
    # - T1 = YOLO detection (best - no occlusion)
    # - T2 = SAM recovered (partial occlusion - player visible but YOLO missed)
    # - T3 = Estimated position (full occlusion - 2m behind blocking player)
    #
    # Golden rule: Players don't vanish. Always 12 unless off-screen.
    # =============================================================
    if draw_tracks:
        # Count by tier for stats overlay
        t1_count = 0  # YOLO
        t2_count = 0  # SAM
        t3_count = 0  # Estimated

        for track in match_data.player_tracks:
            det = track.get_detection_at_frame(frame_idx)
            if det is None:
                continue

            # Determine tracking tier
            is_sam = hasattr(det, 'is_sam_recovered') and det.is_sam_recovered
            is_interpolated = hasattr(det, 'is_interpolated') and det.is_interpolated

            # Get base color from team or track ID
            player_color = get_track_color(track.track_id)
            if track.team != TeamID.UNKNOWN:
                team_color = palette.get(track.team, get_track_color(track.track_id))
            else:
                team_color = player_color

            # Build label with tier indicator
            label_parts = [f"#{track.track_id}"]
            if track.jersey_number is not None:
                label_parts.append(f"J{track.jersey_number}")

            # Assign tier and visual style
            # IMPORTANT: T2 (SAM) uses same base_color as T1 for consistency
            if is_interpolated:
                # T3: Full occlusion - estimated position
                tier = "T3"
                color = (255, 0, 255)  # Magenta (BGR) - distinct color for estimates
                dotted = True
                t3_count += 1
            elif is_sam:
                # T2: Partial occlusion - SAM recovered
                # Use SAME color as T1 (base_color) so mask matches bbox color
                tier = "T2"
                color = team_color  # Team border color
                dotted = True  # Dotted box to indicate SAM recovery
                t2_count += 1
            else:
                # T1: Normal YOLO detection (best)
                tier = "T1"
                color = team_color
                dotted = False
                t1_count += 1

            label_parts.append(tier)
            label = " ".join(label_parts)
            if dotted:
                # T2/T3: keep dotted rectangle
                draw_bbox(frame_bgr, det.bbox, color, label=label, dotted=True, thickness=1)
            else:
                # T1: supervision ellipse annotator
                _det = sv.Detections(xyxy=np.array([[det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]]))
                _color = sv.Color(color[2], color[1], color[0])  # Team ellipse color (BGR -> RGB)
                _label_color = sv.Color(player_color[2], player_color[1], player_color[0])
                frame_bgr = sv.EllipseAnnotator(
                    color=_color, thickness=2, color_lookup=sv.ColorLookup.INDEX
                ).annotate(frame_bgr, _det)
                frame_bgr = sv.LabelAnnotator(
                    color=_label_color, text_color=sv.Color.WHITE, color_lookup=sv.ColorLookup.INDEX
                ).annotate(frame_bgr, _det, labels=[label])

            # Draw mask for SAM detections - use same color as bbox for consistency
            if hasattr(det, 'mask') and det.mask is not None:
                if is_sam:
                    # T2: Prominent filled mask with SAME color as bbox
                    frame_bgr = draw_sam_mask_prominent(frame_bgr, det.mask, color=color, alpha=0.4)
                else:
                    # T1 detections with masks: subtle contour with same color
                    frame_bgr = draw_mask_contour(frame_bgr, det.mask, color, thickness=2, alpha=0.15)

            # Draw SAM mask when in dual-box mode (YOLO + SAM tracking both active)
            if hasattr(det, 'sam_mask') and det.sam_mask is not None:
                frame_bgr = draw_sam_mask_prominent(frame_bgr, det.sam_mask, color=base_color, alpha=0.3)

        # =============================================================
        # BIRD'S EYE VIEW MINIMAP (top-right corner)
        # =============================================================
        # Shows player positions on a 2D pitch representation
        # Uses homography for accurate court coordinate mapping
        # =============================================================
        if draw_minimap:
            # Use homography output size if available, otherwise default
            if homography is not None:
                minimap_w = homography.output_w
                minimap_h = homography.output_h
            else:
                minimap_w, minimap_h = minimap_size
            minimap_margin = 10

            # Create minimap background (green pitch)
            minimap = np.zeros((minimap_h, minimap_w, 3), dtype=np.uint8)
            minimap[:] = (34, 100, 34)  # Dark green

            # Draw pitch markings scaled to actual court dimensions
            if homography is not None:
                scale = homography.output_pixel_scale
                court_l = homography.pitch_width_m  # 40m (X axis)
                court_w = homography.pitch_height_m  # 20m (Y axis)
            else:
                scale = 15
                court_l = 40.0
                court_w = 20.0

            # Pitch outline
            cv2.rectangle(minimap, (0, 0), (minimap_w - 1, minimap_h - 1), (255, 255, 255), 2)

            # Center line (at X = 20m)
            center_x = int(court_l / 2 * scale)
            cv2.line(minimap, (center_x, 0), (center_x, minimap_h), (255, 255, 255), 2)

            # Center circle (3m radius)
            center_radius = int(3.0 * scale)
            cv2.circle(minimap, (center_x, minimap_h // 2), center_radius, (255, 255, 255), 2)

            # Penalty areas (smaller depth, 6m each side of center = 12m total width)
            pa_depth = int(4.0 * scale)  # 4m deep from goal line
            pa_half_width = 6.0  # 6m each side of center
            pa_top = int((court_w / 2 - pa_half_width) * scale)
            pa_bottom = int((court_w / 2 + pa_half_width) * scale)
            # Left penalty area
            cv2.rectangle(minimap, (0, pa_top), (pa_depth, pa_bottom), (255, 255, 255), 2)
            # Right penalty area
            cv2.rectangle(minimap, (minimap_w - pa_depth, pa_top), (minimap_w - 1, pa_bottom), (255, 255, 255), 2)

            # Goals (3m wide, centered)
            goal_top = int((court_w / 2 - 1.5) * scale)
            goal_bottom = int((court_w / 2 + 1.5) * scale)
            cv2.line(minimap, (0, goal_top), (0, goal_bottom), (0, 0, 255), 4)
            cv2.line(minimap, (minimap_w - 1, goal_top), (minimap_w - 1, goal_bottom), (0, 0, 255), 4)

            # Map player positions using homography
            for track in match_data.player_tracks:
                det = track.get_detection_at_frame(frame_idx)
                if det is None:
                    continue

                # Transform player position to court coordinates
                if homography is not None:
                    # Use head + offset for stable positioning
                    court_x, court_y = homography.transform_bbox_to_court(det.bbox)
                    # Convert court meters to minimap pixels, INVERT Y axis
                    mini_x, mini_y = homography.court_to_pixel_2d(court_x, court_y)
                    mini_y = minimap_h - mini_y  # Flip Y so camera-near is at bottom
                else:
                    # Fallback: simple linear mapping
                    frame_h, frame_w = frame_bgr.shape[:2]
                    cx, cy = det.bbox.center
                    mini_x = int((cx / frame_w) * minimap_w)
                    mini_y = int((cy / frame_h) * minimap_h)

                # Clamp to minimap bounds
                mini_x = max(8, min(mini_x, minimap_w - 8))
                mini_y = max(8, min(mini_y, minimap_h - 8))

                # Get color based on track
                player_color = get_track_color(track.track_id)
                if track.team != TeamID.UNKNOWN:
                    team_color = palette.get(track.team, player_color)
                else:
                    team_color = player_color

                # Determine if SAM/interpolated for visual distinction
                is_sam = hasattr(det, 'is_sam_recovered') and det.is_sam_recovered
                is_interp = hasattr(det, 'is_interpolated') and det.is_interpolated

                # Draw player dot
                radius = 14
                inner_radius = max(4, radius - 6)
                if is_interp:
                    # T3: show team color with thin magenta outline to indicate interpolation
                    cv2.circle(minimap, (mini_x, mini_y), radius, team_color, 4)  # Team ring
                    cv2.circle(minimap, (mini_x, mini_y), inner_radius, player_color, -1)  # Player dot
                    cv2.circle(minimap, (mini_x, mini_y), radius + 2, (255, 0, 255), 1)  # Thin magenta outline
                else:
                    # Outer ring = team color, inner fill = player color
                    cv2.circle(minimap, (mini_x, mini_y), radius, team_color, 4)
                    cv2.circle(minimap, (mini_x, mini_y), inner_radius, player_color, -1)

                # Draw track ID
                cv2.putText(minimap, str(track.track_id), (mini_x - 6, mini_y + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Draw ball on minimap
            for ball in match_data.ball_positions:
                if ball.frame_idx == frame_idx:
                    ball_cx, ball_cy = ball.bbox.bottom_center  # use ground contact point for homography

                    if homography is not None:
                        bx, by = homography.pixel_to_court(ball_cx, ball_cy)
                        ball_mini_x, ball_mini_y = homography.court_to_pixel_2d(bx, by)
                        ball_mini_y = minimap_h - ball_mini_y
                    else:
                        frame_h, frame_w = frame_bgr.shape[:2]
                        ball_mini_x = int((ball_cx / frame_w) * minimap_w)
                        ball_mini_y = int((ball_cy / frame_h) * minimap_h)

                    ball_mini_x = max(6, min(ball_mini_x, minimap_w - 6))
                    ball_mini_y = max(6, min(ball_mini_y, minimap_h - 6))

                    # Orange filled circle with black border
                    cv2.circle(minimap, (ball_mini_x, ball_mini_y), 8, BALL_COLOR, -1)
                    cv2.circle(minimap, (ball_mini_x, ball_mini_y), 8, (0, 0, 0), 2)
                    break  # One ball per frame

            # Add border to minimap
            cv2.rectangle(minimap, (0, 0), (minimap_w - 1, minimap_h - 1), (100, 100, 100), 3)

            # Place minimap in top-right corner of frame (with vertical offset to avoid media player)
            minimap_x = frame_bgr.shape[1] - minimap_w - minimap_margin
            minimap_y = minimap_margin + 150  # Offset down to avoid media player controls

            # Blend minimap onto frame with slight transparency
            roi = frame_bgr[minimap_y:minimap_y + minimap_h, minimap_x:minimap_x + minimap_w]
            blended = cv2.addWeighted(roi, 0.3, minimap, 0.7, 0)
            frame_bgr[minimap_y:minimap_y + minimap_h, minimap_x:minimap_x + minimap_w] = blended

            # Position stats box below minimap
            stats_box_y = minimap_y + minimap_h + 10
        else:
            stats_box_y = 10

        # Draw tracking stats overlay
        total = t1_count + t2_count + t3_count
        stats_lines = [
            f"Players: {total}/12",
            f"T1 YOLO: {t1_count}",
        ]
        if t2_count > 0:
            stats_lines.append(f"T2 SAM: {t2_count}")
        if t3_count > 0:
            stats_lines.append(f"T3 Est: {t3_count}")

        # Draw stats box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        line_height = 25
        box_width = 130
        box_height = len(stats_lines) * line_height + 10
        box_x = frame_bgr.shape[1] - box_width - 10
        box_y = stats_box_y

        # Semi-transparent background
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
        frame_bgr = cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0)

        # Draw text with tier colors
        for i, line in enumerate(stats_lines):
            text_y = box_y + 20 + i * line_height
            if "T2" in line:
                text_color = (255, 255, 0)  # Cyan
            elif "T3" in line:
                text_color = (255, 0, 255)  # Magenta
            else:
                text_color = (255, 255, 255)  # White
            cv2.putText(frame_bgr, line, (box_x + 5, text_y), font, font_scale, text_color, font_thickness)

    # Draw frame info
    info_text = f"Frame: {frame_idx}"
    cv2.putText(
        frame_bgr,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    # Draw ball position (with optional trail annotator)
    if draw_ball:
        # Find best ball detection for this frame (highest confidence)
        best_ball = None
        best_conf = 0.0
        for ball in match_data.ball_positions:
            if ball.frame_idx == frame_idx and ball.bbox.confidence > best_conf:
                best_ball = ball
                best_conf = ball.bbox.confidence

        if ball_annotator is not None:
            # Trail annotator: jet-colormap tail with interpolated radius
            center = None
            if best_ball is not None:
                center = (
                    int((best_ball.bbox.x1 + best_ball.bbox.x2) / 2),
                    int((best_ball.bbox.y1 + best_ball.bbox.y2) / 2),
                )
            frame_bgr = ball_annotator.annotate(frame_bgr, center, frame_idx=frame_idx)
        elif best_ball is not None:
            # Fallback: simple filled circle
            center_x = int((best_ball.bbox.x1 + best_ball.bbox.x2) / 2)
            center_y = int((best_ball.bbox.y1 + best_ball.bbox.y2) / 2)
            radius = max(int((best_ball.bbox.x2 - best_ball.bbox.x1) / 2), 8)
            cv2.circle(frame_bgr, (center_x, center_y), radius, BALL_COLOR, -1)
            cv2.circle(frame_bgr, (center_x, center_y), radius, (255, 255, 255), 2)
            label = f"Ball {best_ball.bbox.confidence:.2f}"
            cv2.putText(
                frame_bgr,
                label,
                (center_x - 30, center_y - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                BALL_COLOR,
                1,
            )

    # Convert back to RGB
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def create_court_view(
    width: int = 800,
    height: int = 400,
    court_length: float = 40.0,
    court_width: float = 20.0,
) -> np.ndarray:
    """
    Create a blank court background for tactical view.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        court_length: Court length in meters
        court_width: Court width in meters

    Returns:
        BGR numpy array with court markings
    """
    # Create green background
    court = np.zeros((height, width, 3), dtype=np.uint8)
    court[:] = (34, 139, 34)  # Forest green

    # Scale factors
    scale_x = width / court_length
    scale_y = height / court_width

    # Court outline
    margin = 10
    cv2.rectangle(
        court,
        (margin, margin),
        (width - margin, height - margin),
        (255, 255, 255),
        2,
    )

    # Center line
    cv2.line(
        court,
        (width // 2, margin),
        (width // 2, height - margin),
        (255, 255, 255),
        2,
    )

    # Center circle (radius 3m)
    center_radius = int(3.0 * min(scale_x, scale_y))
    cv2.circle(
        court,
        (width // 2, height // 2),
        center_radius,
        (255, 255, 255),
        2,
    )

    # Penalty areas (6m x 3m from goal line)
    penalty_depth = int(6.0 * scale_x)
    penalty_width = int(12.0 * scale_y)
    penalty_y = (height - penalty_width) // 2

    # Left penalty area
    cv2.rectangle(
        court,
        (margin, penalty_y),
        (margin + penalty_depth, penalty_y + penalty_width),
        (255, 255, 255),
        2,
    )

    # Right penalty area
    cv2.rectangle(
        court,
        (width - margin - penalty_depth, penalty_y),
        (width - margin, penalty_y + penalty_width),
        (255, 255, 255),
        2,
    )

    # Goals (3m wide)
    goal_width = int(3.0 * scale_y)
    goal_y = (height - goal_width) // 2

    cv2.line(court, (margin, goal_y), (margin, goal_y + goal_width), (0, 0, 255), 4)
    cv2.line(
        court,
        (width - margin, goal_y),
        (width - margin, goal_y + goal_width),
        (0, 0, 255),
        4,
    )

    return court


def draw_tactical_view(
    court: np.ndarray,
    player_positions: list[tuple[int, float, float, TeamID]],
    ball_position: Optional[tuple[float, float]] = None,
    court_length: float = 40.0,
    court_width: float = 20.0,
    team_colors: Optional[dict[TeamID, tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Draw players and ball on tactical view.

    Args:
        court: Court background image
        player_positions: List of (track_id, x_meters, y_meters, team)
        ball_position: Ball (x_meters, y_meters) or None
        court_length: Court length in meters
        court_width: Court width in meters

    Returns:
        Annotated court view
    """
    height, width = court.shape[:2]
    frame = court.copy()

    # Scale factors
    margin = 10
    scale_x = (width - 2 * margin) / court_length
    scale_y = (height - 2 * margin) / court_width

    def to_pixels(x_m: float, y_m: float) -> tuple[int, int]:
        px = int(margin + x_m * scale_x)
        py = int(margin + y_m * scale_y)
        return px, py

    palette = team_colors or TEAM_COLORS

    # Draw players
    for track_id, x, y, team in player_positions:
        px, py = to_pixels(x, y)
        player_color = get_track_color(track_id)
        team_color = palette.get(team, player_color)

        # Player marker: dominant team ring, tiny player dot
        outer_radius = 16
        ring_thickness = 4
        inner_radius = 5  # small personal dot so team color dominates
        cv2.circle(frame, (px, py), outer_radius, team_color, ring_thickness)
        cv2.circle(frame, (px, py), inner_radius + 2, (0, 0, 0), 2)  # subtle outline for contrast
        cv2.circle(frame, (px, py), inner_radius, player_color, -1)

        # Track ID with slight shadow for legibility
        cv2.putText(frame, str(track_id), (px - 5, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(frame, str(track_id), (px - 5, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Draw ball (2D view): white filled circle, no outline
    if ball_position:
        px, py = to_pixels(ball_position[0], ball_position[1])
        cv2.circle(frame, (px, py), 8, (255, 255, 255), -1)

    return frame
