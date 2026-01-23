"""
Homography transformation for mapping pixel coordinates to court coordinates.
"""

import numpy as np
import cv2


class CourtHomography:
    """Transforms pixel coordinates to real-world court coordinates (meters)."""

    def __init__(
        self,
        source_points: np.ndarray,
        dest_points: np.ndarray,
        pitch_width_m: float = 40.0,
        pitch_height_m: float = 20.0,
        output_pixel_scale: int = 15,
    ):
        """
        Initialize homography from calibration points.

        Args:
            source_points: Pixel coordinates from video frame (Nx2)
            dest_points: Real-world court coordinates in meters (Nx2)
            pitch_width_m: Court length in meters (X axis)
            pitch_height_m: Court width in meters (Y axis)
            output_pixel_scale: Pixels per meter for 2D output
        """
        self.pitch_width_m = pitch_width_m
        self.pitch_height_m = pitch_height_m
        self.output_pixel_scale = output_pixel_scale
        self.output_w = int(pitch_width_m * output_pixel_scale)
        self.output_h = int(pitch_height_m * output_pixel_scale)

        # Calculate homography matrix
        self.H, _ = cv2.findHomography(source_points, dest_points, cv2.RANSAC, 5.0)
        if self.H is None:
            raise ValueError("Could not calculate homography from provided points")

    def pixel_to_court(self, x: float, y: float) -> tuple[float, float]:
        """
        Transform pixel coordinates to court coordinates (meters).

        Args:
            x: Pixel X coordinate
            y: Pixel Y coordinate

        Returns:
            (court_x, court_y) in meters
        """
        pt = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])

    def court_to_pixel_2d(self, court_x: float, court_y: float) -> tuple[int, int]:
        """
        Transform court coordinates (meters) to 2D pitch pixel coordinates.

        Args:
            court_x: Court X in meters (0 to pitch_width_m)
            court_y: Court Y in meters (0 to pitch_height_m)

        Returns:
            (pixel_x, pixel_y) for 2D pitch visualization
        """
        px = int(court_x * self.output_pixel_scale)
        py = int(court_y * self.output_pixel_scale)
        # Clamp to output bounds
        px = max(0, min(px, self.output_w - 1))
        py = max(0, min(py, self.output_h - 1))
        return px, py

    def transform_bbox_to_court(self, bbox) -> tuple[float, float]:
        """
        Transform a bounding box to court position using bottom-center (feet).

        Args:
            bbox: BoundingBox with x1, y1, x2, y2

        Returns:
            (court_x, court_y) in meters
        """
        # Use bottom-center as feet position
        foot_x = (bbox.x1 + bbox.x2) / 2
        foot_y = bbox.y2
        return self.pixel_to_court(foot_x, foot_y)


def create_homography_from_config(config: dict) -> CourtHomography | None:
    """
    Create CourtHomography from config dict.

    Args:
        config: Config dict with 'homography' section containing 'source_points' and 'dest_points'

    Returns:
        CourtHomography instance or None if not configured
    """
    h_cfg = config.get("homography", {})

    if "source_points" not in h_cfg or "dest_points" not in h_cfg:
        return None

    source = np.array(h_cfg["source_points"], dtype=np.float32)
    dest = np.array(h_cfg["dest_points"], dtype=np.float32)

    return CourtHomography(
        source_points=source,
        dest_points=dest,
        pitch_width_m=h_cfg.get("court_length", 40.0),
        pitch_height_m=h_cfg.get("court_width", 20.0),
        output_pixel_scale=h_cfg.get("output_pixel_scale", 15),
    )
