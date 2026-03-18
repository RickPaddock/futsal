"""2D futsal pitch drawing utilities for bird's-eye renders."""

import numpy as np
import cv2


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
    # Create dark green background for the tactical inset.
    court = np.zeros((height, width, 3), dtype=np.uint8)
    court[:] = (20, 70, 20)

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

    # Futsal penalty areas are 12m wide and 4m deep from the goal line.
    penalty_depth = int(4.0 * scale_x)
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
