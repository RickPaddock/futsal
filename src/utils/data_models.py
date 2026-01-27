"""
Pydantic data models for the futsal tracking system.

Defines structured data types for detections, tracks, events, and analytics.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import numpy as np


class TeamID(str, Enum):
    """Team identifier."""
    TEAM_A = "team_a"
    TEAM_B = "team_b"
    UNKNOWN = "unknown"
    REFEREE = "referee"


class BoundingBox(BaseModel):
    """Bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        """Get center point of bbox."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Get bottom center (feet position estimate)."""
        return ((self.x1 + self.x2) / 2, self.y2)

    def to_xyxy(self) -> list[float]:
        """Convert to [x1, y1, x2, y2] format."""
        return [self.x1, self.y1, self.x2, self.y2]

    def to_xywh(self) -> list[float]:
        """Convert to [x, y, width, height] format."""
        return [self.x1, self.y1, self.width, self.height]

    class Config:
        arbitrary_types_allowed = True


class Detection(BaseModel):
    """Single object detection in a frame."""
    frame_idx: int
    bbox: BoundingBox
    class_id: int = 0  # 0 = person in COCO
    class_name: str = "person"

    class Config:
        arbitrary_types_allowed = True


class PlayerDetection(Detection):
    """Player-specific detection with additional attributes."""
    team: TeamID = TeamID.UNKNOWN
    jersey_number: Optional[int] = None
    jersey_confidence: float = 0.0
    color_histogram: Optional[np.ndarray] = None  # HSV histogram for appearance matching
    is_interpolated: bool = False  # True if position was predicted (Kalman), not detected
    mask: Optional[np.ndarray] = None  # Optional binary mask from segmentation (SAM)
    is_sam_recovered: bool = False  # True if this detection was recovered by SAM (lost player found)
    sam_bbox: Optional[BoundingBox] = None  # SAM bounding box when both YOLO and SAM detect
    sam_mask: Optional[np.ndarray] = None  # SAM mask when both YOLO and SAM detect


class Track(BaseModel):
    """Object track across multiple frames."""
    track_id: int
    detections: list[Detection] = Field(default_factory=list)
    is_active: bool = True
    lost_frames: int = 0

    @property
    def start_frame(self) -> int:
        if not self.detections:
            return -1
        return self.detections[0].frame_idx

    @property
    def end_frame(self) -> int:
        if not self.detections:
            return -1
        return self.detections[-1].frame_idx

    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame + 1

    def get_detection_at_frame(self, frame_idx: int) -> Optional[Detection]:
        """Get detection at specific frame."""
        for det in self.detections:
            if det.frame_idx == frame_idx:
                return det
        return None

    class Config:
        arbitrary_types_allowed = True


class PlayerTrack(Track):
    """Player track with identification info."""
    team: TeamID = TeamID.UNKNOWN
    jersey_number: Optional[int] = None
    jersey_votes: dict[int, int] = Field(default_factory=dict)  # number -> vote count
    embedding: Optional[list[float]] = None  # Re-ID embedding

    def vote_jersey_number(self, number: int, confidence: float = 1.0):
        """Add a vote for a jersey number detection."""
        if number not in self.jersey_votes:
            self.jersey_votes[number] = 0
        self.jersey_votes[number] += int(confidence * 10)

    def get_confirmed_number(self, threshold: int = 5) -> Optional[int]:
        """Get jersey number if vote threshold met."""
        if not self.jersey_votes:
            return None
        best_number = max(self.jersey_votes, key=self.jersey_votes.get)
        if self.jersey_votes[best_number] >= threshold:
            return best_number
        return None




class CourtPosition(BaseModel):
    """Position in court coordinates (meters)."""
    x: float  # 0 to court_length (e.g., 40m)
    y: float  # 0 to court_width (e.g., 20m)

    def distance_to(self, other: "CourtPosition") -> float:
        """Euclidean distance to another position."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class EventType(str, Enum):
    """Types of detected events."""
    POSSESSION_START = "possession_start"
    POSSESSION_END = "possession_end"
    PASS = "pass"
    SHOT = "shot"
    GOAL = "goal"
    OUT_OF_PLAY = "out_of_play"


class Event(BaseModel):
    """Detected game event."""
    event_type: EventType
    frame_idx: int
    timestamp: float  # seconds from video start
    player_track_id: Optional[int] = None
    team: Optional[TeamID] = None
    position: Optional[CourtPosition] = None
    metadata: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class PassEvent(Event):
    """Pass event with source and target."""
    event_type: EventType = EventType.PASS
    from_player_id: Optional[int] = None
    to_player_id: Optional[int] = None
    start_position: Optional[CourtPosition] = None
    end_position: Optional[CourtPosition] = None
    distance: float = 0.0  # meters
    successful: bool = True


class ShotEvent(Event):
    """Shot on goal event."""
    event_type: EventType = EventType.SHOT
    velocity: float = 0.0  # m/s
    on_target: bool = False
    resulted_in_goal: bool = False


class GoalEvent(Event):
    """Goal scored event."""
    event_type: EventType = EventType.GOAL
    scoring_team: Optional[TeamID] = None
    scorer_track_id: Optional[int] = None
    assister_track_id: Optional[int] = None


class FrameData(BaseModel):
    """All tracking data for a single frame."""
    frame_idx: int
    player_detections: list[PlayerDetection] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class MatchData(BaseModel):
    """Complete match tracking data."""
    video_path: str
    fps: float
    total_frames: int
    width: int
    height: int

    player_tracks: list[PlayerTrack] = Field(default_factory=list)
    ball_positions: list[Detection] = Field(default_factory=list)  # Ball detections per frame
    events: list[Event] = Field(default_factory=list)
    
    # Low-confidence detections for debug visualization (filtered out from tracking)
    low_confidence_detections: dict[int, list] = Field(default_factory=dict)

    # Team info
    team_a_name: str = "Team A"
    team_b_name: str = "Team B"

    class Config:
        arbitrary_types_allowed = True

    def get_track_by_id(self, track_id: int) -> Optional[PlayerTrack]:
        """Get player track by ID."""
        for track in self.player_tracks:
            if track.track_id == track_id:
                return track
        return None

    def get_tracks_at_frame(self, frame_idx: int) -> list[PlayerTrack]:
        """Get all active player tracks at a specific frame."""
        return [
            track for track in self.player_tracks
            if track.start_frame <= frame_idx <= track.end_frame
        ]
    
    def get_ball_at_frame(self, frame_idx: int) -> Optional[Detection]:
        """Get ball detection at a specific frame."""
        for ball in self.ball_positions:
            if ball.frame_idx == frame_idx:
                return ball
        return None
