"""
Video I/O utilities using PyAV.

Per CLAUDE.md Section 7 (Utilities):
- Video reading with frame-by-frame iteration
- Video writing with codec configuration
- Frame extraction and manipulation
- Pure I/O (no detection or tracking logic)
"""

import av
import numpy as np
from typing import Optional, Tuple, Generator
from pathlib import Path


class VideoReader:
    """
    Video reader using PyAV.

    Per CLAUDE.md Section 4 (Pass 1):
    - Read frames sequentially for detection
    - Extract metadata (fps, width, height, total frames)
    - Support seeking to specific frames
    """

    def __init__(self, video_path: str):
        """
        Initialize video reader.

        Args:
            video_path: Path to input video file

        Raises:
            FileNotFoundError: If video file doesn't exist
            av.AVError: If video cannot be opened
        """
        self.video_path = str(video_path)

        if not Path(self.video_path).exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        # Open video container
        self.container = av.open(self.video_path)
        self.stream = self.container.streams.video[0]

        # Extract metadata
        self.fps = float(self.stream.average_rate)
        self.width = self.stream.width
        self.height = self.stream.height

        # Estimate total frames
        # Note: Duration can be unreliable for some codecs
        if self.stream.frames > 0:
            self.total_frames = self.stream.frames
        else:
            # Fallback: estimate from duration
            duration_seconds = float(self.stream.duration * self.stream.time_base)
            self.total_frames = int(duration_seconds * self.fps)

        self.current_frame_idx = 0

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Close video container."""
        if self.container:
            self.container.close()
            self.container = None

    def read_frame(self) -> Optional[Tuple[int, np.ndarray]]:
        """
        Read next frame from video.

        Returns:
            (frame_idx, frame) tuple, or None if end of video
            - frame_idx: 0-indexed frame number
            - frame: BGR numpy array (H, W, 3)

        Examples:
            >>> reader = VideoReader("video.mp4")
            >>> frame_idx, frame = reader.read_frame()
            >>> frame.shape
            (1080, 1920, 3)
        """
        try:
            for packet in self.container.demux(self.stream):
                for frame in packet.decode():
                    # Convert to numpy array (BGR format for OpenCV compatibility)
                    img = frame.to_ndarray(format='bgr24')

                    frame_idx = self.current_frame_idx
                    self.current_frame_idx += 1

                    return (frame_idx, img)

        except av.AVError:
            return None

        return None

    def iter_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Iterate over all frames in video.

        Yields:
            (frame_idx, frame) tuples

        Examples:
            >>> reader = VideoReader("video.mp4")
            >>> for frame_idx, frame in reader.iter_frames():
            ...     print(f"Frame {frame_idx}: {frame.shape}")
        """
        while True:
            result = self.read_frame()
            if result is None:
                break
            yield result

    def seek(self, frame_idx: int) -> bool:
        """
        Seek to a specific frame.

        Note: Seeking is not frame-accurate for all codecs.
        Use with caution.

        Args:
            frame_idx: Target frame index

        Returns:
            True if seek successful

        Examples:
            >>> reader = VideoReader("video.mp4")
            >>> reader.seek(100)
            >>> frame_idx, frame = reader.read_frame()
            >>> frame_idx
            100
        """
        try:
            # Convert frame index to timestamp
            timestamp = int(frame_idx / self.fps / self.stream.time_base)
            self.container.seek(timestamp, stream=self.stream)
            self.current_frame_idx = frame_idx
            return True
        except av.AVError:
            return False

    def get_frame_at(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get a specific frame (non-sequential access).

        Warning: This is slow for large seeks. Use iter_frames() for sequential access.

        Args:
            frame_idx: Frame index to retrieve

        Returns:
            Frame as BGR numpy array, or None if frame not found

        Examples:
            >>> reader = VideoReader("video.mp4")
            >>> frame = reader.get_frame_at(100)
            >>> frame.shape
            (1080, 1920, 3)
        """
        if not self.seek(frame_idx):
            return None

        result = self.read_frame()
        if result is None:
            return None

        _, frame = result
        return frame

    def __repr__(self) -> str:
        return (
            f"VideoReader(path={self.video_path}, "
            f"fps={self.fps:.2f}, "
            f"resolution={self.width}x{self.height}, "
            f"frames={self.total_frames})"
        )


class VideoWriter:
    """
    Video writer using PyAV.

    Per CLAUDE.md Section 5 (Visualization):
    - Write frames sequentially
    - Support codec configuration
    - Output high-quality video for review
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = "libx264",
        pix_fmt: str = "yuv420p",
        bitrate: Optional[int] = None,
    ):
        """
        Initialize video writer.

        Args:
            output_path: Path to output video file
            fps: Frames per second
            width: Frame width in pixels
            height: Frame height in pixels
            codec: Video codec (default "libx264")
            pix_fmt: Pixel format (default "yuv420p")
            bitrate: Bitrate in bps (default None = auto)

        Examples:
            >>> writer = VideoWriter("output.mp4", fps=30, width=1920, height=1080)
            >>> writer.write_frame(frame)
            >>> writer.close()
        """
        self.output_path = str(output_path)
        self.fps = fps
        self.width = width
        self.height = height

        # Ensure output directory exists
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        # Open output container
        self.container = av.open(self.output_path, mode='w')

        # Add video stream
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = pix_fmt

        if bitrate:
            self.stream.bit_rate = bitrate

        self.frame_count = 0

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a frame to video.

        Args:
            frame: BGR numpy array (H, W, 3)

        Examples:
            >>> writer = VideoWriter("output.mp4", fps=30, width=1920, height=1080)
            >>> writer.write_frame(frame)
        """
        # Validate frame dimensions
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"Frame shape {frame.shape[:2]} doesn't match "
                f"expected {(self.height, self.width)}"
            )

        # Convert BGR to VideoFrame
        video_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
        video_frame.pts = self.frame_count

        # Encode and write
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

        self.frame_count += 1

    def close(self):
        """Finalize and close video file."""
        if self.container:
            # Flush encoder
            for packet in self.stream.encode():
                self.container.mux(packet)

            # Close container
            self.container.close()
            self.container = None

    def __repr__(self) -> str:
        return (
            f"VideoWriter(path={self.output_path}, "
            f"fps={self.fps:.2f}, "
            f"resolution={self.width}x{self.height}, "
            f"frames_written={self.frame_count})"
        )


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata without reading frames.

    Args:
        video_path: Path to video file

    Returns:
        Dict with keys: fps, width, height, total_frames, duration_seconds

    Examples:
        >>> info = get_video_info("video.mp4")
        >>> info
        {
            'fps': 30.0,
            'width': 1920,
            'height': 1080,
            'total_frames': 900,
            'duration_seconds': 30.0
        }
    """
    with VideoReader(video_path) as reader:
        return {
            'fps': reader.fps,
            'width': reader.width,
            'height': reader.height,
            'total_frames': reader.total_frames,
            'duration_seconds': reader.total_frames / reader.fps,
        }


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_indices: Optional[list] = None,
    format: str = "jpg",
) -> list:
    """
    Extract frames from video and save as images.

    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        frame_indices: Optional list of frame indices to extract (default: all)
        format: Image format (default "jpg")

    Returns:
        List of saved image paths

    Examples:
        >>> extract_frames("video.mp4", "/tmp/frames", frame_indices=[0, 10, 20])
        ['/tmp/frames/frame_0000.jpg', '/tmp/frames/frame_0010.jpg', '/tmp/frames/frame_0020.jpg']
    """
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    with VideoReader(video_path) as reader:
        for frame_idx, frame in reader.iter_frames():
            # Skip if frame_indices specified and current frame not in list
            if frame_indices is not None and frame_idx not in frame_indices:
                continue

            # Save frame
            output_path = output_dir / f"frame_{frame_idx:04d}.{format}"
            cv2.imwrite(str(output_path), frame)
            saved_paths.append(str(output_path))

    return saved_paths
