"""
Video I/O module using PyAV for memory-efficient streaming.

Handles large video files (11GB+) by streaming frames rather than loading all into memory.
"""

from fractions import Fraction
from pathlib import Path
from typing import Iterator, Generator
import numpy as np

try:
    import av
except ImportError:
    raise ImportError("PyAV is required. Install with: pip install av")


class VideoReader:
    """Memory-efficient video reader using PyAV streaming."""

    def __init__(self, video_path: Path | str):
        """
        Initialize video reader.

        Args:
            video_path: Path to the video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # Open container to get metadata
        container = av.open(str(self.video_path))
        stream = container.streams.video[0]

        self.width = stream.width
        self.height = stream.height
        self.fps = float(stream.average_rate)
        self.total_frames = stream.frames
        self.duration = float(stream.duration * stream.time_base) if stream.duration else None
        self.codec = stream.codec_context.name

        container.close()

    def __repr__(self) -> str:
        return (
            f"VideoReader({self.video_path.name}, "
            f"{self.width}x{self.height}, "
            f"{self.fps:.2f}fps, "
            f"{self.total_frames} frames)"
        )

    def frames(
        self,
        start_frame: int = 0,
        end_frame: int | None = None,
        step: int = 1,
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """
        Iterate over video frames as numpy arrays.

        Args:
            start_frame: First frame to yield (0-indexed)
            end_frame: Last frame to yield (exclusive), None for all
            step: Yield every Nth frame

        Yields:
            Tuple of (frame_number, frame_array) where frame_array is RGB uint8
        """
        container = av.open(str(self.video_path))
        stream = container.streams.video[0]

        # Seek to start frame if needed
        if start_frame > 0:
            # Convert frame to timestamp and seek
            timestamp = int(start_frame / self.fps / stream.time_base)
            container.seek(timestamp, stream=stream)

        frame_idx = 0
        for frame in container.decode(video=0):
            # After seeking, calculate actual frame index from PTS
            if frame.pts is not None:
                frame_idx = int(frame.pts * stream.time_base * self.fps)

            # Skip frames before start (seek may land before target)
            if frame_idx < start_frame:
                frame_idx += 1
                continue

            # Stop at end frame
            if end_frame is not None and frame_idx >= end_frame:
                break

            # Apply step
            if (frame_idx - start_frame) % step == 0:
                # Convert to numpy RGB array
                img = frame.to_ndarray(format="rgb24")
                yield frame_idx, img

            frame_idx += 1

        container.close()

    def get_frame(self, frame_number: int) -> np.ndarray:
        """
        Get a specific frame by number.

        Args:
            frame_number: Frame index (0-indexed)

        Returns:
            Frame as RGB numpy array
        """
        for idx, frame in self.frames(start_frame=frame_number, end_frame=frame_number + 1):
            return frame
        raise ValueError(f"Frame {frame_number} not found")

    def batch_frames(
        self,
        batch_size: int,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> Generator[list[tuple[int, np.ndarray]], None, None]:
        """
        Iterate over frames in batches for GPU inference.

        Args:
            batch_size: Number of frames per batch
            start_frame: First frame to yield
            end_frame: Last frame to yield (exclusive)

        Yields:
            List of (frame_number, frame_array) tuples
        """
        batch = []
        for frame_idx, frame in self.frames(start_frame=start_frame, end_frame=end_frame):
            batch.append((frame_idx, frame))
            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield remaining frames
        if batch:
            yield batch


class VideoWriter:
    """Video writer using PyAV."""

    def __init__(
        self,
        output_path: Path | str,
        width: int,
        height: int,
        fps: float = 30.0,
        codec: str = "h264",
        crf: int = 23,
    ):
        """
        Initialize video writer.

        Args:
            output_path: Path for output video
            width: Frame width
            height: Frame height
            fps: Frames per second
            codec: Video codec (h264, hevc, etc.)
            crf: Constant rate factor (quality, lower = better, 18-28 typical)
        """
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps

        self.container = av.open(str(self.output_path), mode="w")
        # Convert fps to Fraction for PyAV compatibility
        fps_fraction = Fraction(fps).limit_denominator(10000)
        self.stream = self.container.add_stream(codec, rate=fps_fraction)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(crf)}

    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to the video.

        Args:
            frame: RGB numpy array (height, width, 3)
        """
        # Ensure correct dimensions
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Frame shape {frame.shape[:2]} doesn't match "
                f"video dimensions ({self.height}, {self.width})"
            )

        # Convert to PyAV frame
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

        # Encode and write
        for packet in self.stream.encode(av_frame):
            self.container.mux(packet)

    def close(self):
        """Finalize and close the video file."""
        # Flush encoder
        for packet in self.stream.encode():
            self.container.mux(packet)

        self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def trim_video(
    input_path: Path,
    output_path: Path,
    start_time: float = 0,
    duration: float | None = None,
    scale: float = 1.0,
):
    """
    Trim and optionally resize a video.

    Uses FFmpeg via PyAV for efficient processing.

    Args:
        input_path: Source video path
        output_path: Output video path
        start_time: Start time in seconds
        duration: Duration in seconds (None for rest of video)
        scale: Scale factor (0.5 = half resolution)
    """
    reader = VideoReader(input_path)

    # Calculate frame range
    start_frame = int(start_time * reader.fps)
    end_frame = None
    if duration:
        end_frame = start_frame + int(duration * reader.fps)

    # Calculate output dimensions
    out_width = int(reader.width * scale)
    out_height = int(reader.height * scale)

    # Ensure even dimensions for video encoding
    out_width = out_width - (out_width % 2)
    out_height = out_height - (out_height % 2)

    with VideoWriter(output_path, out_width, out_height, reader.fps) as writer:
        import cv2

        for frame_idx, frame in reader.frames(start_frame=start_frame, end_frame=end_frame):
            if scale != 1.0:
                frame = cv2.resize(frame, (out_width, out_height))
            writer.write_frame(frame)


def get_video_info(video_path: Path | str) -> dict:
    """
    Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video information
    """
    reader = VideoReader(video_path)
    return {
        "path": str(reader.video_path),
        "width": reader.width,
        "height": reader.height,
        "fps": reader.fps,
        "total_frames": reader.total_frames,
        "duration": reader.duration,
        "codec": reader.codec,
    }
