# PROV: FUSBAL.PIPELINE.VIDEO.01
# REQ: REQ-V1-VIDEO-INGEST-001, SYS-ARCH-15
# WHY: Provide a governed, deterministic video ingest API for local single-camera clips.

from .ingest import VideoFrame, VideoIngestError, VideoMetadata, iter_video_frames_rgb24, probe_video_metadata

__all__ = [
    "VideoFrame",
    "VideoIngestError",
    "VideoMetadata",
    "iter_video_frames_rgb24",
    "probe_video_metadata",
]

