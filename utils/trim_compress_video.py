#!/usr/bin/env python3
"""
Trim and stitch video segments - keeps only specified time ranges.
"""

import subprocess
import os
import sys

# Configuration
INPUT_VIDEO = "videos/input/ORIGINAL/GoPro_Futsal_part1.mp4"
OUTPUT_PATTERN = "videos/input/GoPro_Futsal_part1_CLEANED_clip{}.mp4"  # {} will be replaced with clip number

# Segments to KEEP (start, end) in "MM:SS" or "SS" format
# Use None for end to mean "to end of video"
KEEP_SEGMENTS = [
    ("1:18", "2:25"),
    ("3:05", "3:48"),
    ("3:57", "4:20"),
    ("4:24", "4:52"),
    ("5:00", "5:16"),
    ("5:20", "6:09"),
    ("6:22", "7:31"),
    ("7:40", "7:54"),
    ("7:58", "8:39"),
    ("8:59", "9:25"),
    ("9:34", "10:04"),
    ("10:10", "11:04"),
    ("11:16", "13:36"),
    ("13:42", "14:40"),
    ("14:45", "15:07"),
    ("15:11", "15:32"),
    ("15:39", "17:05")
]


def parse_time(time_str: str) -> float:
    """Convert MM:SS or SS string to seconds."""
    parts = time_str.split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Invalid time format: {time_str}")


def format_time(seconds):
    """Format seconds as MM:SS for display."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:05.2f}"


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def extract_segments():
    """Extract segments and save as separate clip files."""

    # Verify input exists
    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: Input video not found: {INPUT_VIDEO}")
        sys.exit(1)

    # Get video duration for "to end" segments
    duration = get_video_duration(INPUT_VIDEO)
    print(f"Input video duration: {format_time(duration)}")

    # Parse segments
    segments: list[tuple[float, float]] = []
    total_kept = 0.0
    for start_str, end_str in KEEP_SEGMENTS:
        start = parse_time(start_str)
        end = parse_time(end_str) if end_str else duration
        segments.append((start, end))
        total_kept += end - start

    print(f"\nSegments to extract: {len(segments)}")
    print(f"Total duration: {format_time(total_kept)}")
    print()

    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_PATTERN)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract each segment as a separate file
    output_files = []
    for i, (start, end) in enumerate(segments):
        clip_num = i + 1
        output_path = OUTPUT_PATTERN.format(clip_num)
        output_files.append(output_path)

        print(f"Extracting clip {clip_num}/{len(segments)}: {format_time(start)} - {format_time(end)}")

        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start),
            '-i', INPUT_VIDEO,
            '-t', str(end - start),
            '-c', 'copy',  # No re-encoding for speed
            '-avoid_negative_ts', 'make_zero',
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error extracting clip {clip_num}:")
            print(result.stderr)
            sys.exit(1)

    # Report results
    print(f"\nComplete! Created {len(output_files)} clips:")
    for output_path in output_files:
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            output_duration = get_video_duration(output_path)
            print(f"  {output_path}")
            print(f"    Duration: {format_time(output_duration)}, Size: {output_size:.1f} MB")
        else:
            print(f"  Error: {output_path} was not created")
            sys.exit(1)


if __name__ == "__main__":
    extract_segments()
