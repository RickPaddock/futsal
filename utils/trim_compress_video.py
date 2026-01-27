#!/usr/bin/env python3
"""
Trim and stitch video segments - keeps only specified time ranges.
"""

import subprocess
import os
import sys
import tempfile

# Configuration
INPUT_VIDEO = "videos/input/ORIGINAL/GoPro_Futsal_part2.mp4"
OUTPUT_VIDEO = "videos/input/GoPro_Futsal_part2_CLEANED.mp4"

# Segments to KEEP (start, end) in "MM:SS" or "SS" format
# Use None for end to mean "to end of video"
KEEP_SEGMENTS = [
    ("0:22", "1:10"),
    ("1:15", "2:05")
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


def extract_and_concat_segments():
    """Extract segments and concatenate them into one video."""

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

    print(f"\nSegments to keep: {len(segments)}")
    print(f"Total kept duration: {format_time(total_kept)}")
    print(f"Removed: {format_time(duration - total_kept)}")
    print()

    # Create temp directory for segment files
    with tempfile.TemporaryDirectory() as temp_dir:
        segment_files = []
        concat_list_path = os.path.join(temp_dir, "concat_list.txt")

        # Extract each segment
        for i, (start, end) in enumerate(segments):
            segment_path = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
            segment_files.append(segment_path)

            print(f"Extracting segment {i+1}/{len(segments)}: {format_time(start)} - {format_time(end)}")

            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start),
                '-i', INPUT_VIDEO,
                '-t', str(end - start),
                '-c', 'copy',  # No re-encoding for speed
                '-avoid_negative_ts', 'make_zero',
                segment_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error extracting segment {i+1}:")
                print(result.stderr)
                sys.exit(1)

        # Create concat list file
        with open(concat_list_path, 'w') as f:
            for seg_path in segment_files:
                f.write(f"file '{seg_path}'\n")

        # Ensure output directory exists
        output_dir = os.path.dirname(OUTPUT_VIDEO)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Concatenate all segments
        print(f"\nConcatenating {len(segments)} segments...")

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list_path,
            '-c', 'copy',
            OUTPUT_VIDEO
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Error concatenating segments:")
            print(result.stderr)
            sys.exit(1)

    # Report results
    if os.path.exists(OUTPUT_VIDEO):
        output_size = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
        output_duration = get_video_duration(OUTPUT_VIDEO)
        print(f"\nComplete!")
        print(f"Output: {OUTPUT_VIDEO}")
        print(f"Duration: {format_time(output_duration)}")
        print(f"Size: {output_size:.1f} MB")
    else:
        print("Error: Output file was not created")
        sys.exit(1)


if __name__ == "__main__":
    extract_and_concat_segments()
