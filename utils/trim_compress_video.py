#!/usr/bin/env python3
"""
Trim and compress GoPro futsal video for player identification testing.
Extracts middle portion and compresses to ~100MB.
"""

import subprocess
import os
import sys

# Configuration
INPUT_VIDEO = "input/videos/GoPro_Futsal_part2.mp4"
OUTPUT_VIDEO = "output/GoPro_Futsal_part2_trimmed_100mb.mp4"
OUTPUT_DIR = "output"

# Video parameters
DURATION = 142.4  # seconds
TARGET_SIZE_MB = 100
START_TIME = 51  # Start at 51 seconds (middle portion)
TRIM_DURATION = 40  # 40 seconds duration

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def calculate_target_bitrate(duration_seconds, target_mb):
    """Calculate target video bitrate to achieve desired file size."""
    # Formula: bitrate = (target_size_mb * 8192) / duration - audio_bitrate
    # Assuming 128kbps audio
    audio_bitrate_kbps = 128
    target_video_bitrate = (target_mb * 8192) / duration_seconds - audio_bitrate_kbps
    return int(target_video_bitrate)

def trim_and_compress_video():
    """Trim middle portion and compress using ffmpeg."""
    ensure_output_dir()

    # Calculate target bitrate
    target_bitrate = calculate_target_bitrate(TRIM_DURATION, TARGET_SIZE_MB)
    print(f"\nVideo Compression Settings:")
    print(f"  Input: {INPUT_VIDEO}")
    print(f"  Output: {OUTPUT_VIDEO}")
    print(f"  Start time: {START_TIME}s")
    print(f"  Duration: {TRIM_DURATION}s")
    print(f"  Target size: ~{TARGET_SIZE_MB}MB")
    print(f"  Target video bitrate: ~{target_bitrate}kbps")
    print()

    # Build ffmpeg command
    # Using H.265 (HEVC) with CRF for quality-based encoding
    # CRF 28 provides good quality while maintaining small file size
    cmd = [
        'ffmpeg',
        '-ss', str(START_TIME),           # Start time
        '-i', INPUT_VIDEO,                 # Input file
        '-t', str(TRIM_DURATION),          # Duration
        '-c:v', 'libx265',                 # H.265 codec
        '-crf', '17',                      # Constant Rate Factor (lower = better quality)
        '-preset', 'medium',               # Encoding speed vs compression
        '-c:a', 'aac',                     # Audio codec
        '-b:a', '128k',                    # Audio bitrate
        '-movflags', '+faststart',         # Enable streaming
        '-y',                              # Overwrite output
        OUTPUT_VIDEO
    ]

    print("Running ffmpeg...")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Video processing complete!")

        # Check output file size
        if os.path.exists(OUTPUT_VIDEO):
            size_mb = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
            print(f"\nOutput file: {OUTPUT_VIDEO}")
            print(f"Final size: {size_mb:.2f}MB")

            if size_mb > TARGET_SIZE_MB * 1.1:
                print(f"\n⚠ Warning: Output size ({size_mb:.2f}MB) exceeds target by >10%")
                print("Consider adjusting CRF value (higher = smaller file) or duration")
            elif size_mb < TARGET_SIZE_MB * 0.8:
                print(f"\n⚠ Note: Output size ({size_mb:.2f}MB) is significantly smaller than target")
                print("Consider lowering CRF value (lower = better quality) for better quality")

    except subprocess.CalledProcessError as e:
        print(f"✗ Error during video processing:")
        print(e.stderr)
        sys.exit(1)

def trim_and_compress_twopass():
    """Trim and compress using two-pass encoding for exact file size."""
    ensure_output_dir()

    target_bitrate = calculate_target_bitrate(TRIM_DURATION, TARGET_SIZE_MB)

    print(f"\nTwo-Pass Encoding:")
    print(f"  Target bitrate: {target_bitrate}kbps")
    print()

    # Pass 1
    cmd_pass1 = [
        'ffmpeg',
        '-ss', str(START_TIME),
        '-i', INPUT_VIDEO,
        '-t', str(TRIM_DURATION),
        '-c:v', 'libx265',
        '-b:v', f'{target_bitrate}k',
        '-preset', 'medium',
        '-pass', '1',
        '-an',  # No audio in pass 1
        '-f', 'null',
        '/dev/null' if sys.platform != 'win32' else 'NUL'
    ]

    print("Pass 1/2...")
    subprocess.run(cmd_pass1, check=True)

    # Pass 2
    cmd_pass2 = [
        'ffmpeg',
        '-ss', str(START_TIME),
        '-i', INPUT_VIDEO,
        '-t', str(TRIM_DURATION),
        '-c:v', 'libx265',
        '-b:v', f'{target_bitrate}k',
        '-preset', 'medium',
        '-pass', '2',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        '-y',
        OUTPUT_VIDEO
    ]

    print("Pass 2/2...")
    subprocess.run(cmd_pass2, check=True)

    # Clean up pass files
    for f in ['ffmpeg2pass-0.log', 'ffmpeg2pass-0.log.mbtree']:
        if os.path.exists(f):
            os.remove(f)

    print("✓ Two-pass encoding complete!")

    # Check output file size
    if os.path.exists(OUTPUT_VIDEO):
        size_mb = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
        print(f"\nOutput file: {OUTPUT_VIDEO}")
        print(f"Final size: {size_mb:.2f}MB")

if __name__ == "__main__":
    # Use single-pass CRF encoding by default (faster, good quality)
    # Uncomment the line below to use two-pass encoding for more precise file size
    trim_and_compress_video()
    # trim_and_compress_twopass()
