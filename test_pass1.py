"""
Test script for Pass 1: Raw Evidence Extraction

Run this to test the first pass of the pipeline.

Usage:
    python test_pass1.py

This will process the test clip and create:
- videos/output/GoPro_Futsal_part1_CLEANED_clip9/pass1_raw.json
- videos/output/GoPro_Futsal_part1_CLEANED_clip9/pass1_validation.json
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.skills.pass1_extractor import run_pass1
from src.utils.file_utils import get_output_dir

# Configuration
VIDEO_PATH = "videos/input/GoPro_Futsal_part1_CLEANED_clip9.mp4"
VIDEO_NAME = "GoPro_Futsal_part1_CLEANED_clip9"

# Model paths (adjust if needed)
PLAYER_MODEL = "models/PLAYER_MODEL_best_v1.pt"
BALL_MODEL = "models/BALL_MODEL_best_v2.pt"
JERSEY_MODEL = "models/JERSEY_MODEL_best_v1.pt"

# For testing, process only first 100 frames (remove end_frame=100 to process all)
START_FRAME = 0
END_FRAME = 100  # Set to None to process entire video


def main():
    """Run Pass 1 on test clip."""
    print("=" * 80)
    print("Pass 1: Raw Evidence Extraction - Test")
    print("=" * 80)
    print()

    # Check video exists
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        print(f"❌ Error: Video not found: {VIDEO_PATH}")
        print(f"   Please ensure the video is at: {video_path.absolute()}")
        return 1

    # Check models exist
    for model_name, model_path in [
        ("Player", PLAYER_MODEL),
        ("Ball", BALL_MODEL),
        ("Jersey", JERSEY_MODEL),
    ]:
        if not Path(model_path).exists():
            print(f"⚠️  Warning: {model_name} model not found: {model_path}")
            print(f"   This may cause errors during execution")

    # Get output directory
    output_dir = get_output_dir(VIDEO_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "pass1_raw.json"
    validation_path = output_dir / "pass1_validation.json"

    print(f"Input video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Processing frames: {START_FRAME} to {END_FRAME if END_FRAME else 'end'}")
    print()

    # Run Pass 1
    try:
        result = run_pass1(
            video_path=str(video_path),
            output_path=str(output_path),
            validation_output_path=str(validation_path),
            player_model_path=PLAYER_MODEL,
            ball_model_path=BALL_MODEL,
            jersey_model_path=JERSEY_MODEL,
            start_frame=START_FRAME,
            end_frame=END_FRAME,
        )

        print()
        print("=" * 80)
        print("✅ Pass 1 Complete!")
        print("=" * 80)
        print()
        print(f"Player detections: {len(result.detections)}")
        print(f"Ball detections: {len(result.ball_detections)}")
        print()
        print(f"Output files:")
        print(f"  - {output_path}")
        print(f"  - {validation_path}")
        print()
        print("Next steps:")
        print("  1. Check validation report: cat", str(validation_path))
        print("  2. Inspect detections: cat", str(output_path), "| head -50")
        print("  3. Continue with Pass 2 implementation")
        print()

        return 0

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ Pass 1 Failed!")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        print("Check the validation report for details:")
        print(f"  cat {validation_path}")
        print()

        import traceback
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())
