"""
Futsal Tracking System - Main CLI Entry Point

Usage:
    # Run full pipeline
    python -m src.main --input videos/input/clip9.mp4

    # Run specific pass
    python -m src.main --input videos/input/clip9.mp4 --pass 1

    # Process only first 100 frames
    python -m src.main --input videos/input/clip9.mp4 --end-frame 100
"""

import sys
import argparse
from pathlib import Path

from .utils.file_utils import get_output_dir
from .utils.logging_utils import get_logger
from .skills.pass1_extractor import run_pass1

logger = get_logger("main")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Futsal Tracking System - Multi-pass player tracking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Pass 1 only
  python -m src.main --input videos/input/clip9.mp4 --pass 1

  # Run Pass 1 on first 100 frames
  python -m src.main --input videos/input/clip9.mp4 --pass 1 --end-frame 100

  # Run full pipeline (when implemented)
  python -m src.main --input videos/input/clip9.mp4
        """
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input video file"
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory (default: videos/output/<video_name>/)"
    )

    parser.add_argument(
        "--pass",
        dest="pass_number",
        type=int,
        choices=[1, 2, 3],
        help="Run specific pass only (1=raw, 2=fragments, 3=identity). Default: run all passes"
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start frame index (default: 0)"
    )

    parser.add_argument(
        "--end-frame",
        type=int,
        help="End frame index (default: process entire video)"
    )

    parser.add_argument(
        "--debug-pass1-roi-video",
        action="store_true",
        help="Write deterministic Pass 1 debug video with player bbox + jersey ROI overlays"
    )

    parser.add_argument(
        "--player-model",
        default="models/PLAYER_MODEL_best_v1.pt",
        help="Path to player detection model"
    )

    parser.add_argument(
        "--ball-model",
        default="models/BALL_MODEL_best_v2.pt",
        help="Path to ball detection model"
    )

    parser.add_argument(
        "--jersey-model",
        default="models/JERSEY_MODEL_best_v1.pt",
        help="Path to jersey classification model"
    )

    args = parser.parse_args()

    # Validate input video exists
    video_path = Path(args.input)
    if not video_path.exists():
        logger.error(f"Input video not found: {args.input}")
        return 1

    # Get output directory
    video_name = video_path.stem
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(get_output_dir(video_name))

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Futsal Tracking System - Multi-Pass Pipeline")
    logger.info("=" * 80)
    logger.info(f"Input video: {video_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Processing frames: {args.start_frame} to {args.end_frame or 'end'}")
    logger.info(f"Pass 1 ROI debug video: {'enabled' if args.debug_pass1_roi_video else 'disabled'}")
    logger.info("")

    # Determine which passes to run
    if args.pass_number:
        passes_to_run = [args.pass_number]
        logger.info(f"Running Pass {args.pass_number} only")
    else:
        passes_to_run = [1]  # TODO: Add [1, 2, 3] when all passes implemented
        logger.info("Running all implemented passes (currently: Pass 1 only)")

    logger.info("")

    try:
        # Run Pass 1
        if 1 in passes_to_run:
            logger.info("Starting Pass 1: Raw Evidence Extraction")
            logger.info("-" * 80)

            pass1_output = output_dir / "pass1_raw.json"
            pass1_validation = output_dir / "pass1_validation.json"

            result = run_pass1(
                video_path=str(video_path),
                output_path=str(pass1_output),
                validation_output_path=str(pass1_validation),
                player_model_path=args.player_model,
                ball_model_path=args.ball_model,
                jersey_model_path=args.jersey_model,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                debug_roi_video_path=(
                    str(output_dir / "pass1_debug_jersey_roi.mp4")
                    if args.debug_pass1_roi_video
                    else None
                ),
            )

            logger.info("")
            logger.info(f"✅ Pass 1 Complete!")
            logger.info(f"   Player detections: {len(result.detections)}")
            logger.info(f"   Ball detections: {len(result.ball_detections)}")
            logger.info(f"   Output: {pass1_output}")
            logger.info(f"   Validation: {pass1_validation}")
            if args.debug_pass1_roi_video:
                logger.info(f"   Debug video: {output_dir / 'pass1_debug_jersey_roi.mp4'}")
            logger.info("")

        # TODO: Run Pass 2
        if 2 in passes_to_run:
            logger.error("Pass 2 not yet implemented")
            return 1

        # TODO: Run Pass 3
        if 3 in passes_to_run:
            logger.error("Pass 3A and 3B not yet implemented (Pass 3C solver is ready)")
            return 1

        logger.info("=" * 80)
        logger.info("✅ Pipeline Complete!")
        logger.info("=" * 80)
        logger.info(f"All outputs saved to: {output_dir}")
        logger.info("")

        return 0

    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("❌ Pipeline Failed!")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error("")

        import traceback
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())
