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
from typing import Set

from .utils.file_utils import get_output_dir, load_json
from .utils.logging_utils import get_logger
from .skills.pass1_extractor import run_pass1, render_pass1_debug_video_from_artifact
from .skills.pass2a_fragmenter import run_pass2a, render_pass2a_debug_video_from_artifact
from .core.data_models import Pass1Output

logger = get_logger("main")

ALLOWED_VIDEO_OUTPUT_PASSES = {"1", "2", "2a", "2b", "3", "3a", "3b", "3c", "ball", "viz"}
IMPLEMENTED_VIDEO_OUTPUT_PASSES = {"1", "2a"}


def _parse_video_output_option(value: str) -> Set[str]:
    """
    Parse --video-output option into normalized pass keys.

    Accepts comma-separated values such as: "1,2a,2b".
    Aliases:
        2 -> 2a
        3 -> 3c
    """
    if not value:
        return set()

    normalized: Set[str] = set()
    tokens = [token.strip().lower() for token in value.split(",") if token.strip()]

    for token in tokens:
        if token == "2":
            normalized.add("2a")
            continue
        if token == "3":
            normalized.add("3c")
            continue

        if token not in ALLOWED_VIDEO_OUTPUT_PASSES:
            allowed = ", ".join(sorted(ALLOWED_VIDEO_OUTPUT_PASSES))
            raise argparse.ArgumentTypeError(
                f"Invalid --video-output value '{token}'. Allowed values: {allowed}"
            )

        normalized.add(token)

    return normalized


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Futsal Tracking System - Multi-pass player tracking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Pass 1 only
  python -m src.main --input videos/input/clip9.mp4 --pass 1

  # Run Pass 2A only (requires Pass 1 output exists)
  python -m src.main --input videos/input/clip9.mp4 --pass 2

  # Run Pass 1 on first 100 frames
  python -m src.main --input videos/input/clip9.mp4 --pass 1 --end-frame 100

  # Run full pipeline (when fully implemented)
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
        help="Run specific pass only (1=raw, 2A=fragments, 3=identity). Default: run all passes"
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
        "--video-output",
        type=_parse_video_output_option,
        default=set(),
        help=(
            "Comma-separated pass keys for debug video output "
            "(examples: 1,2a,2b,3c,ball,viz). "
            "Currently implemented: 1,2a"
        )
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
    if args.video_output:
        requested_video_outputs = ", ".join(sorted(args.video_output))
        logger.info(f"Debug video outputs requested for passes: {requested_video_outputs}")
    else:
        logger.info("Debug video outputs requested: none")

    unimplemented_video_outputs = sorted(args.video_output - IMPLEMENTED_VIDEO_OUTPUT_PASSES)
    if unimplemented_video_outputs:
        logger.warning(
            f"Debug video outputs not implemented yet for: {', '.join(unimplemented_video_outputs)}"
        )
    logger.info("")

    # Determine which passes to run
    if args.pass_number:
        passes_to_run = [args.pass_number]
        logger.info(f"Running Pass {args.pass_number} only")
    else:
        passes_to_run = [1, 2]  # TODO: Add [1, 2, 3] when Pass 3 implemented
        logger.info("Running all implemented passes (currently: Pass 1, Pass 2A)")

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
                    str(output_dir / "pass1_debug.mp4")
                    if "1" in args.video_output
                    else None
                ),
            )

            logger.info("")
            logger.info(f"[OK] Pass 1 Complete!")
            logger.info(f"   Player detections: {len(result.detections)}")
            logger.info(f"   Ball detections: {len(result.ball_detections)}")
            logger.info(f"   Output: {pass1_output}")
            logger.info(f"   Validation: {pass1_validation}")
            if "1" in args.video_output:
                logger.info(f"   Debug video: {output_dir / 'pass1_debug.mp4'}")
            logger.info("")

        if "1" in args.video_output and 1 not in passes_to_run:
            pass1_output_path = output_dir / "pass1_raw.json"
            if not pass1_output_path.exists():
                logger.error(f"Cannot render Pass 1 debug video: missing {pass1_output_path}")
                logger.error("Run Pass 1 first or run without --pass to generate pass1_raw.json")
                return 1

            debug_video_path = output_dir / "pass1_debug.mp4"
            logger.info("Generating Pass 1 debug video from existing pass1_raw.json")
            render_pass1_debug_video_from_artifact(
                video_path=str(video_path),
                pass1_output_path=str(pass1_output_path),
                debug_video_path=str(debug_video_path),
                start_frame=args.start_frame,
                end_frame=args.end_frame,
            )
            logger.info(f"Pass 1 debug video written: {debug_video_path}")
            logger.info("")

        # Run Pass 2A
        if 2 in passes_to_run:
            logger.info("Starting Pass 2A: Mechanical Fragmentation")
            logger.info("-" * 80)

            # Check Pass 1 output exists
            pass1_output_path = output_dir / "pass1_raw.json"
            if not pass1_output_path.exists():
                logger.error(f"Pass 1 output not found: {pass1_output_path}")
                logger.error("Please run Pass 1 first: python -m src.main --input <video> --pass 1")
                return 1

            # Load Pass 1 data for reference
            pass1_output = load_json(str(pass1_output_path), Pass1Output)
            logger.info(f"Loaded {len(pass1_output.detections)} detections from Pass 1")

            # Run Pass 2A fragmentation
            pass2a_output = run_pass2a(input_dir=output_dir, output_dir=output_dir)

            logger.info("")
            logger.info(f"[OK] Pass 2A Complete!")
            logger.info(f"   Fragments created: {len(pass2a_output.fragments)}")

            # Split reasons breakdown
            split_reasons = {}
            for frag in pass2a_output.fragments:
                reason = frag.split_reason or "initial"
                split_reasons[reason] = split_reasons.get(reason, 0) + 1

            logger.info(f"   Split reasons:")
            for reason, count in sorted(split_reasons.items(), key=lambda x: -x[1]):
                logger.info(f"     {reason}: {count}")

            logger.info(f"   Output: {output_dir / 'pass2_fragments.json'}")
            logger.info(f"   Validation: {output_dir / 'pass2_validation.json'}")
            if "2a" in args.video_output:
                pass2a_debug_path = output_dir / "pass2a_debug.mp4"
                logger.info("   Rendering Pass 2A debug video from artifacts...")
                render_pass2a_debug_video_from_artifact(
                    video_path=str(video_path),
                    pass1_output_path=str(output_dir / "pass1_raw.json"),
                    pass2a_output_path=str(output_dir / "pass2_fragments.json"),
                    debug_video_path=str(pass2a_debug_path),
                    start_frame=args.start_frame,
                    end_frame=args.end_frame,
                )
                logger.info(f"   Debug video: {pass2a_debug_path}")
            logger.info("")

        if "2a" in args.video_output and 2 not in passes_to_run:
            pass1_output_path = output_dir / "pass1_raw.json"
            pass2a_output_path = output_dir / "pass2_fragments.json"

            if not pass1_output_path.exists():
                logger.error(f"Cannot render Pass 2A debug video: missing {pass1_output_path}")
                logger.error("Run Pass 1 and Pass 2 first or run without --pass to generate artifacts")
                return 1

            if not pass2a_output_path.exists():
                logger.error(f"Cannot render Pass 2A debug video: missing {pass2a_output_path}")
                logger.error("Run Pass 2 first or run without --pass to generate pass2_fragments.json")
                return 1

            pass2a_debug_path = output_dir / "pass2a_debug.mp4"
            logger.info("Generating Pass 2A debug video from existing artifacts")
            render_pass2a_debug_video_from_artifact(
                video_path=str(video_path),
                pass1_output_path=str(pass1_output_path),
                pass2a_output_path=str(pass2a_output_path),
                debug_video_path=str(pass2a_debug_path),
                start_frame=args.start_frame,
                end_frame=args.end_frame,
            )
            logger.info(f"Pass 2A debug video written: {pass2a_debug_path}")
            logger.info("")

        # TODO: Run Pass 3
        if 3 in passes_to_run:
            logger.error("Pass 3A and 3B not yet implemented (Pass 3C solver is ready)")
            return 1

        logger.info("=" * 80)
        logger.info("[OK] Pipeline Complete!")
        logger.info("=" * 80)
        logger.info(f"All outputs saved to: {output_dir}")
        logger.info("")

        return 0

    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("[FAIL] Pipeline Failed!")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error("")

        import traceback
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())
