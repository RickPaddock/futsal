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
from .skills.pass2b_fragment_scoring import run_pass2b
from .skills.pass2c_ghost_generator import run_pass2c
from .skills.ball_interpolator import run_ball_interpolation
from .skills.analytics_distance import run_analytics_distance, run_analytics_summary
from .skills.analytics_passes import run_analytics_pass_detection
from .skills.analytics_possession import run_analytics_possession
from .skills.analytics_visualizer import run_analytics_events_visualization, run_analytics_possession_visualization
from .skills.birds_eye_pitch import run_birds_eye_pitch, render_birds_eye_debug_video_from_artifact
from .skills.visualizer import run_visualization
from .skills.pass3a_candidate_generator import run_pass3a
from .skills.pass3b_constraint_builder import run_pass3b
from .skills.pass3_debug_visualizer import render_pass3_debug_video_from_artifact
from .skills.pass3c_identity_solver import run_pass3c
from .core.data_models import Pass1Output

logger = get_logger("main")

ALLOWED_VIDEO_OUTPUT_PASSES = {"1", "2", "3", "4", "analytics", "ball", "events", "poss", "viz"}
IMPLEMENTED_VIDEO_OUTPUT_PASSES = {"1", "2", "3", "4", "analytics", "events", "poss", "viz"}


def _parse_video_output_option(value: str) -> Set[str]:
    """
    Parse --video-output option into normalized pass keys.

    Accepts comma-separated values such as: "1,2,3".
    Pass 2 = unified video (fragments + quality + ghosts)
    """
    if not value:
        return set()

    normalized: Set[str] = set()
    tokens = [token.strip().lower() for token in value.split(",") if token.strip()]

    for token in tokens:
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
        choices=[1, 2, 3, 4],
        help="Run specific pass only (1=raw, 2=fragments, 3=identity, 4=birdseye). Default: run all implemented passes"
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
            "Comma-separated pass keys for debug video output. "
            "Examples: 1,2,3,4,analytics,viz. "
            "Implemented: 1 (raw detections), 2 (fragments + quality + ghosts), 3 (committed identity), 4 (birdseye inset), analytics (unified analytics debug overlay), viz (final visualization)"
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
        passes = ", ".join(sorted(args.video_output))
        logger.info(f"Debug video outputs requested: {passes}")
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
        passes_to_run = [1, 2, 3, 4]
        logger.info("Running all implemented passes (currently: Pass 1, Pass 2, Pass 3, Pass 4, analytics possession, analytics pass detection)")

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

            logger.info(f"   Output (raw): {output_dir / 'pass2_fragments.json'}")
            logger.info(f"   Validation: {output_dir / 'pass2_validation.json'}")

            # Run Pass 2B quality scoring
            logger.info("")
            run_pass2b(
                pass1_json_path=pass1_output_path,
                pass2a_json_path=output_dir / "pass2_fragments.json",
                output_dir=output_dir,
            )
            logger.info(f"[OK] Pass 2B Complete!")
            logger.info(f"   Output (scored): {output_dir / 'pass2b_scored_fragments.json'}")

            # Run Pass 2C ghost generation
            logger.info("")
            pass2c_output = run_pass2c(input_dir=output_dir, output_dir=output_dir)
            logger.info(f"[OK] Pass 2C Complete!")
            logger.info(f"   Total fragments (with ghosts): {len(pass2c_output.fragments)}")
            ghost_count = sum(1 for f in pass2c_output.fragments if f.is_ghost)
            logger.info(f"   Ghosts created: {ghost_count}")
            logger.info(f"   Output: {output_dir / 'pass2_ghosts.json'}")

            if "2" in args.video_output:
                pass2_debug_path = output_dir / "pass2_debug.mp4"
                logger.info("   Rendering Pass 2 debug video (fragments + quality + ghosts)...")
                # Check if ghosts JSON exists (from Pass 2C)
                ghosts_path = output_dir / "pass2_ghosts.json"
                render_pass2a_debug_video_from_artifact(
                    video_path=str(video_path),
                    pass1_output_path=str(output_dir / "pass1_raw.json"),
                    pass2a_output_path=str(output_dir / "pass2_fragments.json"),
                    debug_video_path=str(pass2_debug_path),
                    start_frame=args.start_frame,
                    end_frame=args.end_frame,
                    pass2c_ghosts_path=str(ghosts_path) if ghosts_path.exists() else None,
                )
                logger.info(f"   Debug video: {pass2_debug_path}")
            logger.info("")

        if "2a" in args.video_output and 2 not in passes_to_run:
            pass1_output_path = output_dir / "pass1_raw.json"
            pass2a_output_path = output_dir / "pass2_fragments.json"

            if not pass1_output_path.exists():
                logger.error(f"Cannot render Pass 2 debug video: missing {pass1_output_path}")
                logger.error("Run Pass 1 and Pass 2 first or run without --pass to generate artifacts")
                return 1

            if not pass2a_output_path.exists():
                logger.error(f"Cannot render Pass 2 debug video: missing {pass2a_output_path}")
                logger.error("Run Pass 2 first or run without --pass to generate pass2_fragments.json")
                return 1

            pass2_debug_path = output_dir / "pass2_debug.mp4"
            ghosts_path = output_dir / "pass2_ghosts.json"
            logger.info("Generating Pass 2 debug video from existing artifacts (fragments + quality + ghosts)")
            render_pass2a_debug_video_from_artifact(
                video_path=str(video_path),
                pass1_output_path=str(pass1_output_path),
                pass2a_output_path=str(pass2a_output_path),
                debug_video_path=str(pass2_debug_path),
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                pass2c_ghosts_path=str(ghosts_path) if ghosts_path.exists() else None,
            )
            logger.info(f"Pass 2 debug video written: {pass2_debug_path}")
            logger.info("")

        # TODO: Run Pass 3
        if 3 in passes_to_run:
            logger.info("Starting Pass 3A: Identity Candidate Generation")
            logger.info("-" * 80)

            pass2c_output_path = output_dir / "pass2_ghosts.json"
            if not pass2c_output_path.exists():
                logger.error(f"Pass 2C output not found: {pass2c_output_path}")
                logger.error("Please run Pass 2 first: python -m src.main --input <video> --pass 2")
                return 1

            pass3a_output = run_pass3a(input_dir=output_dir, output_dir=output_dir)
            logger.info("")
            logger.info("[OK] Pass 3A Complete!")
            logger.info(f"   Candidates generated: {len(pass3a_output.candidates)}")
            logger.info(f"   Output: {output_dir / 'pass3_candidates.json'}")
            logger.info(f"   Validation: {output_dir / 'pass3_validation.json'}")
            logger.info("")

            logger.info("Starting Pass 3B: Constraint Graph Construction")
            logger.info("-" * 80)
            pass3b_output = run_pass3b(input_dir=output_dir, output_dir=output_dir)
            logger.info("")
            logger.info("[OK] Pass 3B Complete!")
            logger.info(f"   Constraints generated: {len(pass3b_output.constraints)}")
            logger.info(f"   Output: {output_dir / 'pass3_constraints.json'}")
            logger.info(f"   Validation: {output_dir / 'pass3_validation.json'}")
            logger.info("")

            logger.info("Starting Pass 3C: Identity Commit")
            logger.info("-" * 80)
            pass3c_output = run_pass3c(
                fragments_path=str(output_dir / "pass2_ghosts.json"),
                constraints_path=str(output_dir / "pass3_constraints.json"),
                output_path=str(output_dir / "pass3_identity_commit.json"),
            )
            logger.info("")
            logger.info("[OK] Pass 3C Complete!")
            logger.info(f"   Committed identities: {len(pass3c_output.identities)}")
            logger.info(f"   Output: {output_dir / 'pass3_identity_commit.json'}")
            logger.info(f"   Debug metrics: {output_dir / 'debug_metrics.json'}")
            if "3" in args.video_output:
                pass3_debug_path = output_dir / "pass3_debug.mp4"
                logger.info("   Rendering Pass 3 debug video (committed identity overlays)...")
                render_pass3_debug_video_from_artifact(
                    video_path=str(video_path),
                    pass1_output_path=str(output_dir / "pass1_raw.json"),
                    pass2c_output_path=str(output_dir / "pass2_ghosts.json"),
                    pass3_output_path=str(output_dir / "pass3_identity_commit.json"),
                    debug_video_path=str(pass3_debug_path),
                    start_frame=args.start_frame,
                    end_frame=args.end_frame,
                )
                logger.info(f"   Debug video: {pass3_debug_path}")
            logger.info("")

        if "3" in args.video_output and 3 not in passes_to_run:
            pass1_output_path = output_dir / "pass1_raw.json"
            pass2c_output_path = output_dir / "pass2_ghosts.json"
            pass3_output_path = output_dir / "pass3_identity_commit.json"

            if not pass1_output_path.exists():
                logger.error(f"Cannot render Pass 3 debug video: missing {pass1_output_path}")
                logger.error("Run Pass 1 first or run without --pass to generate artifacts")
                return 1
            if not pass2c_output_path.exists():
                logger.error(f"Cannot render Pass 3 debug video: missing {pass2c_output_path}")
                logger.error("Run Pass 2 first or run without --pass to generate artifacts")
                return 1
            if not pass3_output_path.exists():
                logger.error(f"Cannot render Pass 3 debug video: missing {pass3_output_path}")
                logger.error("Run Pass 3 first or run without --pass to generate artifacts")
                return 1

            pass3_debug_path = output_dir / "pass3_debug.mp4"
            logger.info("Generating Pass 3 debug video from existing artifacts (committed identity)")
            render_pass3_debug_video_from_artifact(
                video_path=str(video_path),
                pass1_output_path=str(pass1_output_path),
                pass2c_output_path=str(pass2c_output_path),
                pass3_output_path=str(pass3_output_path),
                debug_video_path=str(pass3_debug_path),
                start_frame=args.start_frame,
                end_frame=args.end_frame,
            )
            logger.info(f"Pass 3 debug video written: {pass3_debug_path}")
            logger.info("")

        if 4 in passes_to_run:
            logger.info("Starting Pass 4: Bird's-Eye Pitch Projection")
            logger.info("-" * 80)

            pass1_output_path = output_dir / "pass1_raw.json"
            pass2c_output_path = output_dir / "pass2_ghosts.json"
            pass3_output_path = output_dir / "pass3_identity_commit.json"

            if not pass1_output_path.exists():
                logger.error(f"Pass 1 output not found: {pass1_output_path}")
                logger.error("Please run Pass 1 first: python -m src.main --input <video> --pass 1")
                return 1
            if not pass2c_output_path.exists():
                logger.error(f"Pass 2C output not found: {pass2c_output_path}")
                logger.error("Please run Pass 2 first: python -m src.main --input <video> --pass 2")
                return 1
            if not pass3_output_path.exists():
                logger.error(f"Pass 3 output not found: {pass3_output_path}")
                logger.error("Please run Pass 3 first: python -m src.main --input <video> --pass 3")
                return 1

            ball_output_path = output_dir / "ball_interpolation.json"
            if not ball_output_path.exists():
                logger.info("Ball interpolation artifact missing; generating it before Pass 4")
                run_ball_interpolation(input_dir=output_dir, output_dir=output_dir)

            birdseye_output = run_birds_eye_pitch(
                input_dir=output_dir,
                output_dir=output_dir,
                video_path=str(video_path),
                debug_video_path=(
                    str(output_dir / "birdseye_debug.mp4")
                    if "4" in args.video_output
                    else None
                ),
            )

            logger.info("")
            logger.info("[OK] Pass 4 Complete!")
            logger.info(f"   Projected frames: {len(birdseye_output.frames)}")
            logger.info(f"   Output: {output_dir / 'birdseye_projection.json'}")
            logger.info(f"   Validation: {output_dir / 'birdseye_validation.json'}")
            if "4" in args.video_output:
                logger.info(f"   Debug video: {output_dir / 'birdseye_debug.mp4'}")
            logger.info("")

        run_possession_stage = args.pass_number is None
        if run_possession_stage:
            logger.info("Starting Analytics: Possession")
            logger.info("-" * 80)

            pass3_output_path = output_dir / "pass3_identity_commit.json"
            ball_output_path = output_dir / "ball_interpolation.json"
            birdseye_output_path = output_dir / "birdseye_projection.json"

            if not pass3_output_path.exists():
                logger.error(f"Pass 3 output not found: {pass3_output_path}")
                logger.error("Please run Pass 3 first: python -m src.main --input <video> --pass 3")
                return 1
            if not ball_output_path.exists():
                logger.error(f"Ball interpolation output not found: {ball_output_path}")
                logger.error("Please run Pass 4 first or run without --pass to generate analytics possession")
                return 1
            if not birdseye_output_path.exists():
                logger.error(f"Bird's-eye output not found: {birdseye_output_path}")
                logger.error("Please run Pass 4 first: python -m src.main --input <video> --pass 4")
                return 1

            possession_output = run_analytics_possession(input_dir=output_dir, output_dir=output_dir)
            logger.info("")
            logger.info("[OK] Analytics Possession Complete!")
            logger.info(f"   Frames: {len(possession_output.frames)}")
            logger.info(f"   Confirmed possession frames: {int(possession_output.diagnostics.get('confirmed_possession_frames', 0.0))}")
            logger.info(f"   Output: {output_dir / 'analytics_possession.json'}")
            logger.info("")

            logger.info("Starting Analytics: Pass Detection")
            logger.info("-" * 80)
            events_output = run_analytics_pass_detection(input_dir=output_dir, output_dir=output_dir)
            logger.info("")
            logger.info("[OK] Analytics Pass Detection Complete!")
            logger.info(f"   Events: {len(events_output.events)}")
            logger.info(f"   Successful passes: {int(events_output.diagnostics.get('successful_pass_events', 0.0))}")
            logger.info(f"   Output: {output_dir / 'analytics_events.json'}")
            logger.info("")

            logger.info("Starting Analytics: Distance Summary")
            logger.info("-" * 80)
            distance_output = run_analytics_distance(input_dir=output_dir, output_dir=output_dir)
            logger.info("")
            logger.info("[OK] Analytics Distance Summary Complete!")
            logger.info(f"   Reportable players: {len(distance_output.players)}")
            logger.info(f"   Output: {output_dir / 'player_distance_summary.json'}")
            logger.info("")

            logger.info("Starting Analytics: Summary")
            logger.info("-" * 80)
            summary_output = run_analytics_summary(input_dir=output_dir, output_dir=output_dir)
            logger.info("")
            logger.info("[OK] Analytics Summary Complete!")
            logger.info(f"   Reportable players: {len(summary_output.players)}")
            logger.info(f"   Output: {output_dir / 'analytics_summary.json'}")
            logger.info("")

        if "4" in args.video_output and 4 not in passes_to_run:
            birdseye_output_path = output_dir / "birdseye_projection.json"

            if not birdseye_output_path.exists():
                logger.error(f"Cannot render Pass 4 debug video: missing {birdseye_output_path}")
                logger.error("Run Pass 4 first or run without --pass to generate birdseye_projection.json")
                return 1

            birdseye_debug_path = output_dir / "birdseye_debug.mp4"
            logger.info("Generating Pass 4 debug video from existing birdseye_projection.json")
            render_birds_eye_debug_video_from_artifact(
                video_path=str(video_path),
                birdseye_output_path=str(birdseye_output_path),
                debug_video_path=str(birdseye_debug_path),
                start_frame=args.start_frame,
                end_frame=args.end_frame,
            )
            logger.info(f"Pass 4 debug video written: {birdseye_debug_path}")
            logger.info("")

        if "poss" in args.video_output:
            birdseye_output_path = output_dir / "birdseye_projection.json"
            if not birdseye_output_path.exists():
                logger.error(f"Cannot render analytics possession video: missing {birdseye_output_path}")
                logger.error("Run Pass 4 first or include Pass 4 in the current run")
                return 1

            possession_visualization_path = run_analytics_possession_visualization(
                input_dir=output_dir,
                output_dir=output_dir,
                video_path=str(video_path),
            )
            logger.info(f"Analytics debug video written: {possession_visualization_path}")
            logger.info("")

        if "events" in args.video_output or "analytics" in args.video_output:
            birdseye_output_path = output_dir / "birdseye_projection.json"
            if not birdseye_output_path.exists():
                logger.error(f"Cannot render analytics events video: missing {birdseye_output_path}")
                logger.error("Run Pass 4 first or include Pass 4 in the current run")
                return 1

            events_visualization_path = run_analytics_events_visualization(
                input_dir=output_dir,
                output_dir=output_dir,
                video_path=str(video_path),
            )
            logger.info(f"Analytics debug video written: {events_visualization_path}")
            logger.info("")

        if "viz" in args.video_output:
            birdseye_output_path = output_dir / "birdseye_projection.json"
            if not birdseye_output_path.exists():
                logger.error(f"Cannot render visualization video: missing {birdseye_output_path}")
                logger.error("Run Pass 4 first or include Pass 4 in the current run")
                return 1

            visualization_path = run_visualization(
                input_dir=output_dir,
                output_dir=output_dir,
                video_path=str(video_path),
            )
            logger.info(f"Visualization video written: {visualization_path}")
            logger.info("")

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
