"""
3-Pass Futsal Video Analysis Pipeline - CLI Interface

Usage:
    python -m src.cli run-full --input-dir videos/input --output-dir output
    python -m src.cli pass1 --input-dir videos/input --run-dir output/run_...
    python -m src.cli pass2 --run-dir output/run_...
    python -m src.cli pass3 --run-dir output/run_...
"""

import click
from pathlib import Path
from datetime import datetime
import yaml
import sys


@click.group()
def cli():
    """Futsal Video Analysis Pipeline - 3-Pass Architecture"""
    pass


@cli.command()
@click.option('--input-dir', type=Path, required=True, help='Folder with pre-split video clips')
@click.option('--output-dir', type=Path, default='videos/output', help='Base output directory')
@click.option('--config', type=Path, default='config/default.yaml', help='Pipeline config file')
def run_full(input_dir, output_dir, config):
    """Run the complete 3-pass pipeline (Pass 0 → Pass 1 → Pass 2 → Pass 3)"""
    from src.passes.pass0_setup import create_run_folder
    from src.passes.pass1_collect import run_pass1
    from src.passes.pass2_geometry import run_pass2
    from src.passes.pass3_identity import run_pass3

    click.echo("=" * 60)
    click.echo("Futsal Video Analysis - Full Pipeline")
    click.echo("=" * 60)

    # Load config
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Pass 0: Setup
    click.echo("\n[PASS 0] Creating run folder...")
    run_dir = create_run_folder(output_dir)
    click.echo(f"Run folder: {run_dir}")

    # Pass 1: Raw Evidence Collection
    click.echo("\n[PASS 1] Raw evidence collection (NO identity assignment)...")
    run_pass1(input_dir, run_dir, cfg)

    # Pass 2: Geometry + Aggregation
    click.echo("\n[PASS 2] Geometry + track fragment aggregation...")
    run_pass2(run_dir, cfg)

    # Pass 3: Identity Inference
    click.echo("\n[PASS 3] Identity inference (team + jersey assignment)...")
    run_pass3(run_dir, cfg)

    click.echo("\n" + "=" * 60)
    click.echo("Pipeline complete!")
    click.echo(f"Results saved to: {run_dir}")
    click.echo("=" * 60)


@cli.command()
@click.option('--input-dir', type=Path, default=None, help='Folder with pre-split video clips')
@click.option('--clip', type=Path, default=None, help='Single video clip to process')
@click.option('--output-dir', type=Path, default='videos/output', help='Base output directory (auto-creates timestamped run folder)')
@click.option('--config', type=Path, default='config/default.yaml', help='Pipeline config file')
@click.option('--batch-size', type=int, default=None, help='Batch size for GPU inference (overrides config)')
def pass1(input_dir, clip, output_dir, config, batch_size):
    """Pass 1: Raw Evidence Collection (NO identity assignment)

    Automatically creates timestamped run folder in output directory.

    Process either a directory of clips (--input-dir) or a single clip (--clip).

    Hard Constraints:
    - NO team assignment
    - NO jersey assignment
    - NO identity locking
    - Tracking uses ONLY IoU + motion
    """
    from src.passes.pass0_setup import create_run_folder
    from src.passes.pass1_collect import run_pass1

    # Validate: exactly one of input_dir or clip must be provided
    if input_dir is None and clip is None:
        click.echo("Error: Must specify either --input-dir or --clip")
        sys.exit(1)
    if input_dir is not None and clip is not None:
        click.echo("Error: Cannot specify both --input-dir and --clip")
        sys.exit(1)

    click.echo("=" * 60)
    click.echo("[PASS 1] Raw Evidence Collection (GPU ONLY)")
    click.echo("=" * 60)

    # Auto-create run folder
    run_dir = create_run_folder(output_dir)
    click.echo(f"Run folder: {run_dir}\n")

    # Load config
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Override config if specified
    if batch_size is not None:
        cfg['detection']['batch_size'] = batch_size

    # Process single clip or directory
    if clip is not None:
        clip = Path(clip)
        if not clip.exists():
            click.echo(f"Error: Clip not found: {clip}")
            sys.exit(1)
        # Pass single clip as a list
        run_pass1(input_dir=None, run_dir=run_dir, config=cfg, video_files=[clip])
    else:
        run_pass1(input_dir=input_dir, run_dir=run_dir, config=cfg)

    click.echo(f"\nPass 1 complete! Output: {run_dir / 'pass1_raw'}")


@cli.command()
@click.option('--run-dir', type=Path, required=True, help='Run folder (output/run_DDMMYY_HHMMSS)')
@click.option('--config', type=Path, default='config/default.yaml', help='Pipeline config file')
def pass2(run_dir, config):
    """Pass 2: Geometry + Aggregation (build track fragments)

    Key Constraints:
    - Fragment breaks at track ID discontinuity ONLY
    - NO fragment merging under any condition
    - Pass 2 must remain mechanical, not interpretive
    """
    from src.passes.pass2_geometry import run_pass2

    click.echo("=" * 60)
    click.echo("[PASS 2] Geometry + Track Fragment Aggregation")
    click.echo("=" * 60)

    # Load config
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Ensure run_dir exists
    run_dir = Path(run_dir)
    if not run_dir.exists():
        click.echo(f"Error: Run directory {run_dir} does not exist.")
        sys.exit(1)

    pass1_dir = run_dir / 'pass1_raw'
    if not pass1_dir.exists() or not list(pass1_dir.glob('*.json')):
        click.echo(f"Error: Pass 1 output not found in {pass1_dir}")
        sys.exit(1)

    run_pass2(run_dir, cfg)

    click.echo(f"\nPass 2 complete! Output: {run_dir / 'pass2_identity'}")


@cli.command()
@click.option('--run-dir', type=Path, required=True, help='Run folder (output/run_DDMMYY_HHMMSS)')
@click.option('--config', type=Path, default='config/default.yaml', help='Pipeline config file')
@click.option('--min-confidence', type=float, default=None, help='Min confidence for jersey lock (overrides config)')
def pass3(run_dir, config, min_confidence):
    """Pass 3: Identity Inference (team + jersey assignment, divergence detection)

    Key Operations:
    - K-Means (k=2) run ONCE at start for team assignment
    - Pileup detection with unreliable_window marking
    - Jersey inference (bibbed team only)
    - Divergence detection (narrow criteria - ALL must be true)
    - Identity repair (relabeling only, NEVER re-run tracking)
    """
    from src.passes.pass3_identity import run_pass3

    click.echo("=" * 60)
    click.echo("[PASS 3] Identity Inference")
    click.echo("=" * 60)

    # Load config
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Override config if specified
    if min_confidence is not None:
        cfg['jersey']['lock_threshold'] = min_confidence

    # Ensure run_dir exists
    run_dir = Path(run_dir)
    if not run_dir.exists():
        click.echo(f"Error: Run directory {run_dir} does not exist.")
        sys.exit(1)

    pass2_dir = run_dir / 'pass2_identity'
    if not pass2_dir.exists() or not list(pass2_dir.glob('*.json')):
        click.echo(f"Error: Pass 2 output not found in {pass2_dir}")
        sys.exit(1)

    run_pass3(run_dir, cfg)

    click.echo(f"\nPass 3 complete! Output: {run_dir / 'pass3_final'}")


@cli.command()
@click.option('--run-dir', type=Path, required=True, help='Run folder (output/run_DDMMYY_HHMMSS)')
@click.option('--clip', type=str, default=None, help='Specific clip to visualize (default: all clips)')
@click.option('--input-dir', type=Path, default='videos/input', help='Directory with original video files')
@click.option('--output-scale', type=float, default=1.0, help='Output video scale (1.0 = full size)')
@click.option('--config', type=Path, default='config/default.yaml', help='Pipeline config file')
@click.option('--2d', 'mode_2d', type=str, default=None, help='2D pitch views: B=birdseye, V=voronoi (e.g., "B", "V", "BV")')
def visualize(run_dir, clip, input_dir, output_scale, config, mode_2d):
    """Generate annotated video from Pass 3 output

    Automatically detects clips from Pass 1 JSON files.

    Shows:
    - Bounding boxes with team colors (team_a, team_b)
    - Jersey numbers (locked identities only)
    - Divergence markers (from Pass 2)
    - Track IDs and fragment IDs

    2D Pitch Options (--2d):
    - B: Birdseye view (2D pitch with player positions)
    - V: Voronoi overlay (spatial dominance regions)
    - BV or VB: Both birdseye and voronoi views
    - (none): Video only (default)
    """
    from src.passes.pass_visualize import visualize_run

    click.echo("=" * 60)
    click.echo(f"[VISUALIZE] Pass 3 output")
    click.echo("=" * 60)

    # Load config
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Ensure run_dir exists
    run_dir = Path(run_dir)
    if not run_dir.exists():
        click.echo(f"Error: Run directory {run_dir} does not exist.")
        sys.exit(1)

    # Check Pass 1 directory exists
    pass1_dir = run_dir / 'pass1_raw'
    if not pass1_dir.exists():
        click.echo(f"Error: Pass 1 output not found in {pass1_dir}")
        sys.exit(1)

    # Parse 2D mode flags
    render_birdseye = False
    render_voronoi = False
    if mode_2d:
        mode_2d_upper = mode_2d.upper()
        render_birdseye = 'B' in mode_2d_upper
        render_voronoi = 'V' in mode_2d_upper

        if render_birdseye or render_voronoi:
            click.echo(f"\n2D Pitch Rendering:")
            if render_birdseye:
                click.echo("  ✓ Birdseye view (player positions)")
            if render_voronoi:
                click.echo("  ✓ Voronoi overlay (spatial dominance)")

    visualize_run(
        run_dir=run_dir,
        clip_filter=clip,
        input_dir=input_dir,
        output_scale=output_scale,
        config=cfg,
        render_2d_birdseye=render_birdseye,
        render_2d_voronoi=render_voronoi,
    )

    click.echo(f"\nVisualization complete! Output: {run_dir / 'pass3_final'}")


if __name__ == '__main__':
    cli()
