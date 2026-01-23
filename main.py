#!/usr/bin/env python3
"""
Futsal Player Tracking System - CLI Entry Point

Usage:
    python main.py process <video_path> [options]
"""

# Force ONNX Runtime to use GPU (must be set before importing inference)
import os
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CUDAExecutionProvider"

import click
from pathlib import Path
import yaml


def load_config(config_path: Path | None) -> dict:
    """Load configuration file, optionally merged with a custom config."""
    # Load default config
    default_config_path = Path(__file__).parent / "config" / "default.yaml"
    with open(default_config_path) as f:
        config = yaml.safe_load(f)

    # Load custom config if specified
    if config_path:
        with open(config_path) as f:
            custom_config = yaml.safe_load(f)
            config = deep_merge(config, custom_config)

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Futsal Player Tracking System - Analyze futsal match recordings."""
    pass


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--config", "-c", type=click.Path(exists=True), help="Custom config file")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--start-frame", type=int, default=0, help="Start processing from frame N")
@click.option("--end-frame", type=int, default=None, help="Stop processing at frame N")
@click.option("--debug", is_flag=True, help="Enable debug visualization output")
@click.option("--no-gpu", is_flag=True, help="Disable GPU acceleration")
def process(video_path, config, output, start_frame, end_frame, debug, no_gpu):
    """Process a futsal match video.

    VIDEO_PATH: Path to the input video file (MP4, etc.)
    """
    from src.pipeline import Pipeline

    # Load configuration
    config_dict = load_config(Path(config) if config else None)

    # Override device if no-gpu flag set
    if no_gpu:
        config_dict["processing"]["device"] = "cpu"

    # Set output directory
    if output:
        output_dir = Path(output)
    else:
        output_dir = Path(video_path).parent / f"{Path(video_path).stem}_output"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model info
    player_cfg = config_dict["player_detection"]
    if player_cfg.get("use_roboflow"):
        model_name = f"Roboflow: {player_cfg.get('roboflow_model_id', 'unknown')}"
    else:
        model_name = f"Local: {player_cfg.get('model', 'unknown')}"

    click.echo(f"Processing: {video_path}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Device: {config_dict['processing']['device']}")
    click.echo(f"Model: {model_name}")

    # Initialize and run pipeline
    pipeline = Pipeline(config_dict, output_dir, debug=debug)
    pipeline.run(
        video_path=Path(video_path),
        start_frame=start_frame,
        end_frame=end_frame,
    )

    click.echo("Processing complete!")


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output video path")
@click.option("--start", "-s", type=float, default=0, help="Start time in seconds")
@click.option("--duration", "-d", type=float, default=None, help="Duration in seconds")
@click.option("--scale", type=float, default=0.5, help="Scale factor (0.5 = half resolution)")
def trim(video_path, output, start, duration, scale):
    """Trim and compress a video for testing.

    Creates a smaller video clip for development/testing purposes.
    """
    from src.utils.video_io import trim_video

    if output is None:
        output = Path(video_path).stem + "_trimmed.mp4"

    click.echo(f"Trimming: {video_path}")
    click.echo(f"Output: {output}")
    click.echo(f"Start: {start}s, Duration: {duration}s, Scale: {scale}")

    trim_video(
        input_path=Path(video_path),
        output_path=Path(output),
        start_time=start,
        duration=duration,
        scale=scale,
    )

    click.echo("Trim complete!")


@cli.command()
def info():
    """Show system information and GPU status."""
    import torch

    click.echo("Futsal Player Tracking System v0.1.0")
    click.echo("-" * 40)
    click.echo(f"PyTorch version: {torch.__version__}")
    click.echo(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        click.echo(f"CUDA version: {torch.version.cuda}")
        click.echo(f"GPU: {torch.cuda.get_device_name(0)}")
        click.echo(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


if __name__ == "__main__":
    cli()