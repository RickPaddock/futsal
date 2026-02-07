"""
Pass 0: Run Folder Creation

Creates timestamped run folder with subdirectories for each pass output.
No ML logic here - just folder setup.

Output structure:
    output/run_DDMMYY_HHMMSS/
        pass1_raw/          # Track-centric raw evidence JSON
        pass2_identity/     # Track fragments JSON
        pass3_final/        # Final identities JSON
        pass3_final/        # Final identities JSON + visualizations
        logs/               # Execution logs
"""

from pathlib import Path
from datetime import datetime


def create_run_folder(base_output_dir: Path | str = "output") -> Path:
    """
    Create a timestamped run folder with subdirectories.

    Args:
        base_output_dir: Base output directory (default: "output")

    Returns:
        Path to created run folder (e.g., output/run_060226_124530)
    """
    base_output_dir = Path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped folder name
    timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
    run_dir = base_output_dir / f"run_{timestamp}"

    # Create subdirectories
    (run_dir / "pass1_raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "pass2_identity").mkdir(parents=True, exist_ok=True)
    (run_dir / "pass3_final").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    print(f"Created run folder: {run_dir}")
    return run_dir
