"""
Test Pass 2B: Fragment Quality Scoring

Quick test to verify Pass 2B implementation works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from skills.pass2b_fragment_scoring import run_pass2b
from core.constants import VIDEOS_OUTPUT_DIR

def main():
    """Test Pass 2B on a sample output directory."""

    # Use clip8 as test (should have pass1 and pass2a outputs from previous tests)
    clip_name = "GoPro_Futsal_part1_CLEANED_clip8"
    output_dir = VIDEOS_OUTPUT_DIR / clip_name

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        print("Please run test_pass1.py and test_pass2a.py first")
        return 1

    # Check if required input files exist
    pass1_json = output_dir / "pass1_raw.json"
    pass2a_json = output_dir / "pass2_fragments.json"

    if not pass1_json.exists():
        print(f"Error: {pass1_json} not found")
        print("Please run test_pass1.py first")
        return 1

    if not pass2a_json.exists():
        print(f"Error: {pass2a_json} not found")
        print("Please run test_pass2a.py first")
        return 1

    print(f"Testing Pass 2B on {clip_name}...")
    print(f"Input: {pass1_json}")
    print(f"Input: {pass2a_json}")
    print(f"Output dir: {output_dir}")
    print()

    try:
        # Run Pass 2B
        output_path = run_pass2b(
            pass1_json_path=pass1_json,
            pass2a_json_path=pass2a_json,
            output_dir=output_dir,
        )

        print()
        print("=" * 80)
        print(f"✓ PASS 2B COMPLETE")
        print(f"Output: {output_path}")
        print("=" * 80)

        return 0

    except Exception as e:
        print()
        print("=" * 80)
        print(f"✗ PASS 2B FAILED")
        print(f"Error: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
