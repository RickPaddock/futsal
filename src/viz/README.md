# 2D Pitch Visualization

Standalone visualization layer that renders top-down 2D tactical pitch views from Pass 1-3 JSON outputs.

## Overview

This module provides read-only visualization without modifying any pass logic or JSON schemas. Pitch geometry and homography utilities copied from ARCHIVE to `src/geometry/`.

## Features

### Task 1: Player Positions ✅

- **Player circles**: Team-colored circles for each player
- **Jersey numbers**: Displayed in circle center ONLY when locked by Pass 3
- **Deterministic**: Same JSON inputs → same output (no temporal smoothing)
- **Homography**: Uses ARCHIVE pitch geometry for accurate coordinate mapping

### Task 2: Voronoi Overlay ✅

- **Voronoi regions**: Optional overlay showing spatial dominance
- **Team-colored**: Regions use team color with alpha blending (0.25)
- **Court-space computation**: Voronoi computed in meters, NOT pixels
- **Pitch-clipped**: Regions clipped to pitch boundary

## Usage

### Integrated with CLI (Recommended)

The 2D pitch visualization is integrated into the main visualize command:

```bash
# Video only (default)
python -m src.cli visualize --run-dir output/run_070225_123456

# Video + Birdseye (2D pitch)
python -m src.cli visualize --run-dir output/run_070225_123456 --2d B

# Video + Voronoi overlay
python -m src.cli visualize --run-dir output/run_070225_123456 --2d V

# Video + Both (birdseye and voronoi)
python -m src.cli visualize --run-dir output/run_070225_123456 --2d BV
# OR
python -m src.cli visualize --run-dir output/run_070225_123456 --2d VB
```

### Options

- `--2d B`: Birdseye view (player positions on 2D pitch)
- `--2d V`: Voronoi view (spatial dominance regions)
- `--2d BV` or `--2d VB`: Both views
- No `--2d` flag: Video only

## Data Requirements

### Input Files

The visualization requires three JSON files from the 3-pass pipeline:

1. **Pass 1** (`pass1_raw/<clip>.json`):
   - Player tracks with pixel centroids
   - Bounding boxes per frame

2. **Pass 2** (`pass2_identity/<clip>_fragments.json`):
   - Track fragments after divergence detection
   - Court coordinates (meters) per frame via homography

3. **Pass 3** (`pass3_final/<clip>.json`):
   - Team assignments (team_a, team_b, unknown)
   - Jersey number assignments (ONLY if locked)

### Homography Config

The `config/default.yaml` must include a complete homography section:

```yaml
homography:
  court_length: 40.0      # meters
  court_width: 20.0       # meters
  output_pixel_scale: 20  # pixels per meter

  source_points:          # Pixel coordinates from video
    - [680, 595]          # Far Left Corner
    - [843, 638]          # ...
    # ... (13 calibration points)

  dest_points:            # Real-world coordinates (meters)
    - [0.0, 19.0]         # Far Left Corner
    - [3.5, 17.0]         # ...
    # ... (13 corresponding points)
```

## Design Constraints

### Hard Constraints (Do Not Violate)

- ✅ **Read-only**: Does NOT modify Pass 1/2/3 logic or JSON schemas
- ✅ **ARCHIVE reuse**: Uses existing homography and pitch rendering utilities
- ✅ **Jersey locking**: Numbers shown ONLY if Pass 3 reports them as locked
- ✅ **Deterministic**: No temporal smoothing or prediction
- ✅ **Court-space Voronoi**: Computed in meters, not pixels

### Not Implemented (Future Work)

- ❌ Ball weighting for Voronoi
- ❌ Temporal smoothing or interpolation
- ❌ Heatmaps or convex hulls
- ❌ Analytics overlays

## Architecture

### Class: `PitchRenderer2D`

Main renderer class that:
1. Indexes fragments by frame for fast lookup
2. Maps fragment_id → identity (team + jersey)
3. Renders pitch background using ARCHIVE utilities
4. Draws player circles with team colors
5. Shows jersey numbers when locked
6. Optionally draws Voronoi regions

### Functions

- `load_pass_data()`: Load Pass 1-3 JSON files
- `render_pitch_frame()`: Render single frame (main entry point)
- `export_frame_png()`: Save single frame as PNG
- `export_mp4()`: Export frame range as MP4 video

## Team Colors

```python
TEAM_COLORS = {
    "team_a": (100, 255, 100),  # Green (bibbed team)
    "team_b": (100, 100, 255),  # Red (non-bibbed team)
    "unknown": (200, 200, 200), # Gray
}
```

## Jersey Number Display Logic

```python
# Jersey number is shown ONLY if:
if jersey_number is not None:  # Pass 3 locked this fragment
    # Draw number in circle center
    cv2.putText(...)
```

**Critical**: The visualization NEVER guesses or invents jersey numbers. It only displays what Pass 3 has locked.

## Output

- **PNG**: Single frame (800×400 default)
- **MP4**: Video at configurable FPS (default 30 FPS)
- **Resolution**: Configurable via `pitch_width_px` and `pitch_height_px`


## Dependencies

- `numpy`
- `opencv-python` (cv2)
- `scipy` (for Voronoi, Task 2)
- `tqdm` (for progress bars)
- `pyyaml` (for config loading)

## Next Steps (Not Implemented)

After completing Tasks 1 and 2, potential extensions include:

- Ball position overlay
- Ball-weighted Voronoi regions
- Heatmaps (positional density over time)
- Convex hulls (team shape)
- Pass networks and analytics

These are NOT part of the current scope.
