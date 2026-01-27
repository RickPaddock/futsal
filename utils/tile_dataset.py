"""
Tile a YOLO format dataset into 2x2 grid for small object detection.

This script takes a downloaded Roboflow dataset (YOLO format) and:
1. Splits each image into 2x2 tiles
2. Transforms bounding box annotations to match each tile
3. Preserves train/valid/test split structure
4. Outputs a new dataset ready to upload to Roboflow

Usage:
    python utils/tile_dataset.py --input path/to/dataset --output path/to/tiled_dataset

The input dataset should have this structure:
    dataset/
        train/
            images/
            labels/
        valid/
            images/
            labels/
        test/
            images/
            labels/
        data.yaml

Requirements:
    pip install opencv-python numpy pyyaml
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import shutil
import yaml


def parse_yolo_annotation(line: str) -> tuple:
    """Parse a YOLO line supporting both bbox (5 vals) and polygon (>=8 vals).

    Roboflow exports for segmentation use: class x1 y1 x2 y2 ... xn yn (all normalized).
    For tiling we convert polygons to their bounding box; detection training only needs bbox.
    """
    parts = line.strip().split()
    if not parts:
        raise ValueError("Empty annotation line")

    class_id = int(parts[0])
    values = list(map(float, parts[1:]))

    # Standard YOLO bbox: cls cx cy w h
    if len(values) == 4:
        x_center, y_center, width, height = values
        return class_id, x_center, y_center, width, height

    # Polygon format: cls x1 y1 x2 y2 ... xn yn
    if len(values) % 2 != 0:
        raise ValueError(f"Invalid polygon coordinates: {line}")

    xs = values[0::2]
    ys = values[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    return class_id, x_center, y_center, width, height


def bbox_intersects_tile(x_center, y_center, width, height, tile_x, tile_y, tile_w, tile_h):
    """
    Check if a bounding box intersects with a tile region.
    
    Args:
        x_center, y_center, width, height: YOLO bbox (normalized 0-1)
        tile_x, tile_y: Top-left corner of tile (normalized 0-1)
        tile_w, tile_h: Tile dimensions (normalized 0-1)
    
    Returns:
        bool: True if bbox intersects with tile
    """
    # Convert YOLO center format to corners
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    # Tile corners
    tile_x2 = tile_x + tile_w
    tile_y2 = tile_y + tile_h
    
    # Check intersection
    return not (x2 < tile_x or x1 > tile_x2 or y2 < tile_y or y1 > tile_y2)


def transform_bbox_to_tile(x_center, y_center, width, height, tile_x, tile_y, tile_w, tile_h):
    """
    Transform a bounding box from full image coordinates to tile coordinates.
    Clips the bbox to tile boundaries.
    
    Returns:
        Tuple of (x_center, y_center, width, height) in tile coordinates (0-1)
        or None if bbox doesn't intersect tile
    """
    # Convert YOLO center format to corners
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    # Tile corners
    tile_x2 = tile_x + tile_w
    tile_y2 = tile_y + tile_h
    
    # Check intersection
    if not bbox_intersects_tile(x_center, y_center, width, height, tile_x, tile_y, tile_w, tile_h):
        return None
    
    # Clip bbox to tile boundaries
    clipped_x1 = max(x1, tile_x)
    clipped_y1 = max(y1, tile_y)
    clipped_x2 = min(x2, tile_x2)
    clipped_y2 = min(y2, tile_y2)
    
    # Transform to tile coordinate system (0-1 within tile)
    new_x1 = (clipped_x1 - tile_x) / tile_w
    new_y1 = (clipped_y1 - tile_y) / tile_h
    new_x2 = (clipped_x2 - tile_x) / tile_w
    new_y2 = (clipped_y2 - tile_y) / tile_h
    
    # Convert back to YOLO center format
    new_x_center = (new_x1 + new_x2) / 2
    new_y_center = (new_y1 + new_y2) / 2
    new_width = new_x2 - new_x1
    new_height = new_y2 - new_y1
    
    # Ensure valid bbox (not too small)
    if new_width < 0.01 or new_height < 0.01:
        return None
    
    return new_x_center, new_y_center, new_width, new_height


def tile_image_2x2(image: np.ndarray, overlap: int = 100):
    """
    Split image into 2x2 grid of overlapping tiles.

    IMPORTANT: This MUST match supervision.InferenceSlicer behavior exactly!
    InferenceSlicer uses stride-based positioning:
        stride = tile_size - overlap
        positions = [0, stride, 2*stride, ...]

    For 4K (3840x2160) with 100px overlap:
        tile_size = (2020, 1180)
        stride = (1920, 1080)
        Grid: [(0,0), (1920,0), (0,1080), (1920,1080)]

    Args:
        image: Input image (numpy array)
        overlap: Pixel overlap between tiles (default 100, matching InferenceSlicer)

    Returns:
        List of 4 tiles (top-left, top-right, bottom-left, bottom-right)
        and their normalized coordinates (x, y, w, h) for annotation transformation
    """
    h, w = image.shape[:2]

    # Tile dimensions with overlap (matches InferenceSlicer: width//2 + overlap)
    tile_w = w // 2 + overlap
    tile_h = h // 2 + overlap

    # CRITICAL: Use stride-based positioning to match InferenceSlicer exactly
    # InferenceSlicer calculates: stride = slice_size - overlap
    stride_x = tile_w - overlap  # = w // 2
    stride_y = tile_h - overlap  # = h // 2

    # Grid positions (matching InferenceSlicer)
    # Top-left:     (0, 0)
    # Top-right:    (stride_x, 0)
    # Bottom-left:  (0, stride_y)
    # Bottom-right: (stride_x, stride_y)

    tiles = []
    positions = [
        (0, 0),                      # top-left
        (stride_x, 0),               # top-right
        (0, stride_y),               # bottom-left
        (stride_x, stride_y),        # bottom-right
    ]

    for start_x, start_y in positions:
        # Clip to image boundaries (InferenceSlicer does this too)
        end_x = min(start_x + tile_w, w)
        end_y = min(start_y + tile_h, h)

        # Extract tile
        tile_img = image[start_y:end_y, start_x:end_x]

        # Actual tile dimensions (may be smaller at edges)
        actual_tile_w = end_x - start_x
        actual_tile_h = end_y - start_y

        # Normalized coordinates for annotation transformation
        norm_x = start_x / w
        norm_y = start_y / h
        norm_w = actual_tile_w / w
        norm_h = actual_tile_h / h

        tiles.append((tile_img, norm_x, norm_y, norm_w, norm_h))

    return tiles


def process_dataset_split(input_split_dir: Path, output_split_dir: Path, split_name: str, overlap: int = 100, keep_empty_ratio: float = 0.0):
    """Process a single split (train/valid/test) of the dataset.

    Args:
        keep_empty_ratio: Fraction of empty tiles to keep (0.0 = none, 1.0 = all).
                         Recommended: 0.0-0.25 for ball detection to avoid class imbalance.
    """

    images_dir = input_split_dir / "images"
    labels_dir = input_split_dir / "labels"

    output_images_dir = output_split_dir / "images"
    output_labels_dir = output_split_dir / "labels"

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        print(f"  Skipping {split_name} - no images directory found")
        return

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    print(f"\n  Processing {split_name}: {len(image_files)} images (overlap={overlap}px)")

    processed_count = 0

    for img_path in image_files:
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"    Warning: Could not read {img_path.name}")
            continue

        # Read annotations (if they exist)
        label_path = labels_dir / f"{img_path.stem}.txt"
        annotations = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        annotations.append(parse_yolo_annotation(line))

        # Create 2x2 tiles with overlap
        tiles = tile_image_2x2(image, overlap=overlap)

        # Process each tile
        for tile_idx, (tile_img, tile_x, tile_y, tile_w, tile_h) in enumerate(tiles):
            # Generate tile filename
            tile_name = f"{img_path.stem}_tile{tile_idx}"
            tile_img_path = output_images_dir / f"{tile_name}.jpg"
            tile_label_path = output_labels_dir / f"{tile_name}.txt"

            # Transform annotations for this tile
            tile_annotations = []
            for class_id, x_center, y_center, width, height in annotations:
                transformed = transform_bbox_to_tile(
                    x_center, y_center, width, height,
                    tile_x, tile_y, tile_w, tile_h
                )
                if transformed is not None:
                    tile_annotations.append((class_id, *transformed))

            # Save tile annotations (skip empty tiles unless keep_empty_ratio allows)
            if tile_annotations:
                # Tile has ball - always save
                cv2.imwrite(str(tile_img_path), tile_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                with open(tile_label_path, 'w') as f:
                    for class_id, x_c, y_c, w, h in tile_annotations:
                        f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            elif keep_empty_ratio > 0 and np.random.random() < keep_empty_ratio:
                # Empty tile - save only if randomly selected based on ratio
                cv2.imwrite(str(tile_img_path), tile_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                # No label file = negative example

        processed_count += 1
        if processed_count % 100 == 0:
            print(f"    Processed {processed_count}/{len(image_files)} images")

    print(f"  Completed {split_name}: {processed_count} images -> {processed_count * 4} tiles")


def tile_dataset(input_dir: Path, output_dir: Path, overlap: int = 100, keep_empty_ratio: float = 0.0):
    """
    Tile an entire YOLO format dataset.

    Args:
        input_dir: Path to input dataset (with train/valid/test folders)
        output_dir: Path to output tiled dataset
        overlap: Pixel overlap between tiles (default 100, matching InferenceSlicer)
        keep_empty_ratio: Fraction of empty tiles to keep (0.0 = ball tiles only)
    """
    print(f"Tiling dataset: {input_dir}")
    print(f"Output to: {output_dir}")
    print(f"Tile overlap: {overlap}px (matches InferenceSlicer)")
    print(f"Keep empty tiles: {keep_empty_ratio*100:.0f}% (0% = only tiles with ball)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split_name in ["train", "valid", "test"]:
        input_split_dir = input_dir / split_name
        output_split_dir = output_dir / split_name

        if input_split_dir.exists():
            process_dataset_split(input_split_dir, output_split_dir, split_name, overlap=overlap, keep_empty_ratio=keep_empty_ratio)

    # Copy data.yaml if it exists
    data_yaml_path = input_dir / "data.yaml"
    if data_yaml_path.exists():
        print("\nCopying data.yaml...")
        output_yaml_path = output_dir / "data.yaml"

        # Read and potentially update paths in data.yaml
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Update paths if they're relative
        if 'train' in data and not Path(data['train']).is_absolute():
            data['train'] = '../train/images'
        if 'val' in data and not Path(data['val']).is_absolute():
            data['val'] = '../valid/images'
        if 'test' in data and not Path(data['test']).is_absolute():
            data['test'] = '../test/images'

        with open(output_yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    print("\nTiling complete!")
    print(f"\nNext steps:")
    print(f"  1. Verify the tiled dataset in: {output_dir}")
    print(f"  2. Train with: python utils/train_yolo.py --task ball")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tile a YOLO dataset into 2x2 grid for small object detection"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input YOLO dataset directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output tiled dataset directory"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Pixel overlap between tiles (default: 200 = ~10%% overlap, optimal for small objects)"
    )
    parser.add_argument(
        "--keep-empty",
        type=float,
        default=0.0,
        help="Fraction of empty tiles to keep (0.0 = only ball tiles, 0.25 = 25%% of empties). "
             "Recommended: 0.0-0.25 to avoid class imbalance."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input directory not found: {input_path}")
        exit(1)

    tile_dataset(input_path, output_path, overlap=args.overlap, keep_empty_ratio=args.keep_empty)
