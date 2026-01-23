"""
Train YOLO model on custom futsal player dataset.

Usage:
    python utils/train_yolo.py

Requirements:
    pip install ultralytics

Before running:
    1. Export dataset from Roboflow (YOLOv8 format)
    2. Extract the zip to a folder
    3. Update DATASET_PATH below to point to the data.yaml file
"""

from ultralytics import YOLO
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to data.yaml from Roboflow export
DATASET_PATH = "models/datasets/GoPro_v1.v2i.yolov11/data.yaml"

# Base model to fine-tune (will download if not present)
BASE_MODEL = "models/yolo11m.pt"  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt

# Training parameters
EPOCHS = 100
IMAGE_SIZE = 640  # Match your training data resolution
BATCH_SIZE = 16   # Reduce if you run out of GPU memory
DEVICE = "0"      # GPU device (0 for first GPU, "cpu" for CPU)

# Output
PROJECT = "runs/train"
NAME = "futsal_player_detector_GoPro_v1__v2"


# ============================================================================
# TRAINING
# ============================================================================

def train():
    """Train YOLO model on futsal dataset."""

    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("\nSteps to fix:")
        print("1. Export dataset from Roboflow (YOLOv8 format)")
        print("2. Extract the zip file")
        print("3. Update DATASET_PATH in this script")
        return

    print(f"Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    print(f"Starting training...")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Device: {DEVICE}")

    results = model.train(
        data=str(dataset_path),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        # Augmentation (good defaults for sports)
        hsv_h=0.015,      # Hue augmentation
        hsv_s=0.7,        # Saturation augmentation
        hsv_v=0.4,        # Value augmentation
        degrees=0.0,      # No rotation (players should be upright)
        translate=0.1,    # Translation
        scale=0.5,        # Scale augmentation
        fliplr=0.5,       # Horizontal flip
        mosaic=1.0,       # Mosaic augmentation
    )

    # Copy best weights to models folder
    best_weights = Path(PROJECT) / NAME / "weights" / "best.pt"
    if best_weights.exists():
        output_path = Path("models") / "futsal_player_detector.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import shutil
        shutil.copy(best_weights, output_path)

        print(f"\nTraining complete!")
        print(f"Best weights saved to: {output_path}")
        print(f"\nTo use in your pipeline, update config/default.yaml:")
        print(f'  model: "models/futsal_player_detector.pt"')
        print(f'  use_roboflow: false')
    else:
        print(f"\nTraining complete! Check {PROJECT}/{NAME} for results.")


if __name__ == "__main__":
    train()
