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
import argparse


# ============================================================================
# CONFIGURATION (select task at runtime)
# ============================================================================

TASKS = {
    "player": {
        "dataset_path": "models/datasets/GoPro_v1.v2i.yolov11/data.yaml",
        "base_model": "models/yolo11m.pt",  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
        "epochs": 100,
        "image_size": 640,
        "batch_size": 16,
        "device": "0",
        "project": "runs/train",
        "name": "futsal_player_detector_GoPro_v1__v2",
        "output_weights": "models/futsal_player_detector.pt",
    },
    "ball": {
        # Tiled ball dataset WITH OVERLAP (matches InferenceSlicer)
        # Generate with: python utils/tile_dataset.py --input models/datasets/BALL/<ROBOFLOW_EXPORT> --output models/datasets/BALL/GoPro_BALL_v1_tiled_overlap
        "dataset_path": "models/datasets/BALL/GoPro_BALL_v1_tiled_overlap/data.yaml",
        # Use yolo11x for best small object detection (Roboflow blog recommends largest model)
        "base_model": "models/yolo11x.pt",
        "epochs": 50,  # Roboflow blog used 50 epochs; early stopping will apply if converged
        "image_size": 640,
        # Reduce batch size for VRAM headroom (yolo11x is large)
        "batch_size": 8,
        "device": "0",
        "project": "runs/train",
        "name": "futsal_ball_detector_v2",
        "output_weights": "models/BALL_MODEL_best_v2.pt",
    },
}


# ============================================================================
# TRAINING
# ============================================================================

def train(task: str):
    """Train YOLO model on the selected futsal task (player or ball)."""

    if task not in TASKS:
        print(f"Error: Unknown task '{task}'. Choose from: {list(TASKS.keys())}")
        return

    cfg = TASKS[task]

    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("\nSteps to fix:")
        print("1. Export dataset from Roboflow (YOLOv8 format)")
        print("2. Extract the zip file")
        print("3. Update dataset_path for this task in this script")
        return

    print(f"Loading base model: {cfg['base_model']}")
    model = YOLO(cfg["base_model"])

    print(f"Starting training...")
    print(f"  Task: {task}")
    print(f"  Dataset: {cfg['dataset_path']}")
    print(f"  Epochs: {cfg['epochs']}")
    print(f"  Image size: {cfg['image_size']}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  Device: {cfg['device']}")

    results = model.train(
        data=str(dataset_path),
        epochs=cfg["epochs"],
        imgsz=cfg["image_size"],
        batch=cfg["batch_size"],
        device=cfg["device"],
        project=cfg["project"],
        name=cfg["name"],
        exist_ok=True,
        # Augmentation (good defaults for sports)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )

    # Copy best weights to models folder
    best_weights = Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"
    if best_weights.exists():
        output_path = Path(cfg["output_weights"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import shutil
        shutil.copy(best_weights, output_path)

        print(f"\nTraining complete!")
        print(f"Best weights saved to: {output_path}")
        print(f"\nTo use in your pipeline, update config/default.yaml:")
        print(f'  model: "{output_path}"')
        print(f'  use_roboflow: false')
    else:
        print(f"\nTraining complete! Check {cfg['project']}/{cfg['name']} for results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO for futsal tasks (player or ball)")
    parser.add_argument("--task", choices=list(TASKS.keys()), default="player", help="Which task to train")
    args = parser.parse_args()

    train(task=args.task)
