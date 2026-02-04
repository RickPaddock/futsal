# Jersey Digit Dataset Workflow

This guide covers how to pull jersey-focused crops from your match footage, tag digits in Roboflow, and fine-tune the lightweight YOLO classification model that feeds jersey identification in the pipeline.

## 1. Generate Crops

Run the new extractor to harvest torso crops that already respect the same gating rules the pipeline uses (confidence/aspect filters, top-half crop):

```bash
python utils/extract_jersey_crops.py \
  videos/input/GoPro_Futsal_part1_CLEANED.mp4 \
  --config config/default.yaml \
  --output videos/output/jersey_crops \
  --frame-stride 5 \
  --crop-variants top \
  --max-crops 1800
```

Key flags:
- `--crop-variants`: keep `top` for parity with the inference crop (add `torso` if you want a fallback set for experimentation).
- `--min-confidence`, `--min-aspect`: override to widen the gate if you need more samples (defaults come from `jersey_identification`).
- `--dry-run`: sanity-check how many candidates you would save without writing files.

Outputs land in `videos/output/jersey_crops/crops` with a `manifest.csv` that tracks frame index, confidence, and source bbox.

## 2. Curation Checklist

Sort the crops before upload so Roboflow stays clean:
- Keep only crops where the jersey number is clearly visible and centered (front or back).
- Drop frames with heavy motion blur, referee shirts, or obstructed digits.
- For PoC you only need digits **4**, **7**, **10**; keep a handful of representative negatives (no digit) to seed an `unknown` class if you plan to reject ambiguous predictions later.
- Flag double digits such as **10** as a single class label (`10`).

## 3. Roboflow Project Setup

1. Create a new *Single-Label Classification* project (e.g., `futsal-jersey-digits`).
2. Add classes for the full future range (`1`–`12`) so the model head stays stable; you can upload crops for digits 4, 7, 10 now and backfill others later.
3. Upload the curated crops. Roboflow will prompt for the class label per image—assign according to the digit present; use `unknown` for blanks if you kept them.
4. Use Roboflow's split tooling (e.g., 80/10/10) and augment moderately (rotations ±5°, brightness/contrast tweaks). Avoid heavy flips so digits stay readable.

## 4. Training Notes

- Export the dataset in **YOLOv8 Classification** format.
- Update `utils/train_yolo.py` with a new task, for example:
  ```python
  "jersey_digits": {
      "dataset_path": "models/datasets/JERSEY_DIGITS/data.yaml",
      "base_model": "models/yolo11n-cls.pt",
      "epochs": 40,
      "image_size": 224,
      "batch_size": 32,
      "device": "0",
      "project": "runs/classify",
      "name": "jersey_digits_v1",
      "output_weights": "models/JERSEY_DIGITS_best.pt",
  }
  ```
- After training, copy the `best.pt` produced by Ultralytics into `models/` and point `config/default.yaml -> jersey_identification.model` at the new weights.

## 5. Labelling Guidelines

- **Include**: clear digits (front/back), partial overlap with legible digit, mid-action frames, different lighting.
- **Exclude**: goalie numbers on sleeves, digits smaller than ~40 px tall, heavy blur, players with hands covering the number.
- Maintain at least 50–60 examples per digit for the PoC; fill remaining quota with negative/unknown to help the classifier abstain.

Once Roboflow exports are in place you can iterate quickly: rerun the extractor on additional footage, drop new crops into Roboflow, retrain via the helper script, and redeploy the updated weight file.
