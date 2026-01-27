# Futsal Player Tracking

Uses 3 tier tracking hierarchy:
T1 = YOLO Model
T2 = SAM only on missing players
T3 = estimate position behind occluding player

T1 is most accurate and preferred. If a player get occluded and plyer count <12, it tries T2 pixel level SAM2 model to get the outline of the missing player in last known position, likely being occluded behind another player. Finally, T3 is used, which is just an estimate of last known position behind the player who is occluding him.

## 2D Pitch Positioning
Player positions on the 2D pitch are calculated from the **head** (top of bbox) plus a fixed pixel offset downward to estimate foot position. This is stable even when lower body is occluded - the head remains visible. The offset is configurable via `head_to_foot_offset` in homography.py (default 150px). 

# TO DO
1) T2 SAM on area of missing player not full pitch
2) Stable tracking, stop jumping boxes/colours
3) Team segmentation k-nearest
4) Ball tracking

CLI entry: `python main.py` (commands: `process`, `trim`, `info`).

## Install
- Python 3.10+ recommended.
- Create venv and install deps:
	```bash
	python -m venv .venv
	.venv\\Scripts\\activate
	pip install -r requirements.txt
	```
- Install PyTorch with CUDA matching your Legion 5070 GPU (RTX 30/40 series typically CUDA 11.8 or 12.x). Follow https://pytorch.org for the correct `--index-url`.

## SAM2 (kept at current C:\ path)
- Config currently points to your existing checkpoints:
	- `segmentation.sam2.checkpoint_path`: `C:/Users/rickp/Documents/GitRepos/futsal_notebooks/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt`
	- `segmentation.sam2.config_file`: `C:/Users/rickp/Documents/GitRepos/futsal_notebooks/segment-anything-2-real-time/sam2/configs/sam2.1/sam2.1_hiera_s.yaml`
- If you relocate checkpoints later, update `config/default.yaml` accordingly. We only use SAM2 (no SAM v1), model_type `hiera_small` by default.

## Usage
- Process video:
	```bash
	python main.py process "videos/input/GoPro_Futsal_part1.mp4" --output videos/output/run1 --start-frame 10000 --end-frame 18000 --debug
	```
- Trim video:
	```bash
	python main.py trim "videos/input/GoPro_Futsal_part1.mp4" -o videos/output/clip.mp4 -s 10 -d 30 -scale 0.5
	```
- System info:
	```bash
	python main.py info
	```

## Folder layout notes
- `src/utils/` core utilities used by the pipeline (`video_io`, `data_models`, `visualization`).
- `utils/` contains developer tools (click_points, extract_training_frames, train_yolo). The legacy trim script was removed; use `main.py trim` instead.
