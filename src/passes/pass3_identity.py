"""
Pass 3: Identity Inference (Offline)

Output: pass3_final/<clip>.json
"""

from pathlib import Path
import json
import numpy as np
from typing import Any
import cv2
from sklearn.cluster import KMeans

from src.utils.video_io import VideoReader
from src.utils.data_models import BoundingBox
from src.detection.team_clustering import TeamClustering

try:
    import orjson
except ImportError:
    orjson = None


def run_pass3(run_dir: Path, config: dict):
    run_dir = Path(run_dir)
    pass2_dir = run_dir / "pass2_identity"
    pass3_dir = run_dir / "pass3_final"
    pass3_dir.mkdir(parents=True, exist_ok=True)

    pass2_files = list(pass2_dir.glob("*_fragments.json"))
    if not pass2_files:
        print(f"No Pass 2 fragment JSON files found in {pass2_dir}")
        return

    print(f"Found {len(pass2_files)} Pass 2 fragment files")

    for pass2_file in pass2_files:
        print(f"Processing: {pass2_file.name}")
        process_clip_pass3(pass2_file=pass2_file, output_dir=pass3_dir, config=config, run_dir=run_dir)

    print(f"Pass 3 complete! Output: {pass3_dir}")


def process_clip_pass3(pass2_file: Path, output_dir: Path, config: dict, run_dir: Path):
    with open(pass2_file, "r", encoding="utf-8") as f:
        pass2_data = json.load(f)

    clip_name = pass2_data.get("clip_name", pass2_file.stem)
    fragments = pass2_data.get("fragments", [])

    if len(fragments) == 0:
        print(f"  No fragments found")
        return

    team_cfg = config.get("team_clustering", {})
    jersey_cfg = config.get("jersey", {})
    n_clusters = team_cfg.get("n_clusters", 2)
    jersey_lock_threshold = jersey_cfg.get("lock_threshold", 0.7)

    fragment_histograms = []
    valid_fragment_indices = []
    for i, fragment in enumerate(fragments):
        hist = fragment.get("mean_hsv_histogram")
        if hist and len(hist) == 96:
            fragment_histograms.append(hist)
            valid_fragment_indices.append(i)

    if len(fragment_histograms) < n_clusters:
        print(f"  Not enough valid fragments")
        return

    X = np.array(fragment_histograms, dtype=np.float32)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    cluster_variances = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_hist = X[cluster_mask]
        if len(cluster_hist) > 0:
            variance = np.var(cluster_hist, axis=0).mean()
            cluster_variances.append((cluster_id, variance))

    cluster_variances.sort(key=lambda x: x[1])
    bibbed_cluster_id = cluster_variances[0][0]

    print(f"  K-Means: {len(fragments)} fragments to 2 teams")
    print(f"  TEAM_A (bibbed): cluster {bibbed_cluster_id}")

    for i, cluster_id in zip(valid_fragment_indices, cluster_labels):
        if cluster_id == bibbed_cluster_id:
            fragments[i]["team"] = "team_a"
        else:
            fragments[i]["team"] = "team_b"


    jersey_assignments = infer_jersey_numbers(fragments, jersey_lock_threshold)

    print(f"  Jersey assignments (ALL): {jersey_assignments}")

    identities = []
    for fragment in fragments:
        team = fragment.get("team", "unknown")
        fragment_id = fragment.get("fragment_id")
        jersey_number = None

        jersey_number = jersey_assignments.get(fragment_id)
        if jersey_number is not None:
            player_id = f"{team.upper()}_{jersey_number}" if team in ("team_a", "team_b") else f"UNKNOWN_{jersey_number}"
            jersey_timeline = fragment.get("jersey_prob_timeline", {})
            confidence = jersey_timeline.get(str(jersey_number), {}).get("total", 0.0)
        else:
            if team == "team_a":
                player_id = "TEAM_A_UNKNOWN"
            elif team == "team_b":
                player_id = "TEAM_B_UNKNOWN"
            else:
                player_id = "UNKNOWN"
            confidence = 0.0

        identity = {
            "fragment_id": fragment_id,
            "player_id": player_id,
            "team": team,
            "jersey_number": jersey_number,
            "confidence": confidence,
            "start_frame": fragment.get("start_frame"),
            "end_frame": fragment.get("end_frame"),
        }
        identities.append(identity)

    output_data = {
        "clip_name": clip_name,
        "identities": identities,
        "team_summary": {
            "team_a_count": len([i for i in identities if i["team"] == "team_a"]),
            "team_b_count": len([i for i in identities if i["team"] == "team_b"]),
            "team_a_jerseys": list(set(i["jersey_number"] for i in identities if i["jersey_number"] is not None)),
        }
    }

    output_stem = pass2_file.stem.replace('_fragments', '')
    output_path = output_dir / f"{output_stem}.json"
    pretty_json = config.get("pass3", {}).get("pretty_json", True)

    if orjson is not None:
        with open(output_path, "wb") as f:
            if pretty_json:
                f.write(orjson.dumps(output_data, option=orjson.OPT_INDENT_2))
            else:
                f.write(orjson.dumps(output_data))
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            if pretty_json:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            else:
                json.dump(output_data, f, separators=(",", ":"), ensure_ascii=False)

    print(f"  Saved: {output_path}")
    print(f"  Identities: {len(identities)}")
    print(f"  TEAM_A: {output_data['team_summary']['team_a_count']} fragments")
    print(f"  TEAM_B: {output_data['team_summary']['team_b_count']} fragments")

    _export_team_crops(
        run_dir=run_dir,
        clip_name=clip_name,
        output_dir=output_dir,
        output_stem=output_stem,
        fragments=fragments,
        config=config,
    )


def infer_jersey_numbers(fragments: list[dict[str, Any]], lock_threshold: float = 0.7) -> dict[str, int]:
    assignments = {}
    for fragment in fragments:
        fragment_id = fragment.get("fragment_id")
        jersey_timeline = fragment.get("jersey_prob_timeline", {})
        best_jersey = None
        best_confidence = 0.0
        for jersey_id, stats in jersey_timeline.items():
            total_conf = stats.get("total", 0.0)
            if total_conf > best_confidence:
                best_confidence = total_conf
                best_jersey = int(jersey_id)
        if best_jersey is not None and best_confidence >= lock_threshold:
            assignments[fragment_id] = best_jersey
    return assignments


def _crop_jersey_region(
    bbox: list[int],
    frame_shape: tuple[int, int, int],
    top_skip: float = 0.2,
    bottom_cut: float = 0.55,
    width_shrink: float = 0.8,
) -> tuple[int, int, int, int]:
    """Crop upper-torso jersey region from bbox."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    cx = (x1 + x2) / 2.0

    new_w = width * width_shrink
    ny1 = int(max(0, y1 + height * top_skip))
    ny2 = int(min(frame_shape[0], y1 + height * bottom_cut))
    nx1 = int(max(0, cx - new_w / 2.0))
    nx2 = int(min(frame_shape[1], cx + new_w / 2.0))
    return nx1, ny1, nx2, ny2


def _is_valid_bbox(frame_shape: tuple[int, int, int], bbox: BoundingBox) -> bool:
    w = bbox.width
    h = bbox.height
    if w <= 0 or h <= 0:
        return False

    ar = w / h
    min_w, min_h = 20, 35
    max_ar = 3.5
    min_ar = 0.3
    max_h = frame_shape[0] * 0.8

    if w < min_w or h < min_h:
        return False
    if ar > max_ar or ar < min_ar:
        return False
    if h > max_h:
        return False
    return True


def _build_frame_index(tracks: dict[str, Any]) -> dict[int, list[tuple[str, list[int]]]]:
    frame_index: dict[int, list[tuple[str, list[int]]]] = {}
    for track_id, track_data in tracks.items():
        frames = track_data.get("frames", [])
        bboxes = track_data.get("bboxes", [])
        for frame_idx, bbox in zip(frames, bboxes):
            frame_index.setdefault(int(frame_idx), []).append((str(track_id), bbox))
    return frame_index


def _is_isolated_sample(
    bbox: BoundingBox,
    frame_shape: tuple[int, int, int],
    other_bboxes: list[list[int]],
    proximity_thresh: float,
) -> bool:
    if not other_bboxes:
        return True

    crop_region = _crop_jersey_region([bbox.x1, bbox.y1, bbox.x2, bbox.y2], frame_shape)

    def iou(box_a, box_b):
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])
        inter_w = max(0, x_b - x_a)
        inter_h = max(0, y_b - y_a)
        inter_area = inter_w * inter_h
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area

    bbox_center = bbox.center
    for other in other_bboxes:
        other_box = BoundingBox(x1=other[0], y1=other[1], x2=other[2], y2=other[3])
        dx = bbox_center[0] - other_box.center[0]
        dy = bbox_center[1] - other_box.center[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < proximity_thresh:
            return False

        if iou(crop_region, other) > 0.01:
            return False

        cx, cy = other_box.center
        if crop_region[0] <= cx <= crop_region[2] and crop_region[1] <= cy <= crop_region[3]:
            return False

    return True


def _select_sample_indices(total: int, max_samples: int) -> list[int]:
    if total <= 0 or max_samples <= 0:
        return []
    if total <= max_samples:
        return list(range(total))
    return np.linspace(0, total - 1, max_samples, dtype=int).tolist()


def _full_bbox_crop(frame: np.ndarray, bbox: list[int]) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(int(x2), frame.shape[1]), min(int(y2), frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None
    crop_rgb = frame[y1:y2, x1:x2]
    if crop_rgb.size == 0:
        return None
    return cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)




def _export_team_crops(
    run_dir: Path,
    clip_name: str,
    output_dir: Path,
    output_stem: str,
    fragments: list[dict[str, Any]],
    config: dict,
) -> None:
    pass3_cfg = config.get("pass3", {})
    save_team_crops = pass3_cfg.get("save_team_crops", True)
    if not save_team_crops:
        return

    pass1_file = run_dir / "pass1_raw" / f"{Path(clip_name).stem}.json"
    if not pass1_file.exists():
        print(f"  [CROPS] Pass 1 file not found: {pass1_file}")
        return

    input_dir = Path(pass3_cfg.get("crops_input_dir", "videos/input"))
    video_path = input_dir / clip_name
    if not video_path.exists():
        print(f"  [CROPS] Video not found: {video_path}")
        return

    crops_root = output_dir / f"{output_stem}_team_crops"
    team_a_dir = crops_root / "teamA"
    team_b_dir = crops_root / "teamB"
    team_a_dir.mkdir(parents=True, exist_ok=True)
    team_b_dir.mkdir(parents=True, exist_ok=True)

    crops_per_fragment = int(pass3_cfg.get("team_crops_per_fragment", 2))
    min_per_team = int(pass3_cfg.get("team_crops_min_per_team", 20))
    crop_mode = pass3_cfg.get("team_crop_mode", "histogram_source")
    proximity_thresh = float(pass3_cfg.get("team_crop_proximity_px", 40.0))

    team_cfg = config.get("team_clustering", {})
    bins = int(team_cfg.get("bins", 32))
    quality_threshold = float(team_cfg.get("quality_threshold", 0.5))
    clustering = TeamClustering(bins=bins)

    with open(pass1_file, "r", encoding="utf-8") as f:
        pass1_data = json.load(f)

    tracks = pass1_data.get("tracks", {})
    frame_index = _build_frame_index(tracks)
    reader = VideoReader(video_path)
    frame_cache: dict[int, np.ndarray] = {}

    team_fragments = {
        "team_a": [f for f in fragments if f.get("team") == "team_a"],
        "team_b": [f for f in fragments if f.get("team") == "team_b"],
    }
    per_team_target = {}
    for team_key, team_list in team_fragments.items():
        if not team_list:
            continue
        per_fragment_target = int(np.ceil(min_per_team / len(team_list)))
        per_team_target[team_key] = max(crops_per_fragment, per_fragment_target)

    for fragment in fragments:
        team = fragment.get("team")
        if team not in ("team_a", "team_b"):
            continue

        track_id = str(fragment.get("original_track_id"))
        track_data = tracks.get(track_id)
        if not track_data:
            continue

        frames = track_data.get("frames", [])
        bboxes = track_data.get("bboxes", [])
        hsv_histograms = track_data.get("hsv_histograms", [])
        if not frames or not bboxes or len(frames) != len(bboxes):
            continue

        start_frame = fragment.get("start_frame", frames[0])
        end_frame = fragment.get("end_frame", frames[-1])

        samples = [
            (f_idx, bbox, hist)
            for f_idx, bbox, hist in zip(frames, bboxes, hsv_histograms or [None] * len(frames))
            if start_frame <= f_idx <= end_frame
        ]
        if not samples:
            continue

        if crop_mode == "histogram_source":
            samples = [s for s in samples if s[2] is not None]
            if not samples:
                continue

        target_per_fragment = per_team_target.get(team, crops_per_fragment)
        sample_indices = _select_sample_indices(
            len(samples),
            max(target_per_fragment * 3, target_per_fragment)
        )
        fragment_id = fragment.get("fragment_id", "frag_unknown")
        out_dir = team_a_dir if team == "team_a" else team_b_dir

        candidates = []
        fallback_samples = []
        for sample_pos in sample_indices:
            frame_idx, bbox, hist = samples[sample_pos]
            if frame_idx in frame_cache:
                frame = frame_cache[frame_idx]
            else:
                try:
                    frame = reader.get_frame(int(frame_idx))
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"  [CROPS] Failed frame {frame_idx}: {exc}")
                    continue
                frame_cache[frame_idx] = frame

            bbox_obj = BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
            fallback_crop = _full_bbox_crop(frame, bbox)
            if fallback_crop is not None:
                fallback_samples.append((frame_idx, fallback_crop, sample_pos))

            if crop_mode == "histogram_source":
                crop_bgr = fallback_crop
                if crop_bgr is None or crop_bgr.size == 0:
                    continue
                quality = 1.0
                candidates.append((quality, frame_idx, crop_bgr, sample_pos))
                continue

            if not _is_valid_bbox(frame.shape, bbox_obj):
                continue

            others = [b for tid, b in frame_index.get(int(frame_idx), []) if tid != track_id]
            if not _is_isolated_sample(bbox_obj, frame.shape, others, proximity_thresh):
                continue

            if crop_mode == "jersey":
                hist, _, crop_bgr, quality = clustering.extract_features(frame, bbox_obj)
                if quality < quality_threshold:
                    continue
                if crop_bgr is None or crop_bgr.size == 0:
                    continue
            else:
                crop_bgr = fallback_crop
                if crop_bgr is None or crop_bgr.size == 0:
                    continue
                quality = 1.0

            candidates.append((quality, frame_idx, crop_bgr, sample_pos))

        candidates.sort(key=lambda x: x[0], reverse=True)
        for quality, frame_idx, crop_bgr, sample_pos in candidates[:target_per_fragment]:
            filename = f"{fragment_id}_f{frame_idx}_tid{track_id}_{sample_pos}.jpg"
            cv2.imwrite(str(out_dir / filename), crop_bgr)

        if not candidates and fallback_samples:
            fallback_samples.sort(key=lambda x: x[0])
            frame_idx, crop_bgr, sample_pos = fallback_samples[0]
            filename = f"{fragment_id}_f{frame_idx}_tid{track_id}_{sample_pos}_fallback.jpg"
            cv2.imwrite(str(out_dir / filename), crop_bgr)

    print(f"  [CROPS] Saved team crops to: {crops_root}")
