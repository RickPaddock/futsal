"""
Analyze K-means cluster distances to find misclassified players.

Load the team_crops and compute which player samples are furthest from
their assigned cluster center - these are likely misclassifications.
"""

import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import re
from collections import defaultdict


def extract_jersey_features(img_bgr):
    """Extract HSV histogram features (same as team_clustering.py)."""
    if img_bgr is None or img_bgr.size == 0:
        return None
    
    # Convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Use simple mask: exclude very bright pixels (white) and very dark (shadows)
    mask = np.ones(hsv.shape[:2], dtype=np.uint8) * 255
    
    # Calculate histograms
    bins = 32
    hist_h = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], mask, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], mask, [bins], [0, 256])
    
    # Normalize
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    
    return np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)


def main():
    crops_dir = Path("videos/output/team_crops")
    team_a_path = crops_dir / "team_a"
    team_b_path = crops_dir / "team_b"
    
    # Load all crops and extract features
    samples_a = []
    track_ids_a = []
    
    for img_path in team_a_path.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is not None:
            feat = extract_jersey_features(img)
            if feat is not None:
                samples_a.append(feat)
                # Extract track ID
                match = re.search(r'tid(\d+)', img_path.name)
                if match:
                    track_ids_a.append(int(match.group(1)))
                else:
                    track_ids_a.append(-1)
    
    samples_b = []
    track_ids_b = []
    
    for img_path in team_b_path.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is not None:
            feat = extract_jersey_features(img)
            if feat is not None:
                samples_b.append(feat)
                match = re.search(r'tid(\d+)', img_path.name)
                if match:
                    track_ids_b.append(int(match.group(1)))
                else:
                    track_ids_b.append(-1)
    
    print(f"Loaded {len(samples_a)} Team A samples, {len(samples_b)} Team B samples")
    
    # Combine and fit K-means
    all_samples = np.array(samples_a + samples_b)
    all_labels = [0] * len(samples_a) + [1] * len(samples_b)
    all_track_ids = track_ids_a + track_ids_b
    
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    predicted_labels = kmeans.fit_predict(all_samples)
    
    # Compute distances to assigned cluster
    distances = kmeans.transform(all_samples)  # (N, 2) - distance to each cluster
    
    # For each sample, get distance to its assigned cluster
    per_sample_dist = []
    for i, (actual_label, pred_label) in enumerate(zip(all_labels, predicted_labels)):
        dist_to_assigned = distances[i, actual_label]
        dist_to_other = distances[i, 1 - actual_label]
        per_sample_dist.append({
            'index': i,
            'track_id': all_track_ids[i],
            'assigned_team': 'A' if actual_label == 0 else 'B',
            'predicted_team': 'A' if pred_label == 0 else 'B',
            'dist_to_assigned': dist_to_assigned,
            'dist_to_other': dist_to_other,
            'mismatch': actual_label != pred_label
        })
    
    # Sort by distance to assigned cluster (descending) - worst matches first
    per_sample_dist.sort(key=lambda x: x['dist_to_assigned'], reverse=True)
    
    print()
    print("=" * 80)
    print("WORST CLUSTER MATCHES (furthest from assigned cluster)")
    print("=" * 80)
    
    # Group by track ID and show average distance
    track_stats = defaultdict(lambda: {'assigned': None, 'total_dist': 0, 'count': 0, 'mismatches': 0})
    
    for sample in per_sample_dist:
        tid = sample['track_id']
        track_stats[tid]['assigned'] = sample['assigned_team']
        track_stats[tid]['total_dist'] += sample['dist_to_assigned']
        track_stats[tid]['count'] += 1
        if sample['mismatch']:
            track_stats[tid]['mismatches'] += 1
    
    # Compute average distance per track
    track_avg_dist = []
    for tid, stats in track_stats.items():
        if stats['count'] > 0:
            avg_dist = stats['total_dist'] / stats['count']
            track_avg_dist.append({
                'track_id': tid,
                'assigned': stats['assigned'],
                'avg_dist': avg_dist,
                'mismatch_rate': stats['mismatches'] / stats['count']
            })
    
    # Sort by average distance
    track_avg_dist.sort(key=lambda x: x['avg_dist'], reverse=True)
    
    print()
    print("Track ID | Assigned | Avg Distance | Mismatch Rate | Notes")
    print("-" * 80)
    
    for t in track_avg_dist:
        notes = ""
        if t['mismatch_rate'] > 0.5:
            notes = "⚠️  LIKELY MISCLASSIFIED"
        elif t['avg_dist'] > 0.5:
            notes = "⚠️  Weak match"
        
        print(f"   {t['track_id']:2d}    |    {t['assigned']}    |    {t['avg_dist']:.4f}    |     {t['mismatch_rate']:.1%}      | {notes}")
    
    print()
    print("=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    
    # Find Team A players with high mismatch rate
    misclassified_a = [t for t in track_avg_dist if t['assigned'] == 'A' and t['mismatch_rate'] > 0.5]
    misclassified_b = [t for t in track_avg_dist if t['assigned'] == 'B' and t['mismatch_rate'] > 0.5]
    
    if misclassified_a:
        print(f"Players in Team A that K-means thinks should be Team B:")
        for t in misclassified_a:
            print(f"  - Track ID {t['track_id']} (mismatch rate: {t['mismatch_rate']:.1%})")
    
    if misclassified_b:
        print(f"Players in Team B that K-means thinks should be Team A:")
        for t in misclassified_b:
            print(f"  - Track ID {t['track_id']} (mismatch rate: {t['mismatch_rate']:.1%})")


if __name__ == "__main__":
    main()
