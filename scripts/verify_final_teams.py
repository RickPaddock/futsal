"""Final team assignment verification script."""
import json
from pathlib import Path

results_path = Path("videos/output/tracking_results.json")
data = json.load(open(results_path))

tracks = {t['track_id']: t for t in data['tracks']}

print("=" * 70)
print("FINAL TEAM ASSIGNMENTS (from tracking_results.json)")
print("=" * 70)
print()

team_a_ids = []
team_b_ids = []
unknown_ids = []

for tid, track in sorted(tracks.items()):
    team = track.get('team', 'unknown')
    det_count = len(track.get('detections', []))
    
    if team == 'team_a':
        team_a_ids.append(tid)
    elif team == 'team_b':
        team_b_ids.append(tid)
    else:
        unknown_ids.append(tid)
    
    # Show tracks with 0 detections (phantom tracks)
    if det_count == 0:
        print(f"⚠️  Track {tid}: {team}, {det_count} detections (PHANTOM)")

print()
print(f"Team A (black): {len(team_a_ids)} players")
print(f"  Track IDs: {team_a_ids}")
print()

print(f"Team B (orange): {len(team_b_ids)} players")
print(f"  Track IDs: {team_b_ids}")
print()

if unknown_ids:
    print(f"Unknown: {len(unknown_ids)} players")
    print(f"  Track IDs: {unknown_ids}")
    print()

# Filter out phantom tracks
real_team_a = [tid for tid in team_a_ids if len(tracks[tid].get('detections', [])) > 0]
real_team_b = [tid for tid in team_b_ids if len(tracks[tid].get('detections', [])) > 0]

print(f"Real (non-phantom) players:")
print(f"  Team A: {len(real_team_a)} - {real_team_a}")
print(f"  Team B: {len(real_team_b)} - {real_team_b}")
print()

if len(real_team_a) == 6 and len(real_team_b) == 6:
    print("✓ SUCCESS: Both teams have exactly 6 players!")
else:
    print(f"⚠️  Team imbalance:")
    if len(real_team_a) != 6:
        print(f"   Team A has {len(real_team_a)} players (expected 6)")
    if len(real_team_b) != 6:
        print(f"   Team B has {len(real_team_b)} players (expected 6)")
