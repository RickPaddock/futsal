"""
Count final team assignments from tracking results.
This shows EXACTLY how many players are assigned to each team.
"""

import json
from pathlib import Path
from collections import Counter

results_path = Path("videos/output/tracking_results.json")

if not results_path.exists():
    print(f"Error: {results_path} not found")
    exit(1)

with open(results_path) as f:
    data = json.load(f)

tracks = data.get('tracks', [])

if not tracks:
    print("No tracks found in results")
    exit(1)

# Count teams
team_a_tracks = []
team_b_tracks = []
unknown_tracks = []

for track in tracks:
    tid = track['track_id']
    team = track.get('team', 'unknown')
    det_count = len(track.get('detections', []))
    
    if det_count == 0:
        continue  # Skip phantom tracks
    
    if team == 'team_a':
        team_a_tracks.append(tid)
    elif team == 'team_b':
        team_b_tracks.append(tid)
    else:
        unknown_tracks.append(tid)

print("=" * 70)
print("FINAL TEAM COUNTS (real tracks only)")
print("=" * 70)
print()
print(f"Team A (BLACK):  {len(team_a_tracks)} players")
print(f"  Track IDs: {sorted(team_a_tracks)}")
print()
print(f"Team B (ORANGE): {len(team_b_tracks)} players")
print(f"  Track IDs: {sorted(team_b_tracks)}")
print()

if unknown_tracks:
    print(f"Unknown: {len(unknown_tracks)} players")
    print(f"  Track IDs: {sorted(unknown_tracks)}")
    print()

print(f"Total: {len(team_a_tracks) + len(team_b_tracks) + len(unknown_tracks)}")
print()

if len(team_a_tracks) == 6 and len(team_b_tracks) == 6:
    print("✅ SUCCESS! Exactly 6 players per team!")
else:
    print(f"❌ FAILED! Expected 6 per team, got A:{len(team_a_tracks)} B:{len(team_b_tracks)}")
