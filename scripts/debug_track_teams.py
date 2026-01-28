"""Check if player tracks have team assignments."""
import json
from pathlib import Path

results_path = Path("videos/output/tracking_results.json")

if not results_path.exists():
    print(f"Error: {results_path} not found")
    exit(1)

with open(results_path) as f:
    data = json.load(f)

tracks = data.get('tracks', [])

print("=" * 70)
print("TRACK TEAM ASSIGNMENTS")
print("=" * 70)
print()

for track in tracks[:5]:  # Check first 5 tracks
    tid = track['track_id']
    team = track.get('team', 'NOT_SET')
    det_count = len(track.get('detections', []))
    
    print(f"Track {tid}: team={team}, detections={det_count}")

print()
print("Summary:")
team_a = sum(1 for t in tracks if t.get('team') == 'team_a')
team_b = sum(1 for t in tracks if t.get('team') == 'team_b')
unknown = sum(1 for t in tracks if t.get('team') != 'team_a' and t.get('team') != 'team_b')

print(f"  Team A: {team_a}")
print(f"  Team B: {team_b}")
print(f"  Unknown: {unknown}")
