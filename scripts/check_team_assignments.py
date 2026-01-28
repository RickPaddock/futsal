"""
Diagnostic script to check team assignments from team_crops.

This helps identify misclassified players by showing which track IDs
ended up in each team folder during the initial K-means clustering.
"""

import re
from pathlib import Path
from collections import defaultdict


def extract_track_ids_from_crops(crops_dir: Path):
    """Extract track IDs from team crop filenames."""
    team_a_path = crops_dir / "team_a"
    team_b_path = crops_dir / "team_b"
    
    def get_track_ids(path):
        track_ids = set()
        tid_counts = defaultdict(int)
        for f in path.glob("*.jpg"):
            # Extract tid from filename like 'f0_tid10_126.jpg'
            match = re.search(r'tid(\d+|None)', f.name)
            if match:
                tid = match.group(1)
                if tid != 'None':
                    tid_int = int(tid)
                    track_ids.add(tid_int)
                    tid_counts[tid_int] += 1
        return track_ids, tid_counts
    
    team_a_ids, team_a_counts = get_track_ids(team_a_path)
    team_b_ids, team_b_counts = get_track_ids(team_b_path)
    
    return team_a_ids, team_a_counts, team_b_ids, team_b_counts


def main():
    crops_dir = Path("videos/output/team_crops")
    
    if not crops_dir.exists():
        print(f"Error: {crops_dir} does not exist")
        return
    
    team_a_ids, team_a_counts, team_b_ids, team_b_counts = extract_track_ids_from_crops(crops_dir)
    
    print("=" * 70)
    print("TEAM ASSIGNMENT ANALYSIS (from team_crops)")
    print("=" * 70)
    print()
    
    print(f"Team A (BLACK): {len(team_a_ids)} unique players")
    print(f"  Track IDs: {sorted(team_a_ids)}")
    print(f"  Sample counts: {dict(sorted(team_a_counts.items()))}")
    print()
    
    print(f"Team B (ORANGE): {len(team_b_ids)} unique players")
    print(f"  Track IDs: {sorted(team_b_ids)}")
    print(f"  Sample counts: {dict(sorted(team_b_counts.items()))}")
    print()
    
    # Check for issues
    overlap = team_a_ids & team_b_ids
    if overlap:
        print(f"⚠️  WARNING: {len(overlap)} players in BOTH teams: {sorted(overlap)}")
        print()
    
    total_players = len(team_a_ids) + len(team_b_ids)
    expected_players = 12
    
    print(f"Total unique players: {total_players} (expected: {expected_players})")
    
    if len(team_a_ids) > 6:
        print(f"⚠️  WARNING: Team A has {len(team_a_ids)} players (max should be 6)")
        print(f"    Likely {len(team_a_ids) - 6} orange player(s) misclassified as black")
    
    if len(team_b_ids) < 6:
        print(f"⚠️  WARNING: Team B has only {len(team_b_ids)} players (should be 6)")
        print(f"    Missing {6 - len(team_b_ids)} orange player(s)")
    
    if len(team_a_ids) == 6 and len(team_b_ids) == 6:
        print("✓ Both teams have exactly 6 players - assignments look correct!")
    
    print()
    
    # Find missing track IDs (if we expect 12 total)
    all_ids = team_a_ids | team_b_ids
    expected_ids = set(range(1, expected_players + 1))
    missing_ids = expected_ids - all_ids
    
    if missing_ids:
        print(f"Missing track IDs (not in any team): {sorted(missing_ids)}")
        print()


if __name__ == "__main__":
    main()
