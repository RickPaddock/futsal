"""Compare clip-specific Pass 3A/3C outputs against GroundTruth_2a_2b.xlsx.

Usage:
  c:/.../.venv/Scripts/python.exe scripts/compare_pass3a_ground_truth.py \
      --clip 7 \
      --output-dir videos/output/GoPro_Futsal_part1_CLEANED_clip7

This script uses only the Python standard library so it can run in the project
venv without extra installs.
"""

from __future__ import annotations

import argparse
import json
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
}


@dataclass
class GroundTruthRow:
    clip: int
    team: str
    jersey_number: Optional[int]
    name: str
    checkpoints: Dict[int, int]  # frame -> track_id


def _col_to_index(col: str) -> int:
    value = 0
    for ch in col:
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value


def _parse_row_cells(row_elem: ET.Element, shared_strings: List[str]) -> Dict[str, Optional[str]]:
    cells: Dict[str, Optional[str]] = {}
    for cell in row_elem.findall("main:c", NS):
        ref = cell.attrib.get("r", "")
        col = "".join(ch for ch in ref if ch.isalpha())
        cell_type = cell.attrib.get("t")
        value_elem = cell.find("main:v", NS)

        if value_elem is None:
            cells[col] = None
            continue

        raw = value_elem.text
        if raw is None:
            cells[col] = None
            continue

        if cell_type == "s":
            try:
                cells[col] = shared_strings[int(raw)]
            except (ValueError, IndexError):
                cells[col] = raw
        else:
            cells[col] = raw

    return cells


def _load_shared_strings(zip_file: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zip_file.namelist():
        return []

    shared_xml = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
    result: List[str] = []
    for si in shared_xml.findall("main:si", NS):
        t = si.find("main:t", NS)
        if t is not None:
            result.append(t.text or "")
            continue

        # Rich text case.
        parts = [node.text or "" for node in si.findall(".//main:t", NS)]
        result.append("".join(parts))

    return result


def load_ground_truth_rows(xlsx_path: Path, clip: int) -> List[GroundTruthRow]:
    with zipfile.ZipFile(xlsx_path, "r") as zip_file:
        shared_strings = _load_shared_strings(zip_file)
        sheet_xml = ET.fromstring(zip_file.read("xl/worksheets/sheet1.xml"))
        rows = sheet_xml.findall(".//main:sheetData/main:row", NS)

        if len(rows) < 3:
            raise ValueError("Ground-truth sheet does not contain expected header/data rows.")

        parsed_rows: List[Tuple[int, Dict[str, Optional[str]]]] = []
        for row_elem in rows:
            row_idx_raw = row_elem.attrib.get("r")
            if not row_idx_raw or not row_idx_raw.isdigit():
                continue
            parsed_rows.append((int(row_idx_raw), _parse_row_cells(row_elem, shared_strings)))

        def checkpoint_cols_from_header(cells: Dict[str, Optional[str]]) -> List[Tuple[str, int]]:
            cols = sorted(cells.keys(), key=_col_to_index)
            out: List[Tuple[str, int]] = []
            for col in cols:
                header = (cells.get(col) or "").strip()
                if header.isdigit():
                    out.append((col, int(header)))
            return out

        # Workbook may contain multiple clip sections, each with its own frame header row.
        # Detect every section header and pick the one that provides the broadest frame
        # coverage for the requested clip.
        header_rows: List[Tuple[int, List[Tuple[str, int]]]] = []
        for row_idx, cells in parsed_rows:
            col_a = (cells.get("A") or "").strip().lower()
            col_b = (cells.get("B") or "").strip().lower()
            if col_a != "clip" or col_b != "team":
                continue
            checkpoint_cols = checkpoint_cols_from_header(cells)
            if len(checkpoint_cols) < 3:
                continue
            header_rows.append((row_idx, checkpoint_cols))

        if not header_rows:
            raise ValueError("Ground-truth sheet does not contain any valid section headers.")

        header_row_indices = [row_idx for row_idx, _ in header_rows]
        header_lookup = {row_idx: cols for row_idx, cols in header_rows}

        best_rows: List[GroundTruthRow] = []
        best_score: Tuple[int, int, int] = (-1, -1, -1)  # (max_frame, total_checkpoints, row_count)

        for i, header_row_idx in enumerate(header_row_indices):
            checkpoint_columns = header_lookup[header_row_idx]
            next_header_idx = header_row_indices[i + 1] if i + 1 < len(header_row_indices) else 10 ** 9

            block_rows: List[GroundTruthRow] = []
            max_frame_in_block = -1
            total_checkpoints = 0

            for row_idx, cells in parsed_rows:
                if row_idx <= header_row_idx or row_idx >= next_header_idx:
                    continue

                clip_raw = (cells.get("A") or "").strip()
                if not clip_raw.isdigit() or int(clip_raw) != clip:
                    continue

                jersey_raw = (cells.get("D") or "").strip()
                jersey_number = int(jersey_raw) if jersey_raw.isdigit() else None

                checkpoints: Dict[int, int] = {}
                for col, frame_idx in checkpoint_columns:
                    track_raw = (cells.get(col) or "").strip()
                    if track_raw.isdigit():
                        checkpoints[frame_idx] = int(track_raw)

                if checkpoints:
                    max_frame_in_block = max(max_frame_in_block, max(checkpoints.keys()))
                    total_checkpoints += len(checkpoints)

                block_rows.append(
                    GroundTruthRow(
                        clip=int(clip_raw),
                        team=(cells.get("B") or "").strip(),
                        jersey_number=jersey_number,
                        name=(cells.get("E") or "").strip(),
                        checkpoints=checkpoints,
                    )
                )

            if not block_rows:
                continue

            score = (max_frame_in_block, total_checkpoints, len(block_rows))
            if score > best_score:
                best_score = score
                best_rows = block_rows

    return best_rows


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_active_fragment(
    fragments: List[dict],
    frame_idx: int,
    track_id: int,
) -> Optional[dict]:
    for fragment in fragments:
        if bool(fragment.get("is_ghost", False)):
            continue
        if int(fragment.get("track_id", -1)) != track_id:
            continue
        if int(fragment["start_frame"]) <= frame_idx <= int(fragment["end_frame"]):
            return fragment
    return None


def _find_any_active_candidate_tracks(
    fragments: List[dict],
    candidates_by_fragment: Dict[str, dict],
    frame_idx: int,
    jersey_number: int,
) -> List[int]:
    tracks: List[int] = []
    for fragment in fragments:
        if bool(fragment.get("is_ghost", False)):
            continue
        if not (int(fragment["start_frame"]) <= frame_idx <= int(fragment["end_frame"])):
            continue

        candidate = candidates_by_fragment.get(fragment["fragment_id"])
        if candidate is None:
            continue
        if candidate.get("candidate_jersey") == jersey_number:
            tracks.append(int(fragment.get("track_id", -1)))

    return sorted(set(tracks))


def compare(
    gt_rows: List[GroundTruthRow],
    output_dir: Path,
) -> None:
    pass2 = _load_json(output_dir / "pass2_ghosts.json")
    pass3a = _load_json(output_dir / "pass3_candidates.json")
    pass3c = _load_json(output_dir / "pass3_identity_commit.json")

    fragments: List[dict] = pass2["fragments"]
    candidates_by_fragment = {item["fragment_id"]: item for item in pass3a["candidates"]}
    identities_by_fragment = {item["fragment_id"]: item for item in pass3c["identities"]}

    total_checks = 0
    pass3a_track_match = 0
    pass3a_any_match = 0
    pass3c_track_match = 0
    pass3c_team_match = 0

    print("=" * 96)
    print("Ground Truth Comparison (Pass 3A + Pass 3C)")
    print("=" * 96)

    jersey_rows = [row for row in gt_rows if row.jersey_number is not None]
    if not jersey_rows:
        print("No jersey-number rows found for selected clip.")
        return

    for row in jersey_rows:
        expected_team = row.team.lower().replace(" ", "_")
        if row.jersey_number is None:
            continue
        jersey_number = int(row.jersey_number)
        print(f"\n{name_with_case(row.name)} (jersey #{jersey_number}, GT team={row.team})")

        for frame_idx in sorted(row.checkpoints.keys()):
            total_checks += 1
            gt_track = int(row.checkpoints[frame_idx])
            fragment = _find_active_fragment(fragments=fragments, frame_idx=frame_idx, track_id=gt_track)

            if fragment is None:
                print(f"  frame {frame_idx:4d}: GT track {gt_track:>2} not found in pass2 fragments")
                continue

            fragment_id = fragment["fragment_id"]
            candidate = candidates_by_fragment.get(fragment_id, {})
            identity = identities_by_fragment.get(fragment_id, {})

            pass3a_jersey = candidate.get("candidate_jersey")
            pass3c_jersey = identity.get("jersey_number")
            pass3c_team = identity.get("team")

            pass3a_on_track_ok = pass3a_jersey == jersey_number
            pass3c_on_track_ok = pass3c_jersey == jersey_number
            pass3c_team_ok = pass3c_team == expected_team

            if pass3a_on_track_ok:
                pass3a_track_match += 1
            if pass3c_on_track_ok:
                pass3c_track_match += 1
            if pass3c_team_ok:
                pass3c_team_match += 1

            active_tracks_for_jersey = _find_any_active_candidate_tracks(
                fragments=fragments,
                candidates_by_fragment=candidates_by_fragment,
                frame_idx=frame_idx,
                jersey_number=jersey_number,
            )
            pass3a_any_ok = gt_track in active_tracks_for_jersey
            if pass3a_any_ok:
                pass3a_any_match += 1

            print(
                f"  frame {frame_idx:4d}: "
                f"GT track={gt_track:>2} frag={fragment_id} | "
                f"3A(track)={pass3a_jersey} {'OK' if pass3a_on_track_ok else 'X '} | "
                f"3A(active_tracks)={active_tracks_for_jersey} {'OK' if pass3a_any_ok else 'X '} | "
                f"3C(track)={pass3c_jersey} {'OK' if pass3c_on_track_ok else 'X '} | "
                f"3C(team)={pass3c_team} {'OK' if pass3c_team_ok else 'X '}"
            )

    print("\n" + "-" * 96)
    print(f"Total checks: {total_checks}")
    print(f"Pass 3A jersey match on GT track: {pass3a_track_match}/{total_checks}")
    print(f"Pass 3A jersey present on GT track among active candidates: {pass3a_any_match}/{total_checks}")
    print(f"Pass 3C jersey match on GT track: {pass3c_track_match}/{total_checks}")
    print(f"Pass 3C team match on GT track: {pass3c_team_match}/{total_checks}")


def name_with_case(name: str) -> str:
    if not name:
        return "Unknown"
    return name[0].upper() + name[1:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Pass 3A outputs with GroundTruth_2a_2b.xlsx")
    parser.add_argument("--clip", type=int, required=True, help="Clip number in GroundTruth sheet (e.g., 7)")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("docs/GroundTruth_2a_2b.xlsx"),
        help="Path to GroundTruth_2a_2b.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to clip output directory containing pass2/pass3 artifacts",
    )

    args = parser.parse_args()

    if not args.ground_truth.exists():
        raise FileNotFoundError(f"Ground truth file not found: {args.ground_truth}")
    if not args.output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {args.output_dir}")

    gt_rows = load_ground_truth_rows(args.ground_truth, clip=args.clip)
    if not gt_rows:
        raise ValueError(f"No rows found for clip={args.clip} in {args.ground_truth}")

    compare(gt_rows=gt_rows, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
