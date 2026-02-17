# Futsal Tracking System - Master Implementation Contract

**This is the authoritative contract for all implementation work.**

All code must comply with these rules. Violations are invalid and must fail-fast.
This document defines non-negotiable principles, architecture, and validation requirements.

---

## 0. ABSOLUTE PRINCIPLES (CANNOT BE BROKEN)

These apply to all code, all changes, all future modifications.

### P0: Root-Cause Fixes Only
- Never patch downstream symptoms
- If Pass 3 breaks → fix Pass 3 or earlier
- Never "adjust output" to hide earlier errors

### P1: Pass Immutability
- Pass N may read Pass N-1 artifacts
- Pass N may **NEVER** modify Pass N-1 output
- Each pass emits new artifacts, never mutates existing ones

### P2: Single Responsibility Per Pass
- No pass may partially do another pass's job
- Identity is decided ONCE at Pass 3C commit point
- After identity is locked, it is immutable

### P3: Identity Inference at Lock Point
- Identity inference happens at a single commit point (Pass 3C)
- Before Pass 3C: candidates and constraints only
- After Pass 3C: identity is immutable (player_id, team, jersey locked)

### P4: Validation is Blocking
- If a validation rule fails, execution halts immediately
- No partial or "best effort" outputs
- Every rule is executable code (not documentation)

### P5: JSON is Source of Truth
- Everything must be auditable without watching video
- JSON + metrics must fully explain behavior
- Visualization is a confirmation layer, not the source of truth

---

## 1. DIRECTORY & FILE SYSTEM CONTRACT (MANDATORY)

### Input
```
videos/input/
  └── *.mp4  (e.g., GoPro_Futsal_part1_CLEANED_clip9.mp4)
```

### Output (AUTO-CREATED IF NOT EXISTS)
```
videos/output/<clip_name>/
  ├── pass1_raw.json                # Raw detections (YOLO + ByteTrack)
  ├── pass1_validation.json         # Pass 1 validation report
  │
  ├── pass2_fragments.json          # Fragments after splitting
  ├── pass2_ghosts.json             # Ghost fragments
  ├── pass2_validation.json         # Pass 2 validation report
  │
  ├── pass3_candidates.json         # Identity candidates (not locked)
  ├── pass3_constraints.json        # Constraint graph
  ├── pass3_identity_commit.json    # LOCKED identities (immutable)
  ├── pass3_validation.json         # Pass 3 validation report
  │
  ├── ball_interpolation.json       # Ball positions (real + interpolated)
  │
  ├── debug_metrics.json            # Frame-by-frame debug metrics
  └── visualization.mp4             # Final rendered video
```

**Rules:**
- Output folder name = input filename without extension
- No extra subfolders
- Every artifact must be written explicitly
- If a pass fails validation → later files MUST NOT be written

---

## 2. ENTITY MODEL (STRICT)

These identifiers MUST exist and NEVER be confused.

| ID Type       | Meaning                          | Mutable? | Format                       |
|---------------|----------------------------------|----------|------------------------------|
| `track_id`    | ByteTrack temporary ID           | YES      | Integer (e.g., 1, 2, 3)      |
| `fragment_id` | Contiguous physical body segment | NO       | `F{counter:06d}` (e.g., F000001) |
| `player_id`   | Final identity                   | NO       | `P{jersey:02d}_{team}` (e.g., P07_team_a) |
| `detection_id`| Frame-local detection hash       | NO       | `{frame_idx}_{track_id}_{bbox_hash}` |

**Critical Distinction:**
- `track_id`: Temporary, used only in Pass 1, can change/disappear
- `fragment_id`: Immutable, represents a contiguous segment of detections
- `player_id`: Final identity, locked at Pass 3C, never changes

---

## 3. NON-NEGOTIABLE SYSTEM RULES

### R1 — Pass 1 is Raw Truth Only
- **What Pass 1 MUST contain:**
  - YOLO detections (bbox, confidence)
  - ByteTrack temporary track_id
  - Jersey classification probabilities (NOT final assignments)
  - HSV histograms (raw observations)
  - Ball detections
- **What Pass 1 MUST NOT contain:**
  - Teams (no team_a/team_b)
  - Player identity (no player_id)
  - Merged/split tracks (no heuristics beyond pathological filters)
  - Interpretation or inference

**Multi-Layer Defense (Huge Bbox Filter):**
- Layer 1: Reject bbox > 800px height OR > 600px width
- Layer 2: Reject bbox > 25% of frame area
- Rationale: YOLO occasionally hallucinates huge bboxes (floor, shadows)

### R2 — Every Player Has a Team
- **After Pass 3C**, every fragment MUST have `team = "team_a"` OR `team = "team_b"`
- `team = "unknown"` is **FORBIDDEN** after Pass 3C
- Players **NEVER** switch teams mid-game (HARD rule - validation fails if violated)
- Team locked via `_locked_team` field (single source of truth)
- **Team size constraints**:
  - **HARD**: Max 6 players per team 
  - **SOFT**: Imbalance allowed (5v6, 6v5, 4v5, etc.) - partial clips, off-screen starts
  - **Expected balance**: ~6v6 in full game clips
  - **Pipeline does NOT fail on imbalance**, only on exceeding 6 per team

### R3 — Jersey Numbers are Global Identity Keys
- **One jersey = one player forever**
- **Temporal exclusivity**: Same jersey CANNOT appear on different players simultaneously
- **Bidirectional propagation**: Inherit forward AND backward on same track
- **Conditional inheritance**: Inheritance ONLY occurs IF jersey not in use elsewhere at that time
- **Jersey change triggers split**: If jersey changes (#7 → #4), track jumped to different player → SPLIT
- **Jersey first appearance does NOT trigger split**: If jersey appears (None → #4), player turned around → NO SPLIT

### R4 — Players Never Disappear
- **Missing detections → ghost fragments created**
- **Dynamic level (high water mark)**: Level only increases as more players enter, never decreases
- **Target level**: High water mark up to 12 (futsal regulation max)
  - Partial clips: May start with fewer players (e.g., 8 visible)
  - Off-screen starts: Level increases as players enter (10 → 11 → 12)
  - No substitutions in futsal: Once a player enters, they don't leave (except briefly off-screen)
- **Ghosts maintain identity continuity, NOT team symmetry**:
  - Ghosts fill gaps to maintain player count at high-water mark
  - Team balance (6v6) is enforced AFTER Pass 3C team assignment, NOT during ghost creation
- **Track players by `original_track_id`** (NOT `fragment_id`):
  - Fragment splits don't create disappearances
  - Track jumps create new fragments but same player
- **Max 12 concurrent players** (HARD rule - futsal regulation)

### R5 — Ball State Exists at Every Frame
- **HARD rule**: Ball state exists at every frame
- **Ball state** ∈ {`real`, `interpolated`, `out_of_play`}:
  - **`real`**: YOLO detection with bbox and confidence
  - **`interpolated`**: Gap ≤ 30 frames, position interpolated (linear or Kalman)
  - **`out_of_play`**: Gap > 30 frames, ball not on court (no position)
- **Validator checks**: Presence of ball state, NOT presence of ball position
- **Out-of-play frames**: Ball may have no position, but MUST have state = `out_of_play`

---

## 4. PIPELINE OVERVIEW (FIXED ORDER)

```
Pass 1   → Raw Evidence Collection
Pass 2A  → Mechanical Fragmentation
Pass 2B  → Fragment Quality Scoring
Pass 2C  → Ghost Generation
Pass 3A  → Identity Candidate Generation
Pass 3B  → Constraint Graph Construction
Pass 3C  → Identity Commit (🔒 LOCK POINT - identity immutable after this)
Ball     → Ball Interpolation
Viz      → Visualization (confirmation layer)
```

**Each pass is a pure function:**
- Takes explicit input JSON
- Emits output JSON
- Validates output before continuing
- Zero side effects
- **FAIL-FAST** on validation error

---

## 5. PASS RESPONSIBILITIES

### Pass 1: Raw Evidence Collection
**Input**: Video file
**Output**: `pass1_raw.json`, `pass1_validation.json`

**Responsibilities:**
- YOLO player detection (with huge bbox filter)
- ByteTrack tracking (geometry only, temporary track_id)
- YOLO jersey classification (probabilities only, conf ≥ 0.3)
- HSV histogram extraction (8x8x8 = 512 bins)
- YOLO ball detection
- **NO teams, NO identity, NO merges/splits beyond pathological filters**

**Validation (BLOCKING):**
- FAIL IF: bbox out of frame
- FAIL IF: bbox > 25% frame area (huge bbox defense)
- FAIL IF: jersey probabilities invalid
- FAIL IF: duplicate track_id in same frame

### Pass 2A: Mechanical Fragmentation
**Input**: `pass1_raw.json`
**Output**: `pass2_fragments.json`, `pass2_validation.json`

**Responsibilities:**
- Detect divergence points ONLY (no identity logic)
- Split tracks into fragments based on:
  - **Track overlap collision**: Same track_id produces >1 detection in same frame (ByteTrack failure) → immediate split
  - Velocity spikes
  - Appearance drift (HSV histogram change)
  - Jersey inconsistency:
    - ✅ SPLIT: Jersey disappears (#4 → None) - track lost player
    - ✅ SPLIT: Jersey changes (#7 → #4) - track jumped
    - ❌ NO SPLIT: Jersey first appearance (None → #4) - player turned around
  - Jersey temporal exclusivity (same jersey on different tracks)
  - Occlusion confidence drop
  - Spatial crossings
- **Keep ALL fragments** (even < 10 frames)
- Mark short fragments as `quality = "low"`
- **Merge consecutive short fragments** on same track

**Validation (BLOCKING):**
- FAIL IF: Fragment coverage < 100% of Pass 1 frames
- FAIL IF: Fragment overlap detected
- FAIL IF: Player count + ghosts > 12

### Pass 2B: Fragment Quality Scoring
**Input**: `pass2_fragments.json`
**Output**: `pass2_fragments.json` (with quality scores)

**Responsibilities:**
- Compute metadata ONLY (no identity assignment)
- Fragment scores:
  - Appearance stability
  - Jersey observability
  - Motion smoothness
  - Occlusion ratio
- Assign quality: HIGH, MEDIUM, LOW (GHOST assigned in Pass 2C)

### Pass 2C: Ghost Generation
**Input**: `pass2_fragments.json`, `pass1_raw.json`
**Output**: `pass2_ghosts.json`, `pass2_validation.json`

**Responsibilities:**
- Maintain player count continuity
- **Initialize level** from first 10 frames
- **Dynamic level (high water mark)**:
  - Update if more players enter (never decrease)
  - Cap at 12 (futsal regulation)
- **Track players by `original_track_id`** (NOT `fragment_id`)
- Create ghosts when `tracked_count < level`
- **Ghost position**: HOLD last known position (no interpolation)
- **Ghost duration**: Until reappearance or MAX_GAP (60 frames)
- Mark ghosts: `is_ghost = True`, `quality = "ghost"`
- **Exclude ghosts from K-means clustering** (Pass 3C)

**Validation (BLOCKING):**
- FAIL IF: Ghost count unreasonable (> 6 ghosts)
- FAIL IF: Tracked + ghosts > 12

### Pass 3A: Identity Candidate Generation
**Input**: `pass2_ghosts.json`
**Output**: `pass3_candidates.json`

**Responsibilities:**
- For each fragment, generate:
  - Possible teams with scores
  - Possible jerseys with probabilities
  - Possible adjacency links (track continuity)
- **NO LOCKING** (candidates only, no decisions)

### Pass 3B: Constraint Graph Construction
**Input**: `pass3_candidates.json`
**Output**: `pass3_constraints.json`

**Responsibilities:**
- Build constraint graph (nodes = fragments, edges = constraints)
- Constraint types:
  - **MUST_SAME**: Track adjacency + ghost continuity (hard identity constraint)
  - **CANNOT_SAME**: Jersey temporal exclusivity violations (hard exclusion)
  - **SOFT_SAME**: Track continuity preferences (soft preference)
- **Note**: Team assignment happens in Pass 3C AFTER identity resolution

### Pass 3C: Identity Commit (🔒 LOCK POINT)
**Input**: `pass3_constraints.json`, `pass2_ghosts.json`
**Output**: `pass3_identity_commit.json`, `pass3_validation.json`

**This is the ONLY place identity is decided.**

**Algorithm:**
1. **Resolve identity** using MUST_SAME constraints (track adjacency, ghost continuity)
2. **Assign teams** via K-means clustering on resolved identities (exclude ghosts)
3. **Lock teams immediately** via `_locked_team` field (immutable source of truth)
4. **Apply jersey inheritance** (bidirectional with temporal exclusivity check)
5. **Validate CANNOT_SAME constraints** (temporal conflicts)
6. **Optimize SOFT_SAME constraints** (track continuity)
7. **FAIL-FAST if unresolved conflicts**

**Critical Helper Functions:**
- `_assign_and_lock_teams(fragments)`: K-means team assignment (AFTER identity resolution)
- `_apply_jersey_inheritance(graph, fragments, assignments)`: Bidirectional propagation
- `_jersey_available(jersey, fragment, assignments)`: Temporal exclusivity check

**Output Format:**
```json
{
  "identities": [
    {
      "fragment_id": "F000001",
      "player_id": "P07_team_a",
      "team": "team_a",
      "jersey_number": 7,
      "assignment_method": "kmeans",
      "assignment_confidence": 0.95
    }
  ]
}
```

**Validation (BLOCKING):**
- FAIL IF: Any `team = "unknown"`
- FAIL IF: Player switches team
- FAIL IF: Jersey overlaps in time
- FAIL IF: Team balance violated (not 5-7 players per team)
- FAIL IF: Ghost lacks inherited identity

### Ball Interpolation
**Input**: `pass1_raw.json`
**Output**: `ball_interpolation.json`

**Responsibilities:**
- Detect gaps ≤ 30 frames
- Interpolate positions (linear or Kalman)
- Flag interpolated frames
- **FAIL IF**: Ball missing when gap ≤ threshold
- **FAIL IF**: Speed exceeds physical limit

### Visualization
**Input**: `pass3_identity_commit.json`, `ball_interpolation.json`
**Output**: `visualization.mp4`

**Responsibilities:**
- Color bboxes by player_id
- Overlay jersey numbers
- Render ghosts as dashed bboxes
- Render ball (solid = real, dashed = interpolated, no circle = out of play)
- **Deduplication**: Only suppress SAME track_id (different tracks can overlap)

**Visualizer Isolation (CRITICAL - Prevents Truth Contamination):**
- **Visualizer may ONLY consume** `CommittedIdentity` + `ball_interpolation.json`
- **Visualizer CANNOT** infer, fix, suppress, or merge entities
- **Any visual inconsistency** (e.g., overlapping players, missing jerseys) MUST be solved upstream (Pass 1-3C)
- **The video is derived from committed identity, NEVER vice versa**
- **No state leaks**: Visualization logic cannot influence Pass 1-3C decisions

---

## 6. AUTOMATED DEBUGGING (NO HUMAN EYES)

All debugging must be automated via `debug_metrics.json`.

**Required Metrics:**
- Per-frame player count
- Per-frame team counts (team_a, team_b)
- Per-frame jersey conflicts
- Ghost count per frame
- Identity changes (MUST be zero after Pass 3C)

**Stored in**: `debug_metrics.json`

**Visualization is verification only** - the video must be derived from committed identity.

---

## 7. FAILURE POLICY

**Validation Order (CRITICAL):**
- **Validation is run BEFORE writing JSON**
- Failed pass writes NOTHING, not even partial artifacts
- This prevents corrupted downstream state
- Validation JSON is the ONLY file written on failure

**On ANY validation failure:**
1. Stop immediately
2. Write validation JSON with error details
3. Do NOT write pass output JSON
4. Do NOT write later artifacts
5. Exit with non-zero code

**Fail-Fast Points:**

| Pass   | Failure Trigger                                  | Action              |
|--------|--------------------------------------------------|---------------------|
| Pass 1 | Huge bbox detected                               | Filter immediately  |
| Pass 1 | Invalid jersey probability                       | Reject detection    |
| Pass 1 | Bbox out of frame                                | Reject detection    |
| Pass 2 | Fragment overlap                                 | Halt pipeline       |
| Pass 2 | Frame coverage < 100%                            | Halt pipeline       |
| Pass 2 | Ghost count unreasonable                         | Halt pipeline       |
| Pass 3 | `team = "unknown"` after Pass 3C                 | Halt pipeline       |
| Pass 3 | Jersey temporal conflict                         | Halt pipeline       |
| Pass 3 | Unresolved constraint                            | Halt pipeline       |
| Ball   | Gap > threshold, not marked "out of play"        | Halt pipeline       |

---

## 8. CONFIGURATION (MANDATORY VALUES)

**Detection Thresholds:**
- `PLAYER_CONF_THRESHOLD = 0.5`
- `BALL_CONF_THRESHOLD = 0.3`
- `JERSEY_CONF_THRESHOLD = 0.3` (lower per memory learnings)

**Bbox Filtering (Multi-Layer Defense):**
- `MAX_BBOX_HEIGHT_PX = 800`
- `MAX_BBOX_WIDTH_PX = 600`
- `MAX_BBOX_AREA_FRACTION = 0.25`

**ByteTrack Parameters:**
- `TRACK_HIGH_THRESH = 0.6`
- `TRACK_LOW_THRESH = 0.1`
- `TRACK_BUFFER = 30`
- `MIN_TRACK_LENGTH = 5`

**Fragment Parameters:**
- `MIN_FRAGMENT_LENGTH = 10` (but keep shorter ones, mark as low_quality)
- `MAX_FRAGMENT_GAP = 60` (ghost duration)
- `MERGE_CONSECUTIVE_SHORT = True`

**HSV Clustering:**
- `KMEANS_N_CLUSTERS = 2` (team_a vs team_b)
- `HSV_BINS = 8` (8x8x8 = 512 bins)

**Jersey Temporal Exclusivity:**
- `JERSEY_NUMBERS = [1, 2, 3, ..., 12]`
- `MAX_CONCURRENT_PLAYERS = 12`

**Ghost Parameters:**
- `INITIAL_LEVEL_FRAMES = 10` (frames to establish initial level)
- `DYNAMIC_LEVEL_MAX = 12` (cap for futsal)

**Ball Interpolation:**
- `MAX_BALL_GAP_FRAMES = 30`
- `BALL_INTERPOLATION_METHOD = "linear"` (or "kalman")

**Validation Tolerances:**
- `MAX_UNKNOWN_FRAGMENTS = 0` (R2: Every player has a team)
- `MAX_TEAM_SIZE_VIOLATION_FRAMES = 5` (allow brief violations)
- `MAX_CONCURRENT_JERSEY_VIOLATIONS = 0` (R3: One jersey = one player)

---

## 9. CRITICAL IMPLEMENTATION PATTERNS

### Team Assignment Locking (Pass 3C)

```python
# CRITICAL: Team assignment happens AFTER identity resolution
def _assign_and_lock_teams(self, fragments):
    """
    Assign teams via K-means, then lock immediately.
    This happens AFTER identity is resolved via MUST_SAME constraints.
    """
    # K-means clustering (exclude ghosts!)
    real_fragments = [f for f in fragments if not f.is_ghost]
    team_assignments = kmeans_clustering(real_fragments)

    for frag_id, team in team_assignments.items():
        fragments[frag_id]._locked_team = team  # LOCK
        assignments[frag_id] = (team, None)

    return assignments

# ALWAYS use _locked_team, NEVER use team field directly
def _apply_jersey_inheritance(self, graph, fragments, assignments):
    for frag in fragments:
        team = frag._locked_team  # Single source of truth
        # ... inheritance logic
```

### Jersey Temporal Exclusivity Check (Pass 3C)

```python
def _jersey_available(self, jersey: int, target_fragment: Fragment,
                      assignments: Dict, fragments: List[Fragment]) -> bool:
    """
    Check if jersey is available for target_fragment.
    Returns True if NO other fragment has this jersey during target's time range.
    """
    target_frames = set(range(target_fragment.start_frame, target_fragment.end_frame + 1))

    for frag_id, (team, assigned_jersey) in assignments.items():
        if assigned_jersey == jersey and frag_id != target_fragment.fragment_id:
            frag = next(f for f in fragments if f.fragment_id == frag_id)
            frag_frames = set(range(frag.start_frame, frag.end_frame + 1))

            if target_frames & frag_frames:  # Overlap detected
                return False  # Jersey already in use

    return True
```

### Ghost Exclusion from K-Means (Pass 3C)

```python
# CRITICAL: Exclude ghosts from K-means clustering
real_fragments = [f for f in fragments if not f.get('is_ghost', False)]
team_assignments = kmeans_clustering(real_fragments)

# Also exclude ghosts from team size validation
real_identities = [i for i in identities if not fragments[i.fragment_id].is_ghost]
team_a_count = len([i for i in real_identities if i.team == TeamID.TEAM_A])
```

### Ghost Visualization Deduplication

```python
# CRITICAL: Only suppress if SAME track_id
def _dedupe_overlapping_annotations(self, annotations):
    filtered = []
    for i, ann_a in enumerate(annotations):
        suppress = False
        for j, ann_b in enumerate(annotations):
            if i == j:
                continue

            # Only suppress if SAME track_id (duplicate detection)
            # Different tracks can occupy same space (ghost + occluder)
            if ann_a.track_id != ann_b.track_id:
                continue

            if iou(ann_a.bbox, ann_b.bbox) > 0.55:
                if ann_a.rank < ann_b.rank:
                    suppress = True
                    break

        if not suppress:
            filtered.append(ann_a)

    return filtered
```

---

## 10. VALIDATION RULES (EXECUTABLE CODE)

All rules must be implemented as executable checks in `src/validation/`.

### Global Rules (R1-R5)

**R1_Pass1RawTruthOnly**:
- No `team` field in Pass 1 detections
- No `player_id` field in Pass 1 detections
- Only raw observations (bbox, track_id, jersey probabilities, HSV histograms)

**R2_NoUnknownTeams**:
- After Pass 3C, no fragment has `team = "unknown"`
- All fragments have `team = "team_a"` OR `team = "team_b"`

**R3_JerseyTemporalExclusivity**:
- Build timeline: frame → jersey → fragment_id
- FAIL IF: Any frame has same jersey on >1 fragment

**R4_PlayerContinuity**:
- Build timeline: frame → player_ids
- FAIL IF: Any frame has >12 concurrent players

**R5_BallNeverDisappears**:
- FAIL IF: Any frame missing ball position (real or interpolated)

### Pass-Specific Rules

**Pass 1 Rules**:
- Bbox within frame bounds
- Bbox size reasonable (multi-layer defense)
- Jersey probabilities valid
- No duplicate track_id per frame

**Pass 2 Rules**:
- 100% frame coverage (no gaps)
- No fragment overlap
- Ghost count ≤ 6
- Tracked + ghosts ≤ 12

**Pass 3 Rules**:
- Team balance (5-7 players per team)
- Jersey assignment complete
- No team switches
- Constraint satisfaction complete

**Ball Rules**:
- Gap interpolation ≤ 30 frames
- Speed within physical limits
- 100% frame coverage

---

## 11. MODEL PATHS

**Player Detector**: `models/PLAYER_MODEL_best_v1.pt`
**Ball Detector**: `models/BALL_MODEL_best_v2.pt`
**Jersey Classifier**: `models/JERSEY_MODEL_best_v1.pt`

---

## 12. REGRESSION TEST

**Permanent regression test**: `videos/input/GoPro_Futsal_part1_CLEANED_clip9.mp4`

**Expected outcomes**:
- No huge bboxes (Track 14-style hallucinations filtered at Pass 1)
- Zero `team = "unknown"` after Pass 3C
- Zero jersey temporal conflicts
- At most 12 concurrent players per frame
- Ball present at every frame (real or interpolated)
- All validation reports show `passed: true`

---

## FINAL INSTRUCTION

If at any point:
- A rule cannot be satisfied
- Identity cannot be resolved
- Constraints conflict

Then the correct behavior is:

**FAIL FAST AND EXPLAIN WHY.**

Do not guess.
Do not patch.
Do not continue.

---

**This contract is law. All code must comply.**
