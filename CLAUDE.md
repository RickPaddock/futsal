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

## 0.5 Mandatory Pre-Change Gate (REQUIRED)

Before making ANY code or contract change, the following questions MUST be answered
in the terminal or PR description. If any answer is unclear, the change must not proceed.

1. **Is this change fixing the issue at its source, or papering over a downstream symptom?**
   - If downstream: STOP. Re-evaluate.

2. **Which entities are involved in this change?**
   - Explicitly list which of the following are touched:
     players, tracks, fragments, ghosts, frames, splits
   - State the layer for each (identity vs presence).

3. **Which invariant(s) does this change affect?**
   - Must reference invariant IDs (e.g. R4).
   - State whether enforcement is strengthened, weakened, or unchanged.

4. **If this change is wrong, how does it fail?**
   - Loud (validation error)
   - Silent but bounded
   - Silent and propagating (NOT ACCEPTABLE)

5. **What prevents this same class of bug from reappearing?**
   - New validation rule
   - Tightened contract language
   - Explicit non-goal documented

**This is process, not documentation. Any change that bypasses this gate is invalid.**

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

### R4 — Players Never Disappear (ZERO TOLERANCE)

**HARD INVARIANT (Pass 2C Presence Layer)**: At every frame: `tracked_count + ghost_count >= level`

`level` is the dynamic high-water mark. Any deficit (`< level`) is a hard failure.
Overages (`> level`) are treated as unresolved identity collisions and are reconciled in Pass 3.

**Rules**:
1. **Dynamic Level (High Water Mark)**:
   - Initialize from first 10 frames
   - Increase when more players enter (10 → 11 → 12)
   - NEVER decrease (even if players go off-screen)
   - Cap at 12 (futsal regulation)

2. **Ghost Creation (Presence Layer)**:
   - Create ghosts when `real_count < level`
   - Track players by `original_track_id` (NOT `fragment_id`)
   - Fragment splits do NOT trigger ghost creation (identity ≠ presence)
   - Ghost position: HOLD last known position (no interpolation)

3. **Ghost Chaining (CRITICAL - Zero Tolerance)**:
   - When ghost expires (60 frames), check if player reappeared
   - If NOT reappeared: Create new ghost IMMEDIATELY (chain)
   - Chains continue INDEFINITELY until player reappears OR clip ends
   - Gap > 60 frames means "keep chaining", NOT "give up"
   - **ZERO TOLERANCE**: Even 1 frame with count < level is a HARD FAILURE

4. **Validation Enforcement**:
  - FAIL if ANY frame has `tracked_count + ghost_count < level`
   - FAIL if ghost expires without reappearance AND no chain created
   - FAIL if presence gap exists (even 1 frame)
  - If ANY frame has `tracked_count + ghost_count > 12`: record as identity-collision signal for Pass 3
    (non-blocking in Pass 2C, blocking after Pass 3 identity resolution)
   - No tolerance, no exceptions (except clip boundaries)

**Rationale**:
- Futsal has no substitutions (players enter, never leave)
- Off-screen players return (brief edges, not permanent exits)
- Maintaining count invariant enables downstream identity reasoning
- Presence gaps indicate broken ghost chaining, not "acceptable loss"

**Examples**:
- Partial clip: Starts with 8 players → level = 8, maintains 8 throughout
- Players entering: Frame 0-100 level = 10, frame 101+ level = 11 (new player), maintains 11
- Long occlusion: Player missing 200 frames → 3-4 chained ghosts, zero gaps

### R5 — Ball State Exists at Every Frame
- **HARD rule**: Ball state exists at every frame
- **Ball state** ∈ {`real`, `interpolated`, `out_of_play`}:
  - **`real`**: YOLO detection with bbox and confidence
  - **`interpolated`**: Gap ≤ 30 frames, position interpolated (linear or Kalman)
  - **`out_of_play`**: Gap > 30 frames, ball not on court (no position)
- **Validator checks**: Presence of ball state, NOT presence of ball position
- **Out-of-play frames**: Ball may have no position, but MUST have state = `out_of_play`

### 3.5 Fragment vs Ghost Contract (Identity vs Presence Layers)

**CRITICAL DISTINCTION**: Fragments and ghosts serve different purposes and MUST NOT be conflated.

#### Fragment Contract (Identity Layer)
**Purpose**: Represent identity continuity across detection gaps.

**Rules**:
- Fragments span detection gaps caused by:
  - Occlusion (player temporarily hidden)
  - Low confidence (detector missed player)
  - Brief off-screen (player exits frame edge)
- Per Pass 2A: "No fragmentation caused by visibility loss" (line 323)
- Fragments maintain identity through gaps (track_id preserved)
- A fragment can have ZERO detections in some frames (gap), but OWNS those frames for identity purposes

**Example**: Fragment F000001 spans frames 10-50. Pass 1 detections exist at frames 10-15, 25-30, 40-50. Frames 16-24 and 31-39 have NO detections but fragment OWNS them (identity layer). Ghosts fill presence gaps at frames 16-24, 31-39 (presence layer).

**CRITICAL**: Fragments may span frames with no detections, but do NOT imply player presence; presence is satisfied exclusively by real detections or ghosts.

#### Ghost Contract (Presence Layer)
**Purpose**: Maintain R4 presence completeness (tracked + ghosts >= level at EVERY frame).

**Rules**:
- Ghosts created when `real_count < level` (NOT when fragments end)
- Ghosts track players by `original_track_id` (NOT by `fragment_id`)
- Pass 2C does NOT resolve identity across different `track_id` values
- **Ghost Chaining (CRITICAL)**:
  - When a ghost expires, check if player reappeared
  - If player NOT reappeared: create new ghost immediately (chain)
  - Chains continue indefinitely until player reappears OR clip ends
  - Gap > 60 frames does NOT mean "give up" - it means "keep chaining"
- Ghost position: HOLD last known position (no interpolation)
- Ghost metadata: Preserves team, jersey from source fragment

**Layering Guardrail (MANDATORY):**
- Pass 2C MUST NOT terminate ghosts based on spatial similarity to a different track
- Pass 2C MUST NOT merge track IDs
- Pass 2C MUST NOT suppress or kill ghosts due to global player count
- Cross-track reconciliation belongs ONLY to Pass 3

**Ghost Chaining Example**:
- Frame 88: Player T12 last detected
- Frames 89-148: Ghost G000003 active (60 frames)
- Frame 149: Ghost expires, check for reappearance
- Player NOT reappeared → Create Ghost G000004 (chain)
- Frames 149-208: Ghost G000004 active (60 frames)
- Frame 209: Ghost expires, check for reappearance
- Player STILL not reappeared → Create Ghost G000005 (chain)
- ... chains continue until player reappears at frame 300 or clip ends

**Why Separate**:
- Fragment gaps (identity) ≠ presence gaps (player count)
- Fragment splits when identity changes, NOT when visibility lost
- Ghosts maintain count invariant, NOT identity tracking
- Allows fragment to own time range while ghost fills detection gap

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
- Physical plausibility filter: cap player detections to 12 per frame (keep best-12, log warning)
- **NO teams, NO identity, NO merges/splits beyond pathological filters**

**Validation (BLOCKING):**
- FAIL IF: bbox out of frame
- FAIL IF: bbox > 25% frame area (huge bbox defense)
- FAIL IF: mandatory fields missing or structurally invalid
- FAIL IF: duplicate track_id in same frame

**Validation (DIAGNOSTIC / NON-BLOCKING):**
- WARN IF: detector/classifier quality inconsistencies (jersey ROI quality, HSV consistency, low confidence)
- WARN IF: tracker churn diagnostics (short-lived tracks, noisy continuity)

### Pass 2A: Mechanical Fragmentation (HARD CONTRACT)

**Input**: `pass1_raw.json`  
**Output**: `pass2_fragments.json`, `pass2_validation.json`


## Purpose (NON-NEGOTIABLE)
Pass 2A exists **only** to detect points where a tracker has *provably jumped from one physical player to another*.

It **must not**:
- Infer identity
- Infer teams
- Penalise low visibility
- Repair tracking errors

Loss of observability ≠ identity change.

## Responsibility Boundary (HARD LIMIT)

Pass 2A exists solely to detect and split tracker identity jumps — where the same track ID has provably switched from one physical player to another.

**It must not:**
- Split for jersey ambiguity that cannot be linked to a specific tracker jump
- Split to make global jersey uniqueness easier for Pass 3
- Split to help Pass 3 with identity assignment
- Guarantee jersey-resolvable fragments

**Absence of a split does NOT imply identity correctness.**

Global identity feasibility (≤12 players, jersey uniqueness) is enforced ONLY in Pass 3.
Pass 2A may produce fragments that are identity-ambiguous but presence-correct.

---

## Responsibilities (STRICT)
- Detect **divergence points only**
- Split a track into fragments **only when there is hard evidence of identity discontinuity**
- Preserve *all* evidence for downstream passes
- Never “fix” or smooth tracking artefacts

---

## Allowed Split Triggers (EXHAUSTIVE)

A split **MUST occur** if and only if one of the following is true:

1. **Track Collision (ByteTrack failure)**  
   - Same `track_id` produces >1 detection in the same frame  
   → **Immediate split**

2. **Jersey Change (Hard Identity Proof)**  
   - Jersey value changes from one valid number to another  
     - Example: `#7 → #4`  
   → **Split**

3. **Jersey Temporal Exclusivity Violation**  
   - Same jersey number visible on two different tracks in the same frame window  
   → **Split at earliest contradiction point**

4. **Hard Appearance Discontinuity (Guarded)**  
   All conditions must hold:
   - Jersey visible on both sides of the boundary
   - Large HSV distance beyond threshold
   - Incompatible motion (teleport / impossible velocity)  
   → **Split**

---

## Explicit Non-Triggers (MUST NOT SPLIT)

The following **must never cause a split**:

- Jersey disappearance (`#4 → None`)
- Jersey first appearance (`None → #4`)
- Occlusion
- Confidence drops
- Missed detections
- Short gaps
- Spatial crossings without identity contradiction
- Normal appearance drift
- Player turning away from camera

These are **loss-of-observability signals**, not identity changes.

They must be recorded as fragment metadata only.

---

## Fragment Handling Rules

- **All fragments must be kept**, regardless of length
- Fragments < `MIN_FRAGMENT_FRAMES`:
  - Mark as `quality = "low"`
  - Do **not** discard
- **Do NOT merge fragments**
  - Even if consecutive
  - Even if same track_id  
  Fragment boundaries are ground truth evidence and must remain intact

---

## Metadata to Record per Fragment
- `track_id`
- `start_frame`, `end_frame`
- `jersey_visible_ratio`
- `occlusion_ratio`
- `mean_velocity`
- `appearance_stability_score`
- `quality` (`high | low`)

No identity labels allowed.

---

## Validation (BLOCKING — MUST FAIL PIPELINE)

Pass 2A must fail if **any** of the following are true:

- Fragment coverage < **100%** of Pass 1 detections
- Any frame belongs to >1 fragment
- Any detection belongs to no fragment
- Fragment time ranges overlap
- A split occurs without a logged trigger reason
- Player detections + ghosts > 12 in any frame

---

## Output Guarantees
- Fragment boundaries correspond **only** to provable identity discontinuities
- No fragmentation caused by visibility loss
- All downstream identity errors are traceable to real upstream evidence


### Pass 2B: Fragment Quality Scoring
**Input**: `pass2_fragments.json`
**Output**: `pass2_fragments.json` (with quality scores)

**Responsibilities:**
- Compute metadata ONLY (no identity assignment)
- Binary classification ONLY: `real` or `occlusion_candidate`
- Fragment scores:
  - Appearance stability
  - Jersey observability
  - Motion smoothness
  - Occlusion ratio
- Quality tiers (HIGH/MEDIUM/LOW) are metadata only and non-binding

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
- **Ghost duration**: 60 frames per ghost, then chain if player not reappeared
- **Ghost chaining**: When ghost expires, create new ghost if player still missing
- **Ghost termination**: ONLY when player reappears (new detection) OR clip ends
- **Identity scope**: same-track continuity only (`original_track_id`); no cross-track matching
- Mark ghosts: `is_ghost = True`, `quality = "ghost"`
- **Exclude ghosts from K-means clustering** (Pass 3C)

**Validation (BLOCKING):**
- FAIL IF: Ghost count unreasonable (> 6 ghosts)
- FAIL IF: Tracked + ghosts < level (R4 zero tolerance)
- FAIL IF: Ghost expires without reappearance and no chain created
- FAIL IF: Per-track lifespan continuity breaks (gap or overlap for same `original_track_id`)

**Validation (NON-BLOCKING DIAGNOSTIC):**
- WARN IF: Tracked + ghosts > 12 (identity collision; deferred to Pass 3)

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
  - Team assignment uses a single K=2 clustering stage only
  - No secondary/sub-clustering is allowed for team allocation
  - Any optional subcluster analysis is diagnostic-only and must not change team assignment
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

**Pass 3 Debug Visualization Convention (readability):**
- Bibbed team bbox color: ORANGE
- Other team bbox color: BLACK
- Text overlays: WHITE on BLACK background
- Jersey number shown as large `#number` at bottom of bbox

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
| Pass 2 | Tracked + ghosts > 12                            | Log warning (defer to Pass 3) |
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

### Validation Responsibility by Pass

- **Pass 1**: perceptual sanity only; never enforce physical or identity invariants.
- **Pass 2A–2C**: identity-local continuity only; may overcount players.
- **Pass 2 warnings**: evidence of ambiguity, not failure.
- **Pass 3**: first stage allowed to enforce physical reality (`<= level`, unique humans).
- **Any attempt to enforce `<= level` before Pass 3 is a validation layering bug.**
- **Do not fix ghost logic further to hide overcount; fix validator-contract mismatches and defer identity reconciliation to Pass 3.**

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
- FAIL IF: Any frame has presence deficit vs level (`tracked + ghosts < level`)
- NOTE: `tracked + ghosts > 12` at Pass 2 is an identity-collision diagnostic (deferred to Pass 3)

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
- Presence deficit forbidden: `tracked + ghosts < level`
- Physical max exceedance (`>12`) is recorded as warning and deferred to Pass 3 identity resolution

**Pass 3 Rules**:
- Team balance (5-7 players per team)
- Jersey assignment complete
- No team switches
- Constraint satisfaction complete
- Physical cap enforcement (`<= 12` identities per frame)
- Ghost retirement after real match
- Global R4 exact-level enforcement after reconciliation

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
