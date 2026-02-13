# PoC Implementation Checklist - Phase 0

**Status:** 🟡 In Progress
**Started:** 2026-02-11
**Current Task:** Step 0.1 - Testing
**Last Updated:** 2026-02-11

---

## Phase 0: P0 Critical Fixes (2-3 days)

**Goal:** Fix breaking issues - team size violations, occlusion, ball tracking, grey collapse

### ✅ Completed Tasks

#### **Step 0.1: Enforce Team Size Constraint** 🔴 P0 ✅
- [x] Read [src/passes/pass3_identity.py](../src/passes/pass3_identity.py) to understand current validation
- [x] Implement `enforce_team_size_constraint()` function
  - [x] Count concurrent fragments per team per frame
  - [x] Find lowest-confidence violators
  - [x] Mark as "unknown" (NEVER auto-flip to other team)
  - [x] Add violation metadata (`team_constraint_violation`, `violation_reason`)
- [x] Call constraint enforcement FIRST in Pass 3 (before smoothing/inheritance)
- [x] Export violation fields in JSON output

**Files Modified:**
- `src/passes/pass3_identity.py` - Added enforcement function, called before validation

**Implementation Details:**
- Created `enforce_team_size_constraint()` at line 24-85
  - Marks lowest-confidence violators as "unknown" (never auto-flips)
- Created `inherit_team_assignments()` at line 91-183
  - Bidirectional team inheritance (forward + backward passes)
  - "Backdates" team assignments to fill unknown gaps
  - Respects team size constraint during inheritance
- Called at lines 349-360 (right after K-means, BEFORE jersey inference)
- Exports metadata: `team_constraint_violation`, `violation_reason`, `team_confidence`, `team_inherited`, `inherited_from`

**Key Fix:**
Players can't be "unknown" - if a fragment is marked unknown due to constraint violations, but adjacent fragments on the same track have consistent team assignments, the team is inherited backward/forward to fill the gap.

**Ready for Testing:** ✅

---

### 🔄 Current Task

**→ Step 0.1: Test on sample video**

---

### 📋 Pending Tasks

**Step 0.1: Enforce Team Size Constraint** - Testing Phase
- [ ] Test on sample video
- [ ] Verify no team size violations (target: 0)
- [ ] Verify violations properly logged in JSON
- [ ] Verify unknown fragments marked correctly

**Test Results:**
```
[ ] No team size violations after enforcement
[ ] Violations properly logged in JSON with violation_reason
[ ] Unknown fragments marked with team="unknown"
```

**Notes:**
-

---

#### **Step 1.1: Extend Track Buffers + Ghost Tracks** 🔴 P0
- [ ] Read [src/detection/tracking.py](../src/detection/tracking.py) to understand ByteTrack implementation
- [ ] Update track buffer constants
  - [ ] `TRACK_BUFFER_DEFAULT = 60` (2 seconds at 30 FPS)
  - [ ] `TRACK_BUFFER_NUMBERED = 90` (3 seconds for numbered players)
- [ ] Implement `get_track_buffer()` for adaptive buffers
- [ ] Implement `emit_ghost_track()` for predicted positions
  - [ ] Set `occluded=True`, `predicted=True`, `confidence=0.0`
- [ ] Update [src/passes/pass2_geometry.py](../src/passes/pass2_geometry.py) to handle ghost tracks
  - [ ] Ghost tracks extend existing fragments only
  - [ ] Ghost tracks NEVER create new fragments
- [ ] Test on sample video

**Estimated Time:** 3-4 hours
**Files Modified:**
- `src/detection/tracking.py`
- `src/passes/pass2_geometry.py`

**Test Results:**
```
[ ] Players maintain IDs through 30-90 frame occlusions
[ ] Ghost tracks visible in JSON with predicted=true
[ ] No fragment explosion during occlusions
```

**Notes:**
-

---

#### **Step 3.2: Intra-Team Appearance Sub-Clustering (Grey Collapse Fix)** 🔴 P0 ✅
- [x] Lock team assignments immediately after K-means (frag.team_locked = frag.team)
- [x] Implement `_calculate_team_variances()` function
  - [x] Compute intra-team appearance variance per team independently
  - [x] Uses mean Euclidean distance to team centroid (colour-agnostic metric)
- [x] Implement `_detect_multi_appearance_teams()` function
  - [x] **RELATIVE variance detection**: Only auto-detect based on ratio comparison
  - [x] Decision rule: if var(team_i) > ratio_threshold × min(other_vars) → mark as multi-appearance
  - [x] Hard constraint: No absolute thresholds, no team reassignments
- [x] Implement `_subcluster_appearance_modes()` function
  - [x] **METADATA-ONLY**: Adds appearance_mode_id and appearance_mode_confidence
  - [x] Uses MiniBatchKMeans (k ≤ 3) to discover 2-3 modes per high-variance team
  - [x] STRICT RULE: Never modifies frag.team, timelines, or size constraints
- [x] Add config parameters to [config/default.yaml](../config/default.yaml)
  - [x] `appearance_variance_ratio_threshold: 1.5` (relative variance threshold, not absolute)
  - [x] `max_appearance_modes: 3` (hard cap on k for MiniBatchKMeans)
- [x] Export appearance modes to JSON output
  - [x] Per-fragment: appearance_mode_id, appearance_mode_confidence
  - [x] Global metadata: team_a_variance, team_b_variance, multi_appearance_teams
- [x] Wire into process_clip_pass3() with correct call order
  - [x] Step 1: Run existing K-means (unchanged)
  - [x] Step 2: Lock team assignments explicitly
  - [x] Step 3: Calculate team variances
  - [x] Step 4: Detect multi-appearance teams
  - [x] Step 5: Sub-cluster high-variance teams only
  - [x] Step 6: Export metadata (no feedback loops)

**Estimated Time:** 2-3 hours ✅
**Files Modified:**
- `src/passes/pass3_identity.py` - Added 3 functions + integration into process_clip_pass3()
- `config/default.yaml` - Added variance threshold + max_modes parameters

**Implementation Details:**

**Functions Added:**
1. `_calculate_team_variances(fragments) -> Dict[str, float]`
   - Computes intra-team variance independently for each team
   - Metric: mean distance to team centroid across HSV histogram
   
2. `_detect_multi_appearance_teams(team_variances, ratio_threshold) -> Set[str]`
   - Detects high-variance teams using RELATIVE variance only
   - Rule: var(team_i) > ratio_threshold × min(other_vars)
   - Config: appearance_variance_ratio_threshold = 1.5
   
3. `_subcluster_appearance_modes(fragments, team_id, max_modes)`
   - Runs MiniBatchKMeans on high-variance teams only
   - Adds appearance_mode_id and appearance_mode_confidence metadata
   - ZERO feedback: Never touches team assignment or constraints

**Key Safety Guarantees:**
- ✅ One-way data flow (no recursion, no back-edges)
- ✅ Team assignments frozen after K-means (team_locked = team)
- ✅ Appearance modes are informational only (no decision making)
- ✅ Deterministic: Same input = same output
- ✅ Revertible: Can be deleted without side effects (~150 lines)

**Algorithm:**
1. After K-means assigns team_a and team_b → Lock teams
2. Calculate per-team variance independently
3. Detect if var(team_i) > 1.5 × min(var(other_team))
4. If multi-appearance: Run MiniBatchKMeans(k=2-3) within that team only
5. Assign appearance_mode_id metadata (no team change)
6. Export to JSON for debugging/future use

**Test Results:**
```
[ ] No team reassignments during appearance sub-clustering
[ ] Appearance modes visible in JSON output
[ ] Multi-appearance teams correctly identified
[ ] Mixed-shirt teams show 2-3 distinct modes
[ ] Bibbed teams show single mode
[ ] No downstream logic affected (metadata-only)
```

**Notes:**
- CRITICAL: appearance_mode_id is metadata only. No downstream logic depends on it yet.
- Next phase: Use modes for fallback jersey assignment or divergence detection
- Can be extended later without risk (zero feedback loops)

---

#### **Step 2.1: Re-Enable Ball Tracker** 🔴 P0
- [ ] Read [src/detection/ball_detector.py](../src/detection/ball_detector.py) to find disable flag
- [ ] Remove disable flag at line ~243
- [ ] Verify centroid-based tracker re-enabled
- [ ] Test on sample video

**Estimated Time:** 2-3 hours
**Files Modified:**
- `src/detection/ball_detector.py`

**Test Results:**
```
[ ] Ball present in ≥70% of frames
[ ] Ball trajectory mostly continuous (accept some noise)
[ ] Detection gaps logged
```

**Notes:**
-

---
[ ] Mixed-shirt teams no longer collapse to grey
[ ] Both bibbed and non-bibbed teams handled correctly
[ ] Team assignments stable (no flipping)
[ ] Appearance modes visible in JSON
```

**Notes:**
- CRITICAL: Don't assume team_b is always mixed - auto-detect based on variance
- Could be bibbed vs bibbed, bibbed vs mixed, or mixed vs mixed

---

## Test Commands

```bash
# Run full pipeline on test video
python src/main.py --input data/test_video.mp4 --output output/test_run

# Check for team size violations
python scripts/validate_team_size.py output/test_run/fragments.json

# Visualize results
python src/visualization/annotate_video.py output/test_run
```

---

## Success Metrics (Phase 0)

After completing all Phase 0 tasks:

- [ ] **Team size violations:** 0 (hard constraint)
- [ ] **Player ID persistence:** Players tracked through 30-90 frame occlusions
- [ ] **Ball detection rate:** ≥70% of frames
- [ ] **Team assignment stability:** No grey collapse, mixed teams handled correctly
- [ ] **All decisions logged:** JSON contains confidence scores, violation flags

---

## Phase 1 Tasks (Deferred)

These will be tackled after Phase 0 validation:

- **Step 2.2:** Ball Kalman filter (4 hours) 🟡 P1
- **Step 3.3:** Multi-cue team scoring (1 day) 🟡 P1
- **Step 3.4:** Temporal smoothing (4 hours) 🟡 P1
- **Step 4.2:** Export confidence scores (2 hours) 🟡 P1

---

## Recovery Instructions (If Context Closes)

1. **Check this file** for current status and last completed task
2. **Read the notes** section of the last task for any blockers/issues
3. **Continue with next unchecked box** in the pending tasks
4. **Run tests** after completing each task before moving on

**Plan file:** `C:\Users\rickp\.claude\plans\keen-launching-blum.md`
**Full details:** See plan file for algorithm details, code snippets, and architectural rules
