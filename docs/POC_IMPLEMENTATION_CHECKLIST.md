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

#### **Step 3.2: Intra-Team Appearance Sub-Clustering (Grey Collapse Fix)** 🔴 P0
- [ ] Implement `discover_appearance_modes()` function in [src/passes/pass3_identity.py](../src/passes/pass3_identity.py)
  - [ ] **AUTOMATIC detection**: Calculate variance for BOTH teams after K-means
  - [ ] Only sub-cluster teams with variance > threshold (works for team_a, team_b, or both)
  - [ ] Use MiniBatchKMeans (k ≤ 3) to discover 2-3 modes per high-variance team
  - [ ] Assign each fragment to nearest mode centroid
  - [ ] Export mode assignments to JSON
- [ ] Call AFTER initial K-means team assignment, AFTER track continuity enforcement
- [ ] Add config parameters to [config/default.yaml](../config/default.yaml)
  - [ ] `appearance_mode_variance_threshold: 15.0` (auto-detect which teams need it)
  - [ ] `max_modes_per_team: 3`
- [ ] Export `team_appearance_mode`, `mode_confidence` in JSON
- [ ] Test on sample video

**Estimated Time:** 2-3 hours
**Files Modified:**
- `src/passes/pass3_identity.py` - Add intra-team clustering function
- `config/default.yaml` - Add variance threshold config

**Algorithm:**
1. After K-means assigns team_a and team_b
2. Calculate HSV variance for EACH team
3. If team variance > threshold → run sub-clustering on that team
4. Prevents "grey centroid" problem for mixed-shirt teams

**Test Results:**
```
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
