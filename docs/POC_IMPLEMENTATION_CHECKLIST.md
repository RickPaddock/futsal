# PoC Implementation Checklist - Phase 0

**Status:** 🟡 In Progress
**Started:** 2026-02-11
**Current Task:** Step 0.2 - Fix remaining unknown assignments
**Last Updated:** 2026-02-13

---

## Phase 0: P0 Critical Fixes (2-3 days)

**Goal:** Fix breaking issues - team size violations, occlusion, ball tracking, grey collapse

### ✅ Completed Tasks

#### **Step 0.1: Enforce Team Size Constraint** 🔴 P0 ✅
- [x] Read [src/passes/pass3_identity.py](../src/passes/pass3_identity.py) to understand current validation
- [x] Implement `enforce_team_size_constraint()` function
  - [x] Count concurrent fragments per team per frame
  - [x] Find weakest violators deterministically
  - [x] Mark as "unknown" (NEVER auto-flip to other team)
  - [x] Add violation metadata (`team_constraint_violation`, `violation_reason`, `over_capacity_frame_count`)
- [x] Call constraint enforcement after K-means and before team lock
- [x] Export violation fields in JSON output
- [x] Use `team_clustering.team_cap` from config (fallback 6)

**Files Modified:**
- `src/passes/pass3_identity.py` - Added hard cap enforcement + metadata export

**Implementation Details:**
- Added deterministic team-cap enforcer in Pass 3.
  - If concurrent fragments for a team exceed cap, overflow fragments are demoted to `unknown`.
  - No cross-team reassignment is allowed.
  - Enforcement happens before `team_locked` is written.
- Exports metadata: `team_constraint_violation`, `violation_reason`, `over_capacity_frame_count`, `team_confidence`, `label_source`.

**Key Fix:**
Team cap is now hard-enforced in Pass 3: over-cap fragments are demoted to `unknown` instead of silently allowing >6 players on a team.

**Ready for Testing:** ✅

---

#### **Step 0.1a: Unknown Recovery + Retro Backfill** 🔴 P0 ✅
- [x] Implement `recover_unknown_team_assignments()`
  - [x] Reassign unknown-team fragments when anchor/color evidence is strong
  - [x] Enforce full-span capacity check (`team_cap`) before reassignment
  - [x] Export recovery source via `label_source` (`unknown_recovered_anchor` / `unknown_recovered_color`)
- [x] Implement `_retro_backfill_short_unknown_segments()`
  - [x] Backfill short same-track unknown segments from nearby anchors
  - [x] Preserve cap-enforced unknowns (`label_source=team_cap_enforced` is never backfilled)
- [x] Harden jersey assignment around unknown-team fragments
  - [x] Exclude unknown-team fragments from jersey conflict candidacy
  - [x] Use `unknown_team_no_jersey_assignment` when applicable

**Files Modified:**
- `src/passes/pass3_identity.py` - Unknown recovery + retro team/jersey backfill + jersey candidate hardening
- `config/default.yaml` - Added unknown recovery knobs

**Implementation Details:**
- Added post-cap unknown recovery using color centroid margin/ratio and same-track anchor evidence.
- Added retro backfill for short unknown segments to reduce visible identity lag in JSON/video.
- Preserved fragment-first safety rules by never overriding cap-enforced unknown fragments.

**Ready for Testing:** ✅

---

#### **Step 0.1b: Same-Track Jersey Backpopulation on Split Fragments** 🔴 P0 ✅
- [x] Implement `_retro_backfill_same_track_jerseys()`
  - [x] Back-populate jersey backward on same `original_track_id`
  - [x] Require same known team and empty target jersey
  - [x] Block assignment on same-team temporal jersey conflict
- [x] Wire call after `_retro_backfill_short_unknown_segments()` in Pass 3 flow
- [x] Validate target case: `T6:F000045 -> F000045_split` now back-populates jersey `#4`

**Files Modified:**
- `src/passes/pass3_identity.py` - Added conservative same-track jersey retro-backfill pass

**Ready for Testing:** ✅

---

### 🔄 Current Task

**→ Step 0.2: Fix remaining unknown assignments**

---

### 📋 Pending Tasks

**Step 0.1: Enforce Team Size Constraint** - Validation Status
- [x] Test on sample video
- [x] Verify no residual team size violations after enforcement (target: 0)
- [x] Verify violations properly logged in JSON
- [x] Verify unknown fragments marked correctly

**Validation Snapshot:**
```
[x] No team size violations after enforcement (max concurrent = 6/team)
[x] Violations logged in JSON with violation_reason + over_capacity_frame_count
[x] Unknown fragments marked with team="unknown"
[x] Team cap read from config (team_clustering.team_cap)
```

**Step 0.2: Fix Remaining Unknown Assignments** 🔴 P0
- [x] Implement conservative replacement recovery for cap-saturated windows
  - [x] Allow strong unknown to displace weaker incumbent only when evidence is persistent
  - [x] Keep hard cap invariant (never exceed 6)
  - [x] Preserve jersey single-owner temporal invariant
- [x] Re-run Pass 3 + visualize on latest runs
- [x] Validate frame-window outcomes where unknowns remain (starting with frame 1350 case)
- [x] Add audit metadata for replacement decisions (winner/loser fragment + reason)
- [ ] Reduce residual unknown occupancy in target windows (avoid 1:1 swaps that keep unknown count flat at key frames)
- [x] Prevent aggressive replacement demotions of long-established fragments
  - [x] Added age-delta guardrail (`unknown_replacement_max_age_delta_frames`)
  - [x] Verified `T6@frame117` no longer demoted to unknown in clip15 sanity run
- [x] Stabilize jersey continuity in visualization without changing Pass3 JSON semantics
  - [x] Added micro-gap bridge for short fragment holes
  - [x] Added pre-anchor backfill for short unstable windows before strong anchor
  - [x] Added same-track jersey persistence across split fragments (pause/resume on conflict frames)
  - [x] Enforced frame-level single jersey owner arbitration (strongest claim wins)

**Test Results:**
```
[ ] Remaining unknown fragments reduced to 0 in target windows
[x] No team cap violation regressions
[x] No jersey temporal ownership conflicts introduced
[x] Sanity run clip15: `T6 #4` no longer stolen by `T5`
[x] Sanity run clip15: `T6 #4` continuity preserved through split windows
[x] Clip14: frame-level duplicate jersey display suppressed (single owner per frame)
```

**Notes:**
- Replacement recovery is active and audited in identity JSON (`replacement_role`, `replacement_counterpart_fragment_id`, `replacement_reason`, `replacement_score`).
- Validated case: `frag_000087_split` is recovered to `team_b` as replacement winner; loser is `frag_000069` demoted to `unknown`.
- Current state in target run: unknown fragments still 3 (`frag_000037`, `frag_000069`, `frag_000101`); frame 1350 still has 1 unknown due to replacement swap overlap.
- Jersey visualization now includes deterministic continuity + exclusivity arbitration; this is display-layer only and does not mutate Pass3 identity JSON.

---

#### **Step 1.1: Simplified Ghost Tracking (Player Count Invariant)** 🔴 P0 ✅
- [x] Read [src/passes/pass2_geometry.py](../src/passes/pass2_geometry.py) to understand fragment structure
- [x] Implement `create_ghost_fragments()` function
  - [x] Build per-frame player inventory (PlayerInventory class)
  - [x] Initialize level from first 10 frames (dynamic high-water mark)
  - [x] Create ghosts when visible_count < level
  - [x] Position ghosts at nearest REAL player (occluder heuristic)
  - [x] Close ghosts ONLY on reappearance (NEVER auto-expire)
- [x] Ghost identity inheritance from Pass 3
  - [x] Ghosts inherit team/jersey from overlapping real fragments
  - [x] Exclude ghosts from K-means clustering
  - [x] Post-process identity assignment after Pass 3
- [x] Integrate into `process_clip_pass2()` (after line 680)
- [x] Update visualization
  - [x] Ghosts render as T3 with team colors and jersey numbers
  - [x] Map `is_ghost` fragments to visualization layer
  - [x] Ghost deduplication fixed (only same track_id)
- [x] Test on sample video

**Estimated Time:** 4-5 hours ✅ **Completed: 2026-02-16**

**Files Modified:**
- `src/passes/pass2_geometry.py` - Ghost generation logic with occluder following
- `src/passes/pass3_identity.py` - Ghost identity inheritance + ghost exclusion from jersey backfill
- `src/passes/pass_visualize.py` - Ghost visualization mapping
- `config/default.yaml` - Jersey backfill threshold lowered to 1.5

**Test Results:**
```
[x] Level initialized dynamically (high-water mark, starts from first 10 frames)
[x] Ghosts created only when visible_count < level (not on fragment splits)
[x] Ghosts positioned at nearest REAL player and follow occluder
[x] Ghosts inherit team/jersey from source fragment via Pass 3
[x] Ghosts close ONLY on reappearance (NEVER auto-expire)
[x] Ghosts that never reappear persist to clip end
[x] Ghosts render as T3 (dotted) with team colors and jersey numbers
[x] No "ghost explosion" (reasonable ghost count ~44 for test clip)
[x] Player count invariant maintained: tracked + estimated = level
[x] Ghosts excluded from all aggregation (clustering, voting, stats)
[x] Jersey backfill works across fragment splits (ghosts no longer block backfill)
```

**Implementation Details:**
- **Ghost creation**: Pass 2 creates ghost fragments when player count drops below level
- **Dynamic level**: Increases from 10→11→12 as players enter, never decreases
- **Occluder following**: Ghosts track nearest visible player's position (not static)
- **Identity inheritance**: Pass 3 assigns team/jersey to ghosts from overlapping real fragments
- **Jersey backfill fix**: Ghosts now skipped during backfill, preventing them from blocking same-track jersey propagation

**Critical Fixes Applied:**
1. **Ghost identity inheritance**: Ghosts get team/jersey from overlapping source fragments in Pass 3
2. **Jersey backfill**: Ghosts excluded from backfill algorithm (were blocking jersey propagation)
3. **Occluder following**: Ghosts move with nearest visible player instead of staying static
4. **Deduplication**: Different track_ids never suppress each other (ghost vs real player)

**Future Enhancements (Phase 1+):**
- [ ] **Smooth ghost interpolation**: Transition ghost position smoothly from last-seen → occluder → reappearance
  - Currently: Ghost jumps to occluder position immediately, then jumps to reappearance position
  - Enhancement: Interpolate ghost bbox smoothly through the occlusion period
  - Benefits: Eliminates "jumping" bounding boxes, more visually natural transitions
  - Implementation: Store last-seen position, interpolate to occluder, then to reappearance point
- [ ] **Kalman filter prediction**: Use velocity/trajectory instead of simple occluder position
- [ ] **Multi-player occlusion**: Better handling when multiple players overlap (priority ranking)

**Notes:**
- Simple PoC implemented: no Kalman, basic occluder following
- Ghosts follow occluder position frame-by-frame
- Ghosts NEVER auto-expire - persist until reappearance or clip end
- Level initialized from clip (first 10 frames), grows dynamically
- Occluders are REAL visible players only (never ghosts)
- Track PLAYERS (original_track_id), not FRAGMENTS (fragment_id)
- Pass 2 only - no changes to Pass 1

**Testing command:**
```bash
python -m src.cli pass2 --run-dir videos/output/run_130226_183427
python -m src.cli pass3 --run-dir videos/output/run_130226_183427
python -m src.cli visualize --run-dir videos/output/run_130226_183427 --2d B
```

**Ready for Testing:** ✅ **Validated**

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

#### **Step 2.1: Re-Enable Ball Tracker** 🔴 P0 ⏭️ **SKIPPED**
- [x] Explored centroid-based tracker implementation
- [x] Added safeguards (buffer size guard, confidence/size filters)
- [x] Tested tracker - limited benefit with current YOLO model
- [x] Decision: Skip centroid tracker, defer to Kalman filter (Step 2.2, Phase 1)

**Estimated Time:** 2-3 hours
**Files Modified:**
- `src/detection/ball_detector.py` - Tracker remains disabled

**Outcome:**
```
[x] Centroid tracker provides minimal benefit when YOLO model already returns correct ball 97% of time
[x] Tracker cannot fix missing detections (only filters what YOLO provides)
[x] Kalman filter (Step 2.2) is better solution - predicts position, fills gaps
[x] Current approach: max(confidence) selection - simple and effective
```

**Notes:**
- Centroid tracker re-enabled and tested, but removed after analysis
- YOLO model needs better recall (detect ball more consistently) before spatial tracking helps
- Kalman filter (Phase 1) will provide prediction + gap filling - the real solution

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
- **Step 4.3:** Finalize unknown detections (deferred, not imminent) 🟡 P1
  - Consolidate unknown-handling policy (cap enforcement vs replacement vs recovery)
  - Reduce residual unknowns in hard windows without introducing team/jersey theft
  - Promote validated visualization continuity rules into identity layer only if safe

---

## Recovery Instructions (If Context Closes)

1. **Check this file** for current status and last completed task
2. **Read the notes** section of the last task for any blockers/issues
3. **Continue with next unchecked box** in the pending tasks
4. **Run tests** after completing each task before moving on

**Plan file:** `C:\Users\rickp\.claude\plans\keen-launching-blum.md`
**Full details:** See plan file for algorithm details, code snippets, and architectural rules
