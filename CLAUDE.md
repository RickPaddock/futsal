Futsal Tracking System — Master Implementation Contract

This document is the authoritative implementation contract.

All code must comply with these rules.
Violations are invalid and must fail fast.

This document defines non-negotiable architecture, invariants, and validation rules.

0. ABSOLUTE PRINCIPLES (CANNOT BE BROKEN)

These rules override all other instructions.

P0 — Root Cause Fixes Only

Never patch downstream symptoms.

If Pass 3 fails because of earlier data corruption:

Fix Pass 3 or the earlier pass

Never modify downstream outputs to hide upstream errors

P1 — Pass Immutability

Each pipeline pass is append-only.

Rules:

Pass N may read Pass N-1 artifacts

Pass N may never modify Pass N-1 output

Each pass emits new artifacts

Previous artifacts remain untouched

P2 — Single Responsibility Per Pass

Each pass has a strict responsibility boundary.

No pass may partially implement the job of another pass.

Examples:

Pass	Responsibility
Pass 1	raw observation collection
Pass 2A	mechanical fragmentation
Pass 2B	fragment quality scoring
Pass 2C	ghost generation (presence layer)
Pass 3	identity reasoning
P3 — Identity Commit Point

Identity is decided exactly once.

Commit occurs at:

Pass 3C — Identity Commit

Before Pass 3C:

Only candidates and constraints exist

After Pass 3C:

identity is immutable

player_id, team, jersey cannot change

P4 — Validation is Blocking

Every validation rule is executable code.

If any rule fails:

Execution halts immediately

No downstream artifacts are written

Validation report explains failure

No best-effort outputs.

P5 — JSON is the Source of Truth

All behavior must be explainable from artifacts.

Video is not authoritative.

Truth comes from:

JSON artifacts
+
validation metrics

Visualization is only confirmation.

0.5 Mandatory Pre-Change Gate (REQUIRED)

Before making any change to code or contract, the following questions must be answered.

If any answer is unclear → STOP.

1 — Root Cause Check

Is the change fixing the problem at its source?

If the change masks a downstream symptom:

STOP.

2 — Entity Scope

List which entities are affected:

players
tracks
fragments
ghosts
frames
splits

Also specify layer:

identity layer
presence layer
observation layer

3 — Invariant Impact

List invariants affected.

Examples:

R1
R2
R3
R4
R5

State whether the change:

strengthens
weakens
leaves unchanged

4 — Failure Mode

If the change is wrong, how does it fail?

Allowed:

loud failure (validation)
silent but bounded

Not allowed:

silent propagating error

5 — Regression Prevention

State what prevents recurrence:

new validation rule
stronger contract language
documented non-goal

0.6 Mandatory Change Protocol (Claude Enforcement Rules)

This protocol governs every code or contract modification.

Claude must follow this process before, during, and after every change.

Changes that do not follow this protocol are invalid.

Step 1 — Identify Target Section

Before making any change, Claude must explicitly state:

which section of CLAUDE.md governs the behavior being modified

which pipeline pass is responsible

Example:

Target Section: Pass 2A — Mechanical Fragmentation
Affected Pass: Pass 2A
Reason: Fix incorrect split trigger logic

If a change touches multiple passes, it must be split into separate changes.

Step 2 — Re-read Contract Rules

Before editing code, Claude must re-check the rules governing that section.

This includes:

Absolute Principles (Section 0)
Entity Model (Section 2)
Relevant System Rules (R1–R5)
Pass Responsibilities
Validation Rules

Claude must confirm:

The change does not violate any existing contract rule.

If a contradiction exists, the change must stop and report the conflict.

Step 3 — Scope Control (Single Responsibility Change)

Each change must modify only one logical responsibility.

Allowed examples:

fix Pass 2A split trigger
fix ghost chaining logic
add validation rule

Not allowed:

modify Pass 2A and Pass 3 simultaneously
change fragmentation and ghost logic together
refactor identity resolution while adjusting detection thresholds

Large changes must be split into multiple commits.

Step 4 — Responsibility Boundary Check

Claude must verify the change does not bleed into another pass.

Examples of violations:

Pass 2A must NOT:

assign identity
create ghosts
enforce player count
repair tracking errors

Pass 2C must NOT:

merge tracks
resolve identity
terminate ghosts using spatial similarity

If a change violates responsibility boundaries, it must be rejected.

Step 5 — Invariant Compliance Check

Claude must verify the change does not weaken core invariants.

The following invariants must always hold:

R1 Raw Evidence Integrity
R2 Every Player Has a Team
R3 Jersey Temporal Exclusivity
R4 Player Presence Continuity
R5 Ball State Completeness

Claude must explicitly state:

Invariant impact: none / strengthened / unchanged

Weakening an invariant is not allowed.

Step 6 — Architecture Integrity Check

Claude must confirm the change preserves:

Pass immutability
Artifact structure
JSON schema compatibility
Pipeline order

No pass may:

rewrite earlier outputs
skip validation
alter artifact contracts

Step 7 — Produce Change Evidence

Every change must produce a structured change report.

Format:

CHANGE REPORT

Section Modified:
Pass 2A — Fragment Segmentation

Reason for Change:
Fix incorrect split trigger logic for jersey conflicts.

Files Modified:
src/pipeline/pass2_fragmentation.py

Contract Compliance Check:

Pass responsibility: confirmed
No cross-pass bleed: confirmed
R1–R5 invariants: unchanged
Pass immutability: preserved

Expected Behavior Change:

Track collisions will now trigger deterministic splits.

Risk Assessment:

Low — change isolated to Pass 2A.

Step 8 — Post-Change Validation

After implementation, Claude must verify:

All validation rules still pass
No new rule conflicts introduced
Artifact schema unchanged
Pipeline still runs end-to-end

If validation fails:

rollback change
report failure

Step 9 — Forbidden Behaviors

Claude must never:

make silent architecture changes
change multiple passes in one commit
weaken validation rules
remove fragment evidence
invent undocumented heuristics
ignore this protocol

Violating these rules is considered contract breach.

1. DIRECTORY CONTRACT

Input:

videos/input/*.mp4

Output:

videos/output/<clip_name>/

pass1_raw.json
pass1_validation.json

pass2_fragments.json
pass2_ghosts.json
pass2_validation.json

pass3_candidates.json
pass3_constraints.json
pass3_identity_commit.json
pass3_validation.json

ball_interpolation.json

debug_metrics.json
visualization.mp4

Rules:

Output folder = input filename without extension

No nested subfolders

If a pass fails → later artifacts MUST NOT exist

2. ENTITY MODEL

These identifiers must never be confused.

ID	Meaning	Mutable
track_id	ByteTrack temporary ID	YES
fragment_id	contiguous body segment	NO
player_id	final identity	NO
detection_id	frame-local detection	NO

Formats:

fragment_id: F000001
player_id: P07_team_a
detection_id: frame_track_bboxhash

3. SYSTEM RULES
R1 — Pass 1 is Raw Truth Only

Pass 1 may only record observations.

Must include:

bbox
confidence
track_id
HSV histogram
jersey probabilities
ball detection

Must NOT include:

teams
player identity
heuristic merges
heuristic splits

R2 — Every Player Has a Team

After Pass 3C:

team ∈ {team_a, team_b}

Unknown teams forbidden.

Players never switch teams.

Maximum:

6 players per team

Pipeline allows imbalance.

R3 — Jersey Numbers Define Identity

Rules:

one jersey = one player

Temporal exclusivity enforced.

Inheritance allowed only when:

jersey not already used during that time window

Jersey change:

#7 → #4 = identity discontinuity → split

Jersey appearance:

None → #4 = player turned → no split

R4 — Player Presence Invariant

At every frame:

tracked_players + ghosts ≥ level

Level = dynamic high-water mark.

Rules:

initialized from first 10 frames
increases if new players enter
never decreases
capped at 12

Ghosts maintain this invariant.

R5 — Ball State Exists at Every Frame

Ball state must exist:

real
interpolated
out_of_play

Validator checks state presence, not position.

3.5 Fragment vs Ghost Separation

Fragments belong to the identity layer.

Ghosts belong to the presence layer.

They must never be conflated.

Fragments represent identity continuity.

Ghosts represent presence continuity.

4. PIPELINE ORDER

Pass1 Raw Evidence
Pass2A Fragment Segmentation
Pass2B Fragment Scoring
Pass2C Ghost Generation
Pass3A Candidate Generation
Pass3B Constraint Graph
Pass3C Identity Commit
Ball Ball interpolation
Viz Visualization

Each pass is a pure function.

5. PASS RESPONSIBILITIES
Pass 1 — Raw Evidence

Responsibilities:

YOLO detection
ByteTrack tracking
jersey classification probabilities
HSV histograms
ball detection

Validation:

FAIL if:

bbox outside frame
bbox > 25% frame area
duplicate track_id in frame

Pass 2A — Mechanical Fragmentation

Input:

pass1_raw.json

Output:

pass2_fragments.json
pass2_validation.json

Purpose

Pass 2A detects provable tracker identity jumps.

It operates only on mechanical evidence present in raw observations.

It must never infer identity.

Loss of visibility ≠ identity change.

Fragments are evidence of tracking continuity boundaries.

Responsibilities

Pass 2A may only:

detect divergence points
split tracks into fragments
record fragment metadata

Pass 2A must NOT:

assign identity
infer teams
create ghosts
enforce player count
merge fragments
repair tracker errors

Allowed Split Triggers

Splits occur only when hard mechanical evidence exists.

1 Track Collision

Same track_id produces >1 detection in same frame.

Immediate split.

This handles upstream tracker corruption.

2 Jersey Change

Visible jersey number changes.

Example:

#7 → #4

Split.

3 Jersey Temporal Conflict

Same jersey visible simultaneously on two tracks.

Split at earliest contradiction.

4 Hard Appearance Discontinuity (HSV)

Large jersey color discontinuity indicates a tracker identity jump.

This trigger exists specifically to detect ID swaps during player crossings where motion remains smooth.

Required conditions:

jersey ROI valid on both sides
large HSV histogram discontinuity

Motion spike is NOT required.

Reason:

Most tracker ID swaps occur when players cross closely and motion remains smooth.
Requiring motion spikes would suppress correct splits.

HSV comparison window:

±5 frame search window

The algorithm must compare the nearest available HSV samples before and after the boundary.

If HSV samples are missing on one side, the trigger cannot fire.

5 Impossible Motion Spike

Extreme motion inconsistent with human movement.

Example:

large spatial displacement between adjacent frames.

Split.

Motion spike detection is independent from appearance discontinuity.

Explicit Non-Triggers

Never split for:

None → #4
#4 → None
occlusion
missed detection
low confidence
short gaps
crossing players
appearance drift

Fragment Handling

Rules:

fragments are evidence
never merge fragments
never delete fragments

Short fragments:

mark quality = low
keep them

Fragment Metadata

Each fragment records:

track_id
start_frame
end_frame
jersey_visible_ratio
occlusion_ratio
mean_velocity
appearance_stability_score
quality

No identity fields allowed.

Pass 2A Validation (Blocking)

Fail pipeline if:

fragment coverage < 100% of detections
any detection not assigned fragment
fragment ranges overlap
split without trigger reason
frame assigned to multiple fragments

Pass 2A does not validate ghosts or player counts.

Those belong to Pass 2C and Pass 3.

Pass 2B — Fragment Quality Scoring

Adds metadata only.

No identity inference.

Scores:

appearance stability
jersey observability
motion smoothness
occlusion ratio

Quality tiers:

HIGH
MEDIUM
LOW

Pass 2C — Ghost Generation

Maintains presence invariant (R4).

Ghosts created when:

tracked_count < level

Rules:

track by original_track_id

ghost duration = 60 frames

if player not reappeared → chain ghost

ghosts never merge tracks

Ghost termination only when:

player reappears
or clip ends

Pass 3 — Identity Resolution

Purpose:

Resolve player identity continuity across fragments while preserving the invariants:

R1 raw observation integrity
R2 team assignment completeness
R3 jersey exclusivity
R4 player presence continuity

Identity resolution happens exactly once in Pass 3C.

Earlier passes must never assign identity.

Pass 3 operates on fragments produced by Pass2A/B and ghosts from Pass2C.

Pass 3A — Candidate Generation

Purpose:

Generate identity continuity candidates between fragments based on physical and temporal plausibility.

Pass 3A produces evidence edges, not decisions.

Inputs
pass2_fragments_scored.json
pass2_ghost_fragments.json

Fragments include:

fragment_id
track_id
start_frame
end_frame
bbox_series
centroid_series
jersey_number evidence
HSV color evidence
quality score
is_ghost

Ghost fragments may exist but must not generate identity edges.

Outputs
pass3_identity_candidates.json

Each candidate edge contains:

fragment_a
fragment_b
temporal_gap
spatial_distance
jersey_match_score
color_match_score
velocity_consistency_score
overall_candidate_score

Edges represent possible same-player continuity.

Candidate Eligibility Rules

Two fragments are eligible for candidate generation only if:

fragment_a.end_frame < fragment_b.start_frame

and

gap_frames <= MAX_IDENTITY_GAP

Typical value:

MAX_IDENTITY_GAP = 300 frames
Spatial Feasibility Check

Maximum plausible displacement:

distance(fragment_a.last_centroid, fragment_b.first_centroid)
≤ MAX_PLAYER_SPEED * gap_time

If exceeded:

candidate rejected
Evidence Scoring

Candidate score combines:

Jersey Evidence
same jersey number → strong positive
conflicting jersey numbers → hard reject
unknown → neutral
Color Evidence

HSV similarity between fragments.

Produces:

color_match_score ∈ [0,1]
Motion Evidence

Estimate exit velocity from fragment A.

Check whether B's entry location is consistent.

Produces:

velocity_consistency_score
Temporal Gap Penalty

Long gaps reduce confidence.

Final Score

Weighted combination:

overall_candidate_score =
    w1 * jersey_score +
    w2 * color_score +
    w3 * motion_score +
    w4 * temporal_penalty

Candidates below threshold are discarded.

Pass 3B — Constraint Graph Construction

Purpose:

Convert candidate evidence into a global constraint graph.

Nodes:

fragment_id

Edges:

MUST_SAME
CANNOT_SAME
SOFT_SAME
Inputs
pass2 fragments
pass3_identity_candidates
Outputs
pass3_constraint_graph.json
MUST_SAME Constraints

Fragments must belong to the same identity when:

same track_id
and temporal continuity exists

or

candidate_score >= MUST_SAME_THRESHOLD

These edges are hard constraints.

CANNOT_SAME Constraints

Fragments cannot belong to the same identity if:

temporal overlap exists

or

jersey conflict detected

or

spatial impossibility

These edges are hard exclusions.

SOFT_SAME Constraints

Weak evidence linking fragments.

Created when:

candidate_score >= SOFT_THRESHOLD

but below MUST threshold.

These edges influence optimization but do not enforce identity.

Graph Invariants

The constraint graph must satisfy:

no fragment may have MUST_SAME edges to two fragments
that have CANNOT_SAME between them

If detected:

FAIL FAST
Pass 3C — Identity Commit

Purpose:

Resolve the constraint graph into final player identities.

Identity resolution occurs exactly once here.

Inputs
pass3_constraint_graph.json
pass2_fragments
Outputs
pass3_identities.json

Each identity contains:

identity_id
team_id
jersey_number
fragments[]
Identity Resolution Algorithm
Step 1 — Build Identity Groups

Construct connected components using:

MUST_SAME edges

Each component becomes a candidate player identity.

Step 2 — Validate Hard Constraints

Ensure within each component:

no temporal overlaps
no jersey conflicts

If violation exists:

FAIL FAST
Step 3 — Team Assignment

Extract color embeddings for each identity.

Run clustering:

K-means (k=2)

Result:

team assignment for each identity
Step 4 — Lock Team Labels

Once assigned:

team_id immutable

Fragments inherit the identity's team.

Step 5 — Jersey Resolution

For each identity:

jersey = most confident jersey observation

Unknown allowed if insufficient evidence.

Step 6 — Enforce Jersey Exclusivity

For each frame:

no two identities may share the same jersey number on same team

If violation occurs:

resolve via soft constraints

If unresolved:

FAIL FAST
Step 7 — Soft Constraint Optimization

Apply remaining:

SOFT_SAME edges

to maximize global consistency.

Optimization goal:

maximize total candidate scores

subject to:

hard constraints
Step 8 — Identity Finalization

Assign stable identifiers:

identity_id = sequential player ID

Fragments inherit identity.

After this step:

identity assignments immutable
Pass 3 Invariants

After Pass 3C completes:

each fragment belongs to exactly one identity
no identities overlap in time with themselves
team assignment exists for every identity
jersey exclusivity holds
Debug Metrics

Pass 3 must record:

identity_count
fragments_per_identity
candidate_edges
must_edges
cannot_edges
soft_edges
identity_merges
identity_conflicts
Failure Conditions

Pass 3 must stop immediately if:

constraint contradictions detected
identity overlap detected
team assignment fails
jersey exclusivity fails

Required behavior:

write validation_report.json
exit non-zero
do not write pass3 artifacts
Critical Design Rules

Pass 3 must never:

modify fragment boundaries
merge fragments
create new fragments

Pass 3 only:

assigns identities
assigns teams
resolves jerseys

6. Automated Debugging

Metrics recorded in:

debug_metrics.json

Required:

player count per frame
team counts
jersey conflicts
ghost count
identity changes

7. Failure Policy

On validation failure:

stop immediately
write validation report
do not write output artifacts
exit non-zero

8. Configuration

Critical corrections applied:

MERGE_CONSECUTIVE_SHORT = False

Fragments must never merge.

Other parameters unchanged.

9. Implementation Guardrails

Claude must follow these rules when modifying code.

Small Change Rule

Modify only one logical component per change.

Section Responsibility Check

Before implementing:

confirm pass ownership
confirm no responsibility bleed

Architecture Verification

After changes verify:

earlier passes untouched
schemas unchanged
later passes still compatible

Rule Compliance Check

Before commit confirm:

pass immutability preserved
fragment boundaries preserved
identity still resolved only in Pass3C

Stop on Ambiguity

If instructions conflict with this contract:

STOP and report conflict.

Final Rule

If identity cannot be resolved or invariants cannot be satisfied:

FAIL FAST
EXPLAIN WHY

Never guess.

Never patch.