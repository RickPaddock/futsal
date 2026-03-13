# CLAUDE.md — Futsal Player Tracking Pipeline Contract

This document defines the **complete processing pipeline** used to analyze futsal match video and determine player identity continuity and team membership.

The pipeline is designed to answer one core question reliably:

**Does a tracking ID continue to represent the same player, or has it jumped to a different player?**

The pipeline operates in five strict passes.
Each pass has:

• a defined goal
• explicit inputs
• explicit outputs
• validation checks that must pass before the next stage begins

No pass may proceed if its validation fails.

---

# Global System Objective

Given a futsal video:

1. Detect players and track them through time.
2. Extract jersey appearance signals.
3. Discover the dominant shirt colours present in the match.
4. Label each detection with its closest colour cluster.
5. Detect when a track jumps from one player to another.
6. Fragment tracks accordingly.
7. Assign players and teams using fragments.

The system must ensure that **each fragment corresponds to exactly one player**.

---

# Pipeline Overview

Pass 1 — Detection & Tracking
Pass 2 — Global Jersey Colour Clustering
Pass 3 — Observation Colour Labelling
Pass 4 — Track Jump Detection (Fragmentation)
Pass 5 — Identity & Team Resolution

Each pass must complete validation before the next begins.

---

# PASS 1 — Detection and Tracking

## Objective

Detect players and track them across frames. Extract appearance features needed for later processing.

This pass produces **raw observations only**.

It must **not infer teams, identity, or fragment tracks**.

---

## Inputs

Video frames.

---

## Models

• YOLO player detector
• ByteTrack tracker
• Jersey number detector (OCR or detection model)

---

## Processing Steps

For every frame:

1. Detect players using YOLO.
2. Track players using ByteTrack.
3. For each tracked player detection:

   * crop jersey region
   * compute HSV histogram for jersey crop
   * attempt jersey number detection
4. Detect the ball if present.

---

## Observation Record Format

Each detection must produce:

track_id
frame_index
bounding_box
jersey_crop
jersey_histogram (HSV normalized vector)
jersey_number (optional)
ball_detection_flag

---

## Required Data Volume

Pass 1 must produce:

• at least 1 player detection per frame
• jersey histogram for every detection

---

## Validation Checks

Pass 1 must verify:

1. **Tracking coverage**

```
average_players_per_frame >= 6
```

2. **Histogram completeness**

```
jersey_histogram exists for ≥ 95% of detections
```

3. **Track continuity**

Tracks must contain at least:

```
track_length >= 10 frames
```

Shorter tracks should be discarded.

---

## Pass 1 Success Condition

Pass 1 succeeds if:

• player detections exist for the full video
• jersey histograms exist for nearly all detections
• tracks longer than 10 frames exist

If any validation fails, the pipeline must stop.

---

# PASS 2 — Global Jersey Colour Clustering

## Objective

Discover the dominant shirt colours present in the match.

These clusters serve as **appearance prototypes**.

Clusters represent **colour groups only**, not teams.

---

## Inputs

All jersey_histograms produced by Pass 1.

---

## Procedure

1. Gather all jersey_histograms.
2. Normalize each histogram vector.
3. Run k-means clustering.

Cluster count:

```
k = 4
```

Reason:

Typical futsal match contains:

• 1 bib colour cluster
• 3 random shirt colours

---

## Outputs

colour_cluster_centroids
cluster_assignment_counts
cluster_distance_matrix

---

## Validation Checks

1. **Minimum cluster population**

Each cluster must contain at least:

```
≥ 5% of observations
```

2. **Cluster separation**

Minimum centroid distance:

```
centroid_L2_distance ≥ 0.15
```

3. **Cluster stability**

Re-run clustering with different seed.

Centroid drift must be:

```
≤ 0.05
```

---

## Failure Handling

If cluster validation fails:

1. retry clustering with:

```
k = 5
```

2. re-evaluate validation checks

If clustering still fails, pipeline must stop.

---

## Pass 2 Success Condition

Pass 2 succeeds when:

• stable colour clusters exist
• clusters are well separated
• clusters contain sufficient samples

---

# PASS 3 — Observation Colour Labelling

## Objective

Assign each detection to its nearest colour cluster.

This creates a **cluster sequence for each track**.

---

## Inputs

colour_cluster_centroids
jersey_histograms

---

## Procedure

For each detection:

1. compute L2 distance to each centroid
2. assign cluster_id = nearest centroid

Also record distance confidence.

---

## Observation Record Extension

Each observation must now include:

colour_cluster_id
cluster_distance

---

## Validation Checks

1. **Assignment completeness**

```
cluster_id assigned to ≥ 99% of detections
```

2. **Confidence threshold**

Average cluster_distance must satisfy:

```
mean_cluster_distance ≤ 0.25
```

3. **Cluster usage**

Each cluster must still have:

```
≥ 3% of observations
```

---

## Pass 3 Success Condition

Pass 3 succeeds when:

• nearly all detections have a cluster assignment
• cluster distances are reasonable
• clusters remain populated

---

# PASS 4 — Track Jump Detection (Fragmentation)

## Objective

Detect when a tracking ID jumps from one player to another.

Fragment tracks so that **each fragment represents one player only**.

---

## Inputs

track observations
colour_cluster_id sequence
jersey_number sequence

---

## Signal 1 — Colour Cluster Change

A swap candidate occurs when:

cluster_id_before != cluster_id_after

However, changes must persist.

Define persistence window:

```
PERSISTENCE_FRAMES = 15
```

Rule:

If a cluster change persists for ≥15 frames, it is considered real.

---

## Signal 2 — Jersey Number Change

A swap candidate occurs when:

number_before != number_after

Persistence rule identical:

```
≥15 frames
```

Examples:

none → 7
7 → 12
9 → none

---

## Fragmentation Rule

A split occurs if either signal confirms change.

Formal rule:

```
colour_cluster_switch
OR
jersey_number_change
```

Both must satisfy persistence.

---

## Fragment Output

Each fragment must contain:

fragment_id
parent_track_id
start_frame
end_frame
dominant_cluster_id
dominant_number

---

## Validation Checks

1. **Fragment coverage**

All original track frames must belong to exactly one fragment.

2. **Minimum fragment length**

```
fragment_length ≥ 20 frames
```

Fragments shorter than this should merge with neighbors.

3. **Cluster consistency**

Within a fragment:

```
dominant_cluster_ratio ≥ 80%
```

4. **Number consistency**

If number detected:

```
same_number_ratio ≥ 80%
```

---

## Pass 4 Success Condition

Pass 4 succeeds when:

• fragments fully cover all tracks
• fragments show stable colour cluster
• fragments meet minimum length

---

# PASS 5 — Identity and Team Resolution

## Objective

Determine player identities and team membership using fragments.

---

## Inputs

fragments
cluster labels
jersey numbers
temporal frame positions

---

## Identity Rules

1. Players with the same jersey number must share identity.

2. Two fragments with the same identity cannot overlap in time.

3. Players on the same team share consistent colour clusters.

---

## Team Determination

Team clusters are inferred from fragment cluster distributions.

Typical result:

team_A_cluster
team_B_clusters

---

## Output

player_id
team_id
fragment assignments

---

## Validation Checks

1. **Temporal exclusivity**

A player identity must not appear twice in the same frame.

2. **Team consistency**

Each player must belong to one team only.

3. **Roster size**

Teams must have between:

```
3–6 players
```

---

## Pass 5 Success Condition

Pass 5 succeeds when:

• identities obey temporal exclusivity
• team assignments are stable
• fragments map consistently to players

---

# System Completion Condition

Pipeline succeeds when:

• all passes succeed
• fragments represent single players
• teams and identities are consistent

If any pass fails validation, the system must stop and report diagnostics.

---

# Key Design Principle

Fragmentation must prioritize **not mixing players**.

It is acceptable to over-split fragments slightly.

It is unacceptable to allow one fragment to contain multiple players.

---


