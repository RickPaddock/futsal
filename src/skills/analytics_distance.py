"""Analytics distance and lightweight reporting artifacts built from committed outputs."""

from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core import constants as const
from ..core.data_models import (
    AnalyticsPlayerSummary,
    AnalyticsSummaryOutput,
    AnalyticsEventsOutput,
    AnalyticsPossessionOutput,
    BirdseyeProjectionOutput,
    NamedPlayerStatus,
    Pass3COutput,
    PlayerDistanceSummaryOutput,
    PlayerDistanceSummaryRow,
)
from ..core.schemas import ANALYTICS_SUMMARY_OUTPUT_SCHEMA, PLAYER_DISTANCE_SUMMARY_OUTPUT_SCHEMA
from ..core.types import PlayerID, TeamID
from ..utils.file_utils import load_json, save_json
from ..utils.logging_utils import get_logger

logger = get_logger("analytics_distance")


def _distance(a: List[float], b: List[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return float(sqrt(dx * dx + dy * dy))


def _display_name(jersey_number: int) -> Optional[str]:
    return const.PLAYER_NAME_BY_JERSEY.get(int(jersey_number))


def _reportable_player_map(pass3_output: Pass3COutput) -> Tuple[Dict[PlayerID, Tuple[TeamID, int]], set[int]]:
    resolved_candidates: Dict[PlayerID, Tuple[TeamID, int]] = {}
    jersey_to_player_ids: Dict[int, set[PlayerID]] = {}
    for identity in pass3_output.identities:
        if identity.jersey_number is None:
            continue
        jersey_number = int(identity.jersey_number)
        resolved_candidates.setdefault(identity.player_id, (identity.team, jersey_number))
        jersey_to_player_ids.setdefault(jersey_number, set()).add(identity.player_id)

    ambiguous_jerseys = {
        jersey_number
        for jersey_number, player_ids in jersey_to_player_ids.items()
        if len(player_ids) != 1
    }
    reportable_players = {
        player_id: (team_id, jersey_number)
        for player_id, (team_id, jersey_number) in resolved_candidates.items()
        if jersey_number not in ambiguous_jerseys
    }
    return reportable_players, ambiguous_jerseys


def build_player_distance_summary_output(
    pass3_output: Pass3COutput,
    birdseye_output: BirdseyeProjectionOutput,
) -> PlayerDistanceSummaryOutput:
    """Build per-player distance summaries from committed bird's-eye court positions."""
    resolved_players, ambiguous_jerseys = _reportable_player_map(pass3_output)
    distance_by_player: Dict[PlayerID, float] = {player_id: 0.0 for player_id in resolved_players}
    observed_frames: Dict[PlayerID, int] = {player_id: 0 for player_id in resolved_players}
    estimated_frames: Dict[PlayerID, int] = {player_id: 0 for player_id in resolved_players}
    segment_counts: Dict[PlayerID, int] = {player_id: 0 for player_id in resolved_players}
    previous_samples: Dict[PlayerID, Tuple[int, List[float]]] = {}

    diagnostics: Dict[str, float] = {
        "reportable_players": float(len(resolved_players)),
        "ambiguous_jerseys_filtered": float(len(ambiguous_jerseys)),
        "unresolved_player_frames_skipped": 0.0,
        "estimated_frames_skipped": 0.0,
        "gap_breaks": 0.0,
        "distance_segments_counted": 0.0,
        "distance_source_space": 0.0,
    }

    for frame in birdseye_output.frames:
        for player in frame.players:
            player_id = player.player_id
            if player_id not in resolved_players:
                diagnostics["unresolved_player_frames_skipped"] += 1.0
                continue

            if player.is_ghost or player.is_estimated:
                estimated_frames[player_id] += 1
                diagnostics["estimated_frames_skipped"] += 1.0
                previous_samples.pop(player_id, None)
                continue

            observed_frames[player_id] += 1
            prior_sample = previous_samples.get(player_id)
            if prior_sample is not None:
                prior_frame_idx, prior_position = prior_sample
                frame_gap = int(player.frame_idx) - int(prior_frame_idx)
                if frame_gap == 1:
                    distance_by_player[player_id] += _distance(prior_position, player.court_position)
                    segment_counts[player_id] += 1
                    diagnostics["distance_segments_counted"] += 1.0
                else:
                    diagnostics["gap_breaks"] += 1.0

            previous_samples[player_id] = (int(player.frame_idx), list(player.court_position))

    players: List[PlayerDistanceSummaryRow] = []
    for player_id, (team_id, jersey_number) in resolved_players.items():
        observed_count = int(observed_frames[player_id])
        estimated_count = int(estimated_frames[player_id])
        confidence_denominator = max(1, observed_count + estimated_count)
        players.append(
            PlayerDistanceSummaryRow(
                player_id=player_id,
                jersey_number=jersey_number,
                team_id=team_id,
                display_name=_display_name(jersey_number),
                distance_value=float(distance_by_player[player_id]),
                distance_unit="m",
                observed_frame_count=observed_count,
                estimated_frame_count=estimated_count,
                distance_confidence=float(observed_count / confidence_denominator),
            )
        )

    players.sort(key=lambda row: (row.team_id.value, row.jersey_number, row.player_id))
    diagnostics["distance_source_space"] = 1.0

    return PlayerDistanceSummaryOutput(
        video_name=birdseye_output.video_name,
        fps=birdseye_output.fps,
        total_frames=birdseye_output.total_frames,
        processed_start_frame=birdseye_output.processed_start_frame,
        processed_end_frame_exclusive=birdseye_output.processed_end_frame_exclusive,
        players=players,
        diagnostics=diagnostics,
    )


def _build_named_player_statuses(pass3_output: Pass3COutput) -> List[NamedPlayerStatus]:
    resolved_players, _ = _reportable_player_map(pass3_output)
    resolved_by_jersey: Dict[int, Tuple[PlayerID, TeamID]] = {}
    for player_id, (team_id, jersey_number) in resolved_players.items():
        resolved_by_jersey.setdefault(jersey_number, (player_id, team_id))

    named_players: List[NamedPlayerStatus] = []
    for jersey_number, display_name in sorted(const.PLAYER_NAME_BY_JERSEY.items()):
        resolved_entry = resolved_by_jersey.get(int(jersey_number))
        named_players.append(
            NamedPlayerStatus(
                jersey_number=int(jersey_number),
                display_name=display_name,
                resolved=resolved_entry is not None,
                player_id=None if resolved_entry is None else resolved_entry[0],
                team_id=None if resolved_entry is None else resolved_entry[1],
            )
        )
    return named_players


def build_analytics_summary_output(
    pass3_output: Pass3COutput,
    possession_output: AnalyticsPossessionOutput,
    events_output: AnalyticsEventsOutput,
    distance_output: PlayerDistanceSummaryOutput,
) -> AnalyticsSummaryOutput:
    """Build lightweight player and named-jersey summaries from committed artifacts."""
    resolved_players, ambiguous_jerseys = _reportable_player_map(pass3_output)
    distance_by_player = {row.player_id: row for row in distance_output.players}
    player_summaries: Dict[PlayerID, AnalyticsPlayerSummary] = {}

    for player_id, (team_id, jersey_number) in resolved_players.items():
        distance_row = distance_by_player.get(player_id)
        player_summaries[player_id] = AnalyticsPlayerSummary(
            player_id=player_id,
            jersey_number=jersey_number,
            team_id=team_id,
            display_name=_display_name(jersey_number),
            distance_value=0.0 if distance_row is None else float(distance_row.distance_value),
            distance_unit="m" if distance_row is None else distance_row.distance_unit,
            distance_confidence=0.0 if distance_row is None else float(distance_row.distance_confidence),
        )

    for frame in possession_output.frames:
        if frame.player_id is None or frame.player_id not in player_summaries:
            continue
        player_summaries[frame.player_id].confirmed_possession_frames += 1

    for summary in player_summaries.values():
        summary.confirmed_possession_seconds = float(summary.confirmed_possession_frames / max(possession_output.fps, 1.0))

    event_totals: Dict[str, int] = {
        "successful_passes": 0,
        "unsuccessful_passes": 0,
        "loose_ball_passes": 0,
        "shots": 0,
        "goals": 0,
    }
    diagnostics: Dict[str, float] = {
        "reportable_players": float(len(player_summaries)),
        "ambiguous_jerseys_filtered": float(len(ambiguous_jerseys)),
        "audit_only_events_skipped": 0.0,
        "reportable_events_counted": 0.0,
        "named_players_resolved": 0.0,
    }

    for event in events_output.events:
        if event.is_audit_only:
            diagnostics["audit_only_events_skipped"] += 1.0
            continue

        if event.event_type == "pass":
            if event.passer_player_id in player_summaries:
                summary = player_summaries[event.passer_player_id]
                summary.passes_attempted += 1
                diagnostics["reportable_events_counted"] += 1.0
                if event.outcome == "successful":
                    summary.passes_completed += 1
                    event_totals["successful_passes"] += 1
                elif event.outcome == "unsuccessful":
                    summary.unsuccessful_passes += 1
                    event_totals["unsuccessful_passes"] += 1
                elif event.outcome == "loose_ball":
                    summary.loose_ball_releases += 1
                    event_totals["loose_ball_passes"] += 1

            if event.outcome == "successful" and event.receiver_player_id in player_summaries:
                player_summaries[event.receiver_player_id].passes_received += 1

        elif event.event_type == "shot" and event.passer_player_id in player_summaries:
            summary = player_summaries[event.passer_player_id]
            summary.shots_attempted += 1
            diagnostics["reportable_events_counted"] += 1.0
            event_totals["shots"] += 1
            if event.outcome == "goal":
                summary.goals_scored += 1
                event_totals["goals"] += 1

    named_players = _build_named_player_statuses(pass3_output)
    diagnostics["named_players_resolved"] = float(sum(1 for player in named_players if player.resolved))

    players = sorted(
        player_summaries.values(),
        key=lambda row: (row.team_id.value, row.jersey_number, row.player_id),
    )

    return AnalyticsSummaryOutput(
        video_name=distance_output.video_name,
        fps=distance_output.fps,
        total_frames=distance_output.total_frames,
        processed_start_frame=distance_output.processed_start_frame,
        processed_end_frame_exclusive=distance_output.processed_end_frame_exclusive,
        players=players,
        named_players=named_players,
        event_totals=event_totals,
        diagnostics=diagnostics,
    )


def run_analytics_distance(
    input_dir: Path,
    output_dir: Optional[Path] = None,
) -> PlayerDistanceSummaryOutput:
    """Execute the distance summary stage from an artifact directory."""
    if output_dir is None:
        output_dir = input_dir

    pass3_path = input_dir / const.PASS3_IDENTITY_COMMIT_JSON
    birdseye_path = input_dir / const.BIRDSEYE_PROJECTION_JSON
    output_path = output_dir / const.PLAYER_DISTANCE_SUMMARY_JSON

    pass3_output = load_json(pass3_path, Pass3COutput)
    birdseye_output = load_json(birdseye_path, BirdseyeProjectionOutput)
    distance_output = build_player_distance_summary_output(
        pass3_output=pass3_output,
        birdseye_output=birdseye_output,
    )
    save_json(distance_output.model_dump(), str(output_path), PLAYER_DISTANCE_SUMMARY_OUTPUT_SCHEMA)
    logger.info(f"Player distance summary artifact written: {output_path}")
    return distance_output


def run_analytics_summary(
    input_dir: Path,
    output_dir: Optional[Path] = None,
) -> AnalyticsSummaryOutput:
    """Execute the lightweight analytics summary stage from an artifact directory."""
    if output_dir is None:
        output_dir = input_dir

    pass3_path = input_dir / const.PASS3_IDENTITY_COMMIT_JSON
    possession_path = input_dir / const.ANALYTICS_POSSESSION_JSON
    events_path = input_dir / const.ANALYTICS_EVENTS_JSON
    distance_path = input_dir / const.PLAYER_DISTANCE_SUMMARY_JSON
    output_path = output_dir / const.ANALYTICS_SUMMARY_JSON

    pass3_output = load_json(pass3_path, Pass3COutput)
    possession_output = load_json(possession_path, AnalyticsPossessionOutput)
    events_output = load_json(events_path, AnalyticsEventsOutput)
    distance_output = load_json(distance_path, PlayerDistanceSummaryOutput)

    summary_output = build_analytics_summary_output(
        pass3_output=pass3_output,
        possession_output=possession_output,
        events_output=events_output,
        distance_output=distance_output,
    )
    save_json(summary_output.model_dump(), str(output_path), ANALYTICS_SUMMARY_OUTPUT_SCHEMA)
    logger.info(f"Analytics summary artifact written: {output_path}")
    return summary_output