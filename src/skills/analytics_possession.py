"""Analytics possession artifact built from committed identities and ball timeline."""

from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core import constants as const
from ..core.data_models import (
    AnalyticsPossessionOutput,
    BallInterpolationOutput,
    BallPosition,
    BirdseyeBallFrame,
    BirdseyeFrame,
    BirdseyePlayerPosition,
    BirdseyeProjectionOutput,
    Pass3COutput,
    PossessionCandidate,
    PossessionFrame,
)
from ..core.schemas import ANALYTICS_POSSESSION_OUTPUT_SCHEMA
from ..core.types import BallState
from ..utils.file_utils import load_json, save_json
from ..utils.logging_utils import get_logger

logger = get_logger("analytics_possession")


def _distance(a: List[float], b: List[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return float(sqrt(dx * dx + dy * dy))


def _ball_state_value(state: BallState | str) -> str:
    return state.value if isinstance(state, BallState) else str(state)


def _has_opposing_candidate(candidates: List[PossessionCandidate]) -> bool:
    return len(candidates) > 1 and candidates[1].team != candidates[0].team


def _has_secure_loose_ball_control(candidate: PossessionCandidate) -> bool:
    return candidate.distance_to_ball <= float(const.ANALYTICS_POSSESSION_LOOSE_BALL_SECURE_DISTANCE_M)


def _candidate_from_positions(
    player: BirdseyePlayerPosition,
    ball_position: Optional[List[float]],
    source_space: str,
    threshold: float,
    ball_confidence: float,
) -> Optional[PossessionCandidate]:
    if ball_position is None:
        return None

    if player.is_ghost or player.is_estimated:
        return None

    if source_space == "court":
        player_position = player.court_position
        trust = max(0.0, min(1.0, float(player.stabilization_trust)))
    else:
        player_position = player.image_anchor
        trust = 1.0

    distance_to_ball = _distance(player_position, ball_position)
    if distance_to_ball > threshold:
        return None

    base_confidence = max(0.0, 1.0 - (distance_to_ball / max(threshold, 1e-6)))
    confidence = max(0.0, min(1.0, base_confidence * max(0.25, trust) * max(0.25, ball_confidence)))

    return PossessionCandidate(
        player_id=player.player_id,
        team=player.team,
        jersey_number=player.jersey_number,
        distance_to_ball=float(distance_to_ball),
        source_space=source_space,
        confidence=float(confidence),
    )


def _build_candidates(
    frame_projection: BirdseyeFrame,
    ball_state_row: Optional[BallPosition],
    projected_ball: BirdseyeBallFrame,
    valid_player_ids: set[str],
) -> Tuple[List[PossessionCandidate], Optional[str]]:
    if ball_state_row is None:
        return [], None

    state_value = _ball_state_value(ball_state_row.state)
    if state_value == BallState.UNKNOWN.value:
        return [], None

    candidates: List[PossessionCandidate] = []

    if projected_ball.court_position is not None:
        source_space = "court"
        ball_position = projected_ball.court_position
        threshold = float(const.ANALYTICS_POSSESSION_MAX_DISTANCE_M)
    elif ball_state_row.centroid is not None:
        source_space = "image"
        ball_position = ball_state_row.centroid
        threshold = 0.0
    else:
        return [], None

    for player in frame_projection.players:
        if player.player_id not in valid_player_ids:
            continue

        player_threshold = threshold
        if source_space == "image":
            bbox_height = max(1.0, float(player.image_bbox[3]) - float(player.image_bbox[1]))
            player_threshold = max(
                float(const.ANALYTICS_POSSESSION_MIN_IMAGE_THRESHOLD_PX),
                bbox_height * float(const.ANALYTICS_POSSESSION_IMAGE_BBOX_HEIGHT_FACTOR),
            )

        candidate = _candidate_from_positions(
            player=player,
            ball_position=ball_position,
            source_space=source_space,
            threshold=float(player_threshold),
            ball_confidence=float(ball_state_row.confidence),
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda candidate: (candidate.distance_to_ball, -candidate.confidence, candidate.player_id))
    return candidates, source_space


def _build_provisional_frames(
    pass3_output: Pass3COutput,
    ball_output: BallInterpolationOutput,
    birdseye_output: BirdseyeProjectionOutput,
) -> Tuple[List[PossessionFrame], Dict[str, float]]:
    valid_player_ids = {identity.player_id for identity in pass3_output.identities}
    ball_by_frame: Dict[int, BallPosition] = {ball.frame_idx: ball for ball in ball_output.ball_positions}

    diagnostics = {
        "frames_with_ball": 0.0,
        "frames_without_ball": 0.0,
        "ambiguous_frames": 0.0,
        "low_confidence_abstentions": 0.0,
        "contested_duel_frames": 0.0,
        "loose_ball_recovery_frames": 0.0,
        "ghost_blocked_frames": 0.0,
        "court_space_candidate_frames": 0.0,
        "image_space_candidate_frames": 0.0,
    }
    provisional_frames: List[PossessionFrame] = []
    contested_recovery_active = False

    for frame_projection in birdseye_output.frames:
        ball_state_row = ball_by_frame.get(frame_projection.frame_idx)
        state_value = BallState.UNKNOWN
        if ball_state_row is not None:
            state_value = ball_state_row.state

        candidates, source_space = _build_candidates(
            frame_projection=frame_projection,
            ball_state_row=ball_state_row,
            projected_ball=frame_projection.ball,
            valid_player_ids=valid_player_ids,
        )

        if source_space == "court":
            diagnostics["court_space_candidate_frames"] += 1.0
        elif source_space == "image":
            diagnostics["image_space_candidate_frames"] += 1.0

        if state_value == BallState.UNKNOWN or ball_state_row is None:
            contested_recovery_active = False
            diagnostics["frames_without_ball"] += 1.0
            provisional_frames.append(
                PossessionFrame(
                    frame_idx=frame_projection.frame_idx,
                    ball_state=BallState.UNKNOWN,
                    candidates=[],
                )
            )
            continue

        diagnostics["frames_with_ball"] += 1.0
        ghost_only_near_ball = any(
            player.player_id in valid_player_ids and (player.is_ghost or player.is_estimated)
            for player in frame_projection.players
        ) and not candidates
        if ghost_only_near_ball:
            diagnostics["ghost_blocked_frames"] += 1.0

        if not candidates:
            if contested_recovery_active:
                diagnostics["ambiguous_frames"] += 1.0
                diagnostics["loose_ball_recovery_frames"] += 1.0
                provisional_frames.append(
                    PossessionFrame(
                        frame_idx=frame_projection.frame_idx,
                        ball_state=state_value,
                        is_ambiguous=True,
                        source_space=source_space,
                        candidates=[],
                    )
                )
                continue

            provisional_frames.append(
                PossessionFrame(
                    frame_idx=frame_projection.frame_idx,
                    ball_state=state_value,
                    source_space=source_space,
                    candidates=[],
                )
            )
            continue

        best_candidate = candidates[0]
        if best_candidate.confidence < float(const.ANALYTICS_POSSESSION_MIN_CONTROL_CONFIDENCE):
            if contested_recovery_active or _has_opposing_candidate(candidates):
                contested_recovery_active = True
            diagnostics["low_confidence_abstentions"] += 1.0
            provisional_frames.append(
                PossessionFrame(
                    frame_idx=frame_projection.frame_idx,
                    ball_state=state_value,
                    source_space=source_space,
                    candidates=candidates[:3],
                )
            )
            continue

        is_margin_ambiguous = len(candidates) > 1 and (
            best_candidate.confidence - candidates[1].confidence
        ) < float(const.ANALYTICS_POSSESSION_AMBIGUITY_MARGIN)

        is_contested_duel = (
            source_space == "court"
            and len(candidates) > 1
            and candidates[1].team != best_candidate.team
            and candidates[1].distance_to_ball <= float(const.ANALYTICS_POSSESSION_CONTESTED_SECOND_DISTANCE_M)
            and (candidates[1].distance_to_ball - best_candidate.distance_to_ball)
            <= float(const.ANALYTICS_POSSESSION_CONTESTED_DISTANCE_GAP_M)
        )

        if is_margin_ambiguous or is_contested_duel:
            if is_contested_duel:
                contested_recovery_active = True
            diagnostics["ambiguous_frames"] += 1.0
            if is_contested_duel:
                diagnostics["contested_duel_frames"] += 1.0
            provisional_frames.append(
                PossessionFrame(
                    frame_idx=frame_projection.frame_idx,
                    ball_state=state_value,
                    is_ambiguous=True,
                    source_space=source_space,
                    candidates=candidates[:3],
                )
            )
            continue

        if contested_recovery_active and not _has_secure_loose_ball_control(best_candidate):
            diagnostics["ambiguous_frames"] += 1.0
            diagnostics["loose_ball_recovery_frames"] += 1.0
            provisional_frames.append(
                PossessionFrame(
                    frame_idx=frame_projection.frame_idx,
                    ball_state=state_value,
                    is_ambiguous=True,
                    source_space=source_space,
                    candidates=candidates[:3],
                )
            )
            continue

        contested_recovery_active = False

        provisional_frames.append(
            PossessionFrame(
                frame_idx=frame_projection.frame_idx,
                player_id=best_candidate.player_id,
                team=best_candidate.team,
                jersey_number=best_candidate.jersey_number,
                confidence=best_candidate.confidence,
                ball_state=state_value,
                source_space=source_space,
                candidates=candidates[:3],
            )
        )

    return provisional_frames, diagnostics


def _apply_confirmation_window(frames: List[PossessionFrame]) -> Tuple[List[PossessionFrame], float]:
    confirmed_frames: List[PossessionFrame] = []
    pending_owner: Optional[str] = None
    pending_count = 0
    possession_switches = 0.0
    current_owner: Optional[str] = None

    for frame in frames:
        frame_data = frame.model_copy(deep=True)

        if frame.is_ambiguous or frame.player_id is None:
            pending_owner = None
            pending_count = 0
            current_owner = None
            frame_data.player_id = None
            frame_data.team = None
            frame_data.jersey_number = None
            frame_data.confidence = 0.0 if frame.is_ambiguous else frame.confidence
            confirmed_frames.append(frame_data)
            continue

        if frame.player_id == current_owner:
            pending_owner = None
            pending_count = 0
            confirmed_frames.append(frame_data)
            continue

        if frame.player_id == pending_owner:
            pending_count += 1
        else:
            pending_owner = frame.player_id
            pending_count = 1

        if pending_count >= int(const.ANALYTICS_POSSESSION_CONFIRMATION_FRAMES):
            if current_owner is not None and current_owner != frame.player_id:
                possession_switches += 1.0
            current_owner = frame.player_id
            pending_owner = None
            pending_count = 0
            confirmed_frames.append(frame_data)
        else:
            frame_data.player_id = None
            frame_data.team = None
            frame_data.jersey_number = None
            frame_data.confidence = 0.0
            confirmed_frames.append(frame_data)

    return confirmed_frames, possession_switches


def build_analytics_possession_output(
    pass3_output: Pass3COutput,
    ball_output: BallInterpolationOutput,
    birdseye_output: BirdseyeProjectionOutput,
) -> AnalyticsPossessionOutput:
    """Build frame-indexed possession output from committed identities and ball timeline."""
    provisional_frames, diagnostics = _build_provisional_frames(
        pass3_output=pass3_output,
        ball_output=ball_output,
        birdseye_output=birdseye_output,
    )
    confirmed_frames, possession_switches = _apply_confirmation_window(provisional_frames)

    diagnostics.update(
        {
            "confirmed_possession_frames": float(sum(1 for frame in confirmed_frames if frame.player_id is not None)),
            "possession_switches": float(possession_switches),
            "confirmation_window_frames": float(const.ANALYTICS_POSSESSION_CONFIRMATION_FRAMES),
        }
    )

    return AnalyticsPossessionOutput(
        video_name=birdseye_output.video_name,
        fps=birdseye_output.fps,
        total_frames=birdseye_output.total_frames,
        processed_start_frame=birdseye_output.processed_start_frame,
        processed_end_frame_exclusive=birdseye_output.processed_end_frame_exclusive,
        frames=confirmed_frames,
        diagnostics=diagnostics,
    )


def run_analytics_possession(
    input_dir: Path,
    output_dir: Optional[Path] = None,
) -> AnalyticsPossessionOutput:
    """Execute the analytics possession stage from an artifact directory."""
    if output_dir is None:
        output_dir = input_dir

    pass3_path = input_dir / const.PASS3_IDENTITY_COMMIT_JSON
    ball_path = input_dir / const.BALL_INTERPOLATION_JSON
    birdseye_path = input_dir / const.BIRDSEYE_PROJECTION_JSON
    output_path = output_dir / const.ANALYTICS_POSSESSION_JSON

    pass3_output = load_json(pass3_path, Pass3COutput)
    ball_output = load_json(ball_path, BallInterpolationOutput)
    birdseye_output = load_json(birdseye_path, BirdseyeProjectionOutput)

    possession_output = build_analytics_possession_output(
        pass3_output=pass3_output,
        ball_output=ball_output,
        birdseye_output=birdseye_output,
    )
    save_json(possession_output.model_dump(), str(output_path), ANALYTICS_POSSESSION_OUTPUT_SCHEMA)
    logger.info(f"Analytics possession artifact written: {output_path}")
    return possession_output