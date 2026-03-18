import json

import cv2
from src.core.data_models import BallDetection, BallInterpolationOutput, BallPosition, BirdseyeBallFrame, BirdseyeFrame, BirdseyePlayerPosition, BirdseyeProjectionOutput, Detection, Pass1Output, Fragment, Pass2AOutput, Pass2COutput, ScoredFragment, CommittedIdentity, Pass3COutput, IdentityCandidateEdge
from src.core.types import BallState, FragmentQuality, InterpolationMethod, TeamID, AssignmentMethod
from src.skills.ball_interpolator import build_ball_interpolation_output
from src.skills.birds_eye_pitch import build_birds_eye_projection_output, _apply_journey_path_smoothing, _apply_stationary_span_smoothing, _apply_trajectory_segment_smoothing, _blend_directional_motion, _build_journey_activity_lookup, _build_player_height_prior, _classify_motion_span, _expected_player_height, _pose_anchor_from_keypoints, _pose_anchor_from_upper_body_keypoints
from src.skills.pass3_debug_visualizer import _ghost_bbox_for_frame
from src.skills.pass3c_identity_solver import IdentitySolver
from src.skills.visualizer import render_visualization_from_artifact
from src.validation.validator import Validator
from src.skills.pass1_extractor import _load_pass1_intention_lines
from utils.homography import create_homography_from_config
from utils.pitch_drawing import create_court_view
import numpy as np


def _valid_pass1_output() -> Pass1Output:
    detection = Detection(
        detection_id="0_1_deadbeef",
        frame_idx=0,
        bbox=[100, 100, 180, 300],
        centroid=[140, 200],
        confidence=0.90,
        track_id=1,
        jersey_number=None,
        jersey_confidence=0.0,
        jersey_probs=None,
        hsv_histogram_jersey=None,
        jersey_color_sampled=False,
        jersey_roi_valid=False,
        jersey_roi_bbox=None,
    )
    return Pass1Output(
        video_name="unit",
        fps=30.0,
        width=1920,
        height=1080,
        total_frames=1,
        processed_start_frame=0,
        processed_end_frame_exclusive=1,
        detections=[detection],
        ball_detections=[],
    )


def test_pass1_intention_lines_loaded_from_rules_doc():
    lines = _load_pass1_intention_lines()
    assert len(lines) > 0
    joined = " ".join(lines).lower()
    assert "bbox" in joined

def test_pass1_validation_passes_on_minimal_valid_output():
    validator = Validator()
    result = validator.validate_pass1(_valid_pass1_output())
    assert result.passed
    assert len(result.violations) == 0

def test_pass1_validation_fails_on_invalid_bbox():
    invalid = _valid_pass1_output()
    invalid.detections[0].bbox = [2000, 100, 2200, 300]  # outside frame width

    validator = Validator()
    result = validator.validate_pass1(invalid)

    assert not result.passed
    rules = {v.rule for v in result.violations}
    assert "PASS1_BBOX" in rules


def test_pass1_validation_fails_on_invalid_jersey_crop_quality_range():
    invalid = _valid_pass1_output()
    invalid.detections[0].jersey_crop_quality = 1.5

    validator = Validator()
    result = validator.validate_pass1(invalid)

    assert not result.passed
    rules = {v.rule for v in result.violations}
    assert "PASS1_JERSEY_CROP_QUALITY_RANGE" in rules

def test_pass2a_validation_fails_on_temporal_overlap():
    pass1_output = _valid_pass1_output()
    pass1_output.detections.append(
        Detection(
            detection_id="1_1_beadfeed",
            frame_idx=1,
            bbox=[102, 102, 182, 302],
            centroid=[142, 202],
            confidence=0.88,
            track_id=1,
            jersey_number=None,
            jersey_confidence=0.0,
            jersey_probs=None,
            hsv_histogram_jersey=None,
            jersey_color_sampled=False,
            jersey_roi_valid=False,
            jersey_roi_bbox=None,
        )
    )

    pass2a_output = Pass2AOutput(
        fragments=[
            Fragment(
                fragment_id="F000001",
                original_track_id=1,
                start_frame=0,
                end_frame=1,
                detection_ids=["0_1_deadbeef"],
            ),
            Fragment(
                fragment_id="F000002",
                original_track_id=1,
                start_frame=1,
                end_frame=1,
                detection_ids=["1_1_beadfeed"],
            ),
        ],
        split_log=[],
    )

    validator = Validator()
    result = validator.validate_pass2a(pass2a_output, pass1_output)

    assert not result.passed
    rules = {v.rule for v in result.violations}
    assert "PASS2A_TEMPORAL_OVERLAP" in rules


def test_pass3_validation_includes_compactness_diagnostics():
    fragments = [
        ScoredFragment(
            fragment_id="F000001",
            original_track_id=1,
            start_frame=0,
            end_frame=5,
            detection_ids=["0_1_deadbeef"],
            quality=FragmentQuality.HIGH,
            quality_score=0.9,
        ),
        ScoredFragment(
            fragment_id="F000002",
            original_track_id=2,
            start_frame=0,
            end_frame=5,
            detection_ids=["0_2_feedbead"],
            quality=FragmentQuality.HIGH,
            quality_score=0.9,
        )
    ]

    pass3_output = Pass3COutput(
        identities=[
            CommittedIdentity(
                fragment_id="F000001",
                player_id="P07_team_a",
                team=TeamID.TEAM_A,
                jersey_number=7,
                assignment_method=AssignmentMethod.KMEANS,
                assignment_confidence=0.95,
            ),
            CommittedIdentity(
                fragment_id="F000002",
                player_id="P08_team_b",
                team=TeamID.TEAM_B,
                jersey_number=10,
                assignment_method=AssignmentMethod.KMEANS,
                assignment_confidence=0.95,
            )
        ],
        solver_log={
            "team_assignment_mode": "compactness_guided_kmeans",
            "cluster_compactness": {"team_a": 0.10, "team_b": 0.42},
            "compactness_ratio": 4.2,
            "bibbed_team_evidence": "team_b",
        },
        unresolved_conflicts=[],
    )

    validator = Validator()
    result = validator.validate_pass3(pass3_output, fragments)

    assert result.passed
    assert result.diagnostics["team_assignment_mode"] == "compactness_guided_kmeans"
    assert result.diagnostics["cluster_compactness_a"] == 0.10
    assert result.diagnostics["cluster_compactness_b"] == 0.42
    assert result.diagnostics["compactness_ratio"] == 4.2


def test_pass3_validation_allows_matched_ghost_window():
    fragments = [
        ScoredFragment(
            fragment_id="G000001",
            original_track_id=1,
            start_frame=1,
            end_frame=10,
            detection_ids=[],
            quality=FragmentQuality.GHOST,
            quality_score=0.0,
            is_ghost=True,
            exclude_from_clustering=True,
            ghost_last_known_bbox=[100, 100, 180, 300],
            ghost_last_known_centroid=[140, 200],
            ghost_reason="player_occluded",
        ),
        ScoredFragment(
            fragment_id="F000011",
            original_track_id=2,
            start_frame=4,
            end_frame=4,
            detection_ids=["4_2_feedbead"],
            quality=FragmentQuality.HIGH,
            quality_score=0.95,
        ),
        ScoredFragment(
            fragment_id="F000020",
            original_track_id=20,
            start_frame=1,
            end_frame=3,
            detection_ids=["1_20_teamfeed"],
            quality=FragmentQuality.HIGH,
            quality_score=0.95,
        ),
    ]

    pass3_output = Pass3COutput(
        identities=[
            CommittedIdentity(
                fragment_id="G000001",
                player_id="P07_team_a",
                team=TeamID.TEAM_A,
                jersey_number=7,
                assignment_method=AssignmentMethod.GHOST_INHERITED,
                assignment_confidence=0.80,
                assignment_reasons=["MATCHED_GHOST_WINDOW"],
            ),
            CommittedIdentity(
                fragment_id="F000011",
                player_id="P07_team_a",
                team=TeamID.TEAM_A,
                jersey_number=7,
                assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
                assignment_confidence=0.95,
            ),
            CommittedIdentity(
                fragment_id="F000020",
                player_id="P10_team_b",
                team=TeamID.TEAM_B,
                jersey_number=10,
                assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
                assignment_confidence=0.95,
            ),
        ],
        solver_log={
            "team_assignment_mode": "compactness_guided_kmeans",
            "cluster_compactness": {"team_a": 0.10, "team_b": 0.42},
            "compactness_ratio": 4.2,
            "bibbed_team_evidence": "team_b",
            "ghost_active_windows": {
                "G000001": {
                    "start_frame": 1,
                    "end_frame": 3,
                    "matched_reappearance_frame": 4,
                    "target_fragment_id": "F000011",
                    "window_reason": "matched_reappearance",
                }
            },
        },
        unresolved_conflicts=[],
    )

    validator = Validator()
    result = validator.validate_pass3(pass3_output, fragments)

    assert result.passed


def test_birdseye_projection_builds_projected_players_and_ball():
    pass1_output = Pass1Output(
        video_name="unit",
        fps=30.0,
        width=3840,
        height=2160,
        total_frames=2,
        processed_start_frame=0,
        processed_end_frame_exclusive=2,
        detections=[
            Detection(
                detection_id="0_1_deadbeef",
                frame_idx=0,
                bbox=[1000, 700, 1080, 980],
                centroid=[1040, 840],
                confidence=0.95,
                track_id=1,
                jersey_number=7,
                jersey_confidence=0.9,
                jersey_probs={7: 0.9},
                hsv_histogram_jersey=None,
                jersey_color_sampled=False,
                jersey_roi_valid=False,
                jersey_roi_bbox=None,
            )
        ],
        ball_detections=[
            BallDetection(
                frame_idx=0,
                bbox=[1900, 760, 1910, 770],
                centroid=[1905, 765],
                confidence=0.8,
            )
        ],
    )
    pass2_output = Pass2COutput(
        fragments=[
            ScoredFragment(
                fragment_id="F000001",
                original_track_id=1,
                start_frame=0,
                end_frame=0,
                detection_ids=["0_1_deadbeef"],
                quality=FragmentQuality.HIGH,
                quality_score=0.95,
            )
        ]
    )
    pass3_output = Pass3COutput(
        identities=[
            CommittedIdentity(
                fragment_id="F000001",
                player_id="P07_team_a",
                team=TeamID.TEAM_A,
                jersey_number=7,
                assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
                assignment_confidence=0.95,
            )
        ],
        solver_log={},
        unresolved_conflicts=[],
    )
    ball_output = BallInterpolationOutput(
        ball_positions=[
            BallPosition(frame_idx=0, state=BallState.REAL, centroid=[1905, 765], bbox=[1900, 760, 1910, 770], confidence=0.8),
            BallPosition(frame_idx=1, state=BallState.UNKNOWN, centroid=None, bbox=None, confidence=0.0),
        ],
        interpolation_method=InterpolationMethod.LINEAR,
        total_frames=2,
    )
    calibration_config = {
        "homography": {
            "court_length": 40.0,
            "court_width": 20.0,
            "output_pixel_scale": 20,
            "source_points": [[680, 595], [843, 638], [147, 1075], [3136, 611], [2964, 650], [3644, 1100], [1900, 530], [1888, 1592], [303, 772], [128, 875], [3511, 808], [3692, 907], [1891, 760]],
            "dest_points": [[0.0, 19.0], [3.5, 17.0], [3.5, 5.0], [40.0, 19.0], [36.5, 17.0], [36.5, 5.0], [20.0, 19.0], [20.0, 1.0], [0.0, 11.5], [0.0, 8.5], [40.0, 11.5], [40.0, 8.5], [20.0, 10.0]],
        }
    }

    birdseye_output = build_birds_eye_projection_output(
        pass1_output=pass1_output,
        pass2c_output=pass2_output,
        pass3_output=pass3_output,
        ball_output=ball_output,
        calibration_config=calibration_config,
    )

    assert isinstance(birdseye_output, BirdseyeProjectionOutput)
    assert len(birdseye_output.frames) == 2
    assert len(birdseye_output.frames[0].players) == 1
    assert birdseye_output.frames[0].players[0].player_id == "P07_team_a"
    assert birdseye_output.frames[0].players[0].raw_image_anchor == birdseye_output.frames[0].players[0].image_anchor
    assert birdseye_output.frames[0].players[0].raw_court_position == birdseye_output.frames[0].players[0].court_position
    assert birdseye_output.frames[0].players[0].raw_render_position == birdseye_output.frames[0].players[0].render_position
    assert birdseye_output.frames[0].ball.state == BallState.REAL
    assert birdseye_output.frames[0].ball.image_bbox == [1900, 760, 1910, 770]
    assert birdseye_output.frames[0].ball.court_position is not None
    assert birdseye_output.frames[0].ball.render_position is not None
    assert birdseye_output.frames[0].ball.render_position[1] < 200
    assert birdseye_output.frames[1].ball.state == BallState.UNKNOWN
    assert birdseye_output.frames[1].ball.image_bbox is None
    assert birdseye_output.frames[1].ball.court_position is None


def test_birdseye_projection_uses_foot_anchor_and_prediction_only_on_bad_bbox():
    pass1_output = Pass1Output(
        video_name="unit",
        fps=30.0,
        width=3840,
        height=2160,
        total_frames=2,
        processed_start_frame=0,
        processed_end_frame_exclusive=2,
        detections=[
            Detection(
                detection_id="0_1_deadbeef",
                frame_idx=0,
                bbox=[1000, 700, 1080, 980],
                centroid=[1040, 840],
                confidence=0.95,
                track_id=1,
                jersey_number=7,
                jersey_confidence=0.9,
                jersey_probs={7: 0.9},
                hsv_histogram_jersey=None,
                jersey_color_sampled=False,
                jersey_roi_valid=False,
                jersey_roi_bbox=None,
            ),
            Detection(
                detection_id="1_1_deadbeef",
                frame_idx=1,
                bbox=[1010, 780, 1040, 900],
                centroid=[1025.0, 840.0],
                confidence=0.95,
                track_id=1,
                jersey_number=7,
                jersey_confidence=0.9,
                jersey_probs={7: 0.9},
                hsv_histogram_jersey=None,
                jersey_color_sampled=False,
                jersey_roi_valid=False,
                jersey_roi_bbox=None,
            ),
        ],
        ball_detections=[],
    )
    pass2_output = Pass2COutput(
        fragments=[
            ScoredFragment(
                fragment_id="F000001",
                original_track_id=1,
                start_frame=0,
                end_frame=1,
                detection_ids=["0_1_deadbeef", "1_1_deadbeef"],
                quality=FragmentQuality.HIGH,
                quality_score=0.95,
            )
        ]
    )
    pass3_output = Pass3COutput(
        identities=[
            CommittedIdentity(
                fragment_id="F000001",
                player_id="P07_team_a",
                team=TeamID.TEAM_A,
                jersey_number=7,
                assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
                assignment_confidence=0.95,
            )
        ],
        solver_log={},
        unresolved_conflicts=[],
    )
    ball_output = BallInterpolationOutput(
        ball_positions=[
            BallPosition(frame_idx=0, state=BallState.UNKNOWN, centroid=None, bbox=None, confidence=0.0),
            BallPosition(frame_idx=1, state=BallState.UNKNOWN, centroid=None, bbox=None, confidence=0.0),
        ],
        interpolation_method=InterpolationMethod.LINEAR,
        total_frames=2,
    )
    calibration_config = {
        "homography": {
            "court_length": 40.0,
            "court_width": 20.0,
            "output_pixel_scale": 20,
            "source_points": [[680, 595], [843, 638], [147, 1075], [3136, 611], [2964, 650], [3644, 1100], [1900, 530], [1888, 1592], [303, 772], [128, 875], [3511, 808], [3692, 907], [1891, 760]],
            "dest_points": [[0.0, 19.0], [3.5, 17.0], [3.5, 5.0], [40.0, 19.0], [36.5, 17.0], [36.5, 5.0], [20.0, 19.0], [20.0, 1.0], [0.0, 11.5], [0.0, 8.5], [40.0, 11.5], [40.0, 8.5], [20.0, 10.0]],
        }
    }

    birdseye_output = build_birds_eye_projection_output(
        pass1_output=pass1_output,
        pass2c_output=pass2_output,
        pass3_output=pass3_output,
        ball_output=ball_output,
        calibration_config=calibration_config,
    )

    frame0_player = birdseye_output.frames[0].players[0]
    frame1_player = birdseye_output.frames[1].players[0]
    assert frame0_player.raw_image_anchor == [1040.0, 980.0]
    assert frame1_player.raw_image_anchor == [1025.0, 900.0]
    assert frame0_player.image_anchor == [1040.0, 980.0]
    assert frame1_player.image_anchor == [1040.0, 980.0]
    assert frame0_player.raw_court_position == frame0_player.court_position
    assert frame1_player.raw_court_position != frame1_player.court_position
    assert frame0_player.raw_render_position == frame0_player.render_position
    assert frame1_player.raw_render_position != frame1_player.render_position
    assert frame0_player.stabilization_trust == 1.0
    assert frame1_player.stabilization_trust == 0.0
    assert birdseye_output.diagnostics["head_projection_anchors"] == 0.0
    assert birdseye_output.diagnostics["foot_point_projection_anchors"] == 2.0
    assert birdseye_output.diagnostics["pose_anchor_refinements"] == 0.0
    assert birdseye_output.diagnostics["stabilized_player_positions"] == 1.0
    assert birdseye_output.diagnostics["position_stabilizer_bbox_rejections"] == 1.0
    assert birdseye_output.diagnostics["position_stabilizer_prediction_only_frames"] == 1.0
    assert birdseye_output.diagnostics["trajectory_smoothing_frames_modified"] == 0.0
    assert birdseye_output.diagnostics["trajectory_stationary_frames_modified"] == 0.0


def test_pose_anchor_from_keypoints_prefers_ankles():
    keypoints_xy = np.zeros((17, 2), dtype=float)
    keypoints_conf = np.zeros(17, dtype=float)
    keypoints_xy[15] = [42.0, 80.0]
    keypoints_xy[16] = [58.0, 82.0]
    keypoints_conf[15] = 0.9
    keypoints_conf[16] = 0.8

    anchor = _pose_anchor_from_keypoints(
        keypoints_xy=keypoints_xy,
        keypoints_conf=keypoints_conf,
        fallback_anchor=[50.0, 90.0],
        crop_height=120.0,
    )

    assert anchor == [50.0, 82.0]


def test_pose_anchor_from_upper_body_keypoints_prefers_head_and_expected_height():
    keypoints_xy = np.zeros((17, 2), dtype=float)
    keypoints_conf = np.zeros(17, dtype=float)
    keypoints_xy[0] = [46.0, 20.0]
    keypoints_xy[1] = [42.0, 22.0]
    keypoints_xy[2] = [50.0, 21.0]
    keypoints_conf[0] = 0.90
    keypoints_conf[1] = 0.80
    keypoints_conf[2] = 0.85

    anchor = _pose_anchor_from_upper_body_keypoints(
        keypoints_xy=keypoints_xy,
        keypoints_conf=keypoints_conf,
        fallback_anchor=[48.0, 120.0],
        expected_height=100.0,
    )

    assert anchor is not None
    assert round(anchor[0], 1) == 46.0
    assert round(anchor[1], 1) == 115.0


def test_blend_directional_motion_damps_sideways_jitter_more_than_forward_motion():
    stabilized = _blend_directional_motion(
        predicted=[10.0, 5.0],
        measurement=[10.7, 5.8],
        motion_direction=[1.0, 0.0],
        base_alpha=0.5,
    )

    forward_step = stabilized[0] - 10.0
    sideways_step = stabilized[1] - 5.0

    assert forward_step > 0.45
    assert sideways_step < 0.12
    assert forward_step > sideways_step * 4.0


def test_player_height_prior_grows_for_lower_players_on_screen():
    frames = [
        BirdseyeFrame(
            frame_idx=0,
            players=[
                BirdseyePlayerPosition(
                    frame_idx=0,
                    fragment_id="F1",
                    player_id="P1",
                    team=TeamID.TEAM_A,
                    jersey_number=7,
                    track_id=1,
                    is_estimated=False,
                    image_bbox=[100, 100, 140, 220],
                    image_anchor=[120, 220],
                    court_position=[10.0, 10.0],
                    render_position=[10.0, 10.0],
                ),
                BirdseyePlayerPosition(
                    frame_idx=0,
                    fragment_id="F2",
                    player_id="P2",
                    team=TeamID.TEAM_A,
                    jersey_number=8,
                    track_id=2,
                    is_estimated=False,
                    image_bbox=[100, 300, 160, 500],
                    image_anchor=[130, 500],
                    court_position=[10.0, 10.0],
                    render_position=[10.0, 10.0],
                ),
            ],
            ball=BirdseyeBallFrame(frame_idx=0, state=BallState.UNKNOWN, confidence=0.0),
        )
    ]

    prior = _build_player_height_prior(frames)

    assert _expected_player_height(prior, 160.0) < _expected_player_height(prior, 400.0)


def test_trajectory_segment_smoothing_adjusts_only_interior_coherent_frames():
    calibration_config = {
        "homography": {
            "court_length": 40.0,
            "court_width": 20.0,
            "output_pixel_scale": 20,
            "source_points": [[680, 595], [843, 638], [147, 1075], [3136, 611], [2964, 650], [3644, 1100], [1900, 530], [1888, 1592], [303, 772], [128, 875], [3511, 808], [3692, 907], [1891, 760]],
            "dest_points": [[0.0, 19.0], [3.5, 17.0], [3.5, 5.0], [40.0, 19.0], [36.5, 17.0], [36.5, 5.0], [20.0, 19.0], [20.0, 1.0], [0.0, 11.5], [0.0, 8.5], [40.0, 11.5], [40.0, 8.5], [20.0, 10.0]],
        }
    }
    homography = create_homography_from_config(calibration_config)
    assert homography is not None

    y_values = [5.000, 5.011, 5.002, 5.014, 5.008, 5.019]
    frames = []
    for frame_idx, y_value in enumerate(y_values):
        player = BirdseyePlayerPosition(
            frame_idx=frame_idx,
            fragment_id="F1",
            player_id="P1",
            team=TeamID.TEAM_A,
            jersey_number=7,
            track_id=1,
            is_estimated=False,
            image_bbox=[100, 100, 140, 220],
            image_anchor=[120, 220],
            raw_image_anchor=[120, 220],
            raw_court_position=[10.0 + frame_idx * 0.12, y_value],
            raw_render_position=[0.0, 0.0],
            stabilization_trust=0.8,
            court_position=[10.0 + frame_idx * 0.12, y_value],
            render_position=[0.0, 0.0],
        )
        frames.append(BirdseyeFrame(frame_idx=frame_idx, players=[player], ball=BirdseyeBallFrame(frame_idx=frame_idx, state=BallState.UNKNOWN, confidence=0.0)))

    diagnostics = _apply_trajectory_segment_smoothing(
        frames=frames,
        homography=homography,
        court_length_m=40.0,
        court_width_m=20.0,
    )

    assert diagnostics["trajectory_smoothing_segments_smoothed"] == 1.0
    assert diagnostics["trajectory_smoothing_frames_modified"] >= 2.0
    assert frames[0].players[0].court_position[1] == y_values[0]
    assert frames[-1].players[0].court_position[1] == y_values[-1]
    assert abs(frames[2].players[0].court_position[1] - 5.006) < abs(y_values[2] - 5.006)
    assert diagnostics["trajectory_smoothing_max_delta_m"] <= 0.20


def test_journey_path_smoothing_interpolates_long_straight_run_at_constant_speed():
    calibration_config = {
        "homography": {
            "court_length": 40.0,
            "court_width": 20.0,
            "output_pixel_scale": 20,
            "source_points": [[680, 595], [843, 638], [147, 1075], [3136, 611], [2964, 650], [3644, 1100], [1900, 530], [1888, 1592], [303, 772], [128, 875], [3511, 808], [3692, 907], [1891, 760]],
            "dest_points": [[0.0, 19.0], [3.5, 17.0], [3.5, 5.0], [40.0, 19.0], [36.5, 17.0], [36.5, 5.0], [20.0, 19.0], [20.0, 1.0], [0.0, 11.5], [0.0, 8.5], [40.0, 11.5], [40.0, 8.5], [20.0, 10.0]],
        }
    }
    homography = create_homography_from_config(calibration_config)
    assert homography is not None

    raw_points = [
        [10.0, 5.0],
        [11.4, 5.05],
        [12.7, 4.98],
        [14.0, 5.04],
        [15.1, 5.01],
        [16.0, 5.0],
    ]
    frames = []
    for frame_idx, point in enumerate(raw_points):
        player = BirdseyePlayerPosition(
            frame_idx=frame_idx,
            fragment_id="F1",
            player_id="P1",
            team=TeamID.TEAM_A,
            jersey_number=7,
            track_id=1,
            is_estimated=False,
            image_bbox=[100, 100, 140, 220],
            image_anchor=[120, 220],
            raw_image_anchor=[120, 220],
            raw_court_position=list(point),
            raw_render_position=[0.0, 0.0],
            stabilization_trust=1.0,
            court_position=list(point),
            render_position=[0.0, 0.0],
        )
        frames.append(BirdseyeFrame(frame_idx=frame_idx, players=[player], ball=BirdseyeBallFrame(frame_idx=frame_idx, state=BallState.UNKNOWN, confidence=0.0)))

    diagnostics = _apply_journey_path_smoothing(
        frames=frames,
        homography=homography,
        fps=10.0,
        court_length_m=40.0,
        court_width_m=20.0,
    )

    assert diagnostics["journey_segments_smoothed"] == 1.0
    assert diagnostics["journey_frames_modified"] >= 2.0
    assert len(diagnostics["journeys"]) == 1
    assert diagnostics["journeys"][0]["point_a"] == [10.0, 5.0]
    assert diagnostics["journeys"][0]["point_b"] == [16.0, 5.0]
    assert abs(frames[2].players[0].court_position[0] - 12.4) < 1e-6
    assert abs(frames[2].players[0].court_position[1] - 5.0) < 1e-6
    first_step = frames[1].players[0].court_position[0] - frames[0].players[0].court_position[0]
    middle_step = frames[3].players[0].court_position[0] - frames[2].players[0].court_position[0]
    assert abs(first_step - middle_step) < 1e-6


def test_journey_path_smoothing_skips_short_movements():
    calibration_config = {
        "homography": {
            "court_length": 40.0,
            "court_width": 20.0,
            "output_pixel_scale": 20,
            "source_points": [[680, 595], [843, 638], [147, 1075], [3136, 611], [2964, 650], [3644, 1100], [1900, 530], [1888, 1592], [303, 772], [128, 875], [3511, 808], [3692, 907], [1891, 760]],
            "dest_points": [[0.0, 19.0], [3.5, 17.0], [3.5, 5.0], [40.0, 19.0], [36.5, 17.0], [36.5, 5.0], [20.0, 19.0], [20.0, 1.0], [0.0, 11.5], [0.0, 8.5], [40.0, 11.5], [40.0, 8.5], [20.0, 10.0]],
        }
    }
    homography = create_homography_from_config(calibration_config)
    assert homography is not None

    raw_points = [[10.0, 5.0], [10.8, 5.1], [11.6, 5.0], [12.3, 5.05]]
    frames = []
    for frame_idx, point in enumerate(raw_points):
        player = BirdseyePlayerPosition(
            frame_idx=frame_idx,
            fragment_id="F1",
            player_id="P1",
            team=TeamID.TEAM_A,
            jersey_number=7,
            track_id=1,
            is_estimated=False,
            image_bbox=[100, 100, 140, 220],
            image_anchor=[120, 220],
            raw_image_anchor=[120, 220],
            raw_court_position=list(point),
            raw_render_position=[0.0, 0.0],
            stabilization_trust=1.0,
            court_position=list(point),
            render_position=[0.0, 0.0],
        )
        frames.append(BirdseyeFrame(frame_idx=frame_idx, players=[player], ball=BirdseyeBallFrame(frame_idx=frame_idx, state=BallState.UNKNOWN, confidence=0.0)))

    diagnostics = _apply_journey_path_smoothing(
        frames=frames,
        homography=homography,
        fps=10.0,
        court_length_m=40.0,
        court_width_m=20.0,
    )

    assert diagnostics["journey_segments_detected"] == 0.0
    assert diagnostics["journey_segments_smoothed"] == 0.0
    assert diagnostics["journey_frames_modified"] == 0.0
    assert diagnostics["journeys"] == []
    assert [frame.players[0].court_position for frame in frames] == raw_points


def test_build_journey_activity_lookup_marks_only_active_frames():
    lookup = _build_journey_activity_lookup(
        {
            "journeys": [
                {"player_id": "P04_team_b", "start_frame": 210, "end_frame": 326},
                {"player_id": "P01_team_a", "start_frame": 100, "end_frame": 102},
            ]
        }
    )

    assert "P04_team_b" in lookup[210]
    assert "P04_team_b" in lookup[326]
    assert 209 not in lookup
    assert 327 not in lookup
    assert lookup[101] == ["P01_team_a"]


def test_stationary_span_smoothing_holds_near_median_position():
    calibration_config = {
        "homography": {
            "court_length": 40.0,
            "court_width": 20.0,
            "output_pixel_scale": 20,
            "source_points": [[680, 595], [843, 638], [147, 1075], [3136, 611], [2964, 650], [3644, 1100], [1900, 530], [1888, 1592], [303, 772], [128, 875], [3511, 808], [3692, 907], [1891, 760]],
            "dest_points": [[0.0, 19.0], [3.5, 17.0], [3.5, 5.0], [40.0, 19.0], [36.5, 17.0], [36.5, 5.0], [20.0, 19.0], [20.0, 1.0], [0.0, 11.5], [0.0, 8.5], [40.0, 11.5], [40.0, 8.5], [20.0, 10.0]],
        }
    }
    homography = create_homography_from_config(calibration_config)
    assert homography is not None

    x_values = [10.0, 10.015, 10.005, 10.012, 10.001]
    y_values = [5.0, 4.995, 5.006, 5.002, 4.999]
    frames = []
    for frame_idx, (x_value, y_value) in enumerate(zip(x_values, y_values)):
        player = BirdseyePlayerPosition(
            frame_idx=frame_idx,
            fragment_id="F1",
            player_id="P1",
            team=TeamID.TEAM_A,
            jersey_number=7,
            track_id=1,
            is_estimated=False,
            image_bbox=[100, 100, 140, 220],
            image_anchor=[120, 220],
            raw_image_anchor=[120, 220],
            raw_court_position=[x_value, y_value],
            raw_render_position=[0.0, 0.0],
            stabilization_trust=0.85,
            court_position=[x_value, y_value],
            render_position=[0.0, 0.0],
        )
        frames.append(BirdseyeFrame(frame_idx=frame_idx, players=[player], ball=BirdseyeBallFrame(frame_idx=frame_idx, state=BallState.UNKNOWN, confidence=0.0)))

    diagnostics = _apply_stationary_span_smoothing(
        frames=frames,
        homography=homography,
        court_length_m=40.0,
        court_width_m=20.0,
    )

    assert diagnostics["trajectory_stationary_segments_smoothed"] == 1.0
    assert diagnostics["trajectory_stationary_frames_modified"] >= 3.0
    assert max(abs(frame.players[0].court_position[0] - 10.0) for frame in frames) < max(abs(x_value - 10.0) for x_value in x_values)
    assert diagnostics["trajectory_stationary_max_delta_m"] <= 0.12


def test_classify_motion_span_distinguishes_stationary_coherent_and_reactive():
    stationary_span = [
        BirdseyePlayerPosition(
            frame_idx=index,
            fragment_id="F1",
            player_id="P1",
            team=TeamID.TEAM_A,
            jersey_number=7,
            track_id=1,
            is_estimated=False,
            image_bbox=[100, 100, 140, 220],
            image_anchor=[120, 220],
            court_position=[10.0 + x_offset, 5.0 + y_offset],
            render_position=[0.0, 0.0],
            stabilization_trust=0.9,
        )
        for index, (x_offset, y_offset) in enumerate([(0.0, 0.0), (0.01, -0.004), (0.004, 0.006), (0.009, 0.002)])
    ]
    coherent_span = [
        BirdseyePlayerPosition(
            frame_idx=index,
            fragment_id="F1",
            player_id="P1",
            team=TeamID.TEAM_A,
            jersey_number=7,
            track_id=1,
            is_estimated=False,
            image_bbox=[100, 100, 140, 220],
            image_anchor=[120, 220],
            court_position=[10.0 + index * 0.12, 5.0 + index * 0.01],
            render_position=[0.0, 0.0],
            stabilization_trust=0.9,
        )
        for index in range(6)
    ]
    reactive_span = [
        BirdseyePlayerPosition(
            frame_idx=index,
            fragment_id="F1",
            player_id="P1",
            team=TeamID.TEAM_A,
            jersey_number=7,
            track_id=1,
            is_estimated=False,
            image_bbox=[100, 100, 140, 220],
            image_anchor=[120, 220],
            court_position=point,
            render_position=[0.0, 0.0],
            stabilization_trust=0.9,
        )
        for index, point in enumerate([[10.0, 5.0], [10.12, 5.02], [10.24, 5.03], [10.18, 5.16], [10.09, 5.28], [9.98, 5.40]])
    ]

    assert _classify_motion_span(stationary_span) == "stationary"
    assert _classify_motion_span(coherent_span) == "coherent"
    assert _classify_motion_span(reactive_span) == "reactive"


def test_birdseye_validation_fails_when_real_ball_has_no_projection():
    output = BirdseyeProjectionOutput(
        video_name="unit",
        fps=30.0,
        total_frames=1,
        processed_start_frame=0,
        processed_end_frame_exclusive=1,
        court_length_m=40.0,
        court_width_m=20.0,
        output_pixel_scale=20,
        frames=[
            {
                "frame_idx": 0,
                "players": [],
                "ball": {
                    "frame_idx": 0,
                    "state": "real",
                    "confidence": 0.7,
                    "image_position": [1905, 765],
                    "court_position": None,
                    "render_position": None,
                },
            }
        ],
        diagnostics={},
    )

    validator = Validator()
    result = validator.validate_birdseye(output)

    assert not result.passed
    rules = {violation.rule for violation in result.violations}
    assert "BIRDSEYE_BALL_MISSING_POSITION" in rules


def test_create_court_view_uses_futsal_penalty_area_depth():
    court = create_court_view(width=800, height=400, court_length=40.0, court_width=20.0)

    left_penalty_border = tuple(int(v) for v in court[200, 90])
    old_depth_position = tuple(int(v) for v in court[200, 130])

    assert left_penalty_border == (255, 255, 255)
    assert old_depth_position == (20, 70, 20)


def test_visualization_renderer_writes_video_from_birdseye_artifact(tmp_path):
    video_path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    assert writer.isOpened()
    writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    birdseye_output = BirdseyeProjectionOutput(
        video_name="unit",
        fps=10.0,
        total_frames=2,
        processed_start_frame=0,
        processed_end_frame_exclusive=2,
        court_length_m=40.0,
        court_width_m=20.0,
        output_pixel_scale=20,
        frames=[
            BirdseyeFrame(
                frame_idx=0,
                players=[
                    BirdseyePlayerPosition(
                        frame_idx=0,
                        fragment_id="F000001",
                        player_id="P07_team_a",
                        team=TeamID.TEAM_A,
                        jersey_number=7,
                        track_id=1,
                        is_ghost=False,
                        is_estimated=False,
                        image_bbox=[20, 20, 60, 100],
                        image_anchor=[40, 100],
                        raw_image_anchor=[42, 100],
                        raw_court_position=[10.0, 5.0],
                        raw_render_position=[200.0, 200.0],
                        stabilization_trust=1.0,
                        court_position=[10.0, 5.0],
                        render_position=[200.0, 200.0],
                    )
                ],
                ball=BirdseyeBallFrame(
                    frame_idx=0,
                    state=BallState.REAL,
                    confidence=0.8,
                    image_bbox=[75, 60, 85, 70],
                    image_position=[80, 65],
                    court_position=[20.0, 10.0],
                    render_position=[400.0, 200.0],
                ),
            ),
            BirdseyeFrame(
                frame_idx=1,
                players=[],
                ball=BirdseyeBallFrame(
                    frame_idx=1,
                    state=BallState.UNKNOWN,
                    confidence=0.0,
                    image_bbox=None,
                    image_position=None,
                    court_position=None,
                    render_position=None,
                ),
            ),
        ],
        diagnostics={},
    )

    birdseye_output_path = tmp_path / "birdseye_projection.json"
    birdseye_output_path.write_text(json.dumps(birdseye_output.model_dump(mode="json")), encoding="utf-8")

    visualization_path = tmp_path / "visualization.mp4"
    render_visualization_from_artifact(
        video_path=str(video_path),
        birdseye_output_path=str(birdseye_output_path),
        visualization_path=str(visualization_path),
    )

    assert visualization_path.exists()
    assert visualization_path.stat().st_size > 0


def test_pass3_jersey_conflict_resolution_preserves_disjoint_same_number_fragments():
    fragments = [
        ScoredFragment(
            fragment_id="F000001",
            original_track_id=1,
            start_frame=0,
            end_frame=10,
            detection_ids=["0_1_a"],
            quality=FragmentQuality.HIGH,
            quality_score=0.95,
        ),
        ScoredFragment(
            fragment_id="F000002",
            original_track_id=2,
            start_frame=5,
            end_frame=15,
            detection_ids=["5_2_b"],
            quality=FragmentQuality.HIGH,
            quality_score=0.90,
        ),
        ScoredFragment(
            fragment_id="F000003",
            original_track_id=3,
            start_frame=20,
            end_frame=30,
            detection_ids=["20_3_c"],
            quality=FragmentQuality.HIGH,
            quality_score=0.88,
        ),
        ScoredFragment(
            fragment_id="F000004",
            original_track_id=4,
            start_frame=6,
            end_frame=14,
            detection_ids=["6_4_d"],
            quality=FragmentQuality.HIGH,
            quality_score=0.87,
        ),
    ]

    identities = [
        CommittedIdentity(
            fragment_id="F000001",
            player_id="P01_team_a",
            team=TeamID.TEAM_A,
            jersey_number=10,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
            assignment_reasons=["JERSEY_GLOBAL_ASSIGNED"],
        ),
        CommittedIdentity(
            fragment_id="F000002",
            player_id="P02_team_a",
            team=TeamID.TEAM_A,
            jersey_number=10,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.60,
            assignment_reasons=["JERSEY_GLOBAL_ASSIGNED"],
        ),
        CommittedIdentity(
            fragment_id="F000003",
            player_id="P03_team_a",
            team=TeamID.TEAM_A,
            jersey_number=10,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.92,
            assignment_reasons=["JERSEY_GLOBAL_ASSIGNED"],
        ),
        CommittedIdentity(
            fragment_id="F000004",
            player_id="P04_team_b",
            team=TeamID.TEAM_B,
            jersey_number=10,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.91,
            assignment_reasons=["JERSEY_GLOBAL_ASSIGNED"],
        ),
    ]

    dropped = IdentitySolver()._resolve_team_jersey_conflicts(identities, fragments)

    identity_by_fragment = {identity.fragment_id: identity for identity in identities}
    assert dropped == 1
    assert identity_by_fragment["F000001"].jersey_number == 10
    assert identity_by_fragment["F000002"].jersey_number is None
    assert "JERSEY_EXCLUSIVITY_DROPPED" in (identity_by_fragment["F000002"].assignment_reasons or [])
    assert identity_by_fragment["F000003"].jersey_number == 10
    assert identity_by_fragment["F000004"].jersey_number == 10


def test_pass3_ghost_windows_use_pass3a_reappearance_edges_before_cap_pruning():
    fragments = [
        ScoredFragment(
            fragment_id="F_ORANGE",
            original_track_id=1,
            start_frame=470,
            end_frame=604,
            detection_ids=["470_1_orange"],
            quality=FragmentQuality.HIGH,
            quality_score=0.95,
        ),
        ScoredFragment(
            fragment_id="G_ORANGE",
            original_track_id=1,
            start_frame=605,
            end_frame=900,
            detection_ids=[],
            quality=FragmentQuality.GHOST,
            quality_score=0.0,
            is_ghost=True,
            exclude_from_clustering=True,
            ghost_last_known_bbox=[3000, 600, 3050, 720],
            ghost_last_known_centroid=[3025, 660],
            ghost_reason="player_off_screen",
        ),
        ScoredFragment(
            fragment_id="F_BLACK",
            original_track_id=2,
            start_frame=0,
            end_frame=577,
            detection_ids=["0_2_black"],
            quality=FragmentQuality.HIGH,
            quality_score=0.95,
        ),
        ScoredFragment(
            fragment_id="G_BLACK",
            original_track_id=2,
            start_frame=578,
            end_frame=900,
            detection_ids=[],
            quality=FragmentQuality.GHOST,
            quality_score=0.0,
            is_ghost=True,
            exclude_from_clustering=True,
            ghost_last_known_bbox=[3340, 620, 3385, 753],
            ghost_last_known_centroid=[3360, 686],
            ghost_reason="player_off_screen",
        ),
        ScoredFragment(
            fragment_id="F_RETURN_ORANGE",
            original_track_id=3,
            start_frame=610,
            end_frame=700,
            detection_ids=["610_3_return_orange"],
            quality=FragmentQuality.HIGH,
            quality_score=0.93,
        ),
        ScoredFragment(
            fragment_id="F_RETURN_BLACK",
            original_track_id=4,
            start_frame=625,
            end_frame=700,
            detection_ids=["625_4_return_black"],
            quality=FragmentQuality.HIGH,
            quality_score=0.92,
        ),
    ]

    identity_groups = {
        "F_ORANGE": {"F_ORANGE", "G_ORANGE"},
        "F_BLACK": {"F_BLACK", "G_BLACK"},
        "F_RETURN_ORANGE": {"F_RETURN_ORANGE"},
        "F_RETURN_BLACK": {"F_RETURN_BLACK"},
    }

    pass3a_edges = [
        IdentityCandidateEdge(
            fragment_a="F_ORANGE",
            fragment_b="F_RETURN_ORANGE",
            temporal_gap=6,
            spatial_distance=52.0,
            velocity_consistency_score=0.74,
            appearance_similarity=0.83,
            jersey_similarity=0.5,
            temporal_gap_score=0.98,
            overall_candidate_score=0.37,
        ),
        IdentityCandidateEdge(
            fragment_a="F_ORANGE",
            fragment_b="F_RETURN_BLACK",
            temporal_gap=21,
            spatial_distance=32.0,
            velocity_consistency_score=0.87,
            appearance_similarity=0.53,
            jersey_similarity=0.5,
            temporal_gap_score=0.93,
            overall_candidate_score=0.35,
        ),
        IdentityCandidateEdge(
            fragment_a="F_BLACK",
            fragment_b="F_RETURN_ORANGE",
            temporal_gap=33,
            spatial_distance=82.0,
            velocity_consistency_score=0.95,
            appearance_similarity=0.70,
            jersey_similarity=0.5,
            temporal_gap_score=0.89,
            overall_candidate_score=0.39,
        ),
        IdentityCandidateEdge(
            fragment_a="F_BLACK",
            fragment_b="F_RETURN_BLACK",
            temporal_gap=48,
            spatial_distance=68.0,
            velocity_consistency_score=0.98,
            appearance_similarity=0.93,
            jersey_similarity=1.0,
            temporal_gap_score=0.84,
            overall_candidate_score=0.48,
        ),
    ]

    _, retired_ghosts, ghost_windows, _ = IdentitySolver()._validate_identity_collapse_pre_attributes(
        fragments,
        identity_groups,
        pass3a_edges=pass3a_edges,
        fragment_jersey_evidence={"F_BLACK": 7},
    )

    assert retired_ghosts == set()
    assert ghost_windows["G_ORANGE"]["window_reason"] == "matched_reappearance"
    assert ghost_windows["G_ORANGE"]["target_fragment_id"] == "F_RETURN_ORANGE"
    assert ghost_windows["G_ORANGE"]["matched_reappearance_frame"] == 610
    assert ghost_windows["G_ORANGE"]["end_frame"] == 609
    assert ghost_windows["G_ORANGE"]["source_fragment_id"] == "F_ORANGE"
    assert ghost_windows["G_BLACK"]["window_reason"] == "matched_reappearance"
    assert ghost_windows["G_BLACK"]["target_fragment_id"] == "F_RETURN_BLACK"
    assert ghost_windows["G_BLACK"]["matched_reappearance_frame"] == 625
    assert ghost_windows["G_BLACK"]["end_frame"] == 624
    assert ghost_windows["G_BLACK"]["source_fragment_id"] == "F_BLACK"
    assert ghost_windows["G_BLACK"]["defining_jersey_number"] == 7


def test_ghost_bbox_for_frame_extrapolates_source_motion_when_reappearance_unresolved():
    source_fragment = ScoredFragment(
        fragment_id="F_SRC",
        original_track_id=7,
        start_frame=0,
        end_frame=2,
        detection_ids=["0_7_a", "1_7_b", "2_7_c"],
        quality=FragmentQuality.HIGH,
        quality_score=0.95,
    )
    ghost_fragment = ScoredFragment(
        fragment_id="G_SRC",
        original_track_id=7,
        start_frame=3,
        end_frame=10,
        detection_ids=[],
        quality=FragmentQuality.GHOST,
        quality_score=0.0,
        is_ghost=True,
        ghost_last_known_bbox=[120, 100, 200, 300],
        ghost_last_known_centroid=[160, 200],
    )
    detections_by_id = {
        "0_7_a": Detection(
            detection_id="0_7_a",
            frame_idx=0,
            bbox=[100, 100, 180, 300],
            centroid=[140, 200],
            confidence=0.9,
            track_id=7,
            jersey_number=None,
            jersey_confidence=0.0,
            jersey_probs=None,
            hsv_histogram_jersey=None,
            jersey_color_sampled=False,
            jersey_roi_valid=False,
            jersey_roi_bbox=None,
        ),
        "1_7_b": Detection(
            detection_id="1_7_b",
            frame_idx=1,
            bbox=[110, 100, 190, 300],
            centroid=[150, 200],
            confidence=0.9,
            track_id=7,
            jersey_number=None,
            jersey_confidence=0.0,
            jersey_probs=None,
            hsv_histogram_jersey=None,
            jersey_color_sampled=False,
            jersey_roi_valid=False,
            jersey_roi_bbox=None,
        ),
        "2_7_c": Detection(
            detection_id="2_7_c",
            frame_idx=2,
            bbox=[120, 100, 200, 300],
            centroid=[160, 200],
            confidence=0.9,
            track_id=7,
            jersey_number=None,
            jersey_confidence=0.0,
            jersey_probs=None,
            hsv_histogram_jersey=None,
            jersey_color_sampled=False,
            jersey_roi_valid=False,
            jersey_roi_bbox=None,
        ),
    }
    ghost_window = {
        "start_frame": 3,
        "end_frame": 10,
        "matched_reappearance_frame": None,
        "target_fragment_id": None,
        "window_reason": "unmatched_exit",
        "source_fragment_id": "F_SRC",
    }

    bbox = _ghost_bbox_for_frame(
        ghost_fragment,
        frame_idx=5,
        ghost_window=ghost_window,
        fragment_by_id={"F_SRC": source_fragment, "G_SRC": ghost_fragment},
        detections_by_id=detections_by_id,
    )

    assert bbox is not None
    assert bbox == [140.0, 100.0, 220.0, 300.0]


def test_ghost_bbox_for_frame_preserves_size_when_source_boxes_expand():
    source_fragment = ScoredFragment(
        fragment_id="F_SHEAR",
        original_track_id=22,
        start_frame=0,
        end_frame=2,
        detection_ids=["0_22_a", "1_22_b", "2_22_c"],
        quality=FragmentQuality.HIGH,
        quality_score=0.95,
    )
    ghost_fragment = ScoredFragment(
        fragment_id="G_SHEAR",
        original_track_id=22,
        start_frame=3,
        end_frame=15,
        detection_ids=[],
        quality=FragmentQuality.GHOST,
        quality_score=0.0,
        is_ghost=True,
        ghost_last_known_bbox=[100, 200, 150, 320],
        ghost_last_known_centroid=[125, 260],
    )
    detections_by_id = {
        "0_22_a": Detection(
            detection_id="0_22_a",
            frame_idx=0,
            bbox=[80, 180, 120, 250],
            centroid=[100, 215],
            confidence=0.9,
            track_id=22,
            jersey_number=None,
            jersey_confidence=0.0,
            jersey_probs=None,
            hsv_histogram_jersey=None,
            jersey_color_sampled=False,
            jersey_roi_valid=False,
            jersey_roi_bbox=None,
        ),
        "1_22_b": Detection(
            detection_id="1_22_b",
            frame_idx=1,
            bbox=[82, 182, 130, 285],
            centroid=[106, 233.5],
            confidence=0.9,
            track_id=22,
            jersey_number=None,
            jersey_confidence=0.0,
            jersey_probs=None,
            hsv_histogram_jersey=None,
            jersey_color_sampled=False,
            jersey_roi_valid=False,
            jersey_roi_bbox=None,
        ),
        "2_22_c": Detection(
            detection_id="2_22_c",
            frame_idx=2,
            bbox=[84, 184, 144, 340],
            centroid=[114, 262],
            confidence=0.9,
            track_id=22,
            jersey_number=None,
            jersey_confidence=0.0,
            jersey_probs=None,
            hsv_histogram_jersey=None,
            jersey_color_sampled=False,
            jersey_roi_valid=False,
            jersey_roi_bbox=None,
        ),
    }
    ghost_window = {
        "start_frame": 3,
        "end_frame": 15,
        "matched_reappearance_frame": None,
        "target_fragment_id": None,
        "window_reason": "unmatched_exit",
        "source_fragment_id": "F_SHEAR",
    }

    bbox = _ghost_bbox_for_frame(
        ghost_fragment,
        frame_idx=6,
        ghost_window=ghost_window,
        fragment_by_id={"F_SHEAR": source_fragment, "G_SHEAR": ghost_fragment},
        detections_by_id=detections_by_id,
    )

    assert bbox is not None
    assert bbox[2] - bbox[0] == 50.0
    assert bbox[3] - bbox[1] == 120.0


def test_pass3_global_jersey_reassignment_prefers_single_continuous_owner():
    fragments = [
        ScoredFragment(
            fragment_id="F000001",
            original_track_id=1,
            start_frame=0,
            end_frame=40,
            detection_ids=["0_1_a"],
            quality=FragmentQuality.HIGH,
            quality_score=0.90,
        ),
        ScoredFragment(
            fragment_id="F000002",
            original_track_id=2,
            start_frame=41,
            end_frame=80,
            detection_ids=["41_2_b"],
            quality=FragmentQuality.HIGH,
            quality_score=0.90,
        ),
        ScoredFragment(
            fragment_id="F000003",
            original_track_id=3,
            start_frame=0,
            end_frame=200,
            detection_ids=["0_3_c"],
            quality=FragmentQuality.HIGH,
            quality_score=0.95,
        ),
    ]

    identities = [
        CommittedIdentity(
            fragment_id="F000001",
            player_id="P01_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.80,
        ),
        CommittedIdentity(
            fragment_id="F000002",
            player_id="P02_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.80,
        ),
        CommittedIdentity(
            fragment_id="F000003",
            player_id="P03_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
    ]

    updated, diagnostics = IdentitySolver()._reassign_jerseys_globally(
        identities,
        fragments,
        fragment_jersey_scores={
            "F000001": {4: 0.92},
            "F000002": {4: 0.91},
            "F000003": {4: 0.95},
        },
    )

    updated_by_player = {identity.player_id: identity for identity in updated}
    assert diagnostics["jersey_global_assigned_players"]["4"] == ["P03_team_a"]
    assert updated_by_player["P03_team_a"].jersey_number == 4
    assert updated_by_player["P01_team_a"].jersey_number is None
    assert updated_by_player["P02_team_a"].jersey_number is None


def test_pass3_global_jersey_reassignment_keeps_strong_incumbent_label():
    fragments = [
        ScoredFragment(
            fragment_id="F000001",
            original_track_id=5,
            start_frame=0,
            end_frame=200,
            detection_ids=["0_5_a"],
            quality=FragmentQuality.HIGH,
            quality_score=0.95,
        ),
    ]

    identities = [
        CommittedIdentity(
            fragment_id="F000001",
            player_id="P06_team_a",
            team=TeamID.TEAM_A,
            jersey_number=10,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
            assignment_reasons=["JERSEY_INHERITED"],
        ),
    ]

    updated, diagnostics = IdentitySolver()._reassign_jerseys_globally(
        identities,
        fragments,
        fragment_jersey_scores={
            "F000001": {10: 0.9778, 4: 0.8202},
        },
    )

    assert diagnostics["jersey_global_assigned_players"]["10"] == ["P06_team_a"]
    assert updated[0].jersey_number == 10


def test_pass3_global_jersey_exclusivity_drops_overlapping_ghost_label():
    fragments = [
        ScoredFragment(
            fragment_id="F000001",
            original_track_id=12,
            start_frame=600,
            end_frame=700,
            detection_ids=["600_12_real"],
            quality=FragmentQuality.HIGH,
            quality_score=0.95,
        ),
        ScoredFragment(
            fragment_id="G000001",
            original_track_id=99,
            start_frame=640,
            end_frame=680,
            detection_ids=[],
            quality=FragmentQuality.GHOST,
            quality_score=0.0,
            is_ghost=True,
            exclude_from_clustering=True,
            ghost_last_known_bbox=[100, 100, 180, 300],
            ghost_last_known_centroid=[140, 200],
            ghost_reason="player_occluded",
        ),
    ]

    identities = [
        CommittedIdentity(
            fragment_id="F000001",
            player_id="P14_team_a",
            team=TeamID.TEAM_A,
            jersey_number=4,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
        CommittedIdentity(
            fragment_id="G000001",
            player_id="P99_team_b",
            team=TeamID.TEAM_B,
            jersey_number=4,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.40,
        ),
    ]

    dropped = IdentitySolver()._resolve_global_jersey_conflicts(
        committed_identities=identities,
        fragments=fragments,
        ghost_activity_windows={
            "G000001": {
                "start_frame": 650,
                "end_frame": 660,
            }
        },
    )

    assert dropped == 1
    assert identities[0].jersey_number == 4
    assert identities[1].jersey_number is None
    assert "JERSEY_GLOBAL_EXCLUSIVITY_DROPPED" in (identities[1].assignment_reasons or [])


def test_pass3_global_jersey_reassignment_prefers_disjoint_true_sequence_over_long_noisy_interval():
    fragments = [
        ScoredFragment(
            fragment_id="F000006",
            original_track_id=5,
            start_frame=0,
            end_frame=190,
            detection_ids=["0_5_a"],
            quality=FragmentQuality.HIGH,
            quality_score=0.95,
        ),
        ScoredFragment(
            fragment_id="F000012",
            original_track_id=8,
            start_frame=191,
            end_frame=300,
            detection_ids=["191_8_a"],
            quality=FragmentQuality.HIGH,
            quality_score=0.95,
        ),
        ScoredFragment(
            fragment_id="F000019",
            original_track_id=18,
            start_frame=311,
            end_frame=2086,
            detection_ids=["311_18_a"],
            quality=FragmentQuality.HIGH,
            quality_score=0.95,
        ),
        ScoredFragment(
            fragment_id="F000008",
            original_track_id=6,
            start_frame=0,
            end_frame=1552,
            detection_ids=["0_6_b"],
            quality=FragmentQuality.HIGH,
            quality_score=0.95,
        ),
    ]

    identities = [
        CommittedIdentity(
            fragment_id="F000006",
            player_id="P07_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
        CommittedIdentity(
            fragment_id="F000012",
            player_id="P13_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
        CommittedIdentity(
            fragment_id="F000019",
            player_id="P20_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
        CommittedIdentity(
            fragment_id="F000008",
            player_id="P09_team_b",
            team=TeamID.TEAM_B,
            jersey_number=10,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
    ]

    updated, diagnostics = IdentitySolver()._reassign_jerseys_globally(
        identities,
        fragments,
        fragment_jersey_scores={
            "F000006": {10: 0.9678, 4: 0.5276, 7: 0.7141},
            "F000012": {10: 0.8743, 4: 0.6309, 7: 0.4805},
            "F000019": {10: 0.9807, 4: 0.7573, 7: 0.4656},
            "F000008": {10: 0.6417, 4: 0.4529, 7: 0.3800},
        },
    )

    updated_by_player = {identity.player_id: identity for identity in updated}
    assert diagnostics["jersey_global_assigned_players"]["10"] == ["P07_team_a", "P20_team_a"]
    assert updated_by_player["P07_team_a"].jersey_number == 10
    assert updated_by_player["P13_team_a"].jersey_number is None
    assert updated_by_player["P20_team_a"].jersey_number == 10
    assert updated_by_player["P09_team_b"].jersey_number is None


def test_pass3_sanitize_ghost_window_when_target_identity_mismatches():
    ghost_fragment = ScoredFragment(
        fragment_id="G000030",
        track_id=29,
        start_frame=1090,
        end_frame=1313,
        detection_ids=[],
        quality=FragmentQuality.GHOST,
        quality_score=0.0,
        is_ghost=True,
    )
    identities = [
        CommittedIdentity(
            fragment_id="G000030",
            player_id="P22_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.GHOST_INHERITED,
            assignment_confidence=0.80,
            assignment_reasons=["MATCHED_GHOST_WINDOW"],
        ),
        CommittedIdentity(
            fragment_id="F000026",
            player_id="P27_team_b",
            team=TeamID.TEAM_B,
            jersey_number=None,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
    ]

    ghost_windows = {
        "G000030": {
            "start_frame": 1090,
            "end_frame": 1174,
            "matched_reappearance_frame": 1175,
            "target_fragment_id": "F000026",
            "window_reason": "matched_reappearance",
        }
    }

    cleared = IdentitySolver()._sanitize_ghost_activity_windows(
        committed_identities=identities,
        fragments=[ghost_fragment],
        ghost_activity_windows=ghost_windows,
    )

    assert cleared == 1
    assert ghost_windows["G000030"]["end_frame"] == 1313
    assert ghost_windows["G000030"]["matched_reappearance_frame"] is None
    assert ghost_windows["G000030"]["target_fragment_id"] is None
    assert ghost_windows["G000030"]["window_reason"] == "unmatched_exit"
    assert "MATCHED_GHOST_WINDOW" not in (identities[0].assignment_reasons or [])
    assert "UNMATCHED_EXIT" in (identities[0].assignment_reasons or [])
    assert "GHOST_TARGET_IDENTITY_MISMATCH" in (identities[0].assignment_reasons or [])


def test_pass3_sanitize_ghost_window_allows_same_team_fallback_without_defining_jersey():
    ghost_fragment = ScoredFragment(
        fragment_id="G000040",
        track_id=40,
        start_frame=500,
        end_frame=540,
        detection_ids=[],
        quality=FragmentQuality.GHOST,
        quality_score=0.0,
        is_ghost=True,
    )
    target_fragment = ScoredFragment(
        fragment_id="F000041",
        track_id=41,
        start_frame=541,
        end_frame=620,
        detection_ids=["541_41_a"],
        quality=FragmentQuality.HIGH,
        quality_score=0.95,
    )
    identities = [
        CommittedIdentity(
            fragment_id="G000040",
            player_id="P13_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.GHOST_INHERITED,
            assignment_confidence=0.80,
            assignment_reasons=["MATCHED_GHOST_WINDOW"],
        ),
        CommittedIdentity(
            fragment_id="F000041",
            player_id="P20_team_a",
            team=TeamID.TEAM_A,
            jersey_number=10,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
    ]
    ghost_windows = {
        "G000040": {
            "start_frame": 500,
            "end_frame": 540,
            "matched_reappearance_frame": 541,
            "target_fragment_id": "F000041",
            "window_reason": "matched_reappearance",
        }
    }

    cleared = IdentitySolver()._sanitize_ghost_activity_windows(
        committed_identities=identities,
        fragments=[ghost_fragment, target_fragment],
        ghost_activity_windows=ghost_windows,
    )

    assert cleared == 0
    assert ghost_windows["G000040"]["target_fragment_id"] == "F000041"
    assert identities[0].player_id == "P20_team_a"
    assert identities[0].team == TeamID.TEAM_A
    assert identities[0].jersey_number == 10
    assert "GHOST_TEAM_FALLBACK_MATCH" in (identities[0].assignment_reasons or [])
    assert "UNMATCHED_EXIT" not in (identities[0].assignment_reasons or [])


def test_pass3_sanitize_ghost_window_rejects_same_team_fallback_with_defining_jersey():
    ghost_fragment = ScoredFragment(
        fragment_id="G000041",
        track_id=42,
        start_frame=500,
        end_frame=540,
        detection_ids=[],
        quality=FragmentQuality.GHOST,
        quality_score=0.0,
        is_ghost=True,
    )
    target_fragment = ScoredFragment(
        fragment_id="F000042",
        track_id=43,
        start_frame=541,
        end_frame=620,
        detection_ids=["541_43_a"],
        quality=FragmentQuality.HIGH,
        quality_score=0.95,
    )
    identities = [
        CommittedIdentity(
            fragment_id="G000041",
            player_id="P13_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.GHOST_INHERITED,
            assignment_confidence=0.80,
            assignment_reasons=["MATCHED_GHOST_WINDOW"],
        ),
        CommittedIdentity(
            fragment_id="F000042",
            player_id="P20_team_a",
            team=TeamID.TEAM_A,
            jersey_number=10,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
    ]
    ghost_windows = {
        "G000041": {
            "start_frame": 500,
            "end_frame": 540,
            "matched_reappearance_frame": 541,
            "target_fragment_id": "F000042",
            "window_reason": "matched_reappearance",
            "source_fragment_id": "F_SOURCE",
            "defining_jersey_number": 7,
        }
    }

    cleared = IdentitySolver()._sanitize_ghost_activity_windows(
        committed_identities=identities,
        fragments=[ghost_fragment, target_fragment],
        ghost_activity_windows=ghost_windows,
    )

    assert cleared == 1
    assert ghost_windows["G000041"]["target_fragment_id"] is None
    assert ghost_windows["G000041"]["matched_reappearance_frame"] is None
    assert ghost_windows["G000041"]["window_reason"] == "unmatched_exit"
    assert identities[0].player_id == "P13_team_a"
    assert identities[0].team == TeamID.TEAM_A
    assert identities[0].jersey_number == 7
    assert "UNMATCHED_EXIT" in (identities[0].assignment_reasons or [])
    assert "GHOST_TARGET_JERSEY_MISMATCH" in (identities[0].assignment_reasons or [])
    assert "GHOST_TEAM_FALLBACK_MATCH" not in (identities[0].assignment_reasons or [])


def test_pass3_sanitize_ghost_window_keeps_same_team_same_jersey_reappearance():
    ghost_fragment = ScoredFragment(
        fragment_id="G000042",
        track_id=44,
        start_frame=500,
        end_frame=540,
        detection_ids=[],
        quality=FragmentQuality.GHOST,
        quality_score=0.0,
        is_ghost=True,
    )
    target_fragment = ScoredFragment(
        fragment_id="F000043",
        track_id=45,
        start_frame=541,
        end_frame=620,
        detection_ids=["541_45_a"],
        quality=FragmentQuality.HIGH,
        quality_score=0.95,
    )
    identities = [
        CommittedIdentity(
            fragment_id="G000042",
            player_id="P13_team_a",
            team=TeamID.TEAM_A,
            jersey_number=7,
            assignment_method=AssignmentMethod.GHOST_INHERITED,
            assignment_confidence=0.80,
            assignment_reasons=["MATCHED_GHOST_WINDOW"],
        ),
        CommittedIdentity(
            fragment_id="F000043",
            player_id="P20_team_a",
            team=TeamID.TEAM_A,
            jersey_number=7,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
    ]
    ghost_windows = {
        "G000042": {
            "start_frame": 500,
            "end_frame": 540,
            "matched_reappearance_frame": 541,
            "target_fragment_id": "F000043",
            "window_reason": "matched_reappearance",
            "source_fragment_id": "F_SOURCE",
            "defining_jersey_number": 7,
        }
    }

    cleared = IdentitySolver()._sanitize_ghost_activity_windows(
        committed_identities=identities,
        fragments=[ghost_fragment, target_fragment],
        ghost_activity_windows=ghost_windows,
    )

    assert cleared == 0
    assert ghost_windows["G000042"]["target_fragment_id"] == "F000043"
    assert ghost_windows["G000042"]["matched_reappearance_frame"] == 541
    assert ghost_windows["G000042"]["window_reason"] == "matched_reappearance"
    assert identities[0].player_id == "P20_team_a"
    assert identities[0].team == TeamID.TEAM_A
    assert identities[0].jersey_number == 7
    assert "GHOST_JERSEY_LOCK_MATCH" in (identities[0].assignment_reasons or [])
    assert "UNMATCHED_EXIT" not in (identities[0].assignment_reasons or [])


def test_pass3_sanitize_ghost_window_rematches_to_alternate_same_team_target():
    source_fragment = ScoredFragment(
        fragment_id="F000021",
        track_id=21,
        start_frame=1082,
        end_frame=1089,
        detection_ids=["1082_21_a"],
        quality=FragmentQuality.HIGH,
        quality_score=0.95,
    )
    ghost_fragment = ScoredFragment(
        fragment_id="G000030",
        track_id=29,
        start_frame=1090,
        end_frame=1313,
        detection_ids=[],
        quality=FragmentQuality.GHOST,
        quality_score=0.0,
        is_ghost=True,
    )
    invalid_target_fragment = ScoredFragment(
        fragment_id="F000026",
        track_id=26,
        start_frame=1175,
        end_frame=1313,
        detection_ids=["1175_26_a"],
        quality=FragmentQuality.HIGH,
        quality_score=0.95,
    )
    alternate_target_fragment = ScoredFragment(
        fragment_id="F000027",
        track_id=27,
        start_frame=1197,
        end_frame=1210,
        detection_ids=["1197_27_a"],
        quality=FragmentQuality.HIGH,
        quality_score=0.95,
    )
    identities = [
        CommittedIdentity(
            fragment_id="G000030",
            player_id="P22_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.GHOST_INHERITED,
            assignment_confidence=0.80,
            assignment_reasons=["MATCHED_GHOST_WINDOW"],
        ),
        CommittedIdentity(
            fragment_id="F000026",
            player_id="P27_team_b",
            team=TeamID.TEAM_B,
            jersey_number=None,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
        CommittedIdentity(
            fragment_id="F000027",
            player_id="P28_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
    ]
    ghost_windows = {
        "G000030": {
            "start_frame": 1090,
            "end_frame": 1174,
            "matched_reappearance_frame": 1175,
            "target_fragment_id": "F000026",
            "window_reason": "matched_reappearance",
            "source_fragment_id": "F000021",
            "defining_jersey_number": None,
        }
    }
    pass3a_edges = [
        IdentityCandidateEdge(
            fragment_a="F000021",
            fragment_b="F000026",
            temporal_gap=86,
            spatial_distance=552.9,
            velocity_consistency_score=0.88,
            appearance_similarity=0.75,
            jersey_similarity=0.5,
            temporal_gap_score=0.71,
            overall_candidate_score=0.37,
        ),
        IdentityCandidateEdge(
            fragment_a="F000021",
            fragment_b="F000027",
            temporal_gap=108,
            spatial_distance=191.3,
            velocity_consistency_score=0.92,
            appearance_similarity=0.62,
            jersey_similarity=0.5,
            temporal_gap_score=0.64,
            overall_candidate_score=0.36,
        ),
    ]

    cleared = IdentitySolver()._sanitize_ghost_activity_windows(
        committed_identities=identities,
        fragments=[source_fragment, ghost_fragment, invalid_target_fragment, alternate_target_fragment],
        ghost_activity_windows=ghost_windows,
        pass3a_edges=pass3a_edges,
        real_presence_frames={
            "F000021": {1082},
            "F000026": {1175},
            "F000027": {1197},
        },
    )

    assert cleared == 1
    assert ghost_windows["G000030"]["target_fragment_id"] == "F000027"
    assert ghost_windows["G000030"]["matched_reappearance_frame"] == 1197
    assert ghost_windows["G000030"]["window_reason"] == "matched_reappearance"
    assert ghost_windows["G000030"]["end_frame"] == 1196
    assert identities[0].player_id == "P28_team_a"
    assert identities[0].team == TeamID.TEAM_A
    assert "GHOST_POST_COMMIT_REMAP_MATCH" in (identities[0].assignment_reasons or [])
    assert "MATCHED_GHOST_WINDOW" in (identities[0].assignment_reasons or [])


def test_pass3_sanitize_ghost_window_ignores_stale_defining_jersey_when_source_identity_cleared():
    source_fragment = ScoredFragment(
        fragment_id="F000019",
        track_id=19,
        start_frame=914,
        end_frame=977,
        detection_ids=["914_19_a"],
        quality=FragmentQuality.HIGH,
        quality_score=0.95,
    )
    ghost_fragment = ScoredFragment(
        fragment_id="G000024",
        track_id=24,
        start_frame=978,
        end_frame=1313,
        detection_ids=[],
        quality=FragmentQuality.GHOST,
        quality_score=0.0,
        is_ghost=True,
    )
    target_fragment = ScoredFragment(
        fragment_id="F000020",
        track_id=20,
        start_frame=1023,
        end_frame=1043,
        detection_ids=["1023_20_a"],
        quality=FragmentQuality.HIGH,
        quality_score=0.95,
    )
    identities = [
        CommittedIdentity(
            fragment_id="F000019",
            player_id="P20_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
            assignment_reasons=["JERSEY_GLOBAL_REALLOCATED"],
        ),
        CommittedIdentity(
            fragment_id="G000024",
            player_id="P20_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.GHOST_INHERITED,
            assignment_confidence=0.80,
            assignment_reasons=["UNMATCHED_EXIT", "JERSEY_EXCLUSIVITY_DROPPED"],
        ),
        CommittedIdentity(
            fragment_id="F000020",
            player_id="P21_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
    ]
    ghost_windows = {
        "G000024": {
            "start_frame": 978,
            "end_frame": 1081,
            "matched_reappearance_frame": None,
            "target_fragment_id": None,
            "window_reason": "unmatched_exit",
            "source_fragment_id": "F000019",
            "defining_jersey_number": 4,
        }
    }
    pass3a_edges = [
        IdentityCandidateEdge(
            fragment_a="F000019",
            fragment_b="F000020",
            temporal_gap=46,
            spatial_distance=189.4,
            velocity_consistency_score=0.97,
            appearance_similarity=0.72,
            jersey_similarity=0.5,
            temporal_gap_score=0.85,
            overall_candidate_score=0.40,
        ),
    ]

    cleared = IdentitySolver()._sanitize_ghost_activity_windows(
        committed_identities=identities,
        fragments=[source_fragment, ghost_fragment, target_fragment],
        ghost_activity_windows=ghost_windows,
        pass3a_edges=pass3a_edges,
        real_presence_frames={
            "F000019": {914},
            "F000020": {1023},
        },
    )

    assert cleared == 0
    assert ghost_windows["G000024"]["target_fragment_id"] == "F000020"
    assert ghost_windows["G000024"]["matched_reappearance_frame"] == 1023
    assert ghost_windows["G000024"]["end_frame"] == 1022
    assert ghost_windows["G000024"]["window_reason"] == "matched_reappearance"
    assert identities[1].player_id == "P21_team_a"
    assert identities[1].team == TeamID.TEAM_A
    assert identities[1].jersey_number is None
    assert "GHOST_POST_COMMIT_REMAP_MATCH" in (identities[1].assignment_reasons or [])


def test_pass3_sanitize_ghost_window_suppresses_short_duplicate_visible_source():
    source_fragment = ScoredFragment(
        fragment_id="F_DUP_SRC",
        track_id=29,
        start_frame=1082,
        end_frame=1089,
        detection_ids=[f"{frame}_29_a" for frame in range(1082, 1090)],
        quality=FragmentQuality.HIGH,
        quality_score=0.95,
    )
    visible_fragment = ScoredFragment(
        fragment_id="F_VISIBLE",
        track_id=21,
        start_frame=718,
        end_frame=1313,
        detection_ids=["1089_21_a", "1091_21_b"],
        quality=FragmentQuality.HIGH,
        quality_score=0.95,
    )
    ghost_fragment = ScoredFragment(
        fragment_id="G_DUP",
        track_id=29,
        start_frame=1090,
        end_frame=1313,
        detection_ids=[],
        quality=FragmentQuality.GHOST,
        quality_score=0.0,
        is_ghost=True,
    )
    target_fragment = ScoredFragment(
        fragment_id="F_TARGET",
        track_id=35,
        start_frame=1197,
        end_frame=1210,
        detection_ids=["1197_35_a"],
        quality=FragmentQuality.HIGH,
        quality_score=0.95,
    )
    identities = [
        CommittedIdentity(
            fragment_id="F_DUP_SRC",
            player_id="P22_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
        CommittedIdentity(
            fragment_id="F_VISIBLE",
            player_id="P19_team_b",
            team=TeamID.TEAM_B,
            jersey_number=None,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
        CommittedIdentity(
            fragment_id="G_DUP",
            player_id="P28_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.GHOST_INHERITED,
            assignment_confidence=0.80,
            assignment_reasons=["MATCHED_GHOST_WINDOW"],
        ),
        CommittedIdentity(
            fragment_id="F_TARGET",
            player_id="P28_team_a",
            team=TeamID.TEAM_A,
            jersey_number=None,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
    ]
    ghost_windows = {
        "G_DUP": {
            "start_frame": 1090,
            "end_frame": 1196,
            "matched_reappearance_frame": 1197,
            "target_fragment_id": "F_TARGET",
            "window_reason": "matched_reappearance",
            "source_fragment_id": "F_DUP_SRC",
            "defining_jersey_number": None,
        }
    }
    detections_by_id = {
        **{
            f"{frame}_29_a": Detection(
                detection_id=f"{frame}_29_a",
                frame_idx=frame,
                bbox=[2930 + (frame - 1082), 600.0, 2990 + (frame - 1082), 730.0],
                centroid=[2960 + (frame - 1082), 665.0],
                confidence=0.9,
                track_id=29,
                jersey_number=None,
                jersey_confidence=0.0,
                jersey_probs=None,
                hsv_histogram_jersey=None,
                jersey_color_sampled=False,
                jersey_roi_valid=False,
                jersey_roi_bbox=None,
            )
            for frame in range(1082, 1090)
        },
        "1089_21_a": Detection(
            detection_id="1089_21_a",
            frame_idx=1089,
            bbox=[2931.0, 604.0, 2993.0, 732.0],
            centroid=[2962.0, 668.0],
            confidence=0.9,
            track_id=21,
            jersey_number=None,
            jersey_confidence=0.0,
            jersey_probs=None,
            hsv_histogram_jersey=None,
            jersey_color_sampled=False,
            jersey_roi_valid=False,
            jersey_roi_bbox=None,
        ),
        "1091_21_b": Detection(
            detection_id="1091_21_b",
            frame_idx=1091,
            bbox=[2935.0, 605.0, 2996.0, 745.0],
            centroid=[2965.5, 675.0],
            confidence=0.9,
            track_id=21,
            jersey_number=None,
            jersey_confidence=0.0,
            jersey_probs=None,
            hsv_histogram_jersey=None,
            jersey_color_sampled=False,
            jersey_roi_valid=False,
            jersey_roi_bbox=None,
        ),
        "1197_35_a": Detection(
            detection_id="1197_35_a",
            frame_idx=1197,
            bbox=[3127.7, 614.5, 3177.9, 724.1],
            centroid=[3152.8, 669.3],
            confidence=0.9,
            track_id=35,
            jersey_number=None,
            jersey_confidence=0.0,
            jersey_probs=None,
            hsv_histogram_jersey=None,
            jersey_color_sampled=False,
            jersey_roi_valid=False,
            jersey_roi_bbox=None,
        ),
    }

    cleared = IdentitySolver()._sanitize_ghost_activity_windows(
        committed_identities=identities,
        fragments=[source_fragment, visible_fragment, ghost_fragment, target_fragment],
        ghost_activity_windows=ghost_windows,
        pass3a_edges=[],
        real_presence_frames={
            "F_DUP_SRC": set(range(1082, 1090)),
            "F_VISIBLE": {1089, 1091},
            "F_TARGET": {1197},
        },
        detections_by_id=detections_by_id,
    )

    assert cleared == 0
    assert "G_DUP" not in ghost_windows
    assert "GHOST_SUPPRESSED_VISIBLE_DUPLICATE" in (identities[2].assignment_reasons or [])
    assert "MATCHED_GHOST_WINDOW" not in (identities[2].assignment_reasons or [])


def test_pass3_global_jersey_conflicts_after_sanitized_ghost_extension():
    ghost_fragment = ScoredFragment(
        fragment_id="G000006",
        track_id=8,
        start_frame=301,
        end_frame=2086,
        detection_ids=[],
        quality=FragmentQuality.GHOST,
        quality_score=0.0,
        is_ghost=True,
    )
    real_fragment = ScoredFragment(
        fragment_id="F000019",
        track_id=18,
        start_frame=311,
        end_frame=2086,
        detection_ids=["311_18_a"],
        quality=FragmentQuality.HIGH,
        quality_score=0.95,
    )
    identities = [
        CommittedIdentity(
            fragment_id="G000006",
            player_id="P13_team_a",
            team=TeamID.TEAM_A,
            jersey_number=10,
            assignment_method=AssignmentMethod.GHOST_INHERITED,
            assignment_confidence=0.80,
            assignment_reasons=["MATCHED_GHOST_WINDOW"],
        ),
        CommittedIdentity(
            fragment_id="F000019",
            player_id="P20_team_a",
            team=TeamID.TEAM_A,
            jersey_number=10,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
        CommittedIdentity(
            fragment_id="F_TARGET",
            player_id="P27_team_b",
            team=TeamID.TEAM_B,
            jersey_number=None,
            assignment_method=AssignmentMethod.CONSTRAINT_SOLVED,
            assignment_confidence=0.95,
        ),
    ]
    ghost_windows = {
        "G000006": {
            "start_frame": 301,
            "end_frame": 310,
            "matched_reappearance_frame": 311,
            "target_fragment_id": "F_TARGET",
            "window_reason": "matched_reappearance",
        }
    }

    solver = IdentitySolver()
    cleared = solver._sanitize_ghost_activity_windows(
        committed_identities=identities,
        fragments=[ghost_fragment, real_fragment],
        ghost_activity_windows=ghost_windows,
    )
    dropped = solver._resolve_global_jersey_conflicts(
        committed_identities=identities,
        fragments=[ghost_fragment, real_fragment],
        ghost_activity_windows=ghost_windows,
    )

    assert cleared == 1
    assert ghost_windows["G000006"]["end_frame"] == 2086
    assert dropped == 1
    assert identities[0].jersey_number is None
    assert identities[1].jersey_number == 10
    assert "JERSEY_GLOBAL_EXCLUSIVITY_DROPPED" in (identities[0].assignment_reasons or [])


def test_pass3_prunes_sanitized_ghost_windows_back_to_physical_cap():
    ghost_source = ScoredFragment(
        fragment_id="F000012",
        track_id=8,
        start_frame=191,
        end_frame=300,
        detection_ids=["191_8_a"],
        quality=FragmentQuality.HIGH,
        quality_score=0.95,
    )
    ghost_fragment = ScoredFragment(
        fragment_id="G000006",
        track_id=8,
        start_frame=301,
        end_frame=2086,
        detection_ids=[],
        quality=FragmentQuality.GHOST,
        quality_score=0.0,
        is_ghost=True,
    )
    real_fragments = [
        ScoredFragment(
            fragment_id=f"F_REAL_{idx:02d}",
            track_id=100 + idx,
            start_frame=311,
            end_frame=400,
            detection_ids=[f"311_{100 + idx}_a"],
            quality=FragmentQuality.HIGH,
            quality_score=0.95,
        )
        for idx in range(12)
    ]
    refined_identity_groups = {
        "GHOST_GROUP": {"F000012", "G000006"},
        **{f"REAL_GROUP_{idx:02d}": {fragment.fragment_id} for idx, fragment in enumerate(real_fragments)},
    }
    ghost_windows = {
        "G000006": {
            "start_frame": 301,
            "end_frame": 2086,
            "matched_reappearance_frame": None,
            "target_fragment_id": None,
            "window_reason": "unmatched_exit",
        }
    }
    retired_ghost_fragments = set()

    retired_groups = IdentitySolver()._prune_ghost_windows_to_physical_cap(
        fragments=[ghost_source, ghost_fragment, *real_fragments],
        refined_identity_groups=refined_identity_groups,
        retired_ghost_fragments=retired_ghost_fragments,
        ghost_activity_windows=ghost_windows,
    )

    assert not retired_ghost_fragments
    assert ghost_windows["G000006"]["end_frame"] == 310


def test_ball_interpolation_builds_full_timeline_and_interpolates_short_gap():
    pass1_output = _valid_pass1_output()
    pass1_output.total_frames = 5
    pass1_output.processed_end_frame_exclusive = 5
    pass1_output.ball_detections = [
        BallDetection(
            frame_idx=1,
            bbox=[10, 10, 20, 20],
            centroid=[15, 15],
            confidence=0.90,
        ),
        BallDetection(
            frame_idx=3,
            bbox=[30, 30, 40, 40],
            centroid=[35, 35],
            confidence=0.92,
        ),
    ]

    output = build_ball_interpolation_output(pass1_output)

    states = [position.state for position in output.ball_positions]
    assert len(output.ball_positions) == 5
    assert states == [
        BallState.UNKNOWN,
        BallState.REAL,
        BallState.INTERPOLATED,
        BallState.REAL,
        BallState.UNKNOWN,
    ]
    assert output.interpolation_method == InterpolationMethod.LINEAR
    assert output.interpolated_frames == [2]
    assert output.ball_positions[2].centroid == [25.0, 25.0]
    assert output.ball_positions[2].bbox is None

    validator = Validator()
    result = validator.validate_ball_interpolation(output, pass1_output.total_frames)
    assert result.passed


def test_ball_interpolation_marks_long_gap_unknown():
    pass1_output = _valid_pass1_output()
    pass1_output.total_frames = 36
    pass1_output.processed_end_frame_exclusive = 36
    pass1_output.ball_detections = [
        BallDetection(
            frame_idx=0,
            bbox=[10, 10, 20, 20],
            centroid=[15, 15],
            confidence=0.90,
        ),
        BallDetection(
            frame_idx=35,
            bbox=[70, 70, 80, 80],
            centroid=[75, 75],
            confidence=0.94,
        ),
    ]

    output = build_ball_interpolation_output(pass1_output)

    assert output.ball_positions[0].state == BallState.REAL
    assert output.ball_positions[35].state == BallState.REAL
    assert all(position.state == BallState.UNKNOWN for position in output.ball_positions[1:35])

    validator = Validator()
    result = validator.validate_ball_interpolation(output, pass1_output.total_frames)
    assert result.passed


def test_ball_interpolation_collapses_duplicate_same_frame_detections():
    pass1_output = _valid_pass1_output()
    pass1_output.total_frames = 3
    pass1_output.processed_end_frame_exclusive = 3
    pass1_output.ball_detections = [
        BallDetection(
            frame_idx=1,
            bbox=[10, 10, 20, 20],
            centroid=[15, 15],
            confidence=0.55,
        ),
        BallDetection(
            frame_idx=1,
            bbox=[12, 12, 22, 22],
            centroid=[17, 17],
            confidence=0.91,
        ),
    ]

    output = build_ball_interpolation_output(pass1_output)

    assert output.ball_positions[1].state == BallState.REAL
    assert output.ball_positions[1].centroid == [17.0, 17.0]
    assert output.ball_positions[1].bbox == [12.0, 12.0, 22.0, 22.0]
    assert output.ball_positions[1].confidence == 0.91


def test_ball_interpolation_drops_false_capture_outliers_between_supported_detections():
    pass1_output = _valid_pass1_output()
    pass1_output.total_frames = 11
    pass1_output.processed_end_frame_exclusive = 11
    pass1_output.ball_detections = [
        BallDetection(
            frame_idx=0,
            bbox=[0, 0, 10, 10],
            centroid=[5, 5],
            confidence=0.90,
        ),
        BallDetection(
            frame_idx=5,
            bbox=[50, 0, 60, 10],
            centroid=[55, 5],
            confidence=0.88,
        ),
        BallDetection(
            frame_idx=6,
            bbox=[400, 400, 410, 410],
            centroid=[405, 405],
            confidence=0.40,
        ),
        BallDetection(
            frame_idx=7,
            bbox=[410, 400, 420, 410],
            centroid=[415, 405],
            confidence=0.45,
        ),
        BallDetection(
            frame_idx=10,
            bbox=[100, 0, 110, 10],
            centroid=[105, 5],
            confidence=0.91,
        ),
    ]

    output = build_ball_interpolation_output(pass1_output)

    assert output.ball_positions[6].state == BallState.INTERPOLATED
    assert output.ball_positions[7].state == BallState.INTERPOLATED
    assert output.ball_positions[6].bbox is None
    assert output.ball_positions[7].bbox is None
    assert output.ball_positions[6].centroid == [65.0, 5.0]
    assert output.ball_positions[7].centroid == [75.0, 5.0]


def test_ball_interpolation_preserves_stable_consecutive_kick_run_and_rejects_short_side_branch():
    pass1_output = _valid_pass1_output()
    pass1_output.total_frames = 15
    pass1_output.processed_end_frame_exclusive = 15
    pass1_output.ball_detections = [
        BallDetection(
            frame_idx=0,
            bbox=[900, 590, 910, 600],
            centroid=[905, 595],
            confidence=0.90,
        ),
        BallDetection(
            frame_idx=1,
            bbox=[890, 590, 900, 600],
            centroid=[895, 595],
            confidence=0.90,
        ),
        BallDetection(
            frame_idx=2,
            bbox=[880, 590, 890, 600],
            centroid=[885, 595],
            confidence=0.90,
        ),
        BallDetection(
            frame_idx=8,
            bbox=[480, 730, 490, 740],
            centroid=[485, 735],
            confidence=0.74,
        ),
        BallDetection(
            frame_idx=9,
            bbox=[440, 745, 450, 755],
            centroid=[445, 750],
            confidence=0.77,
        ),
        BallDetection(
            frame_idx=10,
            bbox=[400, 760, 410, 770],
            centroid=[405, 765],
            confidence=0.62,
        ),
        BallDetection(
            frame_idx=11,
            bbox=[360, 775, 370, 785],
            centroid=[365, 780],
            confidence=0.86,
        ),
        BallDetection(
            frame_idx=12,
            bbox=[1360, 615, 1370, 625],
            centroid=[1365, 620],
            confidence=0.58,
        ),
        BallDetection(
            frame_idx=13,
            bbox=[1360, 615, 1370, 625],
            centroid=[1364, 620],
            confidence=0.34,
        ),
    ]

    output = build_ball_interpolation_output(pass1_output)

    assert output.ball_positions[8].state == BallState.REAL
    assert output.ball_positions[9].state == BallState.REAL
    assert output.ball_positions[10].state == BallState.REAL
    assert output.ball_positions[11].state == BallState.REAL
    assert output.ball_positions[12].state == BallState.UNKNOWN
    assert output.ball_positions[13].state == BallState.UNKNOWN


def test_ball_interpolation_rejects_detached_short_island_between_stable_regions():
    pass1_output = _valid_pass1_output()
    pass1_output.total_frames = 17
    pass1_output.processed_end_frame_exclusive = 17
    pass1_output.ball_detections = [
        BallDetection(
            frame_idx=0,
            bbox=[95, 95, 105, 105],
            centroid=[100, 100],
            confidence=0.90,
        ),
        BallDetection(
            frame_idx=1,
            bbox=[96, 97, 106, 107],
            centroid=[101, 102],
            confidence=0.90,
        ),
        BallDetection(
            frame_idx=2,
            bbox=[97, 99, 107, 109],
            centroid=[102, 104],
            confidence=0.90,
        ),
        BallDetection(
            frame_idx=3,
            bbox=[98, 101, 108, 111],
            centroid=[103, 106],
            confidence=0.90,
        ),
        BallDetection(
            frame_idx=4,
            bbox=[99, 103, 109, 113],
            centroid=[104, 108],
            confidence=0.90,
        ),
        BallDetection(
            frame_idx=7,
            bbox=[390, 390, 400, 400],
            centroid=[395, 395],
            confidence=0.71,
        ),
        BallDetection(
            frame_idx=8,
            bbox=[395, 395, 405, 405],
            centroid=[400, 400],
            confidence=0.75,
        ),
        BallDetection(
            frame_idx=9,
            bbox=[400, 400, 410, 410],
            centroid=[405, 405],
            confidence=0.69,
        ),
        BallDetection(
            frame_idx=10,
            bbox=[405, 405, 415, 415],
            centroid=[410, 410],
            confidence=0.66,
        ),
        BallDetection(
            frame_idx=13,
            bbox=[108, 110, 118, 120],
            centroid=[113, 115],
            confidence=0.88,
        ),
        BallDetection(
            frame_idx=14,
            bbox=[109, 112, 119, 122],
            centroid=[114, 117],
            confidence=0.88,
        ),
        BallDetection(
            frame_idx=15,
            bbox=[110, 114, 120, 124],
            centroid=[115, 119],
            confidence=0.88,
        ),
        BallDetection(
            frame_idx=16,
            bbox=[111, 116, 121, 126],
            centroid=[116, 121],
            confidence=0.88,
        ),
    ]

    output = build_ball_interpolation_output(pass1_output)

    assert output.ball_positions[7].state == BallState.INTERPOLATED
    assert output.ball_positions[8].state == BallState.INTERPOLATED
    assert output.ball_positions[9].state == BallState.INTERPOLATED
    assert output.ball_positions[10].state == BallState.INTERPOLATED


def test_ball_interpolation_rejects_detached_longer_island_between_stable_regions():
    pass1_output = _valid_pass1_output()
    pass1_output.total_frames = 25
    pass1_output.processed_end_frame_exclusive = 25
    pass1_output.ball_detections = [
        BallDetection(
            frame_idx=0,
            bbox=[2395, 695, 2405, 705],
            centroid=[2400, 700],
            confidence=0.82,
        ),
        BallDetection(
            frame_idx=1,
            bbox=[2397, 697, 2407, 707],
            centroid=[2402, 702],
            confidence=0.83,
        ),
        BallDetection(
            frame_idx=2,
            bbox=[2399, 699, 2409, 709],
            centroid=[2404, 704],
            confidence=0.84,
        ),
        BallDetection(
            frame_idx=3,
            bbox=[2401, 701, 2411, 711],
            centroid=[2406, 706],
            confidence=0.85,
        ),
        BallDetection(
            frame_idx=4,
            bbox=[2403, 703, 2413, 713],
            centroid=[2408, 708],
            confidence=0.84,
        ),
        BallDetection(
            frame_idx=7,
            bbox=[1295, 895, 1305, 905],
            centroid=[1300, 900],
            confidence=0.51,
        ),
        BallDetection(
            frame_idx=8,
            bbox=[1297, 897, 1307, 907],
            centroid=[1302, 902],
            confidence=0.48,
        ),
        BallDetection(
            frame_idx=9,
            bbox=[1299, 899, 1309, 909],
            centroid=[1304, 904],
            confidence=0.52,
        ),
        BallDetection(
            frame_idx=10,
            bbox=[1301, 901, 1311, 911],
            centroid=[1306, 906],
            confidence=0.63,
        ),
        BallDetection(
            frame_idx=11,
            bbox=[1303, 903, 1313, 913],
            centroid=[1308, 908],
            confidence=0.74,
        ),
        BallDetection(
            frame_idx=12,
            bbox=[1305, 905, 1315, 915],
            centroid=[1310, 910],
            confidence=0.70,
        ),
        BallDetection(
            frame_idx=13,
            bbox=[1307, 907, 1317, 917],
            centroid=[1312, 912],
            confidence=0.44,
        ),
        BallDetection(
            frame_idx=20,
            bbox=[2411, 711, 2421, 721],
            centroid=[2416, 716],
            confidence=0.80,
        ),
        BallDetection(
            frame_idx=21,
            bbox=[2413, 713, 2423, 723],
            centroid=[2418, 718],
            confidence=0.81,
        ),
        BallDetection(
            frame_idx=22,
            bbox=[2415, 715, 2425, 725],
            centroid=[2420, 720],
            confidence=0.83,
        ),
        BallDetection(
            frame_idx=23,
            bbox=[2417, 717, 2427, 727],
            centroid=[2422, 722],
            confidence=0.84,
        ),
        BallDetection(
            frame_idx=24,
            bbox=[2419, 719, 2429, 729],
            centroid=[2424, 724],
            confidence=0.82,
        ),
    ]

    output = build_ball_interpolation_output(pass1_output)

    for frame_idx in range(7, 14):
        assert output.ball_positions[frame_idx].state == BallState.INTERPOLATED


def test_ball_interpolation_rejects_short_low_confidence_island_with_plausible_average_edges():
    pass1_output = _valid_pass1_output()
    pass1_output.total_frames = 35
    pass1_output.processed_end_frame_exclusive = 35
    pass1_output.ball_detections = [
        BallDetection(
            frame_idx=0,
            bbox=[1295, 640, 1305, 650],
            centroid=[1300, 645],
            confidence=0.88,
        ),
        BallDetection(
            frame_idx=1,
            bbox=[1297, 641, 1307, 651],
            centroid=[1302, 646],
            confidence=0.88,
        ),
        BallDetection(
            frame_idx=2,
            bbox=[1299, 642, 1309, 652],
            centroid=[1304, 647],
            confidence=0.88,
        ),
        BallDetection(
            frame_idx=20,
            bbox=[665, 883, 675, 893],
            centroid=[670, 888],
            confidence=0.37,
        ),
        BallDetection(
            frame_idx=21,
            bbox=[666, 884, 676, 894],
            centroid=[671, 889],
            confidence=0.40,
        ),
        BallDetection(
            frame_idx=30,
            bbox=[1509, 619, 1519, 629],
            centroid=[1514, 624],
            confidence=0.86,
        ),
        BallDetection(
            frame_idx=31,
            bbox=[1511, 620, 1521, 630],
            centroid=[1516, 625],
            confidence=0.86,
        ),
        BallDetection(
            frame_idx=32,
            bbox=[1513, 621, 1523, 631],
            centroid=[1518, 626],
            confidence=0.86,
        ),
    ]

    output = build_ball_interpolation_output(pass1_output)

    assert output.ball_positions[20].state == BallState.INTERPOLATED
    assert output.ball_positions[21].state == BallState.INTERPOLATED


def test_ball_interpolation_keeps_true_streak_when_false_streaks_exist_on_both_sides():
    pass1_output = _valid_pass1_output()
    pass1_output.total_frames = 25
    pass1_output.processed_end_frame_exclusive = 25
    pass1_output.ball_detections = [
        BallDetection(frame_idx=0, bbox=[2395, 695, 2405, 705], centroid=[2400, 700], confidence=0.60),
        BallDetection(frame_idx=1, bbox=[2397, 697, 2407, 707], centroid=[2402, 702], confidence=0.60),
        BallDetection(frame_idx=2, bbox=[2399, 699, 2409, 709], centroid=[2404, 704], confidence=0.60),
        BallDetection(frame_idx=3, bbox=[1280, 920, 1290, 930], centroid=[1285, 925], confidence=0.75),
        BallDetection(frame_idx=4, bbox=[1282, 921, 1292, 931], centroid=[1287, 926], confidence=0.76),
        BallDetection(frame_idx=5, bbox=[1284, 922, 1294, 932], centroid=[1289, 927], confidence=0.77),
        BallDetection(frame_idx=8, bbox=[2407, 707, 2417, 717], centroid=[2412, 712], confidence=0.85),
        BallDetection(frame_idx=9, bbox=[2409, 709, 2419, 719], centroid=[2414, 714], confidence=0.85),
        BallDetection(frame_idx=10, bbox=[2411, 711, 2421, 721], centroid=[2416, 716], confidence=0.85),
        BallDetection(frame_idx=11, bbox=[2413, 713, 2423, 723], centroid=[2418, 718], confidence=0.85),
        BallDetection(frame_idx=12, bbox=[2415, 715, 2425, 725], centroid=[2420, 720], confidence=0.85),
        BallDetection(frame_idx=15, bbox=[1310, 900, 1320, 910], centroid=[1315, 905], confidence=0.60),
        BallDetection(frame_idx=16, bbox=[1312, 901, 1322, 911], centroid=[1317, 906], confidence=0.72),
        BallDetection(frame_idx=17, bbox=[1314, 902, 1324, 912], centroid=[1319, 907], confidence=0.74),
        BallDetection(frame_idx=20, bbox=[2423, 723, 2433, 733], centroid=[2428, 728], confidence=0.88),
        BallDetection(frame_idx=21, bbox=[2425, 725, 2435, 735], centroid=[2430, 730], confidence=0.88),
        BallDetection(frame_idx=22, bbox=[2427, 727, 2437, 737], centroid=[2432, 732], confidence=0.88),
    ]

    output = build_ball_interpolation_output(pass1_output)

    for frame_idx in range(8, 13):
        assert output.ball_positions[frame_idx].state == BallState.REAL
    for frame_idx in range(3, 6):
        assert output.ball_positions[frame_idx].state == BallState.INTERPOLATED
    for frame_idx in range(15, 18):
        assert output.ball_positions[frame_idx].state == BallState.INTERPOLATED


def test_ball_interpolation_rejects_one_sided_false_streak_and_keeps_bidirectionally_supported_true_streak():
    pass1_output = _valid_pass1_output()
    pass1_output.total_frames = 29
    pass1_output.processed_end_frame_exclusive = 29
    pass1_output.ball_detections = [
        BallDetection(frame_idx=0, bbox=[2395, 695, 2405, 705], centroid=[2400, 700], confidence=0.82),
        BallDetection(frame_idx=1, bbox=[2397, 697, 2407, 707], centroid=[2402, 702], confidence=0.82),
        BallDetection(frame_idx=2, bbox=[2399, 699, 2409, 709], centroid=[2404, 704], confidence=0.82),
        BallDetection(frame_idx=3, bbox=[1280, 920, 1290, 930], centroid=[1285, 925], confidence=0.76),
        BallDetection(frame_idx=4, bbox=[1282, 921, 1292, 931], centroid=[1287, 926], confidence=0.76),
        BallDetection(frame_idx=5, bbox=[1284, 922, 1294, 932], centroid=[1289, 927], confidence=0.76),
        BallDetection(frame_idx=8, bbox=[2407, 707, 2417, 717], centroid=[2412, 712], confidence=0.85),
        BallDetection(frame_idx=9, bbox=[2409, 709, 2419, 719], centroid=[2414, 714], confidence=0.85),
        BallDetection(frame_idx=10, bbox=[2411, 711, 2421, 721], centroid=[2416, 716], confidence=0.85),
        BallDetection(frame_idx=11, bbox=[2413, 713, 2423, 723], centroid=[2418, 718], confidence=0.85),
        BallDetection(frame_idx=12, bbox=[2415, 715, 2425, 725], centroid=[2420, 720], confidence=0.85),
        BallDetection(frame_idx=15, bbox=[1295, 895, 1305, 905], centroid=[1300, 900], confidence=0.72),
        BallDetection(frame_idx=16, bbox=[1297, 897, 1307, 907], centroid=[1302, 902], confidence=0.74),
        BallDetection(frame_idx=17, bbox=[1299, 899, 1309, 909], centroid=[1304, 904], confidence=0.78),
        BallDetection(frame_idx=18, bbox=[1301, 901, 1311, 911], centroid=[1306, 906], confidence=0.84),
        BallDetection(frame_idx=19, bbox=[1303, 903, 1313, 913], centroid=[1308, 908], confidence=0.86),
        BallDetection(frame_idx=20, bbox=[1305, 905, 1315, 915], centroid=[1310, 910], confidence=0.79),
        BallDetection(frame_idx=21, bbox=[2419, 719, 2429, 729], centroid=[2424, 724], confidence=0.88),
        BallDetection(frame_idx=22, bbox=[2421, 721, 2431, 731], centroid=[2426, 726], confidence=0.88),
        BallDetection(frame_idx=23, bbox=[2423, 723, 2433, 733], centroid=[2428, 728], confidence=0.88),
        BallDetection(frame_idx=24, bbox=[2425, 725, 2435, 735], centroid=[2430, 730], confidence=0.88),
        BallDetection(frame_idx=25, bbox=[2427, 727, 2437, 737], centroid=[2432, 732], confidence=0.88),
        BallDetection(frame_idx=26, bbox=[2429, 729, 2439, 739], centroid=[2434, 734], confidence=0.88),
        BallDetection(frame_idx=27, bbox=[2431, 731, 2441, 741], centroid=[2436, 736], confidence=0.88),
        BallDetection(frame_idx=28, bbox=[2433, 733, 2443, 743], centroid=[2438, 738], confidence=0.88),
    ]

    output = build_ball_interpolation_output(pass1_output)

    for frame_idx in range(8, 13):
        assert output.ball_positions[frame_idx].state == BallState.REAL
    for frame_idx in range(15, 21):
        assert output.ball_positions[frame_idx].state == BallState.INTERPOLATED


def test_ball_interpolation_rejects_short_jump_out_and_back_branch():
    pass1_output = _valid_pass1_output()
    pass1_output.total_frames = 24
    pass1_output.processed_end_frame_exclusive = 24
    pass1_output.ball_detections = [
        BallDetection(frame_idx=0, bbox=[2395, 695, 2405, 705], centroid=[2400, 700], confidence=0.84),
        BallDetection(frame_idx=1, bbox=[2397, 697, 2407, 707], centroid=[2402, 702], confidence=0.84),
        BallDetection(frame_idx=2, bbox=[2399, 699, 2409, 709], centroid=[2404, 704], confidence=0.84),
        BallDetection(frame_idx=3, bbox=[2401, 701, 2411, 711], centroid=[2406, 706], confidence=0.84),
        BallDetection(frame_idx=4, bbox=[2403, 703, 2413, 713], centroid=[2408, 708], confidence=0.84),
        BallDetection(frame_idx=5, bbox=[2405, 705, 2415, 715], centroid=[2410, 710], confidence=0.84),
        BallDetection(frame_idx=6, bbox=[2407, 707, 2417, 717], centroid=[2412, 712], confidence=0.84),
        BallDetection(frame_idx=8, bbox=[1575, 975, 1585, 985], centroid=[1580, 980], confidence=0.51),
        BallDetection(frame_idx=9, bbox=[1505, 945, 1515, 955], centroid=[1510, 950], confidence=0.71),
        BallDetection(frame_idx=10, bbox=[1480, 940, 1490, 950], centroid=[1485, 945], confidence=0.72),
        BallDetection(frame_idx=11, bbox=[1515, 955, 1525, 965], centroid=[1520, 960], confidence=0.30),
        BallDetection(frame_idx=14, bbox=[2411, 711, 2421, 721], centroid=[2416, 716], confidence=0.32),
        BallDetection(frame_idx=15, bbox=[2413, 713, 2423, 723], centroid=[2418, 718], confidence=0.39),
        BallDetection(frame_idx=16, bbox=[2415, 715, 2425, 725], centroid=[2420, 720], confidence=0.58),
        BallDetection(frame_idx=17, bbox=[2417, 717, 2427, 727], centroid=[2422, 722], confidence=0.47),
        BallDetection(frame_idx=20, bbox=[2421, 721, 2431, 731], centroid=[2426, 726], confidence=0.82),
        BallDetection(frame_idx=21, bbox=[2423, 723, 2433, 733], centroid=[2428, 728], confidence=0.82),
        BallDetection(frame_idx=22, bbox=[2425, 725, 2435, 735], centroid=[2430, 730], confidence=0.82),
        BallDetection(frame_idx=23, bbox=[2427, 727, 2437, 737], centroid=[2432, 732], confidence=0.82),
    ]

    output = build_ball_interpolation_output(pass1_output)

    for frame_idx in range(8, 12):
        assert output.ball_positions[frame_idx].state == BallState.INTERPOLATED
    for frame_idx in range(14, 18):
        assert output.ball_positions[frame_idx].state == BallState.REAL



def test_ball_interpolation_defaults_to_all_unknown_when_no_detections():
    pass1_output = _valid_pass1_output()
    pass1_output.total_frames = 4
    pass1_output.processed_end_frame_exclusive = 4
    pass1_output.ball_detections = []

    output = build_ball_interpolation_output(pass1_output)

    assert len(output.ball_positions) == 4
    assert all(position.state == BallState.UNKNOWN for position in output.ball_positions)

    validator = Validator()
    result = validator.validate_ball_interpolation(output, pass1_output.total_frames)
    assert result.passed


def test_ball_validation_fails_when_interpolated_state_has_no_centroid():
    output = BallInterpolationOutput(
        ball_positions=[
            BallPosition(
                frame_idx=0,
                state=BallState.INTERPOLATED,
                centroid=None,
                bbox=None,
                confidence=0.5,
            )
        ],
        interpolation_method=InterpolationMethod.LINEAR,
        total_frames=1,
        interpolated_frames=[0],
    )

    validator = Validator()
    result = validator.validate_ball_interpolation(output, total_frames=1)

    assert not result.passed
    rules = {violation.rule for violation in result.violations}
    assert "BALL_INTERPOLATED_NO_CENTROID" in rules
    assert retired_groups == set()
