from src.core.data_models import Detection, Pass1Output, Fragment, Pass2AOutput, ScoredFragment, CommittedIdentity, Pass3COutput, IdentityCandidateEdge
from src.core.types import FragmentQuality, TeamID, AssignmentMethod
from src.skills.pass3c_identity_solver import IdentitySolver
from src.validation.validator import Validator
from src.skills.pass1_extractor import _load_pass1_intention_lines


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
    assert len(result.violations) == 0


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
    )

    assert retired_ghosts == set()
    assert ghost_windows["G_ORANGE"]["window_reason"] == "matched_reappearance"
    assert ghost_windows["G_ORANGE"]["target_fragment_id"] == "F_RETURN_ORANGE"
    assert ghost_windows["G_ORANGE"]["matched_reappearance_frame"] == 610
    assert ghost_windows["G_ORANGE"]["end_frame"] == 609
    assert ghost_windows["G_BLACK"]["window_reason"] == "matched_reappearance"
    assert ghost_windows["G_BLACK"]["target_fragment_id"] == "F_RETURN_BLACK"
    assert ghost_windows["G_BLACK"]["matched_reappearance_frame"] == 625
    assert ghost_windows["G_BLACK"]["end_frame"] == 624


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
    assert retired_groups == set()
