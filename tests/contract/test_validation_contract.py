from src.core.data_models import Detection, Pass1Output, Fragment, Pass2AOutput
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
