"""
Pass 3A: Pairwise Identity Candidate Generation.

Contract behavior:
- Read pass2b_scored_fragments.json and pass2_ghosts.json
- Generate pairwise candidate edges only for non-ghost fragments
- Enforce temporal and spatial feasibility gates
- Emit pass3a_candidates.json with edge scores

Compatibility behavior:
- Also emit legacy pass3_candidates.json so Pass 3B can continue unchanged
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core import constants as const
from ..core.data_models import (
	Detection,
	IdentityCandidate,
	IdentityCandidateEdge,
	Pass1Output,
	Pass2BOutput,
	Pass2COutput,
	Pass3AEdgesOutput,
	Pass3AOutput,
	ScoredFragment,
	ValidationResult,
	ValidationViolation,
)
from ..core.schemas import PASS3A_EDGE_OUTPUT_SCHEMA
from ..core.types import TeamID
from ..utils.file_utils import load_json, save_json
from ..utils.hsv_color import compare_hsv_histograms, is_histogram_valid
from ..utils.logging_utils import get_logger

logger = get_logger("pass3a_candidate_generator")
ALLOWED_JERSEY_NUMBERS = set(const.JERSEY_NUMBERS)


def _clamp01(value: float) -> float:
	return float(max(0.0, min(1.0, value)))


def _l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
	ax, ay = float(a[0]), float(a[1])
	bx, by = float(b[0]), float(b[1])
	return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> Optional[float]:
	if len(vec_a) != len(vec_b) or len(vec_a) == 0:
		return None
	a = np.asarray(vec_a, dtype=float)
	b = np.asarray(vec_b, dtype=float)
	denom = float(np.linalg.norm(a) * np.linalg.norm(b))
	if denom <= 0:
		return None
	cosine = float(np.dot(a, b) / denom)
	# Map [-1, 1] -> [0, 1]
	return _clamp01((cosine + 1.0) / 2.0)


class Pass3ACandidateGenerator:
	"""Generate contract-compliant pairwise identity candidate edges."""

	def __init__(
		self,
		pass2b_output: Pass2BOutput,
		pass2c_output: Pass2COutput,
		pass1_output: Pass1Output,
	):
		self.pass2b_output = pass2b_output
		self.pass2c_output = pass2c_output
		self.pass1_output = pass1_output

		self._detections_by_id: Dict[str, Detection] = {
			detection.detection_id: detection
			for detection in pass1_output.detections
		}

		self._ghost_ids = {
			fragment.fragment_id
			for fragment in pass2c_output.fragments
			if bool(getattr(fragment, "is_ghost", False))
		}

		self._fragments_by_id: Dict[str, ScoredFragment] = {
			fragment.fragment_id: fragment
			for fragment in pass2b_output.fragments
		}

		# Keep a complete source list from Pass 2C for pair iteration / ghost-reject metrics.
		self._all_fragments: List[ScoredFragment] = [
			self._fragments_by_id.get(fragment.fragment_id, fragment)
			for fragment in pass2c_output.fragments
		]

		self.debug_metrics = {
			"candidate_edges_total": 0,
			"candidate_edges_rejected_gap": 0,
			"candidate_edges_rejected_distance": 0,
			"candidate_edges_rejected_ghost": 0,
		}

	def generate(self) -> Pass3AEdgesOutput:
		candidates: List[IdentityCandidateEdge] = []

		ordered = sorted(
			self._all_fragments,
			key=lambda fragment: (fragment.start_frame, fragment.end_frame, fragment.fragment_id),
		)

		for idx_a, fragment_a in enumerate(ordered):
			for idx_b in range(idx_a + 1, len(ordered)):
				fragment_b = ordered[idx_b]

				if fragment_a.fragment_id == fragment_b.fragment_id:
					continue

				# Contract rule 1: temporal ordering
				if fragment_a.end_frame >= fragment_b.start_frame:
					continue

				# Contract rule: ghosts never create identity edges
				if self._is_ghost(fragment_a) or self._is_ghost(fragment_b):
					self.debug_metrics["candidate_edges_rejected_ghost"] += 1
					continue

				temporal_gap = int(fragment_b.start_frame - fragment_a.end_frame)
				if temporal_gap > const.MAX_IDENTITY_GAP:
					self.debug_metrics["candidate_edges_rejected_gap"] += 1
					continue

				end_centroid_a = self._fragment_last_centroid(fragment_a)
				start_centroid_b = self._fragment_first_centroid(fragment_b)
				if end_centroid_a is None or start_centroid_b is None:
					# Missing spatial evidence means this edge is not provable.
					self.debug_metrics["candidate_edges_rejected_distance"] += 1
					continue

				spatial_distance = _l2_distance(end_centroid_a, start_centroid_b)
				max_distance = const.MAX_PLAYER_SPEED * float(temporal_gap)
				if spatial_distance > max_distance:
					self.debug_metrics["candidate_edges_rejected_distance"] += 1
					continue

				velocity_consistency_score = self._velocity_consistency_score(
					fragment_a,
					fragment_b,
					temporal_gap,
					start_centroid_b,
				)
				appearance_similarity = self._appearance_similarity(fragment_a, fragment_b)
				jersey_similarity = self._jersey_similarity(fragment_a, fragment_b)

				overall_candidate_score = _clamp01(
					0.4 * velocity_consistency_score
					+ 0.4 * appearance_similarity
					+ 0.2 * jersey_similarity
				)

				candidates.append(
					IdentityCandidateEdge(
						fragment_a=fragment_a.fragment_id,
						fragment_b=fragment_b.fragment_id,
						temporal_gap=temporal_gap,
						spatial_distance=float(spatial_distance),
						velocity_consistency_score=float(velocity_consistency_score),
						appearance_similarity=float(appearance_similarity),
						jersey_similarity=float(jersey_similarity),
						overall_candidate_score=float(overall_candidate_score),
					)
				)

		self.debug_metrics["candidate_edges_total"] = len(candidates)
		return Pass3AEdgesOutput(candidates=candidates)

	def build_legacy_candidates(self) -> Pass3AOutput:
		"""
		Build backward-compatible per-fragment candidates for Pass 3B.

		This shim preserves existing Pass 3B behavior while Pass 3A starts
		writing contract-compliant pairwise edge candidates.
		"""
		legacy_candidates: List[IdentityCandidate] = []
		for fragment in self._all_fragments:
			jersey_scores = self._fragment_jersey_scores(fragment)
			jersey_evidence = {
				f"{jersey:02d}": {
					"count": int(score_data["count"]),
					"ratio": float(score_data["ratio"]),
				}
				for jersey, score_data in jersey_scores.items()
			}

			dominant_jersey = None
			if jersey_scores:
				dominant_jersey = max(
					jersey_scores,
					key=lambda jersey_num: (
						jersey_scores[jersey_num]["count"],
						jersey_scores[jersey_num]["ratio"],
					),
				)

			legacy_candidates.append(
				IdentityCandidate(
					fragment_id=fragment.fragment_id,
					candidate_team=None,
					candidate_jersey=dominant_jersey,
					candidate_player_id=None,
					team_evidence={
						TeamID.TEAM_A.value: 0.5,
						TeamID.TEAM_B.value: 0.5,
					},
					jersey_evidence=jersey_evidence,
					player_evidence={
						"is_ghost": 1.0 if self._is_ghost(fragment) else 0.0,
					},
				)
			)

		return Pass3AOutput(candidates=legacy_candidates)

	def _is_ghost(self, fragment: ScoredFragment) -> bool:
		return bool(getattr(fragment, "is_ghost", False) or fragment.fragment_id in self._ghost_ids)

	def _resolve_fragment_detections(self, fragment: ScoredFragment) -> List[Detection]:
		detections: List[Detection] = []
		for detection_id in fragment.detection_ids:
			detection = self._detections_by_id.get(detection_id)
			if detection is not None:
				detections.append(detection)

		detections.sort(key=lambda detection: detection.frame_idx)
		return detections

	def _fragment_first_centroid(self, fragment: ScoredFragment) -> Optional[Tuple[float, float]]:
		detections = self._resolve_fragment_detections(fragment)
		if not detections:
			centroid = getattr(fragment, "estimated_centroid", None)
			if centroid is None:
				centroid = getattr(fragment, "ghost_last_known_centroid", None)
			if centroid is None:
				return None
			return (float(centroid[0]), float(centroid[1]))

		first = detections[0]
		return (float(first.centroid[0]), float(first.centroid[1]))

	def _fragment_last_centroid(self, fragment: ScoredFragment) -> Optional[Tuple[float, float]]:
		detections = self._resolve_fragment_detections(fragment)
		if not detections:
			centroid = getattr(fragment, "estimated_centroid", None)
			if centroid is None:
				centroid = getattr(fragment, "ghost_last_known_centroid", None)
			if centroid is None:
				return None
			return (float(centroid[0]), float(centroid[1]))

		last = detections[-1]
		return (float(last.centroid[0]), float(last.centroid[1]))

	def _fragment_last_velocity(self, fragment: ScoredFragment) -> Tuple[float, float]:
		detections = self._resolve_fragment_detections(fragment)
		if len(detections) < 2:
			return (0.0, 0.0)

		prev = detections[-2]
		curr = detections[-1]
		dt = max(1, int(curr.frame_idx - prev.frame_idx))
		vx = (float(curr.centroid[0]) - float(prev.centroid[0])) / dt
		vy = (float(curr.centroid[1]) - float(prev.centroid[1])) / dt
		return (vx, vy)

	def _velocity_consistency_score(
		self,
		fragment_a: ScoredFragment,
		fragment_b: ScoredFragment,
		temporal_gap: int,
		start_centroid_b: Tuple[float, float],
	) -> float:
		end_centroid_a = self._fragment_last_centroid(fragment_a)
		if end_centroid_a is None:
			return 0.0

		vx, vy = self._fragment_last_velocity(fragment_a)
		pred_x = end_centroid_a[0] + vx * temporal_gap
		pred_y = end_centroid_a[1] + vy * temporal_gap
		prediction_error = _l2_distance((pred_x, pred_y), start_centroid_b)

		max_error = max(1.0, const.MAX_PLAYER_SPEED * float(max(1, temporal_gap)))
		return _clamp01(1.0 - (prediction_error / max_error))

	def _appearance_similarity(self, fragment_a: ScoredFragment, fragment_b: ScoredFragment) -> float:
		embedding_a = getattr(fragment_a, "appearance_embedding", None)
		embedding_b = getattr(fragment_b, "appearance_embedding", None)
		if embedding_a is not None and embedding_b is not None:
			embedding_similarity = _cosine_similarity(embedding_a, embedding_b)
			if embedding_similarity is not None:
				return embedding_similarity

		hist_a = self._fragment_last_histogram(fragment_a)
		hist_b = self._fragment_first_histogram(fragment_b)
		if hist_a is None or hist_b is None:
			return 0.5

		correlation = compare_hsv_histograms(hist_a, hist_b)
		return _clamp01((correlation + 1.0) / 2.0)

	def _fragment_first_histogram(self, fragment: ScoredFragment) -> Optional[List[float]]:
		detections = self._resolve_fragment_detections(fragment)
		for detection in detections:
			if is_histogram_valid(detection.hsv_histogram_jersey):
				return list(detection.hsv_histogram_jersey)

		fallback = getattr(fragment, "hsv_histogram_jersey", None)
		if is_histogram_valid(fallback):
			return list(fallback)
		return None

	def _fragment_last_histogram(self, fragment: ScoredFragment) -> Optional[List[float]]:
		detections = self._resolve_fragment_detections(fragment)
		for detection in reversed(detections):
			if is_histogram_valid(detection.hsv_histogram_jersey):
				return list(detection.hsv_histogram_jersey)

		fallback = getattr(fragment, "hsv_histogram_jersey", None)
		if is_histogram_valid(fallback):
			return list(fallback)
		return None

	def _fragment_jersey_scores(self, fragment: ScoredFragment) -> Dict[int, Dict[str, float]]:
		detections = self._resolve_fragment_detections(fragment)
		total = len(detections)
		if total == 0:
			return {}

		counts: Dict[int, int] = defaultdict(int)
		for detection in detections:
			jersey_number = detection.jersey_number
			if (
				jersey_number is not None
				and jersey_number in ALLOWED_JERSEY_NUMBERS
				and detection.jersey_confidence >= const.JERSEY_CONF_THRESHOLD
			):
				counts[int(jersey_number)] += 1
				continue

			if not detection.jersey_probs:
				continue

			best_jersey = None
			best_score = 0.0
			for jersey_key, score in detection.jersey_probs.items():
				try:
					jersey_num = int(jersey_key)
				except (TypeError, ValueError):
					continue

				score_float = float(score)
				if jersey_num in ALLOWED_JERSEY_NUMBERS and score_float > best_score:
					best_jersey = jersey_num
					best_score = score_float

			if best_jersey is not None:
				counts[int(best_jersey)] += 1

		result: Dict[int, Dict[str, float]] = {}
		for jersey_num, count in counts.items():
			ratio = float(count / total)
			if count >= const.MIN_JERSEY_DETECTIONS and ratio >= const.MIN_JERSEY_RATIO:
				result[int(jersey_num)] = {
					"count": float(count),
					"ratio": ratio,
				}

		return result

	def _dominant_jersey(self, fragment: ScoredFragment) -> Optional[int]:
		jersey_scores = self._fragment_jersey_scores(fragment)
		if not jersey_scores:
			return None
		return int(
			max(
				jersey_scores,
				key=lambda jersey_num: (
					jersey_scores[jersey_num]["count"],
					jersey_scores[jersey_num]["ratio"],
				),
			)
		)

	def _jersey_similarity(self, fragment_a: ScoredFragment, fragment_b: ScoredFragment) -> float:
		jersey_a = self._dominant_jersey(fragment_a)
		jersey_b = self._dominant_jersey(fragment_b)

		if jersey_a is None or jersey_b is None:
			return 0.5
		if jersey_a == jersey_b:
			return 1.0
		return 0.0


def _validate_candidate_edges(pass3a_edges_output: Pass3AEdgesOutput) -> List[ValidationViolation]:
	"""Fail-fast validation for Pass 3A candidate edges."""
	violations: List[ValidationViolation] = []

	for candidate in pass3a_edges_output.candidates:
		if candidate.fragment_a == candidate.fragment_b:
			violations.append(
				ValidationViolation(
					rule="PASS3A_EDGE_SELF_REFERENCE",
					severity="error",
					message=(
						f"Invalid candidate edge: fragment_a == fragment_b ({candidate.fragment_a})"
					),
					details={
						"fragment_a": candidate.fragment_a,
						"fragment_b": candidate.fragment_b,
					},
				)
			)

		if candidate.temporal_gap <= 0:
			violations.append(
				ValidationViolation(
					rule="PASS3A_EDGE_INVALID_TEMPORAL_ORDER",
					severity="error",
					message=(
						f"Invalid candidate edge temporal ordering: {candidate.fragment_a} -> {candidate.fragment_b} "
						f"with temporal_gap={candidate.temporal_gap}"
					),
					details={
						"fragment_a": candidate.fragment_a,
						"fragment_b": candidate.fragment_b,
						"temporal_gap": candidate.temporal_gap,
					},
				)
			)

		if not (0.0 <= candidate.overall_candidate_score <= 1.0):
			violations.append(
				ValidationViolation(
					rule="PASS3A_EDGE_INVALID_SCORE",
					severity="error",
					message=(
						f"Invalid candidate edge score for {candidate.fragment_a} -> {candidate.fragment_b}: "
						f"overall_candidate_score={candidate.overall_candidate_score}"
					),
					details={
						"fragment_a": candidate.fragment_a,
						"fragment_b": candidate.fragment_b,
						"overall_candidate_score": candidate.overall_candidate_score,
					},
				)
			)

	return violations


def _write_pass3a_debug_metrics(output_dir: Path, metrics: Dict[str, int]) -> None:
	"""Append Pass 3A candidate-edge diagnostics into debug_metrics.json."""
	debug_path = output_dir / const.DEBUG_METRICS_JSON
	payload = {}
	if debug_path.exists():
		existing = load_json(str(debug_path))
		if isinstance(existing, dict):
			payload = existing

	payload.update(metrics)
	save_json(payload, str(debug_path))


def run_pass3a(
	input_dir: Path,
	output_dir: Optional[Path] = None,
) -> Pass3AEdgesOutput:
	"""Execute Pass 3A pairwise candidate generation with fail-fast validation."""
	if output_dir is None:
		output_dir = input_dir

	pass2b_path = input_dir / const.PASS2B_SCORED_FRAGMENTS_JSON
	pass2c_path = input_dir / const.PASS2_GHOSTS_JSON
	pass1_path = input_dir / const.PASS1_RAW_JSON

	output_edges_path = output_dir / const.PASS3A_CANDIDATES_JSON
	output_legacy_path = output_dir / const.PASS3_CANDIDATES_JSON
	validation_path = output_dir / const.PASS3_VALIDATION_JSON

	if not pass2b_path.exists():
		raise FileNotFoundError(f"Pass 2B output not found: {pass2b_path}")
	if not pass2c_path.exists():
		raise FileNotFoundError(f"Pass 2C output not found: {pass2c_path}")
	if not pass1_path.exists():
		raise FileNotFoundError(
			"Pass 1 output is required for Pass 3A spatial/appearance scoring: "
			f"{pass1_path}"
		)

	pass2b_output = load_json(str(pass2b_path), Pass2BOutput)
	pass2c_output = load_json(str(pass2c_path), Pass2COutput)
	pass1_output = load_json(str(pass1_path), Pass1Output)

	generator = Pass3ACandidateGenerator(
		pass2b_output=pass2b_output,
		pass2c_output=pass2c_output,
		pass1_output=pass1_output,
	)
	pass3a_edges_output = generator.generate()
	legacy_output = generator.build_legacy_candidates()

	violations = _validate_candidate_edges(pass3a_edges_output)
	errors = [violation for violation in violations if violation.severity == "error"]

	validation_result = ValidationResult(
		passed=len(errors) == 0,
		violations=errors,
		warnings=[],
		timestamp=datetime.utcnow().isoformat() + "Z",
		pass_name="pass3a",
		diagnostics=generator.debug_metrics,
	)

	if not validation_result.passed:
		save_json(validation_result.model_dump(), str(validation_path))
		raise ValueError(
			f"Pass 3A validation failed with {len(validation_result.violations)} error(s). "
			f"See {validation_path}"
		)

	save_json(pass3a_edges_output.model_dump(), str(output_edges_path), PASS3A_EDGE_OUTPUT_SCHEMA)
	# Backward-compatible artifact for current Pass 3B reader.
	save_json(legacy_output.model_dump(), str(output_legacy_path))
	save_json(validation_result.model_dump(), str(validation_path))
	_write_pass3a_debug_metrics(output_dir=output_dir, metrics=generator.debug_metrics)

	logger.info(
		"Pass 3A complete: "
		f"edge_candidates={len(pass3a_edges_output.candidates)}, "
		f"rejected_gap={generator.debug_metrics['candidate_edges_rejected_gap']}, "
		f"rejected_distance={generator.debug_metrics['candidate_edges_rejected_distance']}, "
		f"rejected_ghost={generator.debug_metrics['candidate_edges_rejected_ghost']}, "
		f"output_edges={output_edges_path}, output_legacy={output_legacy_path}"
	)

	return pass3a_edges_output
