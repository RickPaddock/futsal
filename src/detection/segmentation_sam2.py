"""
SAM2-based segmentation wrapper for close/overlapping player instances.

Provides SAM2 segmentation for recovery of occluded/lost player detections.
Works with the three-tier system: T1 (YOLO) + T2 (SAM) + T3 (Interpolation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List
import sys
import numpy as np

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _SAM2_AVAILABLE = True
except Exception as e:
    _SAM2_AVAILABLE = False
    print(f"SAM2 import error: {e}")

import cv2
import torch

from src.utils.data_models import BoundingBox


class SamSegmenter2:
    """Wrapper for SAM2 model for segmentation of bounding boxes.
    
    Used in T2 stage of three-tier detection system.
    Attempts to recover occluded players when T1 (YOLO) fails to detect them.
    """

    def __init__(self, model_type: str = "hiera_small", checkpoint_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize SAM2 predictor.

        Args:
            model_type: SAM2 model type (e.g., 'hiera_tiny', 'hiera_small', 'hiera_base_plus', 'hiera_large')
            checkpoint_path: Path to SAM2 checkpoint (if None, downloaded automatically)
            device: Device ('cuda' or 'cpu')
        """
        self.device = device
        self.available = False
        self._predictor = None
        self._current_frame = None
        self._frame_idx = -1

        if not _SAM2_AVAILABLE:
            print("SAM2 import failed: sam2 package not available")
            return

        if checkpoint_path is None:
            print("SAM2: checkpoint_path is None, cannot load model")
            return

        # Build SAM2 with correct config - use installed package name format
        import os
        import sam2
        
        config_map = {
            "hiera_tiny": "sam2.1_hiera_t",
            "hiera_small": "sam2.1_hiera_s",
            "hiera_base_plus": "sam2.1_hiera_b+",
            "hiera_large": "sam2.1_hiera_l",
        }
        config_name = config_map.get(model_type, "sam2.1_hiera_s")
        
        # Get full path to config in sam2.1 subdir of installed package
        sam2_pkg_dir = os.path.dirname(sam2.__file__)
        config_path = os.path.join(sam2_pkg_dir, "configs", "sam2.1", f"{config_name}.yaml")

        try:
            print(f"Loading SAM2 ({model_type}) with config '{config_path}' and checkpoint '{checkpoint_path}'...")
            model = build_sam2(config_path, checkpoint_path, device=device)
            self._predictor = SAM2ImagePredictor(model)
            self.available = True
            print(f"SAM2 loaded successfully")
        except Exception as e:
            self.available = False
            self._predictor = None
            print(f"SAM2 loading failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    def set_image(self, frame: np.ndarray, frame_idx: int):
        """Set image for processing.

        Args:
            frame: BGR image (OpenCV format)
            frame_idx: Frame index (for tracking/logging)
        """
        if not self.available or self._predictor is None:
            return

        self._frame_idx = frame_idx
        self._current_frame = frame

        # Convert BGR to RGB for SAM2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                self._predictor.set_image(rgb_frame)
        except Exception as e:
            print(f"SAM2 set_image failed: {type(e).__name__}: {e}")

    def segment_by_box(
        self,
        bbox: BoundingBox,
        negative_points: list[tuple[float, float]] | None = None,
    ) -> Optional[np.ndarray]:
        """
        Segment using single box prompt, with optional negative point prompts.

        Args:
            bbox: BoundingBox used as a prompt
            negative_points: Optional list of (x, y) coordinates indicating
                locations that should NOT be part of this object (e.g. centers
                of already-segmented overlapping players).

        Returns:
            Binary mask (H x W) as uint8 values {0,1}, or None
        """
        if not self.available or self._predictor is None or self._current_frame is None:
            return None

        try:
            box = np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], dtype=np.float32)

            if negative_points:
                point_coords = np.array(negative_points, dtype=np.float32)
                point_labels = np.zeros(len(negative_points), dtype=np.int32)  # 0 = background/negative
            else:
                point_coords = None
                point_labels = None

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                masks, scores, logits = self._predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=False,
                )
            if masks is not None and len(masks) > 0:
                return masks[0].astype(np.uint8)
        except Exception as e:
            print(f"SAM2 segment_by_box failed: {type(e).__name__}: {e}")

        return None

    def auto_segment_box(self, bbox: BoundingBox, min_mask_area: int = 500) -> list[np.ndarray]:
        """
        Auto-segment a box.

        Args:
            bbox: BoundingBox to segment
            min_mask_area: Minimum mask area in pixels to keep

        Returns:
            List containing mask if valid, empty list otherwise
        """
        mask = self.segment_by_box(bbox)
        if mask is not None and np.sum(mask) >= min_mask_area:
            return [mask]
        return []

    def segment_pair_by_boxes(self, bbox1: BoundingBox, bbox2: BoundingBox) -> tuple:
        """
        Segment two boxes.

        Args:
            bbox1: First bounding box
            bbox2: Second bounding box

        Returns:
            Tuple of two masks (or None if unavailable)
        """
        if not self.available or self._predictor is None or self._current_frame is None:
            return None, None

        mask1 = self.segment_by_box(bbox1)
        mask2 = self.segment_by_box(bbox2)
        return mask1, mask2
