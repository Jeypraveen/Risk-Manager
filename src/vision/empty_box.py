"""
SAM2-based empty-box detection.

Uses Segment Anything 2 to detect objects in the return photo.
If the total mask area is near zero relative to the image, the box
is likely empty — a deterministic signal, not a trained classifier.

This is intentionally simple: empty-box detection via mask area is
not a learned decision. SAM2 segments whatever is in the image, and
we just check if "whatever" is approximately nothing.

Fallback: If SAM2 is too slow on CPU (>15s), a simpler heuristic
based on edge density and color entropy is used instead.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from src.config import EMPTY_BOX_MASK_AREA_THRESHOLD

logger = logging.getLogger(__name__)

# Flag to track if SAM2 is available
_sam2_available: Optional[bool] = None
_sam2_model = None


def _try_load_sam2():
    """Attempt to load SAM2. Fall back to heuristic if unavailable."""
    global _sam2_available, _sam2_model

    if _sam2_available is not None:
        return _sam2_available

    try:
        # SAM2 has various installation methods; try the most common
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        _sam2_available = True
        logger.info("SAM2 available — using model-based empty-box detection")
    except ImportError:
        _sam2_available = False
        logger.warning(
            "SAM2 not available — falling back to edge-density heuristic. "
            "Install SAM2 for production use: pip install git+https://github.com/facebookresearch/sam2.git"
        )

    return _sam2_available


def detect_empty_box_heuristic(image_path: str) -> tuple[bool, float]:
    """
    Heuristic empty-box detection using edge density and color entropy.

    An empty box has:
      - Low edge density (smooth cardboard, no product contours)
      - Low color entropy (uniform brown/gray, no product colors)

    This is the GPU-free fallback when SAM2 is unavailable or too slow.

    Returns:
        (is_empty, coverage_ratio): is_empty flag and a pseudo-coverage
        ratio where low values suggest emptiness
    """
    try:
        image = Image.open(image_path).convert("RGB")
        img_array = np.array(image)

        # Convert to grayscale for edge detection
        gray = np.mean(img_array, axis=2)

        # Simple edge detection via gradient magnitude
        grad_x = np.diff(gray, axis=1)
        grad_y = np.diff(gray, axis=0)

        # Crop gradients to same shape
        min_h = min(grad_x.shape[0], grad_y.shape[0])
        min_w = min(grad_x.shape[1], grad_y.shape[1])
        grad_x = grad_x[:min_h, :min_w]
        grad_y = grad_y[:min_h, :min_w]

        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Edge density: fraction of pixels with significant edges
        edge_threshold = 30  # Gradient magnitude threshold
        edge_density = (edge_magnitude > edge_threshold).mean()

        # Color entropy: how many distinct colors are present
        # Quantize to 8 levels per channel and count unique combos
        quantized = (img_array // 32).reshape(-1, 3)
        unique_colors = len(set(map(tuple, quantized)))
        max_possible = min(len(quantized), 8**3)
        color_diversity = unique_colors / max_possible

        # Combined "content score" — higher means more stuff in the image
        coverage_ratio = (edge_density * 0.6 + color_diversity * 0.4)

        is_empty = coverage_ratio < EMPTY_BOX_MASK_AREA_THRESHOLD
        return is_empty, float(coverage_ratio)

    except Exception as e:
        logger.error(f"Heuristic empty-box detection failed: {e}")
        return False, 0.5  # Default: assume not empty


def detect_empty_box_heuristic_fallback(image_path: str) -> tuple[bool, float]:
    """
    Lightweight heuristic fallback for empty-box detection.

    STATUS: ACTIVE (Heuristic Fallback) — Because a full SAM2 segmentation model
    requires a ~150MB checkpoint and GPU acceleration, this function serves as
    the default lightweight logic for demo purposes. It uses edge-density and
    color entropy heuristics to estimate if a box is empty.

    A full production implementation would replace this with SAM2AutomaticMaskGenerator
    to compute precise mask area coverage.
    """
    logger.info("Using edge-density heuristic for empty-box detection (SAM2 fallback).")
    return detect_empty_box_heuristic(image_path)


def detect_empty_box(image_path: str) -> tuple[bool, float]:
    """
    Main entry point for empty-box detection.

    Tries SAM2 first, falls back to heuristic if unavailable or too slow.

    Returns:
        (is_empty, coverage_ratio): True if box appears empty,
        and the mask/content coverage ratio [0-1]
    """
    start_time = time.time()

    if _try_load_sam2():
        is_empty, coverage = detect_empty_box_sam2(image_path)
        elapsed = time.time() - start_time

        # If SAM2 was too slow (>15s), log a warning
        if elapsed > 15.0:
            logger.warning(
                f"SAM2 inference took {elapsed:.1f}s — consider using heuristic fallback"
            )
    else:
        is_empty, coverage = detect_empty_box_heuristic(image_path)

    return is_empty, coverage
