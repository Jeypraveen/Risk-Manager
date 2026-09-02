"""
Heuristic-based empty-box detection.

Estimates if a return box is empty using edge density and color entropy.
If the content coverage ratio is near zero relative to the image, the box
is likely empty — a deterministic signal, not a trained classifier.

An empty box typically has:
  - Low edge density (smooth cardboard, no product contours)
  - Low color entropy (uniform brown/gray, no product colors)
"""

import logging

import numpy as np
from PIL import Image

from src.config import EMPTY_BOX_MASK_AREA_THRESHOLD

logger = logging.getLogger(__name__)


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
        return bool(is_empty), float(coverage_ratio)

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

    Uses edge-density and color entropy heuristics to estimate if a box is empty.
    
    Returns:
        (is_empty, coverage_ratio): True if box appears empty,
        and the mask/content coverage ratio [0-1]
    """
    return detect_empty_box_heuristic_fallback(image_path)
