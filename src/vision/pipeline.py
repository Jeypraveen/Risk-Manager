"""
Vision pipeline orchestrator.

Coordinates DINOv2 similarity and SAM2 empty-box detection into a
single VisionResult, handling failures via the circuit breaker.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from src.recovery.circuit_breaker import CircuitBreaker, FailureType

logger = logging.getLogger(__name__)


@dataclass
class VisionResult:
    """Complete output from the vision pipeline."""
    semantic_similarity: Optional[float] = None
    empty_box_flag: Optional[bool] = None
    mask_coverage_ratio: Optional[float] = None
    modality_confidence: float = 0.0
    failure_type: Optional[str] = None
    failure_message: Optional[str] = None
    failure_details: Optional[str] = None


def run_vision_pipeline(
    catalog_image_path: Optional[str],
    return_image_path: Optional[str],
    circuit_breaker: Optional[CircuitBreaker] = None,
) -> VisionResult:
    """
    Run the full vision pipeline on a return request.

    Handles three cases:
      1. Both signals computed successfully → modality_confidence = 1.0
      2. One signal fails → modality_confidence = 0.5
      3. Both fail or no image → modality_confidence = 0.0

    The circuit breaker catches exceptions and converts them to
    structured failures instead of crashes.

    Args:
        catalog_image_path: Path to catalog/reference product image
        return_image_path: Path to customer's return photo (or None)
        circuit_breaker: Optional CircuitBreaker instance for failure handling

    Returns:
        VisionResult with all signals and confidence
    """
    if circuit_breaker is None:
        circuit_breaker = CircuitBreaker()

    # ── Check image availability ──
    failure = circuit_breaker.check_image_availability(return_image_path)
    if failure is not None:
        return VisionResult(
            modality_confidence=0.0,
            failure_type=failure.failure_type.value,
            failure_message=failure.message,
            failure_details=failure.details,
        )

    if catalog_image_path is None:
        return VisionResult(
            modality_confidence=0.0,
            failure_type="image_unavailable",
            failure_message="No catalog reference image available",
        )

    signals_computed = 0
    result = VisionResult()

    # ── DINOv2 Similarity ──
    try:
        from src.vision.similarity import compute_similarity

        sim_result, sim_failure = circuit_breaker.wrap_vision_call(
            compute_similarity, catalog_image_path, return_image_path
        )

        if sim_failure is not None:
            result.failure_type = sim_failure.failure_type.value
            result.failure_message = sim_failure.message
            result.failure_details = sim_failure.details
        elif sim_result is not None:
            # Check embedding dimensions via circuit breaker
            from src.vision.similarity import extract_embedding
            test_emb, emb_failure = circuit_breaker.wrap_vision_call(
                extract_embedding, return_image_path
            )
            dim_failure = circuit_breaker.check_embedding_dimensions(test_emb)

            if dim_failure is not None:
                result.failure_type = dim_failure.failure_type.value
                result.failure_message = dim_failure.message
                result.failure_details = dim_failure.details
            else:
                result.semantic_similarity = sim_result
                signals_computed += 1

    except Exception as e:
        logger.error(f"DINOv2 similarity failed: {e}")
        result.failure_type = FailureType.VISION_MODEL_FAILURE.value
        result.failure_message = f"DINOv2 exception: {str(e)}"

    # ── SAM2 / Heuristic Empty-Box Detection ──
    try:
        from src.vision.empty_box import detect_empty_box_heuristic_fallback

        box_result, box_failure = circuit_breaker.wrap_vision_call(
            detect_empty_box_heuristic_fallback, return_image_path
        )

        if box_failure is not None:
            if result.failure_type is None:
                result.failure_type = box_failure.failure_type.value
                result.failure_message = box_failure.message
        elif box_result is not None:
            is_empty, coverage = box_result
            result.empty_box_flag = is_empty
            result.mask_coverage_ratio = coverage
            signals_computed += 1

    except Exception as e:
        logger.error(f"Empty-box detection failed: {e}")

    # ── Compute modality confidence ──
    if signals_computed == 2:
        result.modality_confidence = 1.0
    elif signals_computed == 1:
        result.modality_confidence = 0.5
    else:
        result.modality_confidence = 0.0

    return result
