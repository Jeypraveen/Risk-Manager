"""
Circuit breaker for the vision pipeline.

Detects two failure modes and handles them gracefully:

1. IMAGE UNAVAILABLE: Image upload times out, CDN is down, or no photo
   was provided. This is a routine operational condition.
   Recovery: modality_confidence = 0, tabular-only scoring, raised
   approve threshold.

2. MODEL CHECKPOINT MISMATCH: DINOv2 output embedding dimension doesn't
   match expected dimension (e.g., wrong checkpoint loaded). This is a
   system integrity issue.
   Recovery: modality_confidence = 0, force MANUAL_REVIEW, log as
   VISION_MODEL_FAILURE audit event.

Both failures:
  - Never crash the request pipeline
  - Always produce a valid decision (degrade, never die)
  - Are logged as first-class audit events
  - Can be triggered for demo via API flags
"""

import logging
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.config import DINOV2_EMBEDDING_DIM

logger = logging.getLogger(__name__)


class FailureType(str, Enum):
    """Types of vision pipeline failures."""
    NONE = "none"
    IMAGE_UNAVAILABLE = "image_unavailable"
    VISION_MODEL_FAILURE = "vision_model_failure"


@dataclass
class FailureEvent:
    """A structured record of a vision pipeline failure."""
    failure_type: FailureType
    message: str
    details: Optional[str] = None  # Stack trace, dimension info, etc.
    model_version: Optional[str] = None
    expected_dim: Optional[int] = None
    actual_dim: Optional[int] = None


class CircuitBreaker:
    """
    Circuit breaker for the vision pipeline.

    Wraps vision operations and catches failures, converting them into
    structured FailureEvents instead of letting them crash the request.
    """

    def __init__(self):
        self._simulate_timeout = False
        self._simulate_checkpoint_mismatch = False
        self._failure_log: list[FailureEvent] = []

    # ── Simulation controls (for demo) ──

    def enable_timeout_simulation(self) -> None:
        """Enable simulated image upload timeout for demo."""
        self._simulate_timeout = True
        logger.warning("⚡ Circuit breaker: timeout simulation ENABLED")

    def enable_checkpoint_mismatch_simulation(self) -> None:
        """Enable simulated model checkpoint mismatch for demo."""
        self._simulate_checkpoint_mismatch = True
        logger.warning("⚡ Circuit breaker: checkpoint mismatch simulation ENABLED")

    def reset_simulations(self) -> None:
        """Disable all failure simulations."""
        self._simulate_timeout = False
        self._simulate_checkpoint_mismatch = False
        logger.info("Circuit breaker: all simulations RESET")

    # ── Failure detection ──

    def check_image_availability(self, image_path: Optional[str]) -> Optional[FailureEvent]:
        """
        Check if the return image is available.

        Returns None if image is available, FailureEvent if not.
        """
        # Simulated timeout for demo
        if self._simulate_timeout:
            event = FailureEvent(
                failure_type=FailureType.IMAGE_UNAVAILABLE,
                message="Image upload timed out (simulated for demo)",
                details="Simulated CDN/upload timeout — image service unreachable",
            )
            self._failure_log.append(event)
            logger.warning(f"Circuit breaker triggered: {event.message}")
            return event

        # Real check: no image provided
        if image_path is None:
            event = FailureEvent(
                failure_type=FailureType.IMAGE_UNAVAILABLE,
                message="No return image provided by customer",
                details="Return request submitted without photo attachment",
            )
            self._failure_log.append(event)
            logger.info(f"Circuit breaker: {event.message}")
            return event

        return None

    def check_embedding_dimensions(
        self,
        embedding: Optional[object],
        expected_dim: int = DINOV2_EMBEDDING_DIM,
    ) -> Optional[FailureEvent]:
        """
        Check if the DINOv2 embedding has the expected dimensions.

        Catches the case where a wrong model checkpoint is loaded,
        producing embeddings of the wrong size. This would cause
        downstream cosine similarity to fail silently or crash.

        Returns None if dimensions match, FailureEvent if not.
        """
        # Simulated mismatch for demo
        if self._simulate_checkpoint_mismatch:
            event = FailureEvent(
                failure_type=FailureType.VISION_MODEL_FAILURE,
                message="Embedding dimension mismatch (simulated for demo)",
                details=f"Expected dim={expected_dim}, got dim=768 (wrong checkpoint loaded)",
                model_version="facebook/dinov2-base (WRONG — should be dinov2-small)",
                expected_dim=expected_dim,
                actual_dim=768,
            )
            self._failure_log.append(event)
            logger.error(f"Circuit breaker triggered: {event.message}")
            return event

        if embedding is None:
            event = FailureEvent(
                failure_type=FailureType.VISION_MODEL_FAILURE,
                message="DINOv2 returned null embedding",
                details="Model inference returned None — possible OOM or model corruption",
            )
            self._failure_log.append(event)
            logger.error(f"Circuit breaker triggered: {event.message}")
            return event

        # Check actual dimensions
        import numpy as np
        emb_arr = np.asarray(embedding)
        actual_dim = emb_arr.shape[-1] if len(emb_arr.shape) > 0 else 0
        if actual_dim != expected_dim:
            event = FailureEvent(
                failure_type=FailureType.VISION_MODEL_FAILURE,
                message=f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}",
                details=f"This indicates a wrong model checkpoint was loaded. "
                        f"Expected dinov2-small (dim={expected_dim}), "
                        f"but got a model producing dim={actual_dim}.",
                expected_dim=expected_dim,
                actual_dim=actual_dim,
            )
            self._failure_log.append(event)
            logger.error(f"Circuit breaker triggered: {event.message}")
            return event

        return None

    def wrap_vision_call(self, callable_fn, *args, **kwargs):
        """
        Wrap any vision pipeline call in a try/except.

        If the call raises any exception, catch it, log it,
        and return (None, FailureEvent) instead of crashing.

        Returns:
            (result, failure_event): result is the callable's return value
            if successful, None if failed. failure_event is None if
            successful, FailureEvent if failed.
        """
        try:
            result = callable_fn(*args, **kwargs)
            return result, None
        except Exception as e:
            event = FailureEvent(
                failure_type=FailureType.VISION_MODEL_FAILURE,
                message=f"Vision pipeline exception: {type(e).__name__}: {str(e)}",
                details=traceback.format_exc(),
            )
            self._failure_log.append(event)
            logger.error(f"Circuit breaker caught exception: {event.message}")
            return None, event

    # ── State access ──

    @property
    def failure_log(self) -> list[FailureEvent]:
        """Return all recorded failure events."""
        return self._failure_log.copy()

    @property
    def is_simulating(self) -> bool:
        """Check if any simulation is active."""
        return self._simulate_timeout or self._simulate_checkpoint_mismatch

    def get_simulation_status(self) -> dict:
        """Return current simulation state."""
        return {
            "timeout_simulation": self._simulate_timeout,
            "checkpoint_mismatch_simulation": self._simulate_checkpoint_mismatch,
            "total_failures_recorded": len(self._failure_log),
        }
