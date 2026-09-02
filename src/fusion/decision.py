"""
Three-way decision logic.

Instead of binary approve/reject, the system routes each return to one of:
  - AUTO_APPROVE: high trust → instant refund
  - NUDGE: medium trust → offer store credit or request photo
  - MANUAL_REVIEW: low trust → human investigator reviews with score breakdown

This design is DEFENSE-ONLY: the system never auto-denies. Even the lowest
trust score routes to human review, not automatic rejection. This satisfies
the brief's "strictly defense-only" requirement.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.config import (
    THRESHOLD_AUTO_APPROVE,
    THRESHOLD_MANUAL_REVIEW,
    STORE_CREDIT_THRESHOLD_PENALTY,
    MAX_REPHOTO_REQUESTS,
    VISION_FAILURE_THRESHOLD_RAISE,
    REPHOTO_MESSAGE,
    STORE_CREDIT_MESSAGE,
)


class Decision(str, Enum):
    """The three possible outcomes for a return request."""
    AUTO_APPROVE = "auto_approve"
    MANUAL_REVIEW = "manual_review"
    NUDGE = "nudge"


class NudgeType(str, Enum):
    """Subtypes of the NUDGE decision."""
    REQUEST_PHOTO = "request_photo"
    OFFER_STORE_CREDIT = "offer_store_credit"


@dataclass
class DecisionResult:
    """Complete decision output with all context for audit logging."""
    decision: Decision
    nudge_type: Optional[NudgeType] = None
    nudge_message: Optional[str] = None
    trust_score: float = 0.0
    effective_approve_threshold: float = THRESHOLD_AUTO_APPROVE
    effective_review_threshold: float = THRESHOLD_MANUAL_REVIEW
    reason: str = ""
    forced_by: Optional[str] = None  # If decision was forced (e.g., circuit breaker)


def make_decision(
    trust_score: float,
    modality_confidence: float = 1.0,
    rephoto_count: int = 0,
    prior_store_credit_count: int = 0,
    vision_failed: bool = False,
) -> DecisionResult:
    """
    Apply three-way decision logic to a return request.

    Args:
        trust_score: Fused Return Trust Score [0-1], higher = more trustworthy
        modality_confidence: Vision pipeline confidence [0-1]
        rephoto_count: Number of re-photo requests already made for this return
        prior_store_credit_count: Number of prior returns where this account
                                  accepted store credit (Modification #6)
        vision_failed: Whether the vision pipeline experienced a failure
                      (not just missing image — actual system failure)

    Returns:
        DecisionResult with the decision, reasoning, and all context
    """
    # ── Compute effective thresholds ──
    approve_threshold = THRESHOLD_AUTO_APPROVE
    review_threshold = THRESHOLD_MANUAL_REVIEW

    # Modification #6: Store-credit recipients get a higher bar for auto-approval
    if prior_store_credit_count > 0:
        approve_threshold += STORE_CREDIT_THRESHOLD_PENALTY
        approve_threshold = min(approve_threshold, 0.95)  # Cap at 0.95

    # Modification #7: If vision gracefully degraded or image missing, raise the bar for auto-approval
    if modality_confidence == 0.0:
        approve_threshold += VISION_FAILURE_THRESHOLD_RAISE
        approve_threshold = min(approve_threshold, 1.0)  # Never auto-approve if this pushes it > 1.0

    # ── Forced decisions ──

    # If vision system had an actual FAILURE (not just missing photo),
    # force to manual review — never auto-approve on a broken system
    if vision_failed:
        return DecisionResult(
            decision=Decision.MANUAL_REVIEW,
            trust_score=trust_score,
            effective_approve_threshold=approve_threshold,
            effective_review_threshold=review_threshold,
            reason="Vision pipeline failure — routing to manual review for safety",
            forced_by="circuit_breaker",
        )

    # Modification #5: If re-photo cap reached, force to manual review
    if rephoto_count >= MAX_REPHOTO_REQUESTS:
        return DecisionResult(
            decision=Decision.MANUAL_REVIEW,
            trust_score=trust_score,
            effective_approve_threshold=approve_threshold,
            effective_review_threshold=review_threshold,
            reason=f"Re-photo request cap reached ({rephoto_count}/{MAX_REPHOTO_REQUESTS})",
            forced_by="rephoto_cap",
        )

    # ── Standard three-way logic ──

    if trust_score >= approve_threshold:
        return DecisionResult(
            decision=Decision.AUTO_APPROVE,
            trust_score=trust_score,
            effective_approve_threshold=approve_threshold,
            effective_review_threshold=review_threshold,
            reason=f"Trust score ({trust_score:.3f}) >= approve threshold ({approve_threshold:.3f})",
        )

    if trust_score < review_threshold:
        return DecisionResult(
            decision=Decision.MANUAL_REVIEW,
            trust_score=trust_score,
            effective_approve_threshold=approve_threshold,
            effective_review_threshold=review_threshold,
            reason=f"Trust score ({trust_score:.3f}) < review threshold ({review_threshold:.3f})",
        )

    # ── NUDGE zone: between review and approve thresholds ──

    # If no photo was provided and we haven't hit the cap, request a photo
    # Modification #5: Generic message, no detection logic leakage
    if modality_confidence == 0.0 and rephoto_count < MAX_REPHOTO_REQUESTS:
        return DecisionResult(
            decision=Decision.NUDGE,
            nudge_type=NudgeType.REQUEST_PHOTO,
            nudge_message=REPHOTO_MESSAGE,
            trust_score=trust_score,
            effective_approve_threshold=approve_threshold,
            effective_review_threshold=review_threshold,
            reason="Medium trust + no photo → requesting photo verification",
        )

    # If photo was provided but trust is still medium, offer store credit
    return DecisionResult(
        decision=Decision.NUDGE,
        nudge_type=NudgeType.OFFER_STORE_CREDIT,
        nudge_message=STORE_CREDIT_MESSAGE,
        trust_score=trust_score,
        effective_approve_threshold=approve_threshold,
        effective_review_threshold=review_threshold,
        reason="Medium trust with photo provided → offering store credit alternative",
    )
