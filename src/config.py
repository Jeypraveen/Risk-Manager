"""
Central configuration for the Return-Risk Scorer.

All thresholds, cost parameters, feature lists, and paths live here.
This is the single source of truth — no magic numbers in other modules.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
IMAGES_DIR = DATA_DIR / "images"
CATALOG_DIR = IMAGES_DIR / "catalog"
RETURNS_DIR = IMAGES_DIR / "returns"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
EVAL_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
AUDIT_DB_PATH = DATA_DIR / "audit.db"

# Ensure directories exist at import time
for d in [SYNTHETIC_DIR, CATALOG_DIR, RETURNS_DIR, CACHE_DIR, MODELS_DIR, EVAL_RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Synthetic Data Generation
# ──────────────────────────────────────────────
RANDOM_SEED = 42
TOTAL_SAMPLES = 10_000
FRAUD_RATE = 0.08  # 8% — realistic for Indian e-commerce returns
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
PHOTO_AVAILABILITY_RATE = 0.70  # 70% of returns have a photo

# Fraud subtypes (must sum to 1.0)
FRAUD_SUBTYPE_DIST = {
    "empty_box": 0.40,
    "substitution": 0.40,
    "wardrobing": 0.20,
}

# ──────────────────────────────────────────────
# Tabular Model — Feature Configuration
# ──────────────────────────────────────────────
TABULAR_FEATURES = [
    "order_value",
    "account_age_days",
    "prior_returns_count",
    "prior_return_approval_rate",
    "return_velocity_7d",
    "is_cod",
    "delivery_to_return_hours",
    "address_order_distance_km",
    # Derived features (computed in features.py)
    "return_value_ratio",
    "account_returns_per_month",
    "is_near_policy_deadline",
]

CATEGORICAL_FEATURES = ["item_category"]

# Return policy window (in hours) — used for is_near_policy_deadline
RETURN_POLICY_HOURS = 30 * 24  # 30 days

# Item categories
ITEM_CATEGORIES = ["electronics", "fashion", "home", "beauty", "books", "sports"]

# ──────────────────────────────────────────────
# Cost Parameters (Back-of-Envelope)
# ──────────────────────────────────────────────
# C_FP: Cost of a false positive — flagging a legitimate customer as fraud.
# Components: customer churn risk (~₹1,000 estimated CLV loss for a mid-tier
# customer) + dispute handling labor (~₹200). Conservative estimate.
C_FP = 1200.0  # ₹1,200

# C_FN: Cost of a false negative — missing a fraud case.
# Components: unrecovered item value (median ~₹2,000 for mixed categories)
# + sunk shipping cost (~₹150 forward + ₹150 return).
C_FN = 2300.0  # ₹2,300

# ──────────────────────────────────────────────
# Decision Thresholds
# ──────────────────────────────────────────────
# These are INITIAL values. Phase 6 will compute the cost-optimal threshold.
# trust_score >= THRESHOLD_AUTO_APPROVE  →  auto-approve refund
# trust_score < THRESHOLD_MANUAL_REVIEW  →  route to human review
# in between                             →  nudge (store credit / request photo)

THRESHOLD_AUTO_APPROVE = 0.75
THRESHOLD_MANUAL_REVIEW = 0.30

# When a prior store-credit recipient returns again, we raise the
# auto-approve threshold by this delta — making auto-approval harder.
STORE_CREDIT_THRESHOLD_PENALTY = 0.10

# ──────────────────────────────────────────────
# Vision Pipeline
# ──────────────────────────────────────────────
# DINOv2 — using the smallest distilled variant for CPU feasibility
DINOV2_MODEL_NAME = "facebook/dinov2-small"
DINOV2_EMBEDDING_DIM = 384  # ViT-S/14 output dimension


# Empty-box detection: if the ratio of detected mask area to total image area
# is below this threshold, flag as potentially empty.
EMPTY_BOX_MASK_AREA_THRESHOLD = 0.10  # 10% of image area

# ──────────────────────────────────────────────
# Nudge & Re-Photo Logic
# ──────────────────────────────────────────────
# Maximum number of re-photo requests before forcing manual review.
# After this many requests, the system MUST route to MANUAL_REVIEW.
MAX_REPHOTO_REQUESTS = 2

# Generic message for re-photo nudge — no detection logic leakage.
REPHOTO_MESSAGE = (
    "To process your return, please upload a clear photo of the "
    "returned item. This helps us verify and speed up your refund."
)

# Store credit nudge message
STORE_CREDIT_MESSAGE = (
    "We can process your return as store credit for immediate use, "
    "or continue with a standard refund review (1-3 business days)."
)

# ──────────────────────────────────────────────
# Circuit Breaker
# ──────────────────────────────────────────────
# When vision fails, raise the auto-approve threshold to be more conservative.
# The system should never auto-approve when it can't visually verify.
VISION_FAILURE_THRESHOLD_RAISE = 0.15

# ──────────────────────────────────────────────
# Audit Trail
# ──────────────────────────────────────────────
AUDIT_TABLE_NAME = "decisions"
