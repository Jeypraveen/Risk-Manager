"""
Feature engineering for the tabular risk scorer.

Transforms raw return request data into the feature vector used by LightGBM.
All transformations are stateless and deterministic — no information leaks
from train to test.

Note on return_value_ratio at inference time:
    At training time, category medians are computed from the full batch.
    At inference time on a single row, the median of a 1-row group is just
    the row's own value — always 1.0. To fix this, the training pipeline saves
    per-category medians to the model checkpoint and passes them at inference.
    Pass `category_medians` to compute_derived_features() to use saved medians.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.config import ITEM_CATEGORIES, RETURN_POLICY_HOURS, TABULAR_FEATURES


def compute_derived_features(
    df: pd.DataFrame,
    category_medians: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Add derived features to the raw return data.

    Derived features capture non-obvious signals that a single raw column
    doesn't express:
      - return_value_ratio: how expensive is this return relative to the
        category median? Fraud targets high-value items.
      - account_returns_per_month: normalized return frequency. A 30-day-old
        account with 5 returns is very different from a 2-year-old account
        with 5 returns.
      - is_near_policy_deadline: returns filed close to the policy deadline
        are a known fraud pattern (stretching the window to keep the item
        as long as possible, or timing it so review staff are rushed).
    """
    df = df.copy()

    # Return value ratio: order_value / category median
    # If pre-computed medians are provided (loaded from training checkpoint), use them.
    # This avoids the single-row inference bug where group median == row's own value.
    if category_medians is not None:
        mapped = df["item_category"].map(category_medians)
        # Fill unknown categories with the global median of known medians
        global_fallback = float(np.median(list(category_medians.values()))) if category_medians else 1.0
        mapped = mapped.fillna(global_fallback)
        df["return_value_ratio"] = df["order_value"] / mapped.clip(lower=1.0)
    else:
        # Batch mode (training/evaluation): compute from the dataframe itself
        batch_medians = df.groupby("item_category")["order_value"].transform("median")
        df["return_value_ratio"] = df["order_value"] / batch_medians.clip(lower=1.0)

    # Account returns per month: normalized frequency
    # Clip account_age_days to minimum 1 to avoid division by zero
    months = df["account_age_days"].clip(lower=1) / 30.0
    df["account_returns_per_month"] = df["prior_returns_count"] / months

    # Near policy deadline flag: delivery_to_return_hours > 85% of policy window
    deadline_threshold = RETURN_POLICY_HOURS * 0.85
    df["is_near_policy_deadline"] = (
        df["delivery_to_return_hours"] > deadline_threshold
    ).astype(int)

    return df


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode item_category.

    Uses a fixed category list from config to ensure train/test consistency.
    Unknown categories at inference time get all-zero encoding.
    """
    df = df.copy()
    for cat in ITEM_CATEGORIES:
        df[f"item_category_{cat}"] = (df["item_category"] == cat).astype(int)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the ordered list of feature column names used by the model.

    This is the canonical feature order — the model expects columns in
    exactly this order.
    """
    # Numeric features from config
    numeric_features = [f for f in TABULAR_FEATURES if f in df.columns]

    # One-hot encoded category columns
    category_features = [f"item_category_{cat}" for cat in ITEM_CATEGORIES]

    return numeric_features + category_features


def prepare_features(
    df: pd.DataFrame,
    category_medians: Optional[dict] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Full feature preparation pipeline.

    Args:
        df: Raw return data with columns from generate_data.py
        category_medians: Per-category order_value medians from training data.
            Pass this at inference time to avoid the single-row median bug.

    Returns:
        (feature_df, feature_columns): The feature matrix and ordered column names
    """
    df = compute_derived_features(df, category_medians=category_medians)
    df = encode_categories(df)
    feature_cols = get_feature_columns(df)
    return df, feature_cols


def prepare_single_request(
    request: dict,
    category_medians: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Prepare features for a single return request at inference time.

    Args:
        request: Dict with raw return request fields
        category_medians: Per-category order_value medians saved from training.
            If None, falls back to single-row group median (always 1.0 — incorrect).
            Always pass this at inference time.

    Returns:
        Single-row DataFrame with all features computed
    """
    df = pd.DataFrame([request])
    df, feature_cols = prepare_features(df, category_medians=category_medians)
    return df[feature_cols]
