"""
Prediction interface for the tabular risk scorer.

Loads a trained LightGBM model and provides a clean inference API.
The output is a RANKING score (not a calibrated probability).
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.tabular.features import prepare_single_request, prepare_features


class TabularScorer:
    """
    Tabular return-risk scorer using a trained LightGBM model.

    Usage:
        scorer = TabularScorer()
        score = scorer.predict(return_request_dict)
    """

    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            model_path = MODELS_DIR / "tabular_scorer.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run `python -m src.tabular.train` first."
            )

        checkpoint = joblib.load(model_path)
        self.model = checkpoint["model"]
        self.feature_columns = checkpoint["feature_columns"]
        # Per-category order_value medians from training data.
        # Used to compute return_value_ratio correctly on single-row inference.
        # Falls back to None for checkpoints saved before this fix (safe — just uses row self-median).
        self.category_medians: Optional[dict] = checkpoint.get("category_medians", None)

    def predict(self, request: dict) -> float:
        """
        Score a single return request.

        Args:
            request: Dict with return request fields (order_value,
                     account_age_days, prior_returns_count, etc.)

        Returns:
            float: Risk score in [0, 1]. Higher = more likely legitimate.
                   This is a RANKING score, not a calibrated probability.
                   Thresholds would be tuned on real operational data.
        """
        features_df = prepare_single_request(request, category_medians=self.category_medians)
        # Ensure column order matches training
        features_df = features_df.reindex(columns=self.feature_columns, fill_value=0)
        prob = self.model.predict_proba(features_df.values)[:, 1]
        # Return 1 - fraud_probability as "trust score" (higher = more trustworthy)
        return float(1.0 - prob[0])

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Score a batch of return requests.

        Args:
            df: DataFrame with return request fields

        Returns:
            np.ndarray: Array of trust scores (1 - fraud_probability)
        """
        df_prepared, _ = prepare_features(df, category_medians=self.category_medians)
        df_prepared = df_prepared.reindex(columns=self.feature_columns, fill_value=0)
        probs = self.model.predict_proba(df_prepared.values)[:, 1]
        return 1.0 - probs

    def get_feature_contributions(self, request: dict) -> dict[str, float]:
        """
        Get per-instance feature contributions using LightGBM's built-in
        pred_contrib (leaf-based SHAP values).

        Unlike global feature_importances_, this returns values specific
        to THIS particular return request — different inputs produce
        different contribution rankings.

        Returns:
            Dict mapping feature name to its absolute contribution weight
            for this specific request (normalized to sum to 1.0).
        """
        df = pd.DataFrame([request])
        df_prepared, _ = prepare_features(df, category_medians=self.category_medians)
        df_prepared = df_prepared.reindex(columns=self.feature_columns, fill_value=0)

        # pred_contrib=True returns per-leaf SHAP values: shape (1, n_features + 1)
        # Last column is the base value (bias); we drop it.
        contribs = self.model.predict_proba(
            df_prepared.values, pred_contrib=True
        )
        # LightGBM returns contributions for each class; take class 1 (fraud)
        # Shape: (1, n_features + 1) for binary classification
        if contribs.ndim == 3:
            # Shape (n_samples, n_classes, n_features+1) — take class 1
            row_contribs = contribs[0, 1, :-1]
        else:
            # Shape (n_samples, n_features+1)
            row_contribs = contribs[0, :-1]

        # Use absolute values for ranking (direction doesn't matter for "top features")
        abs_contribs = np.abs(row_contribs)
        total = abs_contribs.sum() + 1e-8
        return {
            name: float(abs_c / total)
            for name, abs_c in zip(self.feature_columns, abs_contribs)
        }

