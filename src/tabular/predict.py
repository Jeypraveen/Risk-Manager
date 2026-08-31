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
        Get per-feature contribution to the prediction using SHAP.

        Returns:
            Dict mapping feature name to its importance weight for THIS specific request.
        """
        df = pd.DataFrame([request])
        df_prepared, _ = prepare_features(df, category_medians=self.category_medians)
        df_prepared = df_prepared.reindex(columns=self.feature_columns, fill_value=0)

        try:
            import shap  # type: ignore
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*LightGBM binary classifier with TreeExplainer shap values output has changed.*")
                # Create a TreeExplainer for the LightGBM model
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(df_prepared)
            # shap_values[1] contains the values for the positive class (fraud)
            # If shap_values is a list, take [1]. If it's a single array (binary clf in some versions), take it.
            if isinstance(shap_values, list):
                vals = shap_values[1][0]
            else:
                vals = shap_values[0]

            # Normalize magnitudes to sum to 1.0 for the UI
            abs_vals = np.abs(vals)
            total = abs_vals.sum() + 1e-8
            return {
                name: float(val / total)
                for name, val in zip(self.feature_columns, abs_vals)
            }
        except Exception as e:
            # Fallback to global importance if SHAP fails
            print(f"SHAP failed: {e}. Falling back to global importance.")
            importances = self.model.feature_importances_
            total = importances.sum() + 1e-8
            return {
                name: float(imp / total)
                for name, imp in zip(self.feature_columns, importances)
            }
