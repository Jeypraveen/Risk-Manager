"""
Late-fusion meta-learner.

Combines tabular risk score with vision signals into a single
Return Trust Score using logistic regression.

Late fusion was chosen deliberately: images are often missing or
low-quality, and late fusion lets any one modality degrade without
collapsing the whole system. The modality_confidence input lets the
meta-learner explicitly weight vision signals based on their availability.
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import MODELS_DIR, RANDOM_SEED


# Input feature names for the meta-learner, in order
META_FEATURES = [
    "tabular_risk_score",
    "semantic_similarity",
    "empty_box_flag",
    "modality_confidence",
]

# Default values when vision signals are unavailable
DEFAULTS = {
    "semantic_similarity": 0.5,  # Neutral — neither similar nor different
    "empty_box_flag": 0,         # Assume not empty (conservative)
    "modality_confidence": 0.0,  # No vision data available
}


class MetaLearner:
    """
    Logistic regression meta-learner for late fusion.

    Takes [tabular_score, semantic_similarity, empty_box_flag,
    modality_confidence] and outputs a Return Trust Score.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model: Optional[LogisticRegression] = None
        self.model_path = model_path or (MODELS_DIR / "meta_learner.joblib")
        self.is_trained = False

        if self.model_path.exists():
            self._load()

    def _load(self) -> None:
        checkpoint = joblib.load(self.model_path)
        self.model = checkpoint["model"]
        self.is_trained = True

    def train(
        self,
        tabular_scores: np.ndarray,
        semantic_similarities: np.ndarray,
        empty_box_flags: np.ndarray,
        modality_confidences: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        """
        Train the meta-learner on combined signals.

        Args:
            tabular_scores: Tabular model trust scores [0-1]
            semantic_similarities: DINOv2 cosine similarities [0-1] or defaults
            empty_box_flags: SAM2 empty-box detection [0/1] or defaults
            modality_confidences: Vision pipeline confidence [0-1]
            labels: Ground truth fraud labels [0/1] (1 = fraud)

        Returns:
            Training metrics dict
        """
        X = np.column_stack([
            tabular_scores,
            semantic_similarities,
            empty_box_flags,
            modality_confidences,
        ])

        # Labels: 1 = fraud. We want trust score where higher = more legit.
        # So we train to predict NOT fraud (0 = fraud, 1 = legit).
        y = 1 - labels

        self.model = LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=1000,
            class_weight="balanced",
        )
        self.model.fit(X, y)
        self.is_trained = True

        # Save
        joblib.dump({"model": self.model}, self.model_path)

        # Report coefficients — interpretable since it's logistic regression
        coefs = dict(zip(META_FEATURES, self.model.coef_[0]))
        intercept = float(np.asarray(self.model.intercept_)[0])

        metrics = {
            "coefficients": {k: float(v) for k, v in coefs.items()},
            "intercept": intercept,
            "train_accuracy": float(self.model.score(X, y)),  # type: ignore
        }

        print("\nMeta-learner coefficients:")
        for feat, coef in coefs.items():
            print(f"  {feat:30s} {coef:+.4f}")
        print(f"  {'intercept':30s} {intercept:+.4f}")

        return metrics

    def predict(
        self,
        tabular_score: float,
        semantic_similarity: Optional[float] = None,
        empty_box_flag: Optional[int] = None,
        modality_confidence: float = 0.0,
    ) -> float:
        """
        Predict the Return Trust Score for a single request.

        Missing vision signals are replaced with neutral defaults,
        and modality_confidence tells the meta-learner how much to
        weight the vision signals.

        Returns:
            float: Trust score in [0, 1]. Higher = more trustworthy.
                   This is a RANKING score, not a calibrated probability.
        """
        if not self.is_trained:
            # Fallback: if meta-learner isn't trained, just return tabular score
            return tabular_score

        features = np.array([[
            tabular_score,
            semantic_similarity if semantic_similarity is not None else DEFAULTS["semantic_similarity"],
            empty_box_flag if empty_box_flag is not None else DEFAULTS["empty_box_flag"],
            modality_confidence,
        ]])

        # predict_proba returns [P(fraud), P(legit)]
        assert self.model is not None
        trust_score = self.model.predict_proba(features)[0, 1]  # type: ignore
        return float(trust_score)

    def predict_batch(
        self,
        tabular_scores: np.ndarray,
        semantic_similarities: np.ndarray,
        empty_box_flags: np.ndarray,
        modality_confidences: np.ndarray,
    ) -> np.ndarray:
        """Predict trust scores for a batch."""
        if not self.is_trained:
            return tabular_scores

        X = np.column_stack([
            tabular_scores,
            semantic_similarities,
            empty_box_flags,
            modality_confidences,
        ])
        assert self.model is not None
        return np.asarray(self.model.predict_proba(X))[:, 1]  # type: ignore
