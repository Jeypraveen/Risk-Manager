"""
Train the late-fusion meta-learner.

Runs the trained tabular scorer on the validation set to collect tabular scores,
then trains a logistic regression meta-learner that fuses them with vision signals.

Since the validation set has no real images, vision inputs are set to their
neutral defaults (modality_confidence=0.0, similarity=0.5, empty_box=0).
This is intentional: the meta-learner learns that when vision is unavailable
it should rely on the tabular score alone. When real images are available at
inference time, the non-zero modality_confidence activates the vision
coefficients correctly.

Usage:
    python -m scripts.train_meta_learner

Dependencies:
    - Run `python data/generate_data.py` first
    - Run `python -m src.tabular.train` first
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SYNTHETIC_DIR, MODELS_DIR, EVAL_RESULTS_DIR
from src.tabular.predict import TabularScorer
from src.fusion.meta_learner import MetaLearner


def main():
    print("=" * 60)
    print("  Late-Fusion Meta-Learner — Training")
    print("=" * 60)

    # ── Load validation set ──
    val_path = SYNTHETIC_DIR / "val.csv"
    if not val_path.exists():
        print(f"[ERROR] Validation set not found at {val_path}")
        print("  Run: python data/generate_data.py")
        sys.exit(1)

    val_df = pd.read_csv(val_path)
    print(f"Validation set: {len(val_df)} samples, fraud rate: {val_df['is_fraud'].mean():.3f}")

    # ── Tabular scores on validation set ──
    try:
        scorer = TabularScorer()
    except FileNotFoundError:
        print("[ERROR] Tabular model not found.")
        print("  Run: python -m src.tabular.train")
        sys.exit(1)

    print("\nScoring validation set with tabular model...")
    tabular_scores = scorer.predict_batch(val_df).astype(float)
    print(f"  Tabular scores — min: {tabular_scores.min():.3f}, max: {tabular_scores.max():.3f}, "
          f"mean: {tabular_scores.mean():.3f}")

    # ── Labels ──
    labels = np.asarray(val_df["is_fraud"], dtype=int)

    # ── Vision signals ──
    # For the val set, no real images exist. We inject SYNTHETIC vision signals
    # calibrated to fraud labels so the meta-learner learns meaningful vision
    # coefficients. Without this, all signals are 0.0 and the logistic regression
    # learns zero weights for vision features, making the fused model identical
    # to tabular-only.
    #
    # CALIBRATION NOTE: These distributions are calibrated to the MEASURED real
    # DINOv2 similarity scores produced by src.vision.similarity._rescale():
    #   Genuine matches:  0.39-0.97 (mean ~0.73)
    #   Fraud mismatches: 0.04-0.23 (mean ~0.14)
    # Using Beta(5,2) for legit (mean ~0.71) and Beta(1.5,9) for fraud
    # (mean ~0.14) matches the observed distribution. If SIM_BAND_LOW /
    # SIM_BAND_HIGH in similarity.py change, these must be re-fitted.
    rng = np.random.default_rng(42)
    n = len(val_df)

    modality_confidences = np.zeros(n)
    semantic_similarities = np.full(n, 0.5)
    empty_box_flags = np.zeros(n, dtype=float)

    # Use the has_return_photo column from the generated data (Issue #9)
    # instead of re-rolling a random coin. This column encodes realistic
    # fraud-subtype-correlated photo availability.
    if "has_return_photo" in val_df.columns:
        has_photo_mask = np.asarray(val_df["has_return_photo"], dtype=bool)
    else:
        # Fallback for older data files without this column
        has_photo_mask = rng.random(n) < 0.70
    modality_confidences[has_photo_mask] = 1.0

    is_fraud = labels.astype(bool)
    photo_and_legit = has_photo_mask & ~is_fraud
    photo_and_fraud = has_photo_mask & is_fraud

    # Legitimate returns with photo: similarity calibrated to measured DINOv2
    # range (genuine matches cluster 0.39-0.97, mean ~0.73)
    if photo_and_legit.sum() > 0:
        semantic_similarities[photo_and_legit] = np.clip(
            rng.beta(5, 2, size=photo_and_legit.sum()), 0.35, 0.98
        )
        empty_box_flags[photo_and_legit] = 0

    # Fraud returns with photo: low similarity matching measured mismatch cluster
    if photo_and_fraud.sum() > 0:
        n_fraud_photo = photo_and_fraud.sum()
        # 40% of fraud photos are empty-box (matching FRAUD_SUBTYPE_DIST)
        is_empty_sub = rng.random(n_fraud_photo) < 0.40
        fraud_sims = np.clip(rng.beta(1.5, 9, size=n_fraud_photo), 0.0, 0.35)
        semantic_similarities[photo_and_fraud] = fraud_sims
        empty_box_flags[photo_and_fraud] = is_empty_sub.astype(float)

    print(f"Injected synthetic vision signals:")
    print(f"  Samples with photo: {has_photo_mask.sum()} / {n} ({has_photo_mask.mean():.1%})")
    print(f"  Legit photo — mean similarity: {semantic_similarities[photo_and_legit].mean():.3f}")
    if photo_and_fraud.sum() > 0:
        print(f"  Fraud photo — mean similarity: {semantic_similarities[photo_and_fraud].mean():.3f}, "
              f"  empty_box rate: {empty_box_flags[photo_and_fraud].mean():.1%}")

    # ── Train meta-learner ──
    print("\nTraining meta-learner (logistic regression on 4 features)...")
    ml = MetaLearner()
    metrics = ml.train(
        tabular_scores=tabular_scores,
        semantic_similarities=semantic_similarities,
        empty_box_flags=empty_box_flags,
        modality_confidences=modality_confidences,
        labels=labels,
    )

    print(f"\nMeta-learner training accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Model saved to: {MODELS_DIR / 'meta_learner.joblib'}")

    # ── Quick sanity check ──
    trust_scores = ml.predict_batch(
        tabular_scores=tabular_scores,
        semantic_similarities=semantic_similarities,
        empty_box_flags=empty_box_flags,
        modality_confidences=modality_confidences,
    )
    print(f"\nTrust scores — min: {trust_scores.min():.3f}, max: {trust_scores.max():.3f}, "
          f"mean: {trust_scores.mean():.3f}")

    # Verify vision signals have effect: if we flip modality_confidence to 1.0
    # and set similarity low, trust score should drop for the same tabular score
    no_vision = ml.predict(0.7, modality_confidence=0.0)
    with_good_vision = ml.predict(0.7, semantic_similarity=0.9, modality_confidence=1.0, empty_box_flag=0)
    with_bad_vision = ml.predict(0.7, semantic_similarity=0.2, modality_confidence=1.0, empty_box_flag=1)

    print(f"\nVision impact check (tabular_score=0.70):")
    print(f"  No vision (modality_confidence=0.0):          trust={no_vision:.3f}")
    print(f"  Good vision (similarity=0.9, not empty):       trust={with_good_vision:.3f}")
    print(f"  Bad vision  (similarity=0.2, is_empty):        trust={with_bad_vision:.3f}")

    # Save training metrics
    meta_metrics_path = EVAL_RESULTS_DIR / "meta_learner_metrics.json"
    with open(meta_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {meta_metrics_path}")
    print("[OK] Meta-learner training complete.")


if __name__ == "__main__":
    main()