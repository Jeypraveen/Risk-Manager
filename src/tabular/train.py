"""
LightGBM training for the tabular risk scorer.

Trains a binary classifier on return metadata to produce a risk score.
The score is a RANKING score, not a calibrated probability — thresholds
for operational use would be tuned on real merchant data.

Usage:
    python -m src.tabular.train
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import MODELS_DIR, SYNTHETIC_DIR, EVAL_RESULTS_DIR, RANDOM_SEED
from src.tabular.features import prepare_features, get_feature_columns


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation sets."""
    train_df = pd.read_csv(SYNTHETIC_DIR / "train.csv")
    val_df = pd.read_csv(SYNTHETIC_DIR / "val.csv")
    return train_df, val_df


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> lgb.LGBMClassifier:
    """
    Train LightGBM with hyperparameter search.

    Uses RandomizedSearchCV with a small search space (20 trials)
    to find reasonable hyperparameters without overfitting.
    """
    # Base model
    # NOTE: n_jobs=1 here is intentional. RandomizedSearchCV below already
    # parallelizes across CV folds/candidates (n_jobs=-1). If the inner
    # model ALSO parallelizes (n_jobs=-1), you get nested parallelism —
    # joblib forks one process per core, and each forked process tries to
    # spin up its own OpenMP thread pool. On Linux (including inside Docker
    # containers, even on Windows/WSL2 hosts), this fork+OpenMP combination
    # can deadlock permanently. Keeping the inner model single-threaded and
    # letting the search own the parallelism avoids this entirely.
    base_model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbose=-1,
        # Handle class imbalance
        is_unbalance=True,
    )

    # Hyperparameter search space
    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, -1],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [5, 10, 20, 50],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [0, 0.01, 0.1, 1.0],
    }

    # Randomized search — 20 trials, scored on average precision (PR-AUC)
    search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=20,
        scoring="average_precision",
        cv=3,
        random_state=RANDOM_SEED,
        verbose=1,
        n_jobs=4,
    )

    print("Running hyperparameter search (20 trials, 3-fold CV)...")
    search.fit(X_train, y_train)  # type: ignore

    best_model = search.best_estimator_
    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV average precision: {search.best_score_:.4f}")

    return best_model


def evaluate_model(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    split_name: str,
) -> dict:
    """Evaluate the model and return metrics."""
    y_prob = np.asarray(model.predict_proba(X))[:, 1]

    # Metrics
    pr_auc = average_precision_score(y, y_prob)
    roc_auc = roc_auc_score(y, y_prob)

    # Precision-recall at various thresholds
    precision, recall, thresholds = precision_recall_curve(y, y_prob)

    # Find best F1 threshold (for reference, not for deployment)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)

    metrics = {
        "split": split_name,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "best_f1": float(f1_scores[best_f1_idx]),
        "best_f1_threshold": float(thresholds[best_f1_idx]) if best_f1_idx < len(thresholds) else 0.5,
        "best_f1_precision": float(precision[best_f1_idx]),
        "best_f1_recall": float(recall[best_f1_idx]),
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "fraud_rate": float(y.mean()),
    }

    print(f"\n{'='*50}")
    print(f"  {split_name} Evaluation")
    print(f"{'='*50}")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Best F1:   {f1_scores[best_f1_idx]:.4f} @ threshold={metrics['best_f1_threshold']:.3f}")
    print(f"  -> Precision: {precision[best_f1_idx]:.4f}, Recall: {recall[best_f1_idx]:.4f}")

    return metrics


def plot_feature_importance(
    model: lgb.LGBMClassifier,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Save feature importance plot."""
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(sorted_idx)), importance[sorted_idx], color="#2196F3")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Feature Importance (split count)")
    ax.set_title("LightGBM Feature Importance — Tabular Risk Scorer")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved to {output_path}")


def plot_pr_curve(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    output_path: Path,
) -> None:
    """Save precision-recall curve."""
    y_prob = np.asarray(model.predict_proba(X))[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#FF5722", linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.axhline(y=y.mean(), color="gray", linestyle="--", alpha=0.7, label=f"Baseline (fraud rate = {y.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {split_name}")
    ax.legend(loc="upper right")
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PR curve saved to {output_path}")


def main():
    print("=" * 60)
    print("  Tabular Risk Scorer — Training Pipeline")
    print("=" * 60)

    # Load data
    train_df, val_df = load_data()
    print(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples")

    # Prepare features
    # Compute medians from training data to avoid single-row inference bugs and ensure validation matches production
    category_medians = train_df.groupby("item_category")["order_value"].median().to_dict()
    train_df, feature_cols = prepare_features(train_df, category_medians=category_medians)
    val_df, _ = prepare_features(val_df, category_medians=category_medians)

    X_train = np.asarray(train_df[feature_cols].values)
    y_train = np.asarray(train_df["is_fraud"], dtype=int)
    X_val = np.asarray(val_df[feature_cols].values)
    y_val = np.asarray(val_df["is_fraud"], dtype=int)

    print(f"Features: {len(feature_cols)}")
    print(f"Train fraud rate: {y_train.mean():.3f}")
    print(f"Val fraud rate:   {y_val.mean():.3f}")

    # Train
    model = train_model(X_train, y_train, X_val, y_val, feature_cols)

    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, feature_cols, "TRAIN")
    val_metrics = evaluate_model(model, X_val, y_val, feature_cols, "VALIDATION")

    # Save model — include category medians for correct inference-time feature engineering
    import joblib
    model_path = MODELS_DIR / "tabular_scorer.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "category_medians": category_medians,
        },
        model_path,
    )
    print(f"\nModel saved to {model_path}")
    print(f"Category medians saved: {category_medians}")

    # Save metrics
    metrics_path = EVAL_RESULTS_DIR / "tabular_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"train": train_metrics, "validation": val_metrics}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Plots
    plot_feature_importance(model, feature_cols, EVAL_RESULTS_DIR / "feature_importance.png")
    plot_pr_curve(model, X_val, y_val, "Validation", EVAL_RESULTS_DIR / "pr_curve_tabular_val.png")

    # Sanity check: is the model too good or too bad?
    pr_auc = val_metrics["pr_auc"]
    if pr_auc > 0.95:
        print("\n[WARNING] PR-AUC > 0.95 -- synthetic data may be too easy.")
        print("   Consider adding more noise to the data generator.")
    elif pr_auc < 0.20:
        print("\n[WARNING] PR-AUC < 0.20 -- features may not be informative enough.")
        print("   Consider adjusting feature distributions in the generator.")
    else:
        print(f"\n[OK] Validation PR-AUC ({pr_auc:.4f}) is in a realistic range.")


if __name__ == "__main__":
    main()
