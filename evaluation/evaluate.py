"""
Evaluation pipeline for the Return-Risk Scorer.

Runs the full pipeline on the held-out test set and produces:
  1. Tabular-only metrics (PR-AUC, precision, recall at cost-optimal threshold)
  2. Cost-weighted threshold optimization
  3. Cost sensitivity analysis
  4. Three-way decision distribution
  5. All plots committed to evaluation/results/

This is the ONLY time the test set is used.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    C_FP,
    C_FN,
    SYNTHETIC_DIR,
    EVAL_RESULTS_DIR,
    THRESHOLD_AUTO_APPROVE,
    THRESHOLD_MANUAL_REVIEW,
)
from src.tabular.predict import TabularScorer
from src.tabular.features import prepare_features


def load_test_data() -> pd.DataFrame:
    """Load the sacred held-out test set."""
    return pd.read_csv(SYNTHETIC_DIR / "test.csv")


def cost_weighted_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    c_fp: float = C_FP,
    c_fn: float = C_FN,
) -> tuple[float, float, dict]:
    """
    Find the threshold that minimizes expected cost.

    Expected_Cost(t) = FP(t) * C_fp + FN(t) * C_fn

    Args:
        y_true: Ground truth (1 = fraud)
        y_scores: Fraud probability scores (higher = more likely fraud)
        c_fp: Cost of a false positive (flagging a legit customer)
        c_fn: Cost of a false negative (missing a fraud case)

    Returns:
        (optimal_threshold, min_cost, metrics_at_threshold)
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    costs = []

    for t in thresholds:
        predictions = (y_scores >= t).astype(int)
        fp = ((predictions == 1) & (y_true == 0)).sum()
        fn = ((predictions == 0) & (y_true == 1)).sum()
        cost = fp * c_fp + fn * c_fn
        costs.append(cost)

    costs = np.array(costs)
    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    min_cost = costs[best_idx]

    # Metrics at optimal threshold
    predictions = (y_scores >= best_threshold).astype(int)
    tp = ((predictions == 1) & (y_true == 1)).sum()
    fp = ((predictions == 1) & (y_true == 0)).sum()
    fn = ((predictions == 0) & (y_true == 1)).sum()
    tn = ((predictions == 0) & (y_true == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    metrics = {
        "optimal_threshold": float(best_threshold),
        "min_expected_cost": float(min_cost),
        "precision_at_threshold": float(precision),
        "recall_at_threshold": float(recall),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "c_fp": c_fp, "c_fn": c_fn,
    }

    return best_threshold, min_cost, metrics


def sensitivity_analysis(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    c_fp_base: float = C_FP,
    c_fn_base: float = C_FN,
) -> list[dict]:
    """
    Show how the optimal threshold shifts when C_fp/C_fn ratio changes.

    Tests: 0.25x, 0.5x, 1x, 2x, 4x of the base C_fp/C_fn ratio.
    """
    multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
    results = []

    for mult in multipliers:
        adjusted_c_fp = c_fp_base * mult
        threshold, cost, metrics = cost_weighted_threshold(
            y_true, y_scores, adjusted_c_fp, c_fn_base
        )
        results.append({
            "c_fp_multiplier": mult,
            "c_fp": adjusted_c_fp,
            "c_fn": c_fn_base,
            "ratio": adjusted_c_fp / c_fn_base,
            **metrics,
        })

    return results


def compute_decision_distribution(
    trust_scores: np.ndarray,
    threshold_approve: float = THRESHOLD_AUTO_APPROVE,
    threshold_review: float = THRESHOLD_MANUAL_REVIEW,
) -> dict:
    """Compute three-way decision distribution."""
    n = len(trust_scores)
    auto_approve = (trust_scores >= threshold_approve).sum()
    manual_review = (trust_scores < threshold_review).sum()
    nudge = n - auto_approve - manual_review

    return {
        "total": n,
        "auto_approve": int(auto_approve),
        "auto_approve_pct": float(auto_approve / n),
        "nudge": int(nudge),
        "nudge_pct": float(nudge / n),
        "manual_review": int(manual_review),
        "manual_review_pct": float(manual_review / n),
    }


# ──────────────────────────────────────────────
# Plotting functions
# ──────────────────────────────────────────────

def plot_pr_curve(y_true, y_scores, pr_auc, output_path):
    """Precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#FF5722", linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.axhline(y=y_true.mean(), color="gray", linestyle="--", alpha=0.7,
               label=f"Baseline (fraud rate = {y_true.mean():.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve -- Tabular Risk Scorer (Test Set)", fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cost_curve(y_true, y_scores, c_fp, c_fn, optimal_threshold, output_path):
    """Expected cost vs. threshold."""
    thresholds = np.linspace(0.01, 0.99, 200)
    costs = []
    for t in thresholds:
        predictions = (y_scores >= t).astype(int)
        fp = ((predictions == 1) & (y_true == 0)).sum()
        fn = ((predictions == 0) & (y_true == 1)).sum()
        costs.append(fp * c_fp + fn * c_fn)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, costs, color="#2196F3", linewidth=2)
    ax.axvline(x=optimal_threshold, color="#FF5722", linestyle="--", linewidth=1.5,
               label=f"Cost-optimal threshold = {optimal_threshold:.3f}")
    ax.set_xlabel("Fraud Detection Threshold", fontsize=12)
    ax.set_ylabel(f"Expected Cost (C_fp={c_fp}, C_fn={c_fn})", fontsize=12)
    ax.set_title("Cost-Weighted Threshold Optimization", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sensitivity(sensitivity_results, output_path):
    """Sensitivity analysis — threshold vs. cost ratio."""
    ratios = [r["ratio"] for r in sensitivity_results]
    thresholds = [r["optimal_threshold"] for r in sensitivity_results]
    precisions = [r["precision_at_threshold"] for r in sensitivity_results]
    recalls = [r["recall_at_threshold"] for r in sensitivity_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(ratios, thresholds, "o-", color="#FF5722", linewidth=2, markersize=8)
    ax1.set_xlabel("C_fp / C_fn Ratio", fontsize=12)
    ax1.set_ylabel("Optimal Threshold", fontsize=12)
    ax1.set_title("How Threshold Shifts with Cost Ratio", fontsize=14)
    ax1.set_xscale("log")

    ax2.plot(ratios, precisions, "s-", color="#2196F3", linewidth=2, markersize=8, label="Precision")
    ax2.plot(ratios, recalls, "^-", color="#4CAF50", linewidth=2, markersize=8, label="Recall")
    ax2.set_xlabel("C_fp / C_fn Ratio", fontsize=12)
    ax2.set_ylabel("Metric Value", fontsize=12)
    ax2.set_title("Precision-Recall Tradeoff vs. Cost Ratio", fontsize=14)
    ax2.set_xscale("log")
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_decision_distribution(dist, output_path):
    """Three-way decision distribution pie chart."""
    labels = ["Auto-Approve", "Nudge", "Manual Review"]
    sizes = [dist["auto_approve_pct"], dist["nudge_pct"], dist["manual_review_pct"]]
    colors = ["#4CAF50", "#FFC107", "#F44336"]
    explode = (0.03, 0.03, 0.03)

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(  # type: ignore
        sizes, explode=explode, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 12}
    )
    ax.set_title("Three-Way Decision Distribution (Test Set)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60)
    print("  Return-Risk Scorer -- Full Evaluation Pipeline")
    print("  (Test set used for the FIRST and ONLY time)")
    print("=" * 60)

    # Load test data
    test_df = load_test_data()
    print(f"Test set: {len(test_df)} samples, fraud rate: {test_df['is_fraud'].mean():.3f}")

    # Load scorer
    scorer = TabularScorer()

    # Get trust scores (higher = more legitimate)
    trust_scores = scorer.predict_batch(test_df)
    # For evaluation, we need fraud probability (1 - trust_score)
    fraud_probs = 1.0 - trust_scores
    y_true = np.asarray(test_df["is_fraud"], dtype=int)

    # PR-AUC
    pr_auc = average_precision_score(y_true, fraud_probs)
    print(f"\nPR-AUC (tabular-only): {pr_auc:.4f}")

    # Cost-weighted threshold
    opt_threshold, min_cost, cost_metrics = cost_weighted_threshold(y_true, fraud_probs)
    print(f"\nCost-optimal threshold: {opt_threshold:.3f}")
    print(f"  Precision: {cost_metrics['precision_at_threshold']:.4f}")
    print(f"  Recall:    {cost_metrics['recall_at_threshold']:.4f}")
    print(f"  Min cost:  {min_cost:.0f}")

    # Sensitivity analysis
    sensitivity = sensitivity_analysis(y_true, fraud_probs)
    print("\nSensitivity Analysis:")
    for r in sensitivity:
        print(f"  C_fp/C_fn = {r['ratio']:.2f} -> threshold = {r['optimal_threshold']:.3f}, "
              f"P = {r['precision_at_threshold']:.3f}, R = {r['recall_at_threshold']:.3f}")

    # Decision distribution (using trust scores)
    dist = compute_decision_distribution(trust_scores)
    print(f"\nDecision Distribution:")
    print(f"  Auto-approve:  {dist['auto_approve']} ({dist['auto_approve_pct']:.1%})")
    print(f"  Nudge:         {dist['nudge']} ({dist['nudge_pct']:.1%})")
    print(f"  Manual review: {dist['manual_review']} ({dist['manual_review_pct']:.1%})")

    # ── Generate plots ──
    plot_pr_curve(y_true, fraud_probs, pr_auc,
                  EVAL_RESULTS_DIR / "pr_curve_tabular.png")
    plot_cost_curve(y_true, fraud_probs, C_FP, C_FN, opt_threshold,
                    EVAL_RESULTS_DIR / "cost_vs_threshold.png")
    plot_sensitivity(sensitivity,
                     EVAL_RESULTS_DIR / "cost_sensitivity.png")
    plot_decision_distribution(dist,
                               EVAL_RESULTS_DIR / "decision_distribution.png")

    # ── Save all metrics ──
    all_metrics = {
        "tabular_only": {
            "pr_auc": float(pr_auc),
            "n_test_samples": len(y_true),
            "n_positive": int(y_true.sum()),
            "fraud_rate": float(y_true.mean()),
        },
        "cost_optimization": cost_metrics,
        "sensitivity_analysis": sensitivity,
        "decision_distribution": dist,
    }

    metrics_path = EVAL_RESULTS_DIR / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nAll plots and metrics saved to {EVAL_RESULTS_DIR}")
    print("[OK] Evaluation complete.")


if __name__ == "__main__":
    main()
