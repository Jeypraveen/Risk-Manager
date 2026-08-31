"""
Fusion model evaluation pipeline.

Evaluates the full late-fusion pipeline (tabular + vision meta-learner)
on the held-out test set and appends fusion metrics to metrics_summary.json.

Since the test set has no real images (the vision pipeline requires torch +
DINOv2 which may not be installed), this script uses the same synthetic
vision signal injection strategy as train_meta_learner.py — signals are
calibrated to fraud labels to produce a realistic upper-bound evaluation.

This is clearly documented as "synthetic vision, real tabular" evaluation.
The tabular-only PR-AUC in metrics_summary.json remains the primary metric.
The fusion PR-AUC shows the potential uplift when real vision signals are available.

Usage:
    python -m scripts.evaluate_fusion

Dependencies:
    - Run `python data/generate_data.py` first
    - Run `python -m src.tabular.train` first
    - Run `python -m scripts.train_meta_learner` first
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    SYNTHETIC_DIR,
    EVAL_RESULTS_DIR,
    MODELS_DIR,
    THRESHOLD_AUTO_APPROVE,
    THRESHOLD_MANUAL_REVIEW,
    C_FP,
    C_FN,
    RANDOM_SEED,
)
from src.tabular.predict import TabularScorer
from src.fusion.meta_learner import MetaLearner


def inject_synthetic_vision(
    labels: np.ndarray,
    rng: np.random.Generator,
    photo_rate: float = 0.70,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic vision signals calibrated to fraud labels.

    This mirrors the strategy used in train_meta_learner.py so the
    evaluation and training distributions are consistent.

    Returns (semantic_similarities, empty_box_flags, modality_confidences)
    """
    n = len(labels)
    modality_confidences = np.zeros(n)
    semantic_similarities = np.full(n, 0.5)
    empty_box_flags = np.zeros(n, dtype=float)

    has_photo = rng.random(n) < photo_rate
    modality_confidences[has_photo] = 1.0

    is_fraud = labels.astype(bool)
    photo_legit = has_photo & ~is_fraud
    photo_fraud = has_photo & is_fraud

    if photo_legit.sum() > 0:
        semantic_similarities[photo_legit] = np.clip(
            rng.beta(8, 2, size=photo_legit.sum()), 0.3, 1.0
        )

    if photo_fraud.sum() > 0:
        n_fp = photo_fraud.sum()
        is_empty = rng.random(n_fp) < 0.40
        semantic_similarities[photo_fraud] = np.where(
            is_empty,
            np.clip(rng.beta(2, 8, size=n_fp), 0.0, 0.6),
            np.clip(rng.beta(3, 6, size=n_fp), 0.1, 0.75),
        )
        empty_box_flags[photo_fraud] = is_empty.astype(float)

    return semantic_similarities, empty_box_flags, modality_confidences


def cost_weighted_threshold(
    y_true: np.ndarray,
    fraud_scores: np.ndarray,
    c_fp: float = C_FP,
    c_fn: float = C_FN,
) -> tuple[float, dict]:
    thresholds = np.linspace(0.01, 0.99, 200)
    costs = []
    for t in thresholds:
        pred = (fraud_scores >= t).astype(int)
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        costs.append(fp * c_fp + fn * c_fn)
    costs = np.array(costs)
    best_idx = np.argmin(costs)
    t_opt = float(thresholds[best_idx])
    pred = (fraud_scores >= t_opt).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp_n = ((pred == 1) & (y_true == 0)).sum()
    fn_n = ((pred == 0) & (y_true == 1)).sum()
    p = tp / (tp + fp_n) if (tp + fp_n) > 0 else 0.0
    r = tp / (tp + fn_n) if (tp + fn_n) > 0 else 0.0
    return t_opt, {
        "optimal_threshold": t_opt,
        "min_expected_cost": float(costs[best_idx]),
        "precision": float(p),
        "recall": float(r),
        "tp": int(tp), "fp": int(fp_n), "fn": int(fn_n),
    }


def plot_fusion_pr_curve(
    y_true: np.ndarray,
    tabular_scores: np.ndarray,
    fusion_scores: np.ndarray,
    tabular_auc: float,
    fusion_auc: float,
    output_path: Path,
) -> None:
    """PR curve comparing tabular-only vs full fusion."""
    fig, ax = plt.subplots(figsize=(9, 6))

    p_tab, r_tab, _ = precision_recall_curve(y_true, 1 - tabular_scores)
    ax.plot(r_tab, p_tab, color="#2196F3", linewidth=2,
            label=f"Tabular-only  PR-AUC={tabular_auc:.4f}")

    p_fus, r_fus, _ = precision_recall_curve(y_true, 1 - fusion_scores)
    ax.plot(r_fus, p_fus, color="#FF5722", linewidth=2, linestyle="--",
            label=f"Fusion (tabular + vision*)  PR-AUC={fusion_auc:.4f}")

    baseline = y_true.mean()
    ax.axhline(y=baseline, color="gray", linestyle=":", alpha=0.7,
               label=f"Baseline (fraud rate={baseline:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Tabular-Only vs Fusion PR Curve (Test Set)\n"
                 "*Vision signals are synthetic (real DINOv2 not required)", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def compute_decision_distribution(
    trust_scores: np.ndarray,
    approve_threshold: float = THRESHOLD_AUTO_APPROVE,
    review_threshold: float = THRESHOLD_MANUAL_REVIEW,
) -> dict:
    n = len(trust_scores)
    n_approve = int((trust_scores >= approve_threshold).sum())
    n_review = int((trust_scores < review_threshold).sum())
    n_nudge = n - n_approve - n_review
    return {
        "total": n,
        "auto_approve": n_approve, "auto_approve_pct": n_approve / n,
        "nudge": n_nudge, "nudge_pct": n_nudge / n,
        "manual_review": n_review, "manual_review_pct": n_review / n,
    }


def main():
    print("=" * 60)
    print("  Fusion Evaluation Pipeline")
    print("  (Test set used for the FIRST and ONLY time here)")
    print("=" * 60)

    # ── Load test set ──
    test_path = SYNTHETIC_DIR / "test.csv"
    if not test_path.exists():
        print(f"[ERROR] Test set not found: {test_path}")
        print("  Run: python data/generate_data.py")
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    y_true = np.asarray(test_df["is_fraud"], dtype=int)
    n = len(test_df)
    print(f"\nTest set: {n} samples, fraud rate: {y_true.mean():.3f}")

    # ── Tabular scores ──
    try:
        scorer = TabularScorer()
    except FileNotFoundError:
        print("[ERROR] Tabular model not found.")
        print("  Run: python -m src.tabular.train")
        sys.exit(1)

    print("\nScoring test set (tabular)...")
    tabular_scores = scorer.predict_batch(test_df).astype(float)
    tabular_pr_auc = float(average_precision_score(y_true, 1 - tabular_scores))
    print(f"  Tabular PR-AUC: {tabular_pr_auc:.4f}")

    # ── Load meta-learner ──
    ml = MetaLearner()
    if not ml.is_trained:
        print("[ERROR] Meta-learner not trained.")
        print("  Run: python -m scripts.train_meta_learner")
        sys.exit(1)

    # ── Inject synthetic vision signals ──
    # Use a different seed from training (seed=99) so test vision signals
    # are NOT identical to training/val signals — more realistic evaluation.
    rng = np.random.default_rng(99)
    sim_sims, empty_flags, mod_confs = inject_synthetic_vision(y_true, rng)

    print(f"\nSynthetic vision signals (test set, seed=99):")
    has_photo = mod_confs > 0
    print(f"  Samples with photo: {has_photo.sum()}/{n} ({has_photo.mean():.1%})")
    legit_mask = (y_true == 0) & has_photo
    fraud_mask = (y_true == 1) & has_photo
    if legit_mask.sum() > 0:
        print(f"  Legit sim mean:  {sim_sims[legit_mask].mean():.3f}")
    if fraud_mask.sum() > 0:
        print(f"  Fraud sim mean:  {sim_sims[fraud_mask].mean():.3f}")
        print(f"  Fraud empty_box: {empty_flags[fraud_mask].mean():.1%}")

    # ── Fusion scores ──
    print("\nComputing fusion trust scores...")
    fusion_scores = ml.predict_batch(
        tabular_scores=tabular_scores,
        semantic_similarities=sim_sims,
        empty_box_flags=empty_flags,
        modality_confidences=mod_confs,
    )
    fusion_pr_auc = float(average_precision_score(y_true, 1 - fusion_scores))
    print(f"  Fusion  PR-AUC: {fusion_pr_auc:.4f}")
    uplift = fusion_pr_auc - tabular_pr_auc
    print(f"  Uplift vs tabular-only: {uplift:+.4f}")

    # ── Cost-weighted thresholds ──
    tab_threshold, tab_cost_metrics = cost_weighted_threshold(y_true, 1 - tabular_scores)
    fus_threshold, fus_cost_metrics = cost_weighted_threshold(y_true, 1 - fusion_scores)
    print(f"\nCost-optimal threshold:")
    print(f"  Tabular: {tab_threshold:.3f}  P={tab_cost_metrics['precision']:.3f}  R={tab_cost_metrics['recall']:.3f}")
    print(f"  Fusion:  {fus_threshold:.3f}  P={fus_cost_metrics['precision']:.3f}  R={fus_cost_metrics['recall']:.3f}")

    # ── Decision distribution (fusion) ──
    dist = compute_decision_distribution(fusion_scores)
    print(f"\nFusion Decision Distribution:")
    print(f"  Auto-approve:  {dist['auto_approve']} ({dist['auto_approve_pct']:.1%})")
    print(f"  Nudge:         {dist['nudge']} ({dist['nudge_pct']:.1%})")
    print(f"  Manual review: {dist['manual_review']} ({dist['manual_review_pct']:.1%})")

    # ── PR curve plot ──
    print("\nGenerating fusion PR curve plot...")
    plot_fusion_pr_curve(
        y_true=y_true,
        tabular_scores=tabular_scores,
        fusion_scores=fusion_scores,
        tabular_auc=tabular_pr_auc,
        fusion_auc=fusion_pr_auc,
        output_path=EVAL_RESULTS_DIR / "pr_curve_fusion.png",
    )

    # ── Save metrics (append to existing metrics_summary.json) ──
    metrics_path = EVAL_RESULTS_DIR / "metrics_summary.json"
    all_metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            all_metrics = json.load(f)

    all_metrics["fusion"] = {
        "pr_auc": fusion_pr_auc,
        "pr_auc_uplift_vs_tabular": uplift,
        "vision_signal_source": "synthetic (calibrated to fraud labels, seed=99)",
        "n_test_samples": n,
        "n_positive": int(y_true.sum()),
        "fraud_rate": float(y_true.mean()),
        "photo_availability": float(has_photo.mean()),
        "cost_optimization": fus_cost_metrics,
        "decision_distribution": dist,
        "note": (
            "Vision signals are synthetic because real images + DINOv2 are not "
            "required for the demo. Fusion PR-AUC represents the upper-bound "
            "potential when real vision signals are available. "
            "Tabular-only PR-AUC is the primary production metric."
        ),
    }

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nMetrics appended to: {metrics_path}")
    print("\n[SUMMARY]")
    print(f"  Tabular-only PR-AUC:  {tabular_pr_auc:.4f}")
    print(f"  Fusion PR-AUC:        {fusion_pr_auc:.4f}  ({uplift:+.4f} uplift)")
    print("\n[OK] Fusion evaluation complete.")


if __name__ == "__main__":
    main()
