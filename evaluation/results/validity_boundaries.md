# Validity Boundaries

**Razorpay AI Buildathon 2026 - Track 02: AI Risk Manager**

This document outlines the strict methodological boundaries of this proof-of-concept. Following the Buildathon requirement for "honest metrics," this section explicitly details what the evaluation metrics *do* and *do not* prove.

## 1. Synthetic Data Generation Assumptions
The test set metrics (PR-AUC, cost-optimal thresholds) are derived from a **synthetic dataset**. While the data generation process is rooted in published Indian e-commerce statistics, it remains synthetic. 

*   **Fraud Rate:** Fixed at 8%. Real-world rates fluctuate by category, season, and platform.
*   **Feature Distributions:** We intentionally modeled heavy overlap between legitimate and fraudulent behavior. For example, not all high-value COD orders are fraud, and not all fast returns are fraudulent. However, real-world correlations between features (e.g., account age and return velocity) are likely more complex than our generative model captures.
*   **Vision Modality Availability:** Modeled at ~68%, assuming standard enforcement policies, but actual rates depend heavily on UI friction and product category.

**Boundary:** The absolute values of PR-AUC (e.g., 0.447) and precision/recall are **directional only**. They prove the system *can* learn the intended decision boundaries, but they do **not** guarantee identical performance on real merchant data.

## 2. Threshold Recalibration
The cost-optimal threshold (e.g., 0.788) was computed using a static estimate:
*   $C_{FP}$ (False Positive Cost): ₹1,200 (customer churn risk + review labor)
*   $C_{FN}$ (False Negative Cost): ₹2,300 (item value + shipping losses)

**Boundary:** A real merchant's optimal threshold would shift drastically based on their specific average order value and lifetime customer value. The system is designed to emit a *ranking score*, allowing merchants to tune the `THRESHOLD_AUTO_APPROVE` and `THRESHOLD_MANUAL_REVIEW` parameters dynamically based on their specific economic realities.

## 3. Vision Pipeline Limitations
The vision pipeline utilizes DINOv2 (ViT-S/14) for similarity and SAM2/Heuristics for empty-box detection.

*   **Camera Quality:** Assumes reasonably well-lit photos. Real-world photos are often blurry, poorly framed, or taken in low light.
*   **Adversarial Evasion:** This system does **not** employ Deepfake/AI-generated image detection. As per the "Strictly Defense-Only" guidelines, developing models to detect generative artifacts involves modeling offensive capabilities, which is out of scope. A determined fraudster using an AI-generated photo of a substitute item could bypass the vision check.
*   **Hardware:** Designed to run on CPU to meet the "No Budget/Solo Build" constraint. 

## 4. Why This Architecture?
Despite these boundaries, the **Late-Fusion Meta-Learner architecture** is highly robust:
1.  **Graceful Degradation:** If vision fails (missing photo, CDN timeout), the system seamlessly falls back to tabular scoring. It never crashes.
2.  **Safety First:** The system is completely defense-only. It never automatically denies a return; the lowest trust score merely routes to `MANUAL_REVIEW`.
3.  **Auditability:** Every decision logs exactly what was known at the time, providing a full trace for compliance and model debugging.
