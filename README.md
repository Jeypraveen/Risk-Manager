# Razorpay AI Buildathon 2026: Return-Risk Scorer

![Risk Manager](https://img.shields.io/badge/Track-AI%20Risk%20Manager-blue)
![Status](https://img.shields.io/badge/Status-Complete-success)

A defense-only, production-grade architecture AI Risk Manager (with demo-scale validation) designed to stop e-commerce return fraud (empty-box, substitution, wardrobing) through a combination of behavioral tabular scoring and visual verification.

Built for **Razorpay AI Buildathon 2026, Track 02**.

## 🚀 Quick Start (Demo)

This project includes a fully functional FastAPI backend and a beautiful glassmorphism HTML/JS dashboard.

1. **Activate the virtual environment:**
   ```powershell
   .\venv\Scripts\Activate
   ```

2. **Start the API Server:**
   ```powershell
   python src/api/server.py
   ```

3. **Open the Demo UI:**
   Simply open `demo/index.html` in any web browser. No frontend build step required!

## 🧠 System Architecture

This system uses a **Late-Fusion Meta-Learner** architecture to ensure graceful degradation when photos are missing or vision services fail.

1. **Tabular Scorer (LightGBM):** Analyzes behavioral metadata (account age, return velocity, COD status, etc.).
2. **Vision Pipeline (DINOv2 + Heuristics):** Analyzes return photos for substitution and empty-box fraud.
3. **Circuit Breaker:** Wraps the vision pipeline to gracefully handle image upload timeouts or model checkpoint mismatches.
4. **Meta-Learner:** Fuses the tabular and vision signals based on `modality_confidence`.
5. **Three-Way Router:** Routes the final trust score to `AUTO_APPROVE`, `NUDGE`, or `MANUAL_REVIEW`.
6. **Audit Trail:** Logs every decision, score, and failure for compliance.

Read the full details in [docs/architecture.md](docs/architecture.md).

## 📊 Honest Metrics & Boundaries

As per the prompt guidelines, we present honest metrics that consider false-positive costs. The system was evaluated on a held-out test set of 2,000 synthetic return requests with an 8% base fraud rate.

*   **Test PR-AUC (Tabular):** `0.447`
*   **Test PR-AUC (Fusion):** `0.858` (using synthetic vision signals)
*   **Cost-Optimal Threshold:** `0.788` (Precision: `0.635`, Recall: `0.338`)

For full transparency regarding what these synthetic metrics *do* and *do not* prove, please read our [Validity Boundaries](evaluation/results/validity_boundaries.md) document.

## 🛡️ Strict Defense-Only

This system adheres strictly to the "defense-only" requirement:
*   **No Auto-Denials:** The system *never* automatically denies a return. The lowest trust scores are safely routed to `MANUAL_REVIEW` for human intervention.
*   **No Attack Vector Generation:** We do not employ GANs or adversarial image detection, as building those models requires training offensive capabilities.
*   **Generic Nudges:** When requesting photos, the system uses generic messaging so fraudsters cannot reverse-engineer the detection logic.

## 📁 Repository Structure

*   `src/`: Core logic (config, tabular, vision, fusion, recovery, api, audit).
*   `data/`: Synthetic data generator (`generate_data.py`).
*   `evaluation/`: Evaluation scripts and resulting plots.
*   `demo/`: HTML/CSS/JS for the interactive dashboard.
*   `docs/`: Additional architecture documentation.
*   `models/`: Saved joblib checkpoints for LightGBM and Logistic Regression.
